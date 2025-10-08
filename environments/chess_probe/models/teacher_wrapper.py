"""Wrapper for DeepMind's JAX action-value model to extract hidden states.

This module provides a wrapper around the searchless_chess action value transformer
that extracts the hidden states (residual stream) before the final projection layer,
and converts them to PyTorch tensors for use in training the Qwen probe.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Callable, List, Optional, Union

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import torch

import sys
_REPO_ROOT = Path(__file__).resolve().parents[3]
_VENDOR_ROOT = _REPO_ROOT / "environments/chess_probe/vendor"
if str(_VENDOR_ROOT) not in sys.path:
    sys.path.append(str(_VENDOR_ROOT))

from searchless_chess.src import tokenizer as slc_tokenizer  # noqa: E402
from searchless_chess.src import transformer  # noqa: E402
from searchless_chess.src import training_utils  # noqa: E402
from searchless_chess.src import utils as slc_utils  # noqa: E402

import haiku as hk  # noqa: E402


def create_transformer_with_layer_extraction(hidden_layer_idx: Union[int, List[int], None] = None):
    """Create a transformer decoder that can extract hidden states from a specific layer.
    
    Args:
        hidden_layer_idx: Which layer to extract hidden states from (None = last layer)
    
    Returns:
        A function that returns (log_probs, hidden_states_at_layer)
    """
    def transformer_decoder_with_hidden_states(
        targets: jax.Array,
        config: transformer.TransformerConfig,
    ) -> tuple[jax.Array, jax.Array]:
        """Modified transformer decoder that returns logits and hidden states from specific layer.
        
        Returns:
            log_probs: [B, T, V] log probabilities from the final layer
            hidden_states: [B, T, H] hidden states from specified layer
        """
        inputs = transformer.shift_right(targets)
        embeddings = transformer.embed_sequences(inputs, config)
        
        h = embeddings
        hidden_to_return = None
        multi_layers: List[jax.Array] | None = None
        # Normalize to a set for fast membership if list provided
        layer_set = None
        if isinstance(hidden_layer_idx, list):
            layer_set = set(hidden_layer_idx)
        
        for layer_idx in range(config.num_layers):
            attention_input = transformer.layer_norm(h)
            attention = transformer._attention_block(attention_input, config)
            h += attention
            
            mlp_input = transformer.layer_norm(h)
            mlp_output = transformer._mlp_block(mlp_input, config)
            h += mlp_output
            
            # Save hidden state from target layer(s) (after residual)
            if hidden_layer_idx is not None:
                if layer_set is not None:
                    if layer_idx in layer_set:
                        if multi_layers is None:
                            multi_layers = []
                        multi_layers.append(h)
                elif layer_idx == hidden_layer_idx:
                    hidden_to_return = h
        
        if config.apply_post_ln:
            h = transformer.layer_norm(h)
        
        # If multiple layers requested, concatenate; if none, use final hidden
        if layer_set is not None:
            if multi_layers is None or len(multi_layers) == 0:
                hidden_to_return = h
            else:
                hidden_to_return = jnp.concatenate(multi_layers, axis=-1)
        else:
            if hidden_to_return is None:
                hidden_to_return = h
        
        logits = hk.Linear(config.output_size)(h)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        
        return log_probs, hidden_to_return
    
    return transformer_decoder_with_hidden_states


class ActionValueTeacher:
    """Wrapper for the action-value model that extracts hidden states.
    
    This class loads the pretrained 270M action value model and provides methods
    to extract hidden states for a given (board, move) pair.
    """
    
    def __init__(
        self,
        model_size: str = "270M",
        checkpoint_step: int = -1,
        use_ema: bool = True,
        hidden_layer_idx: Union[int, List[int], None] = None,
    ):
        """Initialize the action value teacher model.
        
        Args:
            model_size: Model size to load ("9M", "136M", or "270M")
            checkpoint_step: Which checkpoint step to load (-1 for latest)
            use_ema: Whether to use EMA parameters
            hidden_layer_idx: Which layer to extract hidden states from (None = last layer).
                Use -1 for last layer, -2 for second-to-last, etc.
        """
        self.model_size = model_size
        self.checkpoint_step = checkpoint_step
        self.use_ema = use_ema
        
        # Configure model architecture based on size
        if model_size == "9M":
            num_layers = 8
            embedding_dim = 256
            num_heads = 8
        elif model_size == "136M":
            num_layers = 8
            embedding_dim = 1024
            num_heads = 8
        elif model_size == "270M":
            num_layers = 16
            embedding_dim = 1024
            num_heads = 8
        else:
            raise ValueError(f"Unknown model size: {model_size}")
        
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        
        # Validate and set hidden layer index(es)
        resolved: Optional[Union[int, List[int]]] = None
        if hidden_layer_idx is None:
            print(f"✓ Teacher will extract hidden states from final layer (model has {num_layers} layers)")
            resolved = None
        elif isinstance(hidden_layer_idx, list):
            norm: List[int] = []
            for li in hidden_layer_idx:
                lj = li
                if lj < 0:
                    lj = num_layers + lj
                if lj < 0 or lj >= num_layers:
                    raise ValueError(
                        f"hidden_layer_idx={li} is out of range. Model has {num_layers} layers (0-{num_layers-1})."
                    )
                norm.append(lj)
            norm = sorted(set(norm))
            print(f"✓ Teacher will extract hidden states from layers {norm} (0-based)")
            resolved = norm
        else:
            lj = hidden_layer_idx
            if lj < 0:
                lj = num_layers + lj
            if lj < 0 or lj >= num_layers:
                raise ValueError(
                    f"hidden_layer_idx={hidden_layer_idx} is out of range. Model has {num_layers} layers (0-{num_layers-1})."
                )
            print(f"✓ Teacher will extract hidden states from layer {lj}/{num_layers-1}")
            resolved = lj
        
        self.hidden_layer_idx = resolved
        num_return_buckets = 128
        
        self.config = transformer.TransformerConfig(
            vocab_size=slc_utils.NUM_ACTIONS,
            output_size=num_return_buckets,
            pos_encodings=transformer.PositionalEncodings.LEARNED,
            max_sequence_length=slc_tokenizer.SEQUENCE_LENGTH + 2,
            num_heads=num_heads,
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            apply_post_ln=True,
            apply_qk_layernorm=False,
            use_causal_mask=False,
        )
        
        # Build the modified predictor that returns hidden states from specified layer
        transformer_fn = create_transformer_with_layer_extraction(self.hidden_layer_idx)
        model = hk.transform(
            functools.partial(transformer_fn, config=self.config)
        )
        
        # Load checkpoint
        checkpoint_dir = _VENDOR_ROOT / "searchless_chess" / "checkpoints" / model_size
        dummy_params = model.init(
            rng=jrandom.PRNGKey(1),
            targets=np.ones((1, 1), dtype=np.uint32),
        )
        
        self.params = training_utils.load_parameters(
            checkpoint_dir=str(checkpoint_dir),
            params=dummy_params,
            step=checkpoint_step,
            use_ema_params=use_ema,
        )
        
        self.apply_fn = model.apply
        
    def get_hidden_states(
        self,
        fen: str,
        move: str,
        override_layer_idxs: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Extract hidden states for a (board, move) pair.
        
        Args:
            fen: Board state in FEN notation
            move: Move in UCI format (e.g., "e2e4")
            
        Returns:
            PyTorch tensor of shape [embedding_dim] containing the hidden state
            at the final token position (the move position)
        """
        # Tokenize the FEN
        state_tokens = slc_tokenizer.tokenize(fen).astype(np.int32)
        
        # Tokenize the move
        action_idx = slc_utils.MOVE_TO_ACTION[move]
        action_token = np.array([action_idx], dtype=np.int32)
        
        # Dummy return bucket (not used for hidden state extraction)
        dummy_return = np.array([0], dtype=np.int32)
        
        # Concatenate into sequence: [state, action, return]
        sequence = np.concatenate([state_tokens, action_token, dummy_return])
        sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension [1, T]
        
        # Forward pass to get hidden states
        _, hidden_states = self.apply_fn(
            self.params,
            jrandom.PRNGKey(0),
            sequence,
        )
        
        # Extract hidden state at the action token position (second-to-last token)
        # Sequence is [state_tokens, action, return], we want the state at action position
        action_position = len(state_tokens)  # 0-indexed position of action token
        hidden_at_action = hidden_states[0, action_position, :]  # [embedding_dim]
        
        # Convert JAX array to PyTorch tensor
        hidden_np = np.array(hidden_at_action)
        hidden_torch = torch.from_numpy(hidden_np).float()
        
        return hidden_torch
    
    def get_hidden_states_batch(
        self,
        fens: list[str],
        moves: list[str],
        override_layer_idxs: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Extract hidden states for a batch of (board, move) pairs.
        
        Args:
            fens: List of board states in FEN notation
            moves: List of moves in UCI format
            
        Returns:
            PyTorch tensor of shape [batch_size, embedding_dim]
        """
        # Prepare batch of sequences
        batch_sequences = []
        for fen, move in zip(fens, moves):
            state_tokens = slc_tokenizer.tokenize(fen).astype(np.int32)
            action_idx = slc_utils.MOVE_TO_ACTION[move]
            action_token = np.array([action_idx], dtype=np.int32)
            dummy_return = np.array([0], dtype=np.int32)
            sequence = np.concatenate([state_tokens, action_token, dummy_return])
            batch_sequences.append(sequence)
        
        sequences = np.stack(batch_sequences, axis=0)  # [B, T]
        
        # Forward pass
        _, hidden_states = self.apply_fn(
            self.params,
            jrandom.PRNGKey(0),
            sequences,
        )
        
        # Extract hidden states at action position
        # All sequences have same structure, action is at position len(state_tokens)
        action_position = len(slc_tokenizer.tokenize(fens[0]))
        hidden_at_actions = hidden_states[:, action_position, :]  # [B, embedding_dim]
        
        # Convert to PyTorch
        hidden_np = np.array(hidden_at_actions)
        hidden_torch = torch.from_numpy(hidden_np).float()
        
        return hidden_torch
    
    def get_move_win_probs(self, fen: str) -> tuple[list[str], np.ndarray]:
        """Get win probabilities for all legal moves in a position.
        
        Args:
            fen: Board state in FEN notation
            
        Returns:
            moves: List of legal moves in UCI format (ordered)
            win_probs: Array of win probabilities for each move
        """
        import chess
        from searchless_chess.src.engines import engine as slc_engine
        
        board = chess.Board(fen)
        sorted_legal_moves = slc_engine.get_ordered_legal_moves(board)
        
        # Get action indices for all legal moves
        legal_actions = [slc_utils.MOVE_TO_ACTION[m.uci()] for m in sorted_legal_moves]
        legal_actions = np.array(legal_actions, dtype=np.int32)
        legal_actions = np.expand_dims(legal_actions, axis=-1)
        
        # Tokenize board state
        state_tokens = slc_tokenizer.tokenize(fen).astype(np.int32)
        
        # Create sequences for all legal moves: [state, action, dummy_return]
        dummy_returns = np.zeros((len(legal_actions), 1), dtype=np.int32)
        sequences = np.stack([state_tokens] * len(legal_actions))
        sequences = np.concatenate([sequences, legal_actions, dummy_returns], axis=1)
        
        # Get log probs for return buckets
        log_probs, _ = self.apply_fn(
            self.params,
            jrandom.PRNGKey(0),
            sequences,
        )
        
        # Extract log probs for last token (return bucket predictions)
        return_buckets_log_probs = log_probs[:, -1, :]  # [num_moves, num_buckets]
        
        # Convert to probabilities
        return_buckets_probs = np.exp(np.array(return_buckets_log_probs))
        
        # Get bucket values for computing expected win probability
        _, bucket_values = slc_utils.get_uniform_buckets_edges_values(128)
        
        # Compute win probability for each move
        win_probs = np.inner(return_buckets_probs, bucket_values)
        
        # Return moves as UCI strings
        moves_uci = [m.uci() for m in sorted_legal_moves]
        
        return moves_uci, win_probs
