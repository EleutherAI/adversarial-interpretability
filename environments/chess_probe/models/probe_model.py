"""Probe model that combines frozen Qwen with trainable linear probe.

This module defines:
1. A simple linear probe MLP that maps teacher hidden states to embeddings
2. A wrapper model that combines frozen Qwen with the trainable probe
3. Support for dual input modes: regular text OR text + teacher hidden states
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, List
import math
import torch.nn.functional as F


class LinearProbe(nn.Module):
    """Simple linear probe that maps teacher hidden states to Qwen embedding space."""
    
    def __init__(self, teacher_hidden_size: int, qwen_hidden_size: int):
        """Initialize the linear probe.
        
        Args:
            teacher_hidden_size: Dimension of teacher model hidden states
            qwen_hidden_size: Dimension of Qwen model embeddings
        """
        super().__init__()
        self.probe = nn.Linear(teacher_hidden_size, qwen_hidden_size)
        
    def forward(self, teacher_hidden: torch.Tensor) -> torch.Tensor:
        """Map teacher hidden states to Qwen embedding space.
        
        Args:
            teacher_hidden: [batch_size, teacher_hidden_size]
            
        Returns:
            Embeddings in Qwen space: [batch_size, qwen_hidden_size]
        """
        return self.probe(teacher_hidden)


class LowRankProbe(nn.Module):
    """Low-rank (LoRA-style) probe mapping teacher hidden → Qwen embedding space.
    
    Implements W ≈ B @ A with rank r to reduce parameters. Optionally applies
    LayerNorm on teacher hidden and a learned global scale.
    """
    
    def __init__(
        self,
        teacher_hidden_size: int,
        qwen_hidden_size: int,
        rank: int = 64,
        use_norm: bool = True,
        nonlinearity: Optional[str] = None,
        init_scale: float = 1.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(teacher_hidden_size) if use_norm else nn.Identity()
        self.A = nn.Linear(teacher_hidden_size, rank, bias=False)
        self.B = nn.Linear(rank, qwen_hidden_size, bias=True)
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        self._act = None
        if nonlinearity is None:
            self._act = None
        elif nonlinearity.lower() == "relu":
            self._act = nn.ReLU()
        elif nonlinearity.lower() == "tanh":
            self._act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")
        # Kaiming init for A; small bias for B
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.bias)
    
    def forward(self, teacher_hidden: torch.Tensor) -> torch.Tensor:
        h = self.norm(teacher_hidden)
        z = self.A(h)
        if self._act is not None:
            z = self._act(z)
        z = self.B(z)
        return self.scale * z


class QwenWithProbe(PreTrainedModel):
    """Wrapper model combining frozen Qwen with trainable probe.
    
    This model supports two forward modes:
    1. Normal mode: Standard text input, passes through Qwen normally
    2. Probe mode: Text input + teacher hidden states, where probe embedding
       is concatenated to the sequence before prediction
    
    The Qwen model is frozen; only the probe is trainable.
    """
    
    def __init__(
        self,
        qwen_model: AutoModelForCausalLM,
        teacher_hidden_size: int,
        freeze_qwen: bool = True,
        probe_layer_idx: Optional[int] = None,
        probe_token: str = " um",
        probe_type: str = "linear",
        lowrank_rank: int = 64,
        probe_use_norm: bool = True,
        probe_nonlinearity: Optional[str] = None,
        per_layer_scale: bool = True,
        static_rank_gate: bool = False,
        rank_gate_slots: Optional[int] = None,
    ):
        """Initialize the QwenWithProbe model.
        
        Args:
            qwen_model: Pretrained Qwen causal LM model
            teacher_hidden_size: Dimension of teacher hidden states
            freeze_qwen: Whether to freeze Qwen parameters (default: True)
            probe_layer_idx: Layer index to inject probe (None = concatenate to sequence)
                If specified, probe is added to residual stream at this layer,
                at the position of the probe_token.
                Use -1 for the last layer, -2 for second-to-last, etc.
            probe_token: Token string to use as injection point when using layer injection.
                Default: " um" (neutral filler word). Only used if probe_layer_idx is not None.
        """
        super().__init__(qwen_model.config)
        
        self.qwen = qwen_model
        self.qwen_hidden_size = qwen_model.config.hidden_size
        self.probe_token = probe_token
        
        # Get number of layers
        self.num_layers = qwen_model.config.num_hidden_layers
        
        # Validate and set single probe layer
        if probe_layer_idx is None:
            probe_layer_idx = -1
        li = probe_layer_idx
        if li < 0:
            li = self.num_layers + li
        if li < 0 or li >= self.num_layers:
            raise ValueError(
                f"probe_layer_idx={probe_layer_idx} is out of range. "
                f"Model has {self.num_layers} layers (indices 0-{self.num_layers-1})."
            )
        print(f"✓ Probe will inject at layer {li}/{self.num_layers-1} at token '{probe_token}'")
        self.probe_layer_idx: int = li
        
        # Initialize the trainable probe
        if probe_type not in {"linear", "lowrank"}:
            raise ValueError(f"Unsupported probe_type: {probe_type}")
        if probe_type == "linear":
            self.probe = LinearProbe(teacher_hidden_size, self.qwen_hidden_size)
        else:
            self.probe = LowRankProbe(
                teacher_hidden_size=teacher_hidden_size,
                qwen_hidden_size=self.qwen_hidden_size,
                rank=lowrank_rank,
                use_norm=probe_use_norm,
                nonlinearity=probe_nonlinearity,
                init_scale=1.0,
            )
        self.probe_type = probe_type
        self.lowrank_rank = lowrank_rank
        # Align probe weights dtype to Qwen to avoid bf16/float mismatches
        try:
            _model_dtype = next(self.qwen.parameters()).dtype
            self.probe.to(_model_dtype)
        except StopIteration:
            pass
        
        # Freeze Qwen if requested
        if freeze_qwen:
            for param in self.qwen.parameters():
                param.requires_grad = False
        
        # Only probe parameters (and optional layer scales) should be trainable
        for param in self.probe.parameters():
            param.requires_grad = True
        self.per_layer_scale = per_layer_scale
        if self.per_layer_scale:
            # One learned scalar for the single injection layer
            self.layer_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        else:
            self.register_parameter("layer_scale", None)

        # Static rank-wise gates per probe_index
        self.use_static_rank_gate = static_rank_gate and (self.probe_type == "lowrank")
        if self.use_static_rank_gate:
            if rank_gate_slots is None:
                raise ValueError("rank_gate_slots must be provided when static_rank_gate=True")
            if rank_gate_slots <= 0:
                raise ValueError("rank_gate_slots must be positive")
            self.rank_gate_slots = int(rank_gate_slots)
            # Initialize to ones (pass-through); learn deviations during training
            gate = torch.ones(self.rank_gate_slots, self.lowrank_rank, dtype=torch.float32)
            self.rank_gate = nn.Parameter(gate)
        else:
            self.rank_gate_slots = 0
            self.register_parameter("rank_gate", None)
        
        # Storage for injection state during forward pass
        self._probe_output = None
        self._probe_token_position = None
        self._active_hooks: List[torch.utils.hooks.RemovableHandle] = []
        # For static gating path
        self._probe_rank_output = None  # [B, r]
        self._probe_B_weight = None
        self._probe_B_bias = None
    
    def _create_injection_hook(self):
        """Create a forward hook that adds probe output to hidden states at probe token positions."""
        def hook(module, input, output):
            # Only inject if we have probe output and know where to inject
            if self._probe_output is None or self._probe_token_position is None:
                return output
            
            # output is typically a tuple (hidden_states, ...)
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Clone to avoid in-place modification
            modified_hidden = hidden_states.clone()
            
            # Add probe output at all probe token positions
            # probe_output: [B, H], probe_token_position: list of lists of position indices
            batch_size = hidden_states.shape[0]
            if not self.use_static_rank_gate:
                if self.per_layer_scale and self.layer_scale is not None:
                    # Scale for the single layer
                    scale_val = self.layer_scale.to(modified_hidden.dtype)
                    probe_to_add = self._probe_output * scale_val
                else:
                    probe_to_add = self._probe_output
                for b in range(batch_size):
                    positions = self._probe_token_position[b]
                    for pos in positions:
                        if 0 <= pos < hidden_states.shape[1]:
                            modified_hidden[b, pos, :] = modified_hidden[b, pos, :] + probe_to_add[b, :]
            else:
                # Static low-rank scaling per token position
                # Requirements: self._probe_rank_output [B, r], B weight/bias, rank_gate
                if self._probe_rank_output is None or self._probe_B_weight is None:
                    return output
                layer_scale_val = None
                if self.per_layer_scale and self.layer_scale is not None:
                    layer_scale_val = self.layer_scale.to(modified_hidden.dtype)
                # LowRankProbe overall scale
                probe_scale = self.probe.scale.to(modified_hidden.dtype) if hasattr(self.probe, "scale") else None
                for b in range(batch_size):
                    positions = self._probe_token_position[b]
                    # rank vector for this example
                    z_b = self._probe_rank_output[b, :]  # [r]
                    for pos_idx, pos in enumerate(positions):
                        if not (0 <= pos < hidden_states.shape[1]):
                            continue
                        scaled_rank = z_b
                        # Apply static rank-wise gate per (probe_index)
                        if self.use_static_rank_gate and self.rank_gate is not None and self.rank_gate_slots > 0:
                            gate_idx = pos_idx
                            if gate_idx >= self.rank_gate_slots:
                                gate_idx = self.rank_gate_slots - 1
                            gate_vec = self.rank_gate[gate_idx, :].to(modified_hidden.dtype)  # [r]
                            scaled_rank = scaled_rank * gate_vec
                        # Project via B: y = B(scaled_rank) + bias
                        y = F.linear(scaled_rank, self._probe_B_weight, self._probe_B_bias)  # [H]
                        if probe_scale is not None:
                            y = y * probe_scale
                        if layer_scale_val is not None:
                            y = y * layer_scale_val
                        modified_hidden[b, pos, :] = modified_hidden[b, pos, :] + y
            
            if isinstance(output, tuple):
                return (modified_hidden,) + output[1:]
            else:
                return modified_hidden
        
        return hook
    
    def _find_probe_token_positions(
        self,
        input_ids: torch.Tensor,
        tokenizer,
    ) -> list[list[int]]:
        """Find all positions of the probe token in each sequence.
        
        Returns list of lists: for each batch element, a list of position indices
        where the probe token appears. Empty list if probe token not found.
        """
        # Get the token ID for the probe token
        probe_token_ids = tokenizer.encode(self.probe_token, add_special_tokens=False)
        if len(probe_token_ids) == 0:
            raise ValueError(f"Probe token '{self.probe_token}' encodes to empty sequence")
        
        # For simplicity, use the first token if it tokenizes to multiple
        probe_token_id = probe_token_ids[0]
        
        batch_size, seq_len = input_ids.shape
        positions = []
        
        for b in range(batch_size):
            # Find all occurrences of probe token
            matches = (input_ids[b] == probe_token_id).nonzero(as_tuple=True)[0]
            positions.append(matches.tolist())
        
        return positions
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        teacher_hidden_states: Optional[torch.Tensor] = None,
        tokenizer=None,  # Required for layer injection to find probe token
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Forward pass through Qwen with optional probe injection.
        
        Args:
            input_ids: [batch_size, seq_len] token IDs
            attention_mask: [batch_size, seq_len] attention mask
            labels: [batch_size, seq_len] labels for language modeling loss
            teacher_hidden_states: Optional [batch_size, teacher_hidden_size]
                If provided, probe is injected based on probe_layer_idx
            tokenizer: Required when probe_layer_idx is not None, used to find probe token
            **kwargs: Additional arguments passed to Qwen
            
        Returns:
            CausalLMOutputWithPast with loss and logits
        """
        # If no teacher hidden states, just pass through Qwen normally
        if teacher_hidden_states is None:
            return self.qwen(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )
        
        # Process teacher hidden states through probe using probe's dtype
        try:
            _probe_dtype = next(self.probe.parameters()).dtype
        except StopIteration:
            _probe_dtype = teacher_hidden_states.dtype
        teacher_hidden_states = teacher_hidden_states.to(_probe_dtype)
        if self.use_static_rank_gate and self.probe_type == "lowrank":
            # Compute rank-space vector before final B projection
            # Use LowRankProbe internals: norm -> A -> act
            z = self.probe.norm(teacher_hidden_states)
            z = self.probe.A(z)
            if self.probe._act is not None:
                z = self.probe._act(z)
            # Store rank vector and B params for hook-time per-token scaling
            try:
                _model_dtype = next(self.qwen.parameters()).dtype
            except StopIteration:
                _model_dtype = z.dtype
            self._probe_rank_output = z.to(_model_dtype)  # [B, r]
            self._probe_B_weight = self.probe.B.weight
            self._probe_B_bias = self.probe.B.bias
            # For completeness, keep a default full projection too (unused in gating path)
            probe_output = self.probe(teacher_hidden_states)
        else:
            probe_output = self.probe(teacher_hidden_states)  # [B, H]
        
        # Always inject at the specified layer via hook
        if tokenizer is None:
            raise ValueError(
                "tokenizer must be provided when using probe injection"
            )
        # Find probe token positions
        self._probe_token_position = self._find_probe_token_positions(input_ids, tokenizer)
        # Ensure probe output matches model hidden dtype during injection
        try:
            _model_dtype = next(self.qwen.parameters()).dtype
        except StopIteration:
            _model_dtype = probe_output.dtype
        self._probe_output = probe_output.to(_model_dtype)
        # Register hooks for all target layers
        self._active_hooks = []
        target_layer = self.qwen.model.layers[self.probe_layer_idx]
        h = target_layer.register_forward_hook(self._create_injection_hook())
        self._active_hooks.append(h)
        try:
            output = self.qwen(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )
        finally:
            for h in self._active_hooks:
                h.remove()
            self._probe_output = None
            self._probe_token_position = None
            self._active_hooks = []
            self._probe_rank_output = None
            self._probe_B_weight = None
            self._probe_B_bias = None
        return output
    
    def generate(
        self,
        input_ids: torch.Tensor,
        teacher_hidden_states: Optional[torch.Tensor] = None,
        tokenizer=None,
        **kwargs,
    ):
        """Generate text with optional probe injection.
        
        Args:
            input_ids: [batch_size, seq_len] token IDs
            teacher_hidden_states: Optional [batch_size, teacher_hidden_size]
            **kwargs: Additional generation arguments
            
        Returns:
            Generated token IDs
        """
        if teacher_hidden_states is None:
            return self.qwen.generate(input_ids=input_ids, **kwargs)
        
        # Process probe with dtype alignment
        try:
            _probe_dtype = next(self.probe.parameters()).dtype
        except StopIteration:
            _probe_dtype = teacher_hidden_states.dtype
        teacher_hidden_states = teacher_hidden_states.to(_probe_dtype)
        if self.use_static_rank_gate and self.probe_type == "lowrank":
            z = self.probe.norm(teacher_hidden_states)
            z = self.probe.A(z)
            if self.probe._act is not None:
                z = self.probe._act(z)
            try:
                _model_dtype = next(self.qwen.parameters()).dtype
            except StopIteration:
                _model_dtype = z.dtype
            self._probe_rank_output = z.to(_model_dtype)
            self._probe_B_weight = self.probe.B.weight
            self._probe_B_bias = self.probe.B.bias
            probe_output = self.probe(teacher_hidden_states)
        else:
            probe_output = self.probe(teacher_hidden_states)
        
        # Always use layer injection during generation
        if tokenizer is None:
            raise ValueError(
                "tokenizer must be provided when using probe injection during generate()"
            )
        # Prepare positions and dtype-aligned probe output for injection
        self._probe_token_position = self._find_probe_token_positions(input_ids, tokenizer)
        try:
            _model_dtype = next(self.qwen.parameters()).dtype
        except StopIteration:
            _model_dtype = probe_output.dtype
        self._probe_output = probe_output.to(_model_dtype)
        self._active_hooks = []
        target_layer = self.qwen.model.layers[self.probe_layer_idx]
        h = target_layer.register_forward_hook(self._create_injection_hook())
        self._active_hooks.append(h)
        try:
            return self.qwen.generate(input_ids=input_ids, **kwargs)
        finally:
            for h in self._active_hooks:
                h.remove()
            self._probe_output = None
            self._probe_token_position = None
            self._active_hooks = []
            self._probe_rank_output = None
            self._probe_B_weight = None
            self._probe_B_bias = None
    
    def get_trainable_parameters(self) -> list:
        """Get only the trainable (probe) parameters."""
        params = [p for p in self.probe.parameters() if p.requires_grad]
        if self.per_layer_scale and self.layer_scale is not None:
            params.append(self.layer_scale)
        if self.use_static_rank_gate and self.rank_gate is not None:
            params.append(self.rank_gate)
        return params
    
    def num_trainable_parameters(self) -> int:
        """Count trainable parameters (probe only)."""
        return sum(p.numel() for p in self.get_trainable_parameters())
    
    def num_total_parameters(self) -> int:
        """Count all parameters (Qwen + probe)."""
        return sum(p.numel() for p in self.parameters())
