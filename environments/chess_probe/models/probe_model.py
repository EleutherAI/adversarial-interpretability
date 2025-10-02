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
from typing import Optional


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
        
        # Validate and set probe layer
        if probe_layer_idx is not None:
            # Handle negative indexing
            if probe_layer_idx < 0:
                probe_layer_idx = self.num_layers + probe_layer_idx
            
            if probe_layer_idx < 0 or probe_layer_idx >= self.num_layers:
                raise ValueError(
                    f"probe_layer_idx={probe_layer_idx} is out of range. "
                    f"Model has {self.num_layers} layers (indices 0-{self.num_layers-1}). "
                    f"Use None to concatenate to sequence instead."
                )
            
            print(f"✓ Probe will inject at layer {probe_layer_idx}/{self.num_layers-1} at token '{probe_token}'")
        else:
            print(f"✓ Probe will concatenate to sequence (model has {self.num_layers} layers)")
        
        self.probe_layer_idx = probe_layer_idx
        
        # Initialize the trainable probe
        self.probe = LinearProbe(teacher_hidden_size, self.qwen_hidden_size)
        
        # Freeze Qwen if requested
        if freeze_qwen:
            for param in self.qwen.parameters():
                param.requires_grad = False
        
        # Only probe parameters should be trainable
        for param in self.probe.parameters():
            param.requires_grad = True
        
        # Storage for injection state during forward pass
        self._probe_output = None
        self._probe_token_position = None
    
    def _create_injection_hook(self):
        """Create a forward hook that adds probe output to hidden states at probe token position."""
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
            
            # Add probe output only at the probe token position
            # probe_output: [B, H], probe_token_position: [B] (indices)
            batch_size = hidden_states.shape[0]
            for b in range(batch_size):
                pos = self._probe_token_position[b]
                if 0 <= pos < hidden_states.shape[1]:
                    modified_hidden[b, pos, :] = modified_hidden[b, pos, :] + self._probe_output[b, :]
            
            if isinstance(output, tuple):
                return (modified_hidden,) + output[1:]
            else:
                return modified_hidden
        
        return hook
    
    def _find_probe_token_positions(
        self,
        input_ids: torch.Tensor,
        tokenizer,
    ) -> torch.Tensor:
        """Find the position of the probe token in each sequence.
        
        Returns tensor of shape [batch_size] with the position index of the probe token.
        Returns -1 if probe token not found in a sequence.
        """
        # Get the token ID for the probe token
        probe_token_ids = tokenizer.encode(self.probe_token, add_special_tokens=False)
        if len(probe_token_ids) == 0:
            raise ValueError(f"Probe token '{self.probe_token}' encodes to empty sequence")
        
        # For simplicity, use the first token if it tokenizes to multiple
        probe_token_id = probe_token_ids[0]
        
        batch_size, seq_len = input_ids.shape
        positions = torch.full((batch_size,), -1, dtype=torch.long, device=input_ids.device)
        
        for b in range(batch_size):
            # Find last occurrence of probe token (in case it appears multiple times)
            matches = (input_ids[b] == probe_token_id).nonzero(as_tuple=True)[0]
            if len(matches) > 0:
                positions[b] = matches[-1]
        
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
        
        # Process teacher hidden states through probe
        probe_output = self.probe(teacher_hidden_states)  # [B, H]
        
        # If probe_layer_idx is specified, inject at that layer via hook
        if self.probe_layer_idx is not None:
            if tokenizer is None:
                raise ValueError(
                    "tokenizer must be provided when using probe_layer_idx for layer injection"
                )
            
            # Find probe token positions
            self._probe_token_position = self._find_probe_token_positions(input_ids, tokenizer)
            self._probe_output = probe_output
            
            # Register hook
            target_layer = self.qwen.model.layers[self.probe_layer_idx]
            hook = target_layer.register_forward_hook(self._create_injection_hook())
            
            try:
                output = self.qwen(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    **kwargs,
                )
            finally:
                hook.remove()
                self._probe_output = None
                self._probe_token_position = None
            
            return output
        
        # Otherwise, concatenate probe to sequence (original behavior)
        batch_size = input_ids.shape[0]
        
        # Get embeddings for the input tokens
        inputs_embeds = self.qwen.get_input_embeddings()(input_ids)  # [B, T, H]
        probe_embeds = probe_output.unsqueeze(1)  # [B, 1, H]
        
        # Concatenate probe embedding to the end of the sequence
        inputs_embeds = torch.cat([inputs_embeds, probe_embeds], dim=1)  # [B, T+1, H]
        
        # Extend attention mask to include probe token
        if attention_mask is not None:
            probe_attention = torch.ones(
                (batch_size, 1),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([attention_mask, probe_attention], dim=1)
        
        # Extend labels if provided (probe token doesn't contribute to loss by default)
        if labels is not None:
            probe_labels = torch.full(
                (batch_size, 1),
                -100,  # Ignore index
                dtype=labels.dtype,
                device=labels.device,
            )
            labels = torch.cat([labels, probe_labels], dim=1)
        
        # Forward through Qwen with modified embeddings
        return self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
    
    def generate(
        self,
        input_ids: torch.Tensor,
        teacher_hidden_states: Optional[torch.Tensor] = None,
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
        
        # Process probe
        probe_output = self.probe(teacher_hidden_states)
        
        # If using layer injection, apply hook during generation
        if self.probe_layer_idx is not None:
            target_layer = self.qwen.model.layers[self.probe_layer_idx]
            hook = target_layer.register_forward_hook(
                self._create_injection_hook(probe_output)
            )
            
            try:
                return self.qwen.generate(input_ids=input_ids, **kwargs)
            finally:
                hook.remove()
        
        # Otherwise, concatenate to sequence
        batch_size = input_ids.shape[0]
        inputs_embeds = self.qwen.get_input_embeddings()(input_ids)
        probe_embeds = probe_output.unsqueeze(1)
        inputs_embeds = torch.cat([inputs_embeds, probe_embeds], dim=1)
        
        return self.qwen.generate(inputs_embeds=inputs_embeds, **kwargs)
    
    def get_trainable_parameters(self) -> list:
        """Get only the trainable (probe) parameters."""
        return [p for p in self.probe.parameters() if p.requires_grad]
    
    def num_trainable_parameters(self) -> int:
        """Count trainable parameters (probe only)."""
        return sum(p.numel() for p in self.get_trainable_parameters())
    
    def num_total_parameters(self) -> int:
        """Count all parameters (Qwen + probe)."""
        return sum(p.numel() for p in self.parameters())
