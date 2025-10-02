# Chess Probe Models

This directory contains the modular components for training a Qwen model with a linear probe using an action-value teacher.

## Components

### `teacher_wrapper.py`
Wrapper for DeepMind's JAX-based action-value model that:
- Loads pretrained checkpoints (9M, 136M, or 270M parameters)
- Extracts hidden states from a specified layer's residual stream
- Supports layer selection with validation (e.g., layer 8 of 16, or -1 for last layer)
- Converts JAX arrays to PyTorch tensors
- Supports single and batch inference
- Can get win probabilities for all legal moves

### `probe_model.py`
Defines the trainable probe and Qwen wrapper:
- `LinearProbe`: Simple linear layer mapping teacher hidden states to Qwen embedding space
- `QwenWithProbe`: Wrapper combining frozen Qwen with trainable probe
  - Supports dual input modes: regular text OR text + teacher hidden states
  - Supports two injection methods:
    - **Sequence concatenation** (`probe_layer_idx=None`): Probe output is appended as a token
    - **Layer injection** (`probe_layer_idx=N`): Probe output is added to residual stream at layer N
  - Validates layer indices and prints helpful messages (e.g., "Model has 28 layers")
  - Supports negative indexing (e.g., `-1` for last layer, `-2` for second-to-last)
  - Only probe parameters are trainable; Qwen is frozen

## Architecture

```
Training Example:
┌─────────────────────────────────────────────┐
│ Previous Board State + Action               │
│   ↓                                         │
│ Teacher Model (Action Value, JAX)           │
│   ↓                                         │
│ Hidden States [embedding_dim]               │
└─────────────────────────────────────────────┘
                    │
                    ↓
┌─────────────────────────────────────────────┐
│ Step game forward → New Board State         │
│   ↓                                         │
│ Stockfish → Best Move                       │
└─────────────────────────────────────────────┘
                    │
                    ↓
┌─────────────────────────────────────────────┐
│ Training:                                   │
│   Input: New Board + Teacher Hidden States  │
│   Target: Stockfish Best Move              │
│   Loss: Cross-Entropy (Behavioral Cloning)  │
└─────────────────────────────────────────────┘
```

## Usage

### Creating the Model

```python
from models import QwenWithProbe
from transformers import AutoModelForCausalLM

qwen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B-Base")

# Sequence concatenation (default)
model = QwenWithProbe(qwen, teacher_hidden_size=1024)
# ✓ Probe will concatenate to sequence (model has 28 layers)

# Layer injection at specific layer
model = QwenWithProbe(qwen, teacher_hidden_size=1024, probe_layer_idx=14)
# ✓ Probe will inject at layer 14/27

# Negative indexing (last layer)
model = QwenWithProbe(qwen, teacher_hidden_size=1024, probe_layer_idx=-1)
# ✓ Probe will inject at layer 27/27

# Invalid layer raises helpful error
model = QwenWithProbe(qwen, teacher_hidden_size=1024, probe_layer_idx=50)
# ValueError: probe_layer_idx=50 is out of range. Model has 28 layers (indices 0-27).
```

### Forward Pass

```python
# Normal mode (no probe)
output = model(input_ids=tokens, attention_mask=mask)

# Probe mode (with teacher information)
teacher_hidden = teacher.get_hidden_states(fen, move)
output = model(
    input_ids=tokens,
    attention_mask=mask,
    teacher_hidden_states=teacher_hidden
)
```

### Training Script

```bash
# Train with sequence concatenation
python scripts/train_action_value_probe.py \
  --num_train_data 500 \
  --probe_layer_idx None

# Train with layer injection (middle layer of Qwen)
python scripts/train_action_value_probe.py \
  --num_train_data 500 \
  --probe_layer_idx 14

# Extract teacher hidden states from middle layer (layer 8 of 16)
python scripts/train_action_value_probe.py \
  --num_train_data 500 \
  --teacher_model_size 270M \
  --teacher_hidden_layer_idx 8

# Combine: Teacher layer 8, Qwen probe at layer 14
python scripts/train_action_value_probe.py \
  --num_train_data 500 \
  --teacher_hidden_layer_idx 8 \
  --probe_layer_idx 14 \
  --sampling_temperature 0.5
```
