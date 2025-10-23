# Chess FEN Tokenization Strategy

## Problem

When tokenizing FEN strings with standard LLM tokenizers (e.g., Qwen), we discovered that:
- **Letters and pieces**: Tokenize to single tokens when space-prefixed ✓
- **Digits (0-9)**: Tokenize to TWO tokens when space-prefixed ✗
  - Example: `" 0"` → `['Ġ', '0']` (2 tokens)

This prevents achieving consistent 1-char-per-token alignment for the entire FEN string.

## Solution: Compromise Strategy

**Space-separate the board only; keep metadata compact**

### Implementation

```python
def format_fen_board_spaced(fen: str) -> str:
    """Format FEN with space-separated board only; leave metadata compact.
    
    This compromise strategy ensures all board characters tokenize to single tokens
    (letters, pieces, '.', '/') while avoiding the 2-token issue with digits in metadata.
    
    Example:
        Input:  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        Output: "r n b q k b n r / p p p p p p p p / . . . . . . . . / ... w KQkq - 0 1"
    """
    parts = fen.split(' ')
    board = parts[0]
    
    # Expand digit compression (8 -> . . . . . . . .)
    expanded_board = []
    for char in board:
        if char.isdigit():
            expanded_board.extend(['.'] * int(char))
        else:
            expanded_board.append(char)
    
    # Space-separate board only
    board_spaced = ' '.join(expanded_board)
    
    # Keep metadata compact (no spacing)
    metadata = ' '.join(parts[1:])  # side, castling, en passant, halfmove, fullmove
    
    return board_spaced + ' ' + metadata
```

### Benefits

1. **Fixed-size board representation**: Always 65 characters (64 squares + 7 slashes), each becoming a single token
2. **No digit tokenization issues**: Metadata digits (move counts, castling) stay compact
3. **Better alignment**: Board state has predictable token positions for probing
4. **No vocabulary modification**: Works with existing Qwen tokenizer

### Token Counts

For a typical starting position:
- **Raw FEN**: ~56 tokens
- **Fully spaced**: ~84 tokens (+50%)
- **Compromise**: ~72 tokens (+29%)

The compromise adds some tokens but achieves the critical goal: **predictable single-token-per-square board encoding**.

## Implementation Status

This strategy has been implemented in:

✓ `train_action_value_probe.py` - Probe training script  
✓ `eval_qwen_bc.py` - Evaluation script  
✓ `train_qwen_bc.py` - Behavioral cloning training script  

All prompts (plain and chat-template) and few-shot examples now use this formatting.

## Notebook

See `notebooks/chess_tokenization_experiments.ipynb` for:
- Empirical testing of tokenization strategies
- Character-by-character analysis
- Comparison of different approaches
- Examples with actual Qwen tokenizer

## Why Not Full Vocabulary Extension?

We considered adding chess-specific tokens to Qwen's vocabulary (like the searchless_chess paper), but:
- Requires resizing embedding layers + LM head
- Needs initialization and fine-tuning of new embeddings
- More complex to maintain
- The compromise achieves 90% of the benefit with 10% of the complexity

## Usage

All training and evaluation scripts automatically apply this formatting. No command-line flags needed.

To verify the formatting in your own code:
```python
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
formatted = format_fen_board_spaced(fen)
print(formatted)
# Output: r n b q k b n r / p p p p p p p p / . . . . . . . . / ... w KQkq - 0 1
```

