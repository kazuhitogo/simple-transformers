# simple-transformer

A minimal decoder-only Transformer trained to perform integer addition, used to investigate whether neural networks memorize or generalize.

## Experiment

The model is trained on addition problems like `27+46=73` using curriculum learning (1-digit → 10-digit). At each stage, the model is evaluated on numbers **one digit larger** than its training range (OOD). The key question: can it generalize to digit counts it has never seen?

**Result**: Within the training digit range, the model achieves ~95% accuracy on unseen pairs. The moment digit count increases by one, accuracy drops to 0%.

## Requirements

- [uv](https://docs.astral.sh/uv/)

## Usage

```bash
# Train
uv run transformer.py

# Evaluate in-distribution vs OOD (after training stage 3)
uv run eval_indist.py transformer_stage3.pt
```

## Model

- Architecture: Decoder-only Transformer (GPT-style)
- Parameters: ~3.17M
- Layers: 4, d_model: 256, heads: 8, d_ff: 1024
- Tokenizer: character-level (`0123456789+=\n`, 13 tokens)
- Context length: 36

## Curriculum stages

| Stage | Train range | OOD eval range |
|-------|------------|----------------|
| 0 | 0 ~ 9 | 10 ~ 99 |
| 1 | 0 ~ 99 | 100 ~ 999 |
| ... | ... | ... |
| 9 | 0 ~ 9,999,999,999 | 10,000,000,000 ~ 99,999,999,999 |

Advance condition: train accuracy ≥ 95%. Clear condition: OOD accuracy ≥ 90%.
