"""
保存済みモデルを使って、4桁（0~9999）の in-distribution 大規模評価と OOD 評価を行うスクリプト。

usage:
  uv run eval_indist.py                          # transformer.pt を使用
  uv run eval_indist.py transformer_stage3.pt   # ステージ3のチェックポイントを使用
"""
import sys
import random
import torch
from transformer import (
    DecoderTransformer, vocab_size, BLOCK_SIZE,
    encode, decode, val_range,
)

N_INDIST = 10_000   # in-distribution 評価サンプル数
N_OOD    = 1_000    # OOD 評価サンプル数
MAX_NUM  = 9_999    # 4桁

ckpt = sys.argv[1] if len(sys.argv) > 1 else "transformer.pt"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

model = DecoderTransformer(vocab_size, 256, 8, 4, 1024, BLOCK_SIZE).to(device)
model.load_state_dict(torch.load(ckpt, map_location=device))
print(f"loaded: {ckpt}")
model.eval()
print(f"model loaded  device={device}")

@torch.no_grad()
def predict(a, b):
    prompt = encode(f"{a}+{b}=")
    ctx    = torch.tensor([prompt], dtype=torch.long, device=device)
    out    = model.generate(ctx, max_new_tokens=15)
    return decode(out[0, len(prompt):].tolist()).split('\n')[0]

# ── in-distribution（4桁、未見ペア）──────────────────────────────────────
print(f"\n[in-dist]  0~{MAX_NUM:,} のランダム {N_INDIST:,} ペアを評価中...")
correct = sum(
    predict(a := random.randint(0, MAX_NUM), b := random.randint(0, MAX_NUM)) == str(a + b)
    for _ in range(N_INDIST)
)
print(f"[in-dist]  正答率: {correct / N_INDIST:.1%}  ({correct}/{N_INDIST})")

# ── OOD（5桁）────────────────────────────────────────────────────────────
v_min, v_max = val_range(MAX_NUM)
print(f"\n[OOD]      {v_min:,}~{v_max:,} のランダム {N_OOD:,} ペアを評価中...")
correct = sum(
    predict(a := random.randint(v_min, v_max), b := random.randint(v_min, v_max)) == str(a + b)
    for _ in range(N_OOD)
)
print(f"[OOD]      正答率: {correct / N_OOD:.1%}  ({correct}/{N_OOD})")
