import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── tokenizer ──────────────────────────────────────────────────────────────
CHARS = "0123456789+=\n"
vocab_size = len(CHARS)
stoi = {c: i for i, c in enumerate(CHARS)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)

# ── RoPE ───────────────────────────────────────────────────────────────────
def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_len=512, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, q, k):
        T = q.shape[2]
        cos = self.cos_cached[:, :, :T, :]
        sin = self.sin_cached[:, :, :T, :]
        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin

# ── data ───────────────────────────────────────────────────────────────────
# format: "A+B=C\n"  (variable length, no zero-padding)
# max possible (11-digit validation): "99999999999+99999999999=199999999998\n" = 37 chars
BLOCK_SIZE = 36
EQ_ID      = stoi['=']
PAD_ID     = stoi['0']

# カリキュラム段階: 1桁〜10桁の最大値
STAGES = [9, 99, 999, 9_999, 99_999, 999_999, 9_999_999,
          99_999_999, 999_999_999, 9_999_999_999]
ADVANCE_THRESHOLD = 0.95   # train_acc がこれを超えたら次ステージ
CLEAR_THRESHOLD   = 0.90   # val_acc  がこれを超えたら学習終了

def val_range(max_num):
    """学習データの1桁上の範囲 [10^n, 10^(n+1)-1] を返す"""
    n = len(str(max_num))  # 現ステージの桁数
    return 10**n, 10**(n+1) - 1

def make_problem(max_num, a=None, b=None):
    a = a if a is not None else random.randint(0, max_num)
    b = b if b is not None else random.randint(0, max_num)
    return f"{a}+{b}={a+b}\n"

def get_batch(batch_size, device, max_num):
    probs = [make_problem(max_num) for _ in range(batch_size)]
    toks  = [encode(p) for p in probs]
    padded = [t + [PAD_ID] * max(0, BLOCK_SIZE + 1 - len(t)) for t in toks]
    x = torch.tensor([t[:BLOCK_SIZE]    for t in padded], dtype=torch.long)
    y = torch.tensor([t[1:BLOCK_SIZE+1] for t in padded], dtype=torch.long)
    loss_mask = torch.zeros(batch_size, BLOCK_SIZE, dtype=torch.bool)
    for i, tok in enumerate(toks):
        eq_pos = tok.index(EQ_ID)
        loss_mask[i, eq_pos : len(tok) - 1] = True
    return x.to(device), y.to(device), loss_mask.to(device)

# ── model ──────────────────────────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_len):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.qkv  = nn.Linear(d_model, 3 * d_model)
        self.out  = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(self.head_dim, max_len=max_len)
        mask = torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out  = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, max_len):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, max_len)
        self.ff   = FeedForward(d_model, d_ff)
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class DecoderTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        # No pos_emb — position information is encoded via RoPE in each attention layer
        self.blocks  = nn.Sequential(*[
            DecoderBlock(d_model, n_heads, d_ff, max_len) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len

    def forward(self, idx):
        x = self.tok_emb(idx)
        x = self.blocks(x)
        return self.head(self.ln_f(x))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_len:]
            logits   = self(idx_cond)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx

# ── evaluation ─────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, device, max_num, n_random=300):
    model.eval()

    def predict(a, b):
        prompt = encode(f"{a}+{b}=")
        ctx    = torch.tensor([prompt], dtype=torch.long, device=device)
        out    = model.generate(ctx, max_new_tokens=15)
        return decode(out[0, len(prompt):].tolist()).split('\n')[0]

    # 学習範囲内で評価（train_acc）
    correct = 0
    for _ in range(n_random):
        a = random.randint(0, max_num)
        b = random.randint(0, max_num)
        if predict(a, b) == str(a + b):
            correct += 1
    train_acc = correct / n_random

    # 1桁上の範囲で評価（val_acc / OOD）
    v_min, v_max = val_range(max_num)
    correct = 0
    for _ in range(n_random):
        a = random.randint(v_min, v_max)
        b = random.randint(v_min, v_max)
        if predict(a, b) == str(a + b):
            correct += 1
    val_acc = correct / n_random

    model.train()
    return train_acc, val_acc

# ── training ───────────────────────────────────────────────────────────────
def train():
    device     = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    D_MODEL    = 256
    N_HEADS    = 8
    N_LAYERS   = 4
    D_FF       = 1024
    BATCH_SIZE = 128
    LR         = 1e-3
    EVAL_EVERY = 500
    MAX_STEPS  = 300_000

    print(f"device={device}  vocab={vocab_size}  block={BLOCK_SIZE}")
    print(f"stages (max operand): {STAGES}")
    print(f"advance threshold: {ADVANCE_THRESHOLD:.0%}  (on OOD = 1-digit-higher numbers)")

    model = DecoderTransformer(
        vocab_size, D_MODEL, N_HEADS, N_LAYERS, D_FF, BLOCK_SIZE
    ).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    stage_idx = 0
    max_num   = STAGES[stage_idx]

    v_min, v_max = val_range(max_num)
    print(f"{'─'*72}")
    print(f"Stage {stage_idx}: train 0~{max_num:,}  |  val {v_min:,}~{v_max:,}  (OOD)")
    print(f"{'─'*72}")

    for step in range(1, MAX_STEPS + 1):
        x, y, loss_mask = get_batch(BATCH_SIZE, device, max_num)
        logits = model(x)
        loss   = F.cross_entropy(logits[loss_mask], y[loss_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"step {step:6d} | loss {loss.item():.4f}", end="\r")

        if step % EVAL_EVERY == 0:
            train_acc, val_acc = evaluate(model, device, max_num)
            v_min, v_max = val_range(max_num)
            print(f"step {step:6d} | loss {loss.item():.4f} | "
                  f"train(0~{max_num:,}) {train_acc:5.1%} | "
                  f"val({v_min:,}~{v_max:,}) {val_acc:5.1%}")

            # val_acc が閾値を超えたら学習終了
            if val_acc >= CLEAR_THRESHOLD:
                print(f"\n{'='*72}")
                print(f"  CLEARED! val_acc={val_acc:.1%} at step {step}  (stage {stage_idx})")
                print(f"{'='*72}\n")
                break

            # train_acc が閾値を超えたら次ステージへ
            if train_acc >= ADVANCE_THRESHOLD and stage_idx < len(STAGES) - 1:
                torch.save(model.state_dict(), f"transformer_rope_stage{stage_idx}.pt")
                print(f"  checkpoint saved: transformer_rope_stage{stage_idx}.pt")
                stage_idx += 1
                max_num = STAGES[stage_idx]
                v_min, v_max = val_range(max_num)
                print(f"{'─'*72}")
                print(f"*** Stage {stage_idx}: train 0~{max_num:,}  |  val {v_min:,}~{v_max:,}  (OOD) ***")
                print(f"{'─'*72}")

    print()
    torch.save(model.state_dict(), "transformer_rope.pt")
    print(f"Model saved to transformer_rope.pt  (final stage: {stage_idx}, max_num: {max_num:,})")

if __name__ == "__main__":
    train()
