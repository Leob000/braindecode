import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class MultiHeadAttention_new(nn.Module):
    def __init__(self, emb_size, num_heads, dropout=0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            emb_size, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, x, mask=None):
        return self.mha(x, x, x, attn_mask=mask, need_weights=False)[0]


class MultiHeadAttention_old(nn.Module):
    def __init__(self, emb_size, num_heads, dropout=0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

        self.rearrange_stack = Rearrange(
            "b n (h d) -> b h n d",
            h=num_heads,
        )
        self.rearrange_unstack = Rearrange(
            "b h n d -> b n (h d)",
        )

    def forward(self, x, mask=None):
        queries = self.rearrange_stack(self.queries(x))
        keys = self.rearrange_stack(self.keys(x))
        values = self.rearrange_stack(self.values(x))
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        if mask is not None:
            fill_value = float("-inf")
            energy = energy.masked_fill(~mask, fill_value)

        # scaling = self.emb_size ** (1 / 2)
        scaling = (self.emb_size // self.num_heads) ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum("bhal, bhlv -> bhav ", att, values)
        out = self.rearrange_unstack(out)
        out = self.projection(out)
        return out


def run_comparison():
    print("--- Comparing Old vs New Attention ---")
    B, L, E, H = 4, 10, 16, 4
    torch.manual_seed(42)

    mha_old = MultiHeadAttention_old(E, H)
    mha_new = MultiHeadAttention_new(E, H)

    # Sync weights
    with torch.no_grad():
        w_q, b_q = mha_old.queries.weight, mha_old.queries.bias
        w_k, b_k = mha_old.keys.weight, mha_old.keys.bias
        w_v, b_v = mha_old.values.weight, mha_old.values.bias
        mha_new.mha.in_proj_weight.copy_(torch.cat([w_q, w_k, w_v], dim=0))
        mha_new.mha.in_proj_bias.copy_(torch.cat([b_q, b_k, b_v], dim=0))
        mha_new.mha.out_proj.weight.copy_(mha_old.projection.weight)
        mha_new.mha.out_proj.bias.copy_(mha_old.projection.bias)

    x = torch.randn(B, L, E)
    x_old = x.clone().detach().requires_grad_(True)
    x_new = x.clone().detach().requires_grad_(True)

    # Test 1: Forward (No Mask)
    out_old = mha_old(x_old)
    out_new = mha_new(x_new)
    diff = (out_old - out_new).abs()
    print(
        f"Forward Diff (No Mask): Max: {diff.max().item():.2e} | Mean: {diff.mean().item():.2e}"
    )

    # Test 2: Backward (No Mask)
    target = torch.randn_like(out_old)
    F.mse_loss(out_old, target).backward()
    F.mse_loss(out_new, target).backward()
    assert x_old.grad is not None and x_new.grad is not None
    diff_grad = (x_old.grad - x_new.grad).abs()
    print(
        f"Grad Diff (No Mask):    Max: {diff_grad.max().item():.2e} | Mean: {diff_grad.mean().item():.2e}"
    )

    # Test 3: Forward (With Mask - Raw / Wrong Convention)
    mask = torch.ones(L, L, dtype=torch.bool)
    mask[:, -3:] = False  # Mask last 3 positions
    out_old_masked = mha_old(x_old, mask=mask)

    # This demonstrates that Old uses True=Keep while New uses True=Ignore
    out_new_raw_mask = mha_new(x_new, mask=mask)
    diff_raw = (out_old_masked - out_new_raw_mask).abs()
    print(
        f"Forward Diff (Masked, Raw):      Max: {diff_raw.max().item():.2e} | Mean: {diff_raw.mean().item():.2e} (High diff confirms mask convention mismatch)"
    )

    # Test 4: Forward (With Mask - Inverted / Correct Convention)
    # Old: True=Keep, New: True=Ignore (so we pass ~mask)
    out_new_masked = mha_new(x_new, mask=~mask)
    diff_masked = (out_old_masked - out_new_masked).abs()
    print(
        f"Forward Diff (Masked, Inverted): Max: {diff_masked.max().item():.2e} | Mean: {diff_masked.mean().item():.2e}"
    )


if __name__ == "__main__":
    run_comparison()
