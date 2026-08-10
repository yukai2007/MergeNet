#!/usr/bin/env python3
"""CUDA parity gate for biased FlashAttention and its DTEM caller."""

import math

import torch

from opentome.timm.bias_local_attn import biased_local_attention
from opentome.timm.dtem import DTEMAttention


def reference_bhnd(q, k, v, bias, window):
    """Independent fp32 reference: softmax(qk/sqrt(D) + per-key bias)."""
    q32, k32, v32 = q.float(), k.float(), v.float()
    scores = q32 @ k32.transpose(-1, -2) / math.sqrt(q.shape[-1])
    scores = scores + bias.float()[:, None, None, :]
    if window >= 0:
        tokens = q.shape[-2]
        positions = torch.arange(tokens, device=q.device)
        allowed = (positions[:, None] - positions[None, :]).abs() <= window
        scores = scores.masked_fill(
            ~allowed.view(1, 1, tokens, tokens), float("-inf")
        )
    return torch.softmax(scores, dim=-1) @ v32


def parity_case(layout, tokens=11, heads=3, window=2, check_grad=True):
    batch, dim = 2, 16
    q = (torch.randn(batch, heads, tokens, dim, device="cuda", dtype=torch.float16) * 0.25).requires_grad_()
    k = (torch.randn_like(q) * 0.25).requires_grad_()
    v = torch.randn_like(q).requires_grad_()
    bias = (torch.randn(batch, tokens, device="cuda", dtype=torch.float16) * 0.25).requires_grad_()

    if layout == "BHND":
        q_in, k_in, v_in = q, k, v
    else:
        q_in = q.transpose(1, 2).contiguous()
        k_in = k.transpose(1, 2).contiguous()
        v_in = v.transpose(1, 2).contiguous()

    actual = biased_local_attention(
        q_in,
        k_in,
        v_in,
        bias,
        local_window=window,
        training=check_grad,
        x_dtype=torch.float16,
        input_layout=layout,
    )
    actual_bhnd = actual if layout == "BHND" else actual.transpose(1, 2)
    expected = reference_bhnd(q, k, v, bias, window)
    max_abs = (actual_bhnd.float() - expected).abs().max().item()
    if not torch.allclose(actual_bhnd.float(), expected, atol=1.5e-2, rtol=1.5e-2):
        raise AssertionError(
            f"{layout} tokens={tokens} heads={heads} window={window} "
            f"forward mismatch: max_abs={max_abs}"
        )

    grad_max_abs = None
    if check_grad:
        probe = torch.randn_like(expected)
        actual_grads = torch.autograd.grad(
            (actual_bhnd.float() * probe).sum(), (q, k, v, bias), retain_graph=True
        )
        expected_grads = torch.autograd.grad(
            (expected * probe).sum(), (q, k, v, bias)
        )
        grad_max_abs = max(
            (got.float() - want.float()).abs().max().item()
            for got, want in zip(actual_grads, expected_grads)
        )
        for got, want in zip(actual_grads, expected_grads):
            if not torch.allclose(got.float(), want.float(), atol=3e-2, rtol=3e-2):
                raise AssertionError(
                    f"{layout} tokens={tokens} heads={heads} window={window} "
                    f"gradient mismatch: max_abs={grad_max_abs}"
                )
    return max_abs, grad_max_abs


@torch.inference_mode()
def locality_case():
    batch, heads, tokens, dim = 1, 3, 11, 16
    window, target = 2, 5
    q = torch.randn(batch, heads, tokens, dim, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    bias = torch.randn(batch, tokens, device="cuda", dtype=torch.float16)

    def run(value):
        return biased_local_attention(
            q, k, value, bias, window,
            training=False, input_layout="BHND",
        )

    baseline = run(v)
    inside_v = v.clone()
    inside_v[:, :, target + 1] += 1
    outside_v = v.clone()
    outside_v[:, :, target + window + 2] += 1
    inside_delta = (run(inside_v)[:, :, target] - baseline[:, :, target]).abs().max().item()
    outside_delta = (run(outside_v)[:, :, target] - baseline[:, :, target]).abs().max().item()
    if inside_delta <= 1e-4:
        raise AssertionError(f"inside-window perturbation had no effect: {inside_delta}")
    if outside_delta > 1e-6:
        raise AssertionError(f"outside-window perturbation leaked: {outside_delta}")
    return inside_delta, outside_delta


@torch.inference_mode()
def dtem_caller_case():
    """Catch caller-side q pre-scaling as well as helper layout regressions."""
    batch, tokens, dim, heads = 2, 11, 48, 3
    module = DTEMAttention(
        dim=dim,
        num_heads=heads,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    )
    module.patch(feat_dim=8)
    module._tome_info = {"swa_size": 2, "window_size": 2, "r": [1]}
    module = module.cuda().half().eval()
    x = torch.randn(batch, tokens, dim, device="cuda", dtype=torch.float16)
    size = torch.rand(batch, tokens, 1, device="cuda", dtype=torch.float16) + 0.5

    packed, _ = module.qkv(x)
    qkv = packed.reshape(batch, tokens, 3, heads, dim // heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = module.q_norm(q), module.k_norm(k)
    expected_attn = reference_bhnd(q, k, v, size.squeeze(-1).log(), window=2)
    expected = module.proj(expected_attn.transpose(1, 2).reshape(batch, tokens, dim).half())
    actual, _ = module(x, size=size)
    max_abs = (actual.float() - expected.float()).abs().max().item()
    if not torch.allclose(actual.float(), expected.float(), atol=2e-2, rtol=2e-2):
        raise AssertionError(f"DTEMAttention caller mismatch: max_abs={max_abs}")
    return max_abs


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the biased FlashAttention gate")
    try:
        import flash_attn  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("flash-attn is required for this gate") from exc

    torch.manual_seed(20260809)
    results = []
    for layout in ("BHND", "BNHD"):
        results.append((layout, 11, 3, *parity_case(layout, 11, 3, 2, True)))
        # N < H explicitly guards against the old size-based layout heuristic.
        results.append((layout, 3, 5, *parity_case(layout, 3, 5, 1, False)))
        results.append((layout, 11, 3, *parity_case(layout, 11, 3, -1, False)))

    ambiguous = torch.randn(1, 4, 4, 16, device="cuda", dtype=torch.float16)
    try:
        biased_local_attention(
            ambiguous, ambiguous, ambiguous,
            torch.zeros(1, 4, device="cuda", dtype=torch.float16),
            1,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous N == H layout must require input_layout")

    inside_delta, outside_delta = locality_case()
    dtem_max_abs = dtem_caller_case()
    print(
        "BIASED_LOCAL_ATTENTION_TEST_PASS",
        {"parity": results, "inside_delta": inside_delta,
         "outside_delta": outside_delta, "dtem_max_abs": dtem_max_abs},
    )


if __name__ == "__main__":
    main()
