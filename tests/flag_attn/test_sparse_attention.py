import math
import torch
import pytest

import flag_attn

torch.random.manual_seed(42)


def max_diff(a, b):
    return (a - b).abs().max().item()


def report(name, actual, expected):
    md = max_diff(actual, expected)
    print(f"{name}: \tmax_diff: {md:0.6f}")


def generate_q2k_indices(B, Hkv, num_q_blocks, topK, num_kv_blocks, device="cuda"):
    """Generate random q2k block indices for testing."""
    # Each Q-block randomly selects topK KV blocks (no duplicates per row)
    q2k = torch.full((B, Hkv, num_q_blocks, topK), -1, dtype=torch.int32, device=device)
    for b in range(B):
        for h in range(Hkv):
            for qb in range(num_q_blocks):
                available = min(topK, num_kv_blocks)
                perm = torch.randperm(num_kv_blocks, device=device)[:available]
                # Sort for better memory access patterns
                perm, _ = perm.sort()
                q2k[b, h, qb, :available] = perm.to(torch.int32)
    return q2k


@pytest.mark.parametrize('device_id', [0])
@pytest.mark.parametrize('B, Hq, Hkv, Sq, Sk, D, topK, blk_kv', [
    # Basic cases
    (1, 4, 4, 128, 1024, 128, 4, 128),
    (1, 4, 4, 256, 2048, 128, 8, 128),
    (1, 4, 4, 512, 4096, 128, 16, 128),
    (2, 4, 4, 128, 1024, 128, 4, 128),
    # GQA cases
    (1, 8, 2, 128, 1024, 128, 4, 128),
    (1, 16, 4, 256, 2048, 128, 8, 128),
    (2, 8, 1, 128, 1024, 128, 4, 128),
    # Larger topK
    (1, 4, 4, 256, 4096, 128, 32, 128),
    # Larger sequences
    (1, 4, 4, 1024, 8192, 128, 16, 128),
])
@pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_sparse_attention_fwd(B, Hq, Hkv, Sq, Sk, D, topK, blk_kv, causal, dtype, device_id):
    device = f"cuda:{device_id}"
    BLOCK_M = 128  # Must match kernel's BLOCK_M for q2k_indices alignment
    num_q_blocks = (Sq + BLOCK_M - 1) // BLOCK_M
    # Adjust Sq to be divisible by BLOCK_M for simplicity
    Sq = num_q_blocks * BLOCK_M
    num_kv_blocks = Sk // blk_kv

    # Skip if topK > num_kv_blocks
    if topK > num_kv_blocks:
        pytest.skip("topK > num_kv_blocks")

    q = torch.randn((B, Sq, Hq, D), dtype=dtype, device=device)
    k = torch.randn((B, Sk, Hkv, D), dtype=dtype, device=device)
    v = torch.randn((B, Sk, Hkv, D), dtype=dtype, device=device)

    q2k_indices = generate_q2k_indices(B, Hkv, num_q_blocks, topK, num_kv_blocks, device=device)

    # Reference
    o_ref = flag_attn.testing.sparse_attention(
        q, k, v, q2k_indices, topK, blk_kv=blk_kv, causal=causal, upcast=True
    )

    # Triton kernel
    o_hyp = flag_attn.sparse_attention(
        q, k, v, q2k_indices, topK, blk_kv=blk_kv, causal=causal
    )

    # Torch reference (no upcast, for tolerance comparison)
    o_torch = flag_attn.testing.sparse_attention(
        q, k, v, q2k_indices, topK, blk_kv=blk_kv, causal=causal, upcast=False
    )

    torch_max_diff = max_diff(o_torch, o_ref)
    triton_max_diff = max_diff(o_hyp, o_ref)
    report("o_triton", o_hyp, o_ref)
    report("o_torch", o_torch, o_ref)

    # Triton should be within 2x of torch's error + small epsilon
    assert triton_max_diff <= 2 * torch_max_diff + 1e-3, \
        f"triton_max_diff={triton_max_diff:.6f} > 2*torch_max_diff={2*torch_max_diff:.6f} + 1e-3"


@pytest.mark.parametrize('B, Hq, Hkv, Sq, Sk, D, topK, blk_kv', [
    (1, 4, 4, 128, 512, 128, 4, 128),
    (1, 8, 2, 256, 1024, 128, 8, 128),
])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_sparse_attention_all_blocks(B, Hq, Hkv, Sq, Sk, D, topK, blk_kv, dtype):
    """When topK covers all KV blocks, sparse attention should match dense."""
    device = "cuda"
    BLOCK_M = 128
    num_q_blocks = (Sq + BLOCK_M - 1) // BLOCK_M
    Sq = num_q_blocks * BLOCK_M
    num_kv_blocks = Sk // blk_kv

    # topK = all blocks (dense attention)
    topK_full = num_kv_blocks
    q = torch.randn((B, Sq, Hq, D), dtype=dtype, device=device)
    k = torch.randn((B, Sk, Hkv, D), dtype=dtype, device=device)
    v = torch.randn((B, Sk, Hkv, D), dtype=dtype, device=device)

    # All blocks selected
    q2k_full = torch.zeros((B, Hkv, num_q_blocks, topK_full), dtype=torch.int32, device=device)
    for i in range(topK_full):
        q2k_full[:, :, :, i] = i

    o_sparse = flag_attn.sparse_attention(
        q, k, v, q2k_full, topK_full, blk_kv=blk_kv, causal=False
    )

    # Dense reference (standard flash attention, transposed to BSHD)
    q_bhsd = q.transpose(1, 2)  # [B, Hq, Sq, D]
    k_bhsd = k.transpose(1, 2)
    v_bhsd = v.transpose(1, 2)
    # expand for GQA
    num_groups = Hq // Hkv
    if num_groups > 1:
        k_bhsd = k_bhsd.repeat_interleave(num_groups, dim=1)
        v_bhsd = v_bhsd.repeat_interleave(num_groups, dim=1)
    scores = torch.matmul(q_bhsd.float(), k_bhsd.float().transpose(-2, -1)) / math.sqrt(D)
    attn = torch.softmax(scores, dim=-1)
    o_dense = torch.matmul(attn, v_bhsd.float()).transpose(1, 2).to(dtype)

    md = max_diff(o_sparse, o_dense)
    print(f"sparse_all_blocks vs dense: max_diff = {md:.6f}")
    assert md < 1e-2, f"max_diff too large: {md}"
