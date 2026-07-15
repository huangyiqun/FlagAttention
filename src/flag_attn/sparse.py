import math
import torch
import triton
import triton.language as tl

__all__ = ["attention"]


def maybe_contiguous(x):
    return x.contiguous() if x.stride(-1) != 1 else x


# ============================================================================
# Triton kernel: block-sparse forward attention (MSA style)
#
# Each Q token selects topK KV blocks (of size BLK_KV each) to attend to.
# The kernel processes one Q tile (BLOCK_M rows) per program, iterating over
# topK KV blocks per row. Since block selections are defined at Q-block
# granularity (all Q tokens in a BLOCK_M tile share the same topK blocks),
# we load one block index per topK slot and process the whole tile together.
# ============================================================================

@triton.jit
def _fwd_sparse_attention_kernel(
    Q, K, V, Out,
    Q2K,  # [B, Hkv, num_q_blocks, topK] block indices per Q-tile
    sm_scale,
    stride_qb, stride_qs, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,
    stride_ob, stride_os, stride_oh, stride_od,
    stride_ib, stride_ih, stride_iq, stride_ik,  # q2k strides
    B, Sq, Sk, Hq, Hkv,
    num_groups: tl.constexpr,
    TOPK: tl.constexpr,
    BLK_KV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
):
    # Grid: (cdiv(Sq, BLOCK_M), Hkv, B)
    pid_m = tl.program_id(0)
    pid_hkv = tl.program_id(1)
    pid_b = tl.program_id(2)

    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * log2e

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # Pointer to q2k_indices for this Q-tile
    q2k_base = Q2K + pid_b * stride_ib + pid_hkv * stride_ih + pid_m * stride_iq

    # Process each Q head in this KV head group (GQA)
    for g in range(num_groups):
        off_hq = pid_hkv * num_groups + g

        # Load Q tile: [BLOCK_M, BLOCK_D]
        q_ptrs = (Q + pid_b * stride_qb + off_hq * stride_qh +
                  offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd)
        if DIVISIBLE_M:
            q = tl.load(q_ptrs)
        else:
            mask_m = offs_m < Sq
            q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

        # Online softmax accumulators
        m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        # Iterate over topK selected KV blocks
        for tk in range(TOPK):
            # Load block index for this Q-tile
            blk_idx = tl.load(q2k_base + tk * stride_ik)

            # Skip padding blocks (index == -1)
            if blk_idx >= 0:
                kv_start = blk_idx * BLK_KV
                offs_n = kv_start + tl.arange(0, BLOCK_N)

                # Load K: [BLOCK_N, BLOCK_D]
                k_ptrs = (K + pid_b * stride_kb + pid_hkv * stride_kh +
                          offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd)
                if DIVISIBLE_N:
                    k = tl.load(k_ptrs)
                else:
                    mask_n = offs_n < Sk
                    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)

                # QK^T: [BLOCK_M, BLOCK_N]
                qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
                qk += tl.dot(q, tl.trans(k))
                qk *= qk_scale

                # Causal mask
                if IS_CAUSAL:
                    causal_mask = offs_m[:, None] >= offs_n[None, :]
                    qk = tl.where(causal_mask, qk, float("-inf"))

                # Boundary mask
                if not DIVISIBLE_N:
                    qk = tl.where(mask_n[None, :], qk, float("-inf"))

                # Online softmax update
                m_ij = tl.max(qk, axis=1)
                new_m_i = tl.maximum(m_i, m_ij)
                # Safe exp2: when m_i or new_m_i is -inf, avoid -inf - (-inf) = NaN
                safe_new_m = tl.where(new_m_i > float("-inf"), new_m_i, 0.0)
                alpha = tl.where(m_i > float("-inf"),
                                 tl.math.exp2(m_i - safe_new_m), 0.0)
                p = tl.where(new_m_i[:, None] > float("-inf"),
                             tl.math.exp2(qk - safe_new_m[:, None]), 0.0)
                l_i = l_i * alpha + tl.sum(p, axis=1)
                acc = acc * alpha[:, None]

                # Load V: [BLOCK_N, BLOCK_D]
                v_ptrs = (V + pid_b * stride_vb + pid_hkv * stride_vh +
                          offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd)
                if DIVISIBLE_N:
                    v = tl.load(v_ptrs)
                else:
                    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

                # P @ V accumulation
                acc += tl.dot(p.to(v.dtype), v)
                m_i = new_m_i

        # Finalize: divide by normalizer (handle all-masked rows)
        l_i = tl.where(l_i == 0.0, 1.0, l_i)
        acc = acc / l_i[:, None]

        # Store output: [BLOCK_M, BLOCK_D]
        o_ptrs = (Out + pid_b * stride_ob + off_hq * stride_oh +
                  offs_m[:, None] * stride_os + offs_d[None, :] * stride_od)
        if DIVISIBLE_M:
            tl.store(o_ptrs, acc.to(Out.dtype.element_ty))
        else:
            tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=mask_m[:, None])


# ============================================================================
# Public API
# ============================================================================

def get_fwd_config(D):
    """Heuristic kernel config for sparse attention forward.

    Returns (BLOCK_M, BLOCK_N, num_stages, num_warps).
    BLOCK_M is set to 128 to match the standard blk_kv=128 granularity
    used in MSA-style block-sparse attention.
    """
    if D <= 64:
        return 128, 128, 2, 4
    else:
        return 128, 128, 2, 4


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: torch.Tensor,
    topK: int,
    blk_kv: int = 128,
    causal: bool = False,
    sm_scale: float = None,
) -> torch.Tensor:
    """Block-sparse attention forward (MSA style).

    Each Q-tile (BLOCK_M queries) selects topK KV blocks to attend to.
    Only the selected KV blocks participate in the attention computation,
    enabling sub-quadratic attention for long sequences.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor, shape [B, Sq, Hq, D].
    k : torch.Tensor
        Key tensor, shape [B, Sk, Hkv, D].
    v : torch.Tensor
        Value tensor, shape [B, Sk, Hkv, D].
    q2k_indices : torch.Tensor
        Block selection indices, shape [B, Hkv, num_q_blocks, topK].
        Each entry is a KV block index in [0, num_kv_blocks). Padding with -1.
        num_q_blocks = cdiv(Sq, BLOCK_M).
    topK : int
        Number of KV blocks selected per Q-tile. Supported: 4, 8, 16, 32.
    blk_kv : int
        KV block size (number of tokens per block). Default 128.
    causal : bool
        Whether to apply causal masking within selected blocks.
    sm_scale : float, optional
        Softmax scale. Default: 1/sqrt(D).

    Returns
    -------
    torch.Tensor
        Output tensor, shape [B, Sq, Hq, D].
    """
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4
    B, Sq, Hq, D = q.shape
    _, Sk, Hkv, _ = k.shape
    assert Hq % Hkv == 0, "Hq must be a multiple of Hkv"
    assert D in {64, 128}, f"head_dim must be 64 or 128, got {D}"
    assert k.shape[-1] == D and v.shape[-1] == D
    assert q2k_indices.dim() == 4  # [B, Hkv, num_q_blocks, topK]
    assert q2k_indices.shape[3] == topK

    num_groups = Hq // Hkv

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    q, k, v = maybe_contiguous(q), maybe_contiguous(k), maybe_contiguous(v)
    q2k_indices = q2k_indices.contiguous()

    BLOCK_M, BLOCK_N, num_stages, num_warps = get_fwd_config(D)
    assert BLOCK_N == blk_kv or BLOCK_N <= blk_kv, \
        f"BLOCK_N ({BLOCK_N}) must equal blk_kv ({blk_kv}) for block-sparse attention"
    # Force BLOCK_N = blk_kv for correctness (one KV block per iteration)
    BLOCK_N = blk_kv

    num_q_blocks = triton.cdiv(Sq, BLOCK_M)
    assert q2k_indices.shape[2] == num_q_blocks, \
        f"q2k_indices.shape[2] ({q2k_indices.shape[2]}) must equal num_q_blocks ({num_q_blocks})"

    divisible_m = Sq % BLOCK_M == 0
    divisible_n = Sk % BLOCK_N == 0

    o = torch.empty_like(q)

    grid = (num_q_blocks, Hkv, B)

    _fwd_sparse_attention_kernel[grid](
        q, k, v, o,
        q2k_indices,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q2k_indices.stride(0), q2k_indices.stride(1),
        q2k_indices.stride(2), q2k_indices.stride(3),
        B, Sq, Sk, Hq, Hkv,
        num_groups=num_groups,
        TOPK=topK,
        BLK_KV=blk_kv,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=D,
        IS_CAUSAL=causal,
        DIVISIBLE_M=divisible_m,
        DIVISIBLE_N=divisible_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return o
