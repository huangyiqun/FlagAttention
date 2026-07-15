import math
import torch


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: torch.Tensor,
    topK: int,
    blk_kv: int = 128,
    causal: bool = False,
    sm_scale: float = None,
    upcast: bool = True,
) -> torch.Tensor:
    """PyTorch reference for block-sparse attention (MSA style).

    Constructs an explicit block mask from q2k_indices and computes
    full masked attention for correctness verification.

    Args:
        q: [B, Sq, Hq, D] query tensor
        k: [B, Sk, Hkv, D] key tensor
        v: [B, Sk, Hkv, D] value tensor
        q2k_indices: [B, Hkv, num_q_blocks, topK] int32, KV block indices per Q-tile.
                     Padding with -1 for unused slots.
        topK: number of selected KV blocks per Q-tile
        blk_kv: KV block size (default 128)
        causal: whether to apply causal masking
        sm_scale: softmax scale factor
        upcast: whether to upcast to float32 for reference computation

    Returns:
        output: [B, Sq, Hq, D]
    """
    B, Sq, Hq, D = q.shape
    Sk = k.shape[1]
    Hkv = k.shape[2]
    num_groups = Hq // Hkv
    num_kv_blocks = Sk // blk_kv
    num_q_blocks = q2k_indices.shape[2]
    BLOCK_M = Sq // num_q_blocks

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    input_dtype = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()

    # Build block-level mask: [B, Hkv, Sq, num_kv_blocks]
    # Expand q2k_indices from per-Q-block to per-Q-token
    # q2k_indices: [B, Hkv, num_q_blocks, topK]
    block_mask = torch.zeros(B, Hkv, Sq, num_kv_blocks, dtype=torch.bool, device=q.device)

    for qb in range(num_q_blocks):
        q_start = qb * BLOCK_M
        q_end = min(q_start + BLOCK_M, Sq)
        # indices for this Q-block: [B, Hkv, topK]
        indices = q2k_indices[:, :, qb, :]
        for b in range(B):
            for h in range(Hkv):
                for tk in range(topK):
                    idx = indices[b, h, tk].item()
                    if idx >= 0 and idx < num_kv_blocks:
                        block_mask[b, h, q_start:q_end, idx] = True

    # Expand block mask to token mask: [B, Hkv, Sq, Sk]
    token_mask = block_mask.repeat_interleave(blk_kv, dim=3)
    # Trim if Sk is not perfectly divisible
    token_mask = token_mask[:, :, :, :Sk]

    # Expand for GQA: [B, Hq, Sq, Sk]
    if num_groups > 1:
        token_mask = token_mask.repeat_interleave(num_groups, dim=1)
        k_expanded = k.repeat_interleave(num_groups, dim=2)
        v_expanded = v.repeat_interleave(num_groups, dim=2)
    else:
        k_expanded = k
        v_expanded = v

    # q: [B, Sq, Hq, D] -> [B, Hq, Sq, D]
    q_t = q.transpose(1, 2)
    k_t = k_expanded.transpose(1, 2)  # [B, Hq, Sk, D]
    v_t = v_expanded.transpose(1, 2)  # [B, Hq, Sk, D]

    # Compute scores: [B, Hq, Sq, Sk]
    scores = torch.matmul(q_t, k_t.transpose(2, 3)) * sm_scale

    # Apply causal mask
    if causal:
        q_pos = torch.arange(Sq, device=q.device)
        k_pos = torch.arange(Sk, device=q.device)
        causal_mask = q_pos[:, None] >= k_pos[None, :]
        token_mask = token_mask & causal_mask[None, None, :, :]

    # Apply sparse mask
    scores = scores.masked_fill(~token_mask, float("-inf"))

    # Softmax
    attn = torch.softmax(scores, dim=-1, dtype=torch.float32)
    # Handle all-inf rows (no valid KV tokens)
    attn = torch.where(torch.isnan(attn), torch.zeros_like(attn), attn)

    # Attention output
    out = torch.matmul(attn.to(v_t.dtype), v_t)

    # Back to [B, Sq, Hq, D]
    out = out.transpose(1, 2).contiguous().to(input_dtype)
    return out
