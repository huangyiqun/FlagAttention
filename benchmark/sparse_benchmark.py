import math
import torch
import triton

import flag_attn


def generate_q2k_indices(B, Hkv, num_q_blocks, topK, num_kv_blocks, device="cuda"):
    """Generate random q2k block indices for benchmarking."""
    q2k = torch.full((B, Hkv, num_q_blocks, topK), -1, dtype=torch.int32, device=device)
    for b in range(B):
        for h in range(Hkv):
            for qb in range(num_q_blocks):
                available = min(topK, num_kv_blocks)
                perm = torch.randperm(num_kv_blocks, device=device)[:available]
                perm, _ = perm.sort()
                q2k[b, h, qb, :available] = perm.to(torch.int32)
    return q2k


configs = [triton.testing.Benchmark(
    x_names=['SEQ_LEN'],
    x_vals=[1024, 2048, 4096, 8192, 16384, 32768],
    line_arg='provider',
    line_vals=['flag_attn_sparse', 'torch_ref'],
    line_names=['flag_attn (sparse)', 'torch (masked)'],
    styles=[('red', '-'), ('green', '-')],
    ylabel='tflop/s',
    plot_name=f'sparse_attention_topk-{TOPK}_d-{D}_causal-{causal}_dtype-{dtype}',
    args={'D': D, 'dtype': dtype, 'causal': causal, 'TOPK': TOPK}
) for causal in [False, True]
    for D in [128]
    for dtype in [torch.bfloat16]
    for TOPK in [8, 16]]


@triton.testing.perf_report(configs)
def bench_sparse_attention(SEQ_LEN, D, causal, TOPK, provider, dtype=torch.bfloat16, device="cuda"):
    warmup = 25
    rep = 100

    B = 1
    Hq = 16
    Hkv = 4
    blk_kv = 128
    BLOCK_M = 128

    Sq = SEQ_LEN
    Sk = SEQ_LEN
    num_q_blocks = Sq // BLOCK_M
    num_kv_blocks = Sk // blk_kv

    if TOPK > num_kv_blocks:
        return float('nan')

    q = torch.randn((B, Sq, Hq, D), dtype=dtype, device=device)
    k = torch.randn((B, Sk, Hkv, D), dtype=dtype, device=device)
    v = torch.randn((B, Sk, Hkv, D), dtype=dtype, device=device)
    q2k_indices = generate_q2k_indices(B, Hkv, num_q_blocks, TOPK, num_kv_blocks, device=device)

    # FLOPs for sparse attention: only topK blocks are computed per Q-tile
    # Each Q token: 2 * D * topK * blk_kv (for QK) + 2 * topK * blk_kv * D (for PV)
    flops = 2.0 * B * Hq * Sq * TOPK * blk_kv * D * 2  # *2 for QK + PV

    if provider == 'flag_attn_sparse':
        fn = lambda: flag_attn.sparse_attention(q, k, v, q2k_indices, TOPK, blk_kv=blk_kv, causal=causal)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    elif provider == 'torch_ref':
        fn = lambda: flag_attn.testing.sparse_attention(
            q, k, v, q2k_indices, TOPK, blk_kv=blk_kv, causal=causal, upcast=False
        )
        try:
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        except torch.cuda.OutOfMemoryError:
            ms = float('inf')

    return flops / ms * 1e-9  # TFLOP/s


if __name__ == "__main__":
    bench_sparse_attention.run(print_data=True, save_path="./sparse_benchmark_results")
