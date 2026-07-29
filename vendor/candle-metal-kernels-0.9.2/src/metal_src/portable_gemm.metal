#include <metal_stdlib>

using namespace metal;

struct PortableGemmParams {
  uint m;
  uint n;
  uint k;
  uint batch;
  uint lhs_stride_b;
  uint lhs_stride_m;
  uint lhs_stride_k;
  uint rhs_stride_b;
  uint rhs_stride_k;
  uint rhs_stride_n;
};

kernel void portable_gemm_f32(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant PortableGemmParams& p [[buffer(3)]],
    uint3 tid [[thread_position_in_grid]],
    uint3 local_tid [[thread_position_in_threadgroup]]) {
  constexpr uint TILE = 16;
  threadgroup float lhs_tile[TILE][TILE];
  threadgroup float rhs_tile[TILE][TILE];

  const uint col = tid.x;
  const uint row = tid.y;
  const uint batch = tid.z;
  const uint local_col = local_tid.x;
  const uint local_row = local_tid.y;
  float acc = 0.0f;
  const uint lhs_base = batch * p.lhs_stride_b + row * p.lhs_stride_m;
  const uint rhs_base = batch * p.rhs_stride_b + col * p.rhs_stride_n;

  for (uint tile = 0; tile < p.k; tile += TILE) {
    const uint lhs_k = tile + local_col;
    const uint rhs_k = tile + local_row;
    lhs_tile[local_row][local_col] =
        (row < p.m && lhs_k < p.k && batch < p.batch)
            ? lhs[lhs_base + lhs_k * p.lhs_stride_k]
            : 0.0f;
    rhs_tile[local_row][local_col] =
        (col < p.n && rhs_k < p.k && batch < p.batch)
            ? rhs[rhs_base + rhs_k * p.rhs_stride_k]
            : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kk = 0; kk < TILE; ++kk) {
      acc += lhs_tile[local_row][kk] * rhs_tile[kk][local_col];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (col < p.n && row < p.m && batch < p.batch) {
    out[(batch * p.m + row) * p.n + col] = acc;
  }
}
