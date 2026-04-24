// Source for permlane16_swap_divergent_exec_gfx1250.hsaco —
// wave32-source test kernel that exercises the per-destination-
// lane EXEC-gate contract of `v_permlane16_swap_b32`.
//
// Why this fixture exists
// =======================
//
// The MI400 Shader Programming Guide § V_PERMLANE16_SWAP_B32
// pragma distinguishes the write gate per destination lane:
//
//   for lane in 0:15 do tmp[lane] = VGPR[lane][SRC0] endfor;
//   for lane in 0:15 do
//     if EXEC[lane]:    VGPR[lane][SRC0]  = VGPR[lane+16][VDST]
//     if EXEC[lane+16]: VGPR[lane+16][VDST] = tmp[lane]
//   endfor
//
// i.e. the temp-read of `VGPR[lane][SRC0]` for lanes 0..15 is
// UNCONDITIONAL on EXEC, but each WRITE is gated by the
// destination lane's EXEC bit independently.  The sibling
// `permlane16_swap_wave32_kernel` fixture only exercises the
// EXEC=all-active convergent case; this fixture drops EXEC on
// odd lanes across the swap so the per-lane write gate is what
// determines whether lane L keeps its initial value or picks up
// its partner's data.
//
// Build:
//   hipcc --offload-arch=gfx1250 --genco \
//     test_data/gfx1250/permlane16_swap_divergent_exec_kernel.hip \
//     -o test_data/gfx1250/permlane16_swap_divergent_exec_gfx1250.hsaco
//   /opt/rocm-7.2.1/lib/llvm/bin/clang-offload-bundler \
//     --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx1250 \
//     --input=test_data/gfx1250/permlane16_swap_divergent_exec_gfx1250.hsaco \
//     --output=test_data/gfx1250/permlane16_swap_divergent_exec_gfx1250.hsaco \
//     --unbundle
//
// Per-lane setup:
//   vdst_in[L]  = L
//   src0_in[L]  = 1000 + L
//
// EXEC at the swap site:
//   active lanes: L even (L & 1 == 0)
//   inactive lanes: L odd
//
// Expected after the swap (EXEC-gated asymmetric per-lane
// semantic, combining the pragma's `if EXEC[lane]:` /
// `if EXEC[lane+16]:` writes with the `isLow` split of the
// sibling fixture):
//
//   isLow[L]   = (L & 16) == 0
//   active[L]  = (L & 1) == 0
//
// If active[L]:
//   new_vdst[L]      = isLow ? L                : 1000 + (L XOR 16)
//   new_src0_out[L]  = isLow ? (L XOR 16)       : 1000 + L
// else (inactive):
//   new_vdst[L]      = L                        // initial kept
//   new_src0_out[L]  = 1000 + L                 // initial kept
//
// For even L the partner `L XOR 16` is ALSO even (XOR 16
// preserves bit 0), so every active lane has an active partner
// and the swap proceeds exactly as the convergent sibling.
// The asymmetry shows up only on ODD lanes: their initial
// values must survive unchanged across the swap — if the
// emulation's writes leak through the EXEC gate, those lanes
// would pick up swapped data instead.

#include <hip/hip_runtime.h>

extern "C" __global__ __launch_bounds__(32)
    void permlane16_swap_divergent_exec_kernel(int *out_vdst,
                                                int *out_src0_out) {
  int L = threadIdx.x;
  int v = L;
  int u = 1000 + L;
  if ((L & 1) == 0) {
    // Only even lanes participate in the swap.  Odd lanes take
    // the else-branch (which runs no instructions) so their
    // EXEC bit is 0 at the swap site.
    asm volatile(
        "v_permlane16_swap_b32 %[v], %[u]\n\t"
        : [v] "+v"(v), [u] "+v"(u));
  }
  out_vdst[L] = v;
  out_src0_out[L] = u;
}
