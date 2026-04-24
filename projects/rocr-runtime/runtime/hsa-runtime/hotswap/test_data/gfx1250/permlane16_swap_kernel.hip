// Source for permlane16_swap_gfx1250.hsaco — pinned end-to-end test
// for the P4 rewrite (`v_permlane16_swap_b32` lift; see the
// permlane16_swap row of hotswap/docs/wave-size-translation.md
// §5.3).
//
// This is the source kernel for a Gfx1250Gpu GTest that:
//   1. Reads the pre-built gfx1250 (wave32) `.hsaco`.
//   2. Lifts it through the transpiler pipeline to gfx942 (wave64).
//   3. Runs the lifted kernel on gfx942 hardware.
//   4. Verifies per-lane output matches the expected XOR-16 swap.
//
// The test exists to give the P4 emulation a CI-resident regression
// gate. Without it, the only check that the lifted ds_bpermute
// chain actually swaps correctly is the embedded probe results in
// the handler comment block — documentation, not a test.
//
// Build (regenerate when the kernel is changed):
//   hipcc --offload-arch=gfx1250 --genco \
//     test_data/gfx1250/permlane16_swap_kernel.hip \
//     -o test_data/gfx1250/permlane16_swap_gfx1250.hsaco
//   /opt/rocm-7.2.1/lib/llvm/bin/clang-offload-bundler \
//     --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx1250 \
//     --input=test_data/gfx1250/permlane16_swap_gfx1250.hsaco \
//     --output=test_data/gfx1250/permlane16_swap_gfx1250.hsaco \
//     --unbundle
//
// Per-lane setup:
//   vdst_in[L]  = L
//   src0_in[L]  = 1000 + L
//
// Expected after the swap — ASYMMETRIC per-destination-lane
// semantic from the MI400 Shader Programming Guide pragma §
// V_PERMLANE16_SWAP_B32 (only two of the four 16-lane rows
// move; the other two retain their initial values, see
// hotswap/docs/matrix-translation.md §12.4.7):
//
//   isLow[L]         = (L & 16) == 0
//   new_vdst[L]      = isLow ? vdst_in[L]       : src0_in[L XOR 16]
//                    = isLow ? L                : 1000 + (L XOR 16)
//   new_src0_out[L]  = isLow ? vdst_in[L XOR 16]: src0_in[L]
//                    = isLow ? (L XOR 16)       : 1000 + L
//
// On the lifted gfx942 wave64 kernel each 32-lane half performs
// this swap in isolation (bit 5 of the lane id is preserved by
// `L XOR 16`), so the per-lane formulae above apply to L in
// [0, 15], [16, 31], [32, 47], [48, 63] identically — the test
// verifies all 64 entries.  Pre-Session-8 (2026-04-23) the
// emulation was symmetric, swapping ALL four rows, and this
// test's expectations were `new_vdst[L] = 1000 + (L XOR 16)` /
// `new_src0_out[L] = (L XOR 16)` across all lanes; that
// symmetric over-swap was the root cause of the matmul_fp16
// divergence.

#include <hip/hip_runtime.h>

extern "C" __global__ __launch_bounds__(64) void permlane16_swap_kernel(
    int *out_vdst, int *out_src0_out) {
  int L = threadIdx.x;
  int v = L;
  int u = 1000 + L;
  asm volatile(
      "v_permlane16_swap_b32 %[v], %[u]\n\t"
      : [v] "+v"(v), [u] "+v"(u));
  out_vdst[L] = v;
  out_src0_out[L] = u;
}
