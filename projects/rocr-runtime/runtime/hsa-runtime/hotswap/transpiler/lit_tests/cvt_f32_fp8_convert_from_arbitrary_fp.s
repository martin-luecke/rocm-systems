; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx1250 --emit-ir=cvt_fp8_kernel 2>/dev/null | %FileCheck %s
;
; Lift test for the FP8 read-side conversion family on an OCP-FP8 source
; ISA (gfx1250). The four hardware shapes covered:
;
;   v_cvt_f32_fp8        — scalar one-byte FP8 → f32 (E4M3FN)
;   v_cvt_f32_bf8        — scalar one-byte BF8 → f32 (E5M2)
;   v_cvt_pk_f32_fp8     — packed two-byte FP8 → v2f32 (E4M3FN), word_sel=0
;   v_cvt_pk_f32_bf8     — packed two-byte BF8 → v2f32 (E5M2), word_sel=1
;
; INVARIANTS PINNED:
;
;   1. On OCP-FP8 sources (`ISAProfile::hasOcpFp8 == true`), the four
;      read-side conversions lift to the portable
;      `llvm.convert.from.arbitrary.fp` intrinsic — NOT the AMDGCN
;      `llvm.amdgcn.cvt.{,pk_}f32.{fp8,bf8}` family. The portable
;      shape is recognised by AMDGPU custom lowering (PR
;      llvm/llvm-project#194144) and selects back to the same hardware
;      instructions on capable targets, while remaining target-agnostic
;      for cross-ISA re-lowering.
;
;   2. The format metadata distinguishes FP8 (`Float8E4M3FN`) from BF8
;      (`Float8E5M2`). A regression that swapped the two would silently
;      decode mantissa/exponent wrong.
;
;   3. Scalar form passes a plain `i8` argument; packed form passes a
;      `<2 x i8>` extracted from the lower or upper 16 bits of the
;      source dword per `op_sel:[0]`. Anything else means the byte
;      slicing is wrong.
;
; NEGATIVE PINS:
;
;   * NO `llvm.amdgcn.cvt.f32.fp8`, `llvm.amdgcn.cvt.f32.bf8`,
;     `llvm.amdgcn.cvt.pk.f32.fp8`, or `llvm.amdgcn.cvt.pk.f32.bf8`
;     anywhere — those would mean the OCP dispatch silently fell
;     through to the legacy AMDGCN intrinsic emission.

; CHECK-LABEL: define amdgpu_kernel void @cvt_fp8_kernel(

; Scalar v_cvt_f32_fp8 → convert.from.arbitrary.fp.f32.i8 with Float8E4M3FN.
; CHECK-DAG: call float @llvm.convert.from.arbitrary.fp.f32.i8(i8 %{{[^,]+}}, metadata !"Float8E4M3FN")

; Scalar v_cvt_f32_bf8 → convert.from.arbitrary.fp.f32.i8 with Float8E5M2.
; CHECK-DAG: call float @llvm.convert.from.arbitrary.fp.f32.i8(i8 %{{[^,]+}}, metadata !"Float8E5M2")

; Packed v_cvt_pk_f32_fp8 → convert.from.arbitrary.fp.v2f32.v2i8.
; CHECK-DAG: call <2 x float> @llvm.convert.from.arbitrary.fp.v2f32.v2i8(<2 x i8> %{{[^,]+}}, metadata !"Float8E4M3FN")

; Packed v_cvt_pk_f32_bf8 → convert.from.arbitrary.fp.v2f32.v2i8.
; CHECK-DAG: call <2 x float> @llvm.convert.from.arbitrary.fp.v2f32.v2i8(<2 x i8> %{{[^,]+}}, metadata !"Float8E5M2")

; Negative: the legacy AMDGCN FP8 read-side intrinsics must NOT appear
; on OCP sources — if they do, the dispatch in handle_valu.cpp regressed.
; CHECK-NOT: @llvm.amdgcn.cvt.f32.fp8
; CHECK-NOT: @llvm.amdgcn.cvt.f32.bf8
; CHECK-NOT: @llvm.amdgcn.cvt.pk.f32.fp8
; CHECK-NOT: @llvm.amdgcn.cvt.pk.f32.bf8

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	cvt_fp8_kernel
	.p2align	8
	.type	cvt_fp8_kernel,@function
cvt_fp8_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_mov_b32_e32 v0, 0x12345678
	s_wait_kmcnt 0x0
	;;#ASMSTART
	v_cvt_f32_fp8_e64 v1, v0
	v_cvt_f32_bf8_e64 v2, v0
	v_cvt_pk_f32_fp8 v[4:5], v0
	v_cvt_pk_f32_bf8 v[6:7], v0 op_sel:[1,0]
	;;#ASMEND
	v_add_f32_e32 v1, v1, v2
	v_add_f32_e32 v3, v4, v6
	v_add_f32_e32 v1, v1, v3
	global_store_b32 v0, v1, s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel cvt_fp8_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 4
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 2
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           cvt_fp8_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         cvt_fp8_kernel.kd
    .vgpr_count:     8
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
