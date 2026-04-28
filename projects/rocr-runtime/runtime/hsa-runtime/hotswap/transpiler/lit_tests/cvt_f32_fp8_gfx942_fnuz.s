; RUN: %llvm_mc -mcpu=gfx942 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=cvt_fp8_fnuz_kernel 2>/dev/null | %FileCheck %s
;
; gfx9.4.0 (gfx942) FP8 read-side counter-test for the OCP path in
; cvt_f32_fp8_convert_from_arbitrary_fp.s.
;
; gfx942 hardware reads FP8 lanes with FNUZ semantics
; (`Float8E4M3FNUZ` / `Float8E5M2FNUZ`). Today's
; `APFloatBase::getArbitraryFPSemantics` and the SelectionDAG
; CONVERT_FROM_ARBITRARY_FP expansion in `LegalizeDAG.cpp` only
; recognise the OCP set; emitting a `convert.from.arbitrary.fp` with
; FNUZ metadata against current LLVM would error out at legalization.
;
; Therefore: gfx942 lifts MUST stay on the AMDGCN
; `llvm.amdgcn.cvt.f32.{fp8,bf8}` / `llvm.amdgcn.cvt.pk.f32.{fp8,bf8}`
; intrinsics, whose semantics are target-defined and silently track
; the source / target subtarget bits. The dispatch toggle lives on
; `ISAProfile::hasOcpFp8` (false for gfx942 because the FP8Insts /
; GFX950Insts feature combo flags it as the FNUZ generation).
;
; INVARIANTS PINNED:
;
;   1. Scalar v_cvt_f32_fp8 lifts to the AMDGCN intrinsic, NOT the
;      portable convert.from.arbitrary.fp shape.
;   2. Packed v_cvt_pk_f32_fp8 lifts to amdgcn.cvt.pk.f32.fp8 with
;      its existing (i32, i1 word_sel) signature.

; CHECK-LABEL: define amdgpu_kernel void @cvt_fp8_fnuz_kernel(

; Scalar lift stays on the AMDGCN intrinsic.
; CHECK-DAG: call float @llvm.amdgcn.cvt.f32.fp8
; CHECK-DAG: call float @llvm.amdgcn.cvt.f32.bf8

; Packed lift stays on the AMDGCN intrinsic.
; CHECK-DAG: call <2 x float> @llvm.amdgcn.cvt.pk.f32.fp8
; CHECK-DAG: call <2 x float> @llvm.amdgcn.cvt.pk.f32.bf8

; Negative: portable shape MUST NOT appear on FNUZ sources — neither
; the OCP nor the FNUZ format strings round-trip through it on
; today's LLVM.
; CHECK-NOT: @llvm.convert.from.arbitrary.fp

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	cvt_fp8_fnuz_kernel
	.p2align	8
	.type	cvt_fp8_fnuz_kernel,@function
cvt_fp8_fnuz_kernel:
	s_load_dwordx2 s[0:1], s[0:1], 0x0
	v_mov_b32_e32 v0, 0x12345678
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	v_cvt_f32_fp8_e64 v1, v0
	v_cvt_f32_bf8_e64 v2, v0
	v_cvt_pk_f32_fp8 v[4:5], v0
	v_cvt_pk_f32_bf8 v[6:7], v0
	;;#ASMEND
	v_add_f32_e32 v1, v1, v2
	v_add_f32_e32 v3, v4, v6
	v_add_f32_e32 v1, v1, v3
	global_store_dword v0, v1, s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel cvt_fp8_fnuz_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 4
		.amdhsa_accum_offset 8
		.amdhsa_float_denorm_mode_32 3
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
    .name:           cvt_fp8_fnuz_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         cvt_fp8_fnuz_kernel.kd
    .vgpr_count:     8
    .wavefront_size: 64
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
