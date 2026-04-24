; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=s_load_b96_kernarg_kernel 2>/dev/null | %FileCheck %s
;
; Lift test for `s_load_b96` over a kernarg buffer that contains a
; `by_value` aggregate larger than 8 bytes.  Pins:
;
;   1. The 16-byte `by_value` Big16 struct is decomposed into four
;      i32 IR arguments at byte offsets 0/4/8/12, followed by the
;      `global_buffer` ptr at offset 16 — not a single i128 / [16 x i8]
;      / aggregate type.  See the `byValueSize > 8 && % 4 == 0` branch
;      in raiser.cpp's `paramTypes` build.
;
;   2. The `s_load_b96 s[0:2], s[0:1], 0x4` reaches the three kernarg
;      dwords at offsets 4/8/12 and the result flows into the per-VGPR
;      phi-under-EXEC shape (proves the lift didn't refuse and stub
;      the kernel body).
;
; The phi RHS basic-block names and intermediate SSA names are LLVM-
; printer-renumbered, so we use `{{[a-zA-Z_0-9]+}}` placeholders.

; The kernel signature must decompose the 16-byte by_value into four
; i32 slots followed by the global_buffer ptr.
; CHECK-LABEL: define amdgpu_kernel void @s_load_b96_kernarg_kernel(
; CHECK-SAME: i32 %arg0
; CHECK-SAME: i32 %arg1
; CHECK-SAME: i32 %arg2
; CHECK-SAME: i32 %arg3
; CHECK-SAME: ptr addrspace(1) %arg4

; Kernarg fetches go through `llvm.amdgcn.kernarg.segment.ptr` + a
; real load.
; CHECK: call ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()

; The s_load_b96 result must reach the per-VGPR phi-under-EXEC shape
; via real kernarg loads (proves the lift didn't refuse and stub the
; body).
; CHECK-DAG: phi i32 [ %smem_load{{[0-9]*}}, %{{[a-zA-Z_0-9]+}} ]

; A refusal or zero/undef substitution on the active (do-block) arm
; would surface as one of these.  `undef` is tolerated only on the
; inactive arm of the EXEC predicate.
; CHECK-NOT: phi i32 [ i32 0, %{{[a-zA-Z_0-9]+}} ]
; CHECK-NOT: phi i32 [ i32 undef, %{{[a-zA-Z_0-9]+}} ]

; The by_value decomposition must NOT collapse back into a single
; argument of any of these non-decomposed shapes.
; CHECK-NOT: define amdgpu_kernel void @s_load_b96_kernarg_kernel(i32 %arg0, ptr addrspace(1) %arg1)
; CHECK-NOT: define amdgpu_kernel void @s_load_b96_kernarg_kernel({{[^,]+\[16 x i8\]}}
; CHECK-NOT: define amdgpu_kernel void @s_load_b96_kernarg_kernel(i128

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	s_load_b96_kernarg_kernel
	.p2align	8
	.type	s_load_b96_kernarg_kernel,@function
s_load_b96_kernarg_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b64 s[2:3], s[0:1], 0x10
	s_wait_xcnt 0x0
	;;#ASMSTART
	s_load_b96 s[0:2], s[0:1], 4
	s_wait_kmcnt 0
	
	;;#ASMEND
	v_dual_mov_b32 v7, 0 :: v_dual_mov_b32 v4, s0
	v_dual_mov_b32 v5, s1 :: v_dual_mov_b32 v6, s8
	s_wait_kmcnt 0x0
	v_mov_b64_e32 v[0:1], s[4:5]
	v_mov_b64_e32 v[2:3], s[6:7]
	s_clause 0x1
	global_store_b96 v7, v[4:6], s[2:3]
	global_store_b128 v7, v[0:3], s[2:3] offset:12
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel s_load_b96_kernarg_kernel
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 9
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .offset:         0, .size:           16, .value_kind:     by_value }
      - { .address_space:  global, .offset:         16, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .max_flat_workgroup_size: 1024
    .name:           s_load_b96_kernarg_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     9
    .symbol:         s_load_b96_kernarg_kernel.kd
    .vgpr_count:     8
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
