; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=c2_dpp_quad_perm_kernel 2>/dev/null | %FileCheck %s
;
; The P5 rewrite (DPP modifier intrinsic lift) has landed — see
; the DPP row of hotswap/docs/wave-size-translation.md §5.3. This
; test was originally a refuse-loud fixture asserting
; `cross-wave-shuffle-rewrite-pending`. Per its MAINTENANCE block we
; flipped it to a positive test that asserts:
;
;   1. The raise now succeeds (no `%not`; the classifier accepts
;      `DppCrossLane` sites as outcome (b) rewrite-implemented, see
;      `wave_size_obstruction.cpp`'s DPP case).
;   2. The emitted IR contains a call to `llvm.amdgcn.update.dpp.i32`
;      with the DPP16 operand set from the source instruction.
;   3. The intrinsic is overloaded on i32 (the corpus-wide 32-bit
;      DPP assumption documented in `raise_context.cpp:emitUpdateDpp`
;      — 64-bit DPP `report_fatal_error`s pending a future lift).
;
; The DPP modifier values in the fixture are
; `quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1` —
; which encode to `dpp_ctrl = 0xB1 = 177`, `row_mask = 0xF = 15`,
; `bank_mask = 0xF = 15`, `bound_ctrl = true`.

; CHECK-LABEL: define amdgpu_kernel void @c2_dpp_quad_perm_kernel(

; The update.dpp call: 6 args — %old, %src, ctrl, row_mask, bank_mask,
; bound_ctrl. %old and %src reference the same SSA value here because
; the inline-asm fixture uses an in/out `+v` constraint on the only
; operand.
; CHECK: call i32 @llvm.amdgcn.update.dpp.i32(i32 %{{[^,]+}}, i32 %{{[^,]+}}, i32 177, i32 15, i32 15, i1 true)

; Declaration of the intrinsic with the correct overload signature.
; CHECK: declare i32 @llvm.amdgcn.update.dpp.i32(i32, i32, i32 immarg, i32 immarg, i32 immarg, i1 immarg)

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	c2_dpp_quad_perm_kernel
	.p2align	8
	.type	c2_dpp_quad_perm_kernel,@function
c2_dpp_quad_perm_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_clause 0x1
	s_load_b32 s4, s[0:1], 0x14
	s_load_b64 s[2:3], s[0:1], 0x0
	s_wait_xcnt 0x0
	s_bfe_u32 s0, ttmp6, 0x4000c
	s_and_b32 s1, ttmp6, 15
	s_add_co_i32 s0, s0, 1
	s_getreg_b32 s5, hwreg(HW_REG_IB_STS2, 6, 4)
	s_mul_i32 s0, ttmp9, s0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s1, s1, s0
	s_wait_kmcnt 0x0
	s_and_b32 s4, s4, 0xffff
	s_cmp_eq_u32 s5, 0
	s_cselect_b32 s0, ttmp9, s1
	v_mad_u32 v1, s0, s4, v0
	global_load_b32 v0, v1, s[2:3] scale_offset
	s_wait_loadcnt 0x0
	;;#ASMSTART
	v_mov_b32_dpp v0, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	
	;;#ASMEND
	global_store_b32 v1, v0, s[2:3] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel c2_dpp_quad_perm_kernel
		.amdhsa_kernarg_size 264
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 6
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
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         12
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         16
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         20
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         22
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         24
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         26
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         28
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         30
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         48
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         56
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         72
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 264
    .max_flat_workgroup_size: 1024
    .name:           c2_dpp_quad_perm_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         c2_dpp_quad_perm_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
