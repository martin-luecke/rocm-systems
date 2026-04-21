; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=c2_permlane_swap_kernel 2>/dev/null | %FileCheck %s
;
; The P4 rewrite (v_permlane16_swap_b32 lift) has landed — see
; the permlane16_swap row of hotswap/docs/wave-size-translation.md
; §5.3. The classifier's LaneGroupShuffle site accepts
; V_PERMLANE16_SWAP_B32 as outcome (b) because
; `handle_valu_cross_lane.cpp` emulates the two-VGPR exchange
; through paired `llvm.amdgcn.ds.bpermute` calls (the same target-
; independent path the P2 permlane16/permlanex16 emulation uses,
; for the same reason: gfx942 lacks native isel for
; `llvm.amdgcn.permlane16.swap`, per upstream LLVM's
; `test/CodeGen/AMDGPU/llvm.amdgcn.permlane16.swap.ll` ERR-SDAG
; assertion).
;
; This test asserts:
;   1. The raise succeeds (the classifier marks
;      V_PERMLANE16_SWAP_B32 [implemented]).
;   2. The emitted IR contains TWO calls to `llvm.amdgcn.ds.bpermute`
;      — one per output VGPR (vdst and src0_out). The P4 handler
;      reuses the partner-lane / byte-address chain across both
;      calls, so the bpermutes share their first operand under CSE
;      but each consumes a different second operand (vdst_in vs
;      src0_in).
;   3. The signature property is the partner-lane XOR with 0x10
;      (= 16): each lane's source-lane index is `lane_id XOR 16`.
;      Matching `xor i32 %{{.*}}, 16` in the byte-address chain
;      pins the swap-partner semantics without asserting on SSA
;      names.
;   4. The intrinsic declaration is present.

; CHECK-LABEL: define amdgpu_kernel void @c2_permlane_swap_kernel(

; The XOR-16 partner computation must precede the bpermutes.
; CHECK:      xor i32 %{{[^,]+}}, 16

; Two ds_bpermute calls, one per output VGPR.
; CHECK:      call i32 @llvm.amdgcn.ds.bpermute(
; CHECK:      call i32 @llvm.amdgcn.ds.bpermute(

; The intrinsic declaration must be present.
; CHECK:      declare {{.*}}i32 @llvm.amdgcn.ds.bpermute(i32, i32)

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	c2_permlane_swap_kernel
	.p2align	8
	.type	c2_permlane_swap_kernel,@function
c2_permlane_swap_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b32 s4, s[0:1], 0x1c
	s_bfe_u32 s5, ttmp6, 0x4000c
	s_wait_xcnt 0x0
	s_load_b128 s[0:3], s[0:1], 0x0
	s_add_co_i32 s5, s5, 1
	s_and_b32 s6, ttmp6, 15
	s_mul_i32 s5, ttmp9, s5
	s_getreg_b32 s7, hwreg(HW_REG_IB_STS2, 6, 4)
	s_add_co_i32 s6, s6, s5
	s_wait_kmcnt 0x0
	s_and_b32 s4, s4, 0xffff
	s_cmp_eq_u32 s7, 0
	s_cselect_b32 s5, ttmp9, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	v_mad_u32 v2, s5, s4, v0
	s_clause 0x1
	global_load_b32 v0, v2, s[0:1] scale_offset
	global_load_b32 v1, v2, s[2:3] scale_offset
	s_wait_loadcnt 0x0
	;;#ASMSTART
	v_permlane16_swap_b32 v0, v1
	
	;;#ASMEND
	s_clause 0x1
	global_store_b32 v2, v0, s[0:1] scale_offset
	global_store_b32 v2, v1, s[2:3] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel c2_permlane_swap_kernel
		.amdhsa_kernarg_size 272
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 8
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 2
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
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         20
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         24
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         28
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         30
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         32
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         34
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         36
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         38
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         56
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         80
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 272
    .max_flat_workgroup_size: 1024
    .name:           c2_permlane_swap_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         c2_permlane_swap_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
