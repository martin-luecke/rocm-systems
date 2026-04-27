# GPT-OSS Kernel Fixture Correctness - 2026-04-27

## Summary

We added and exercised an exact per-kernel fixture workflow for the GPT-OSS
SGLang path.  The intent is to separate two questions:

1. Did Salmon translate the code object?
2. Does the exact SGLang kernel launch compute the same final tensor state as
   native gfx942?

This report covers the first full-value capture/replay pass using real SGLang
launch values.  These are debug artifacts, not lightweight committed tests.

## 2026-04-27 Update: Schema v2 Replay

After the first report snapshot, the fixture schema and replay tool were
tightened:

- tensor strides and tensor descriptors are reconstructed explicitly;
- `triton_kernels.tensor.Tensor` wrappers are serialized structurally;
- dynamic Triton kernels now capture launch-time generated source;
- replay statuses distinguish `pass`, `wrong-result`,
  `salmon-translate-fail`, `salmon-runtime-fail`,
  `native-runtime-fail`, and `replay-unsupported`;
- per-run fixture indexes are generated as `index.md` / `index.json`.

Current primary indexes:

- Full capture index: `/home/mluecke/gpt_oss_capture/fixtures_full_v4/index.md`
- Dynamic-kernel index: `/home/mluecke/gpt_oss_capture/fixtures_dynamic_v5/index.md`

Storage after cleanup is about `57G` under `/home/mluecke/gpt_oss_capture`.
Older v2/v3 captures were removed; v4 and dynamic_v5 are the current artifacts.

## Artifact Locations

- Full fixture directory: `/home/mluecke/gpt_oss_capture/fixtures_full_v4`
- Full GPT-OSS comparison run: `/home/mluecke/gpt_oss_capture/compare_full_v4`
- Full replay results: `/home/mluecke/gpt_oss_capture/fixtures_full_v4/replay_all_v2.jsonl`
- Dynamic-only fixture directory: `/home/mluecke/gpt_oss_capture/fixtures_dynamic_v5`
- Dynamic-only replay results: `/home/mluecke/gpt_oss_capture/fixtures_dynamic_v5/replay_all_v2.jsonl`
- Total artifact size after cleanup: about `57G`

The older temporary capture under `/tmp/gpt_oss_kernel_fixtures_full` was
removed to avoid leaving an extra 46 GB on the machine.

## Capture Command Shape

The capture was run from:

```bash
cd /home/mluecke/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/hotswap/transpiler/tools/triton_corpus_runner
TRITON_ALWAYS_COMPILE=1 \
TRITON_CACHE_DIR=/home/mluecke/gpt_oss_capture/triton_cache_full_v4 \
GPT_OSS_KERNEL_FIXTURE_DIR=/home/mluecke/gpt_oss_capture/fixtures_full_v4 \
GPT_OSS_KERNEL_FIXTURE_FILTER='*' \
GPT_OSS_KERNEL_FIXTURE_MAX_PER_KERNEL=1 \
GPT_OSS_KERNEL_FIXTURE_MAX_TENSOR_BYTES=4294967296 \
GPT_OSS_SGLANG_DECOMPOSE_MXFP4_FOR_GFX1250=1 \
HSA_SALMON_TOOL_TIMEOUT_S=600 \
HSA_SALMON_LAUNCH_LOG=/home/mluecke/gpt_oss_capture/compare_full_v4/launch.jsonl \
GPT_OSS_SGLANG_RETURN_HIDDEN_STATES=0 \
GPT_OSS_SGLANG_PROMPTS_JSON='[{"name":"identity_salmon","prompt":"Answer with exactly one word: salmon","expected_text":"salmon"}]' \
python3 compare_gpt_oss_sglang.py \
  --native-mxfp4 \
  --out-dir /home/mluecke/gpt_oss_capture/compare_full_v4 \
  --timeout 1200
```

The GPT-OSS comparison did not complete end-to-end, but fixture capture reached
the target kernel frontier and produced independent replay artifacts.

## Translation Proof

From `/home/mluecke/gpt_oss_capture/compare_full_v4/compare_report.md`:

| Check | Value |
|---|---:|
| Salmon proof valid | yes |
| Child process forced gfx1250 target | yes |
| Saw gfx1250 code object (`orig_mach=0x49`) | yes |
| Saw loader transpile decision | yes |
| Salmon OK count | 16 |
| Salmon FAILED count | 0 |
| First unsupported instruction | n/a |

Note: the full fixture replay itself triggers additional Salmon compilation
and failure/success evidence per fixture.  The table above only describes the
original fixture-capture GPT-OSS run.

## Captured Fixtures

The v4 full capture plus dynamic_v5 capture produced exact launch fixtures for
the current reached GPT-OSS Triton kernel frontier:

| Kernel | Fixture status |
|---|---|
| `create_flashinfer_kv_indices_triton` | captured |
| `_fused_qk_rope_reshape_and_cache_kernel` | captured |
| `_fwd_grouped_kernel_stage1` | captured |
| `_fwd_kernel_stage2` | captured |
| `_topk_forward` | captured |
| `_sum_bitmatrix_rows` | captured |
| `_bitmatrix_metadata_compute_stage1` | captured |
| `_bitmatrix_metadata_compute_stage2` | captured |
| `_ragged_tensor_metadata_memset` | captured |
| `_ragged_tensor_metadata_compute` | captured |
| `_matmul_ogs` | captured and replay-classified via dynamic_v5 |
| `_reduce` | captured and replay-classified via dynamic_v5 |
| `write_req_to_token_pool_triton` | captured |
| `compute_position_kernel` | captured |
| `_fwd_kernel` from `extend_attention` | captured |
| `get_num_kv_splits_triton` | captured |
| `_upcast_from_mxfp` | captured and replay-classified via dynamic_v5 |

## Replay Results

Replay command:

```bash
/home/mluecke/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/hotswap/transpiler/tools/triton_corpus_runner/.venv-sglang-rocm720/bin/python \
  /home/mluecke/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/hotswap/transpiler/tools/compare_correctness/gpt_oss_fixture_replay.py \
  /home/mluecke/gpt_oss_capture/fixtures_full_v4 \
  --json /home/mluecke/gpt_oss_capture/fixtures_full_v4/replay_all_v2.jsonl \
  --timeout 900
```

Full v4 aggregate result:

| Status | Count |
|---|---:|
| pass | 13 |
| wrong-result | 11 |
| native-runtime-fail | 4 |
| salmon-translate-fail | 1 |

Dynamic v5 aggregate result:

| Status | Count |
|---|---:|
| pass | 4 |
| salmon-translate-fail | 1 |

Per-kernel result:

| Status | Kernel | Detail |
|---|---|---|
| pass | `create_flashinfer_kv_indices_triton` | exact fixtures match native |
| pass | `_fused_qk_rope_reshape_and_cache_kernel` | exact fixtures match native |
| pass | `get_num_kv_splits_triton` | exact fixture matches native |
| pass | `write_req_to_token_pool_triton` | exact fixture matches native |
| pass | `compute_position_kernel` | exact fixture matches native |
| pass | `_sum_bitmatrix_rows` | exact fixtures match native |
| pass | `_ragged_tensor_metadata_compute` | exact fixtures match native |
| pass | `_upcast_from_mxfp` | dynamic_v5 exact fixture matches native |
| pass | `_reduce` | dynamic_v5 exact fixtures match native |
| pass | `_matmul_ogs` | dynamic_v5 Salmon-path decomposed-BF16 fixture matches native |
| wrong-result | `_fwd_grouped_kernel_stage1` | `mismatches=1817088`, tensor `arg6` |
| wrong-result | `_fwd_kernel_stage2` | one fixture fails with `mismatches=760256`, tensor `arg2`; another fixture passes |
| wrong-result | `_topk_forward` | `mismatches=2048`, tensor `arg2[0]` |
| wrong-result | `_bitmatrix_metadata_compute_stage1` | `mismatches=21`, tensor `arg5` |
| wrong-result | `_bitmatrix_metadata_compute_stage2` | `mismatches=1536` / `1424`, tensor `arg0` |
| wrong-result | `_ragged_tensor_metadata_memset` | `mismatches=108` / `132`, tensor `arg2` |
| salmon-translate-fail | `extend_attention._fwd_kernel` | unsupported `v_min_i64` |
| salmon-translate-fail | `_matmul_ogs` | native MXFP4 fixture cannot lower forced-gfx1250 path: `PassManager::run failed`; the decomposed-BF16 Salmon-path matmul fixture passes |

## Interpretation

This is now a usable per-kernel correctness triage view for the captured
frontier.  Dynamic modules are replay-classified:

- `_reduce` and `_upcast_from_mxfp` pass.
- `_matmul_ogs` has two meaningful statuses:
  - the decomposed-BF16 Salmon-path matmul fixture passes;
  - the native MXFP4/mixed-dot fixture still fails forced-gfx1250 Triton
    lowering with `PassManager::run failed`, which is a compile/translation
    limitation rather than a replay-loader gap.

The current correctness failures are concentrated in decode attention stage
kernels, top-k, bitmatrix metadata, and ragged metadata memset.

## Immediate Next Work

1. Investigate the mismatching metadata/topk/attention kernels first:
   `_topk_forward`, bitmatrix metadata, ragged metadata, and decode attention
   stage1/stage2.
2. Add `v_min_i64` Salmon support for `extend_attention._fwd_kernel`.
3. Decide whether the native MXFP4/mixed-dot `_matmul_ogs` fixture should stay
   as a known forced-gfx1250 Triton compile limitation while the decomposed
   BF16 path is the correctness target.
4. Add focused smaller correctness-suite recipes for every mismatch that can
   be reduced while preserving the same generated kernel path.
5. Keep these full-value fixtures as debug artifacts under
   `/home/mluecke/gpt_oss_capture`, not as committed test data.
