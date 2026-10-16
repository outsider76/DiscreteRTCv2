# Stacking-cups 30 Hz policy deployment handoff

Status date: 2026-09-16 UTC  
Repository: `/scratch/wangpc/DiscreteRTCv2`  
Task instruction: `Stack the cups.`

This document covers the four policies trained from
`data/stacking_cups_action_30hz`. It is intentionally explicit about which
deployment paths work now and which one still needs a representation-aware
adapter.

## Current state

All four training jobs finished successfully. There are no active training
processes. The pipeline completion record is
[`pipeline_resume.log`](results/training_logs/stacking_cups_action_30hz/pipeline_resume.log).

| Policy | Framework | Final local checkpoint | Deployment status |
| --- | --- | --- | --- |
| Raw base | `QwenPI_v3` | [`steps_50000_pytorch_model.pt`](results/Checkpoints/stacking_cups_action_30hz_qwenpi_raw_h30_2gpu_gb32_50k/checkpoints/steps_50000_pytorch_model.pt) | Standard synchronous server/client path is ready for dry-run validation. |
| Uniform-left B-spline base | `QwenPI_v3_UniformBSpline` | [`steps_50000_pytorch_model.pt`](results/Checkpoints/stacking_cups_action_30hz_qwenpi_uniform_left_s2_h30_c18_2gpu_gb32_50k/checkpoints/steps_50000_pytorch_model.pt) | Standard server/client path is ready; decoding happens inside the framework. |
| Raw ttRTC | `QwenPI_v3_TrainingRTC` | [`steps_5000_pytorch_model.pt`](results/Checkpoints/stacking_cups_action_30hz_qwenpi_raw_h30_training_rtc_prefix1to15_2gpu_gb32_ft5k/checkpoints/steps_5000_pytorch_model.pt) | Existing TrainingRTC server and async client are compatible, with the 30 Hz/H=30 overrides below. |
| Uniform-left B-spline ttRTC | `QwenPI_v3_UniformBSpline_TrainingRTC` | [`steps_5000_pytorch_model.pt`](results/Checkpoints/stacking_cups_action_30hz_qwenpi_uniform_left_s2_h30_c18_training_rtc_prefix4to8_2gpu_gb32_ft5k/checkpoints/steps_5000_pytorch_model.pt) | Vanilla chunk inference works. Learned RTC inference is **not connected to the current decoded-action client/server protocol**; see “B-spline ttRTC gap.” |

Each checkpoint is approximately 10.92 GB. Each run directory also contains
`config.yaml`, `config.full.yaml`, `dataset_statistics.json`, `summary.jsonl`,
and `final_model/pytorch_model.pt`. Use the numbered checkpoint paths above for
an unambiguous handoff.

No `checkpoint_upload.json` or `rtc_checkpoint_upload.json` exists in these
four new run directories as of the status date. Therefore these exact
`stacking_cups_action_30hz` artifacts are locally complete but must not yet be
described as verified Hugging Face uploads.

## Policy contract

- Observation images: two RGB images in this exact order: `global`, then
  `hand`.
- Image size: 224 x 224.
- Language instruction: `Stack the cups.`
- State: normalized 7-vector `[joint_1, ..., joint_6, gripper]`.
- Returned actions: unnormalized absolute 7-vector targets
  `[joint_1, ..., joint_6, gripper]` in robot coordinates.
- Execution frequency: 30 Hz.
- Executable chunk: 30 steps, or one second.
- Normalization key: `new_embodiment`.
- Gripper action data: binary `0/1`. The standard recording server projects
  its decoded gripper output to `0/1` at threshold `0.3` by default.

The generic wrapper loads the framework and its adjacent configuration at
[`policy_wrapper.py#L54-L79`](deployment/model_server/policy_wrapper.py#L54-L79), reports
the executable horizon and representation contract at
[`policy_wrapper.py#L86-L115`](deployment/model_server/policy_wrapper.py#L86-L115), and
uses the original training transforms to unnormalize returned actions at
[`policy_wrapper.py#L217-L274`](deployment/model_server/policy_wrapper.py#L217-L274).
The normalization processor reconstructs those transforms from the checkpoint
metadata at
[`policy_norm_processor.py#L238-L315`](deployment/model_server/policy_norm_processor.py#L238-L315).

The dataset registry entries used during training are
[`data_config.py#L421-L434`](examples/realRobots/Piper/train_files/data_registry/data_config.py#L421-L434).

## B-spline representation contract

The base B-spline action head predicts **18 normalized control points**, each
with 7 values. These are decoded with a cubic, left-clamped B-spline basis into
30 normalized executable actions. It is not a polyline and must never be
implemented by connecting control points.

The exact representation is:

- mode: `uniform_left`
- degree: 3
- native/executable horizon: 30
- control-point count: 18
- knot span: 2 action steps
- frequency: 30 Hz
- regularization: `1e-4`
- end padding: enabled
- implementation: `uniform_left_direct_fit_v1`

The deployed decoder is constructed from these fields at
[`QwenPI_v3_UniformBSpline.py#L25-L68`](starVLA/model/framework/VLM4A/QwenPI_v3_UniformBSpline.py#L25-L68).
The actual curve evaluation is the basis-matrix multiplication
`[30,18] @ [B,18,7] -> [B,30,7]` at
[`QwenPI_v3_UniformBSpline.py#L70-L91`](starVLA/model/framework/VLM4A/QwenPI_v3_UniformBSpline.py#L70-L91).

The encoded dataset sidecars are:

- [`encoder.json`](data/stacking_cups_action_30hz_UniformBSpline/encoder.json)
- [`manifest.json`](data/stacking_cups_action_30hz_UniformBSpline/manifest.json)
- [`verification.json`](data/stacking_cups_action_30hz_UniformBSpline/verification.json)

Representation verification passed for all 52 episodes and all 31,706
observations. Across 951,180 decoded action samples, reconstruction RMSE is
`0.00421537`, MAE is `0.00086256`, and the observation-step gripper maximum
absolute error is `0.00743919`. The verification also checks finite values,
observation alignment, fixed knots, exact left clamping, and partition of
unity.

## Training provenance

The base policies used 2 GPUs per variant, per-device batch 16, gradient
accumulation 1, and global batch 32 per variant. Raw and B-spline variants ran
concurrently, consuming 4 GPUs total. Training used bf16, DeepSpeed ZeRO-2,
native flow matching, and no L1 loss.

- Base: 50,000 steps, 500 warmup steps, checkpoint every 10,000 steps.
- ttRTC: strict warm-start from its matching step-50k base, 5,000 steps,
  125 warmup steps, final checkpoint at 5,000.
- Raw ttRTC samples completed action prefixes uniformly from 1 through 15.
- B-spline ttRTC samples completed **parameter-row** prefixes uniformly from
  4 through 8 out of 18 control rows.

The authoritative configs are:

- [Raw base config](examples/realRobots/Piper/train_files/starvla_qwenpi_stacking_cups_action_30hz_raw_h30_2gpu_gb32_50k.yaml)
- [B-spline base config](examples/realRobots/Piper/train_files/starvla_qwenpi_stacking_cups_action_30hz_uniform_left_s2_h30_c18_2gpu_gb32_50k.yaml)
- [Raw ttRTC config](examples/realRobots/Piper/train_files/starvla_qwenpi_stacking_cups_action_30hz_raw_h30_training_rtc_prefix1to15_2gpu_gb32_ft5k.yaml)
- [B-spline ttRTC config](examples/realRobots/Piper/train_files/starvla_qwenpi_stacking_cups_action_30hz_uniform_left_s2_h30_c18_training_rtc_prefix4to8_2gpu_gb32_ft5k.yaml)

The ttRTC loss samples prefix lengths uniformly at
[`TrainingRTC_LayerwiseFM_ActionHeader.py#L212-L220`](starVLA/model/modules/action_model/TrainingRTC_LayerwiseFM_ActionHeader.py#L212-L220),
keeps that prefix clean, and computes loss only on the postfix at
[`TrainingRTC_LayerwiseFM_ActionHeader.py#L222-L284`](starVLA/model/modules/action_model/TrainingRTC_LayerwiseFM_ActionHeader.py#L222-L284).
The learned hard-prefix sampler is
[`TrainingRTC_LayerwiseFM_ActionHeader.py#L286-L387`](starVLA/model/modules/action_model/TrainingRTC_LayerwiseFM_ActionHeader.py#L286-L387).

Both base jobs were resumed from their durable step-20k checkpoints after an
external SIGHUP. The checkpoint format restores weights and training step but
does not contain AdamW moments, so optimizer momentum restarted at resume.
Final step-50k files and summaries are present.

W&B runs:

- Raw base final/resumed run: <https://wandb.ai/401910710-university-of-california-berkeley/starVLA_piper/runs/lro4155r>
- B-spline base final/resumed run: <https://wandb.ai/401910710-university-of-california-berkeley/starVLA_piper/runs/q16k429d>
- Raw ttRTC: <https://wandb.ai/401910710-university-of-california-berkeley/starVLA_piper/runs/v9szxqsq>
- B-spline ttRTC: <https://wandb.ai/401910710-university-of-california-berkeley/starVLA_piper/runs/u5861p5v>
- Interrupted base run IDs, retained for provenance: raw `99kje221`, B-spline
  `373yoefy`.

## Standard deployment: raw base or B-spline base

The same server works for both base checkpoints because the B-spline framework
decodes its controls before the wrapper sees the action chunk.

Run from the repository root. Override `STARVLA_PYTHON`, because the generic
launcher currently defaults to `/home/tams/...`, while this workspace trained
with `/home/wangpc/miniconda3/envs/starVLA/bin/python`.

Raw base server:

```bash
cd /scratch/wangpc/DiscreteRTCv2
STARVLA_PYTHON=/home/wangpc/miniconda3/envs/starVLA/bin/python \
GPU_ID=0 PORT=10093 \
CKPT=/scratch/wangpc/DiscreteRTCv2/results/Checkpoints/stacking_cups_action_30hz_qwenpi_raw_h30_2gpu_gb32_50k/checkpoints/steps_50000_pytorch_model.pt \
examples/realRobots/Piper/eval_files/run_policy_server.sh
```

B-spline base server:

```bash
cd /scratch/wangpc/DiscreteRTCv2
STARVLA_PYTHON=/home/wangpc/miniconda3/envs/starVLA/bin/python \
GPU_ID=0 PORT=10093 \
CKPT=/scratch/wangpc/DiscreteRTCv2/results/Checkpoints/stacking_cups_action_30hz_qwenpi_uniform_left_s2_h30_c18_2gpu_gb32_50k/checkpoints/steps_50000_pytorch_model.pt \
examples/realRobots/Piper/eval_files/run_policy_server.sh
```

The launcher enters the recording server at
[`run_policy_server.sh#L23-L30`](examples/realRobots/Piper/eval_files/run_policy_server.sh#L23-L30).
The server builds the binary gripper constraint and wrapper at
[`piper_recording_policy_server.py#L39-L62`](examples/realRobots/Piper/eval_files/piper_recording_policy_server.py#L39-L62).

### Live dry-run before robot execution

The live client is dry-run unless `--execute` is explicitly supplied. Its
default statistics, start statistics, task, and 50 Hz rate belong to the older
white-block setup, so they must not be used unchanged for this policy.

First derive and review a stacking-cups-specific start-state JSON containing
`min`, `max`, and `median` over the first state of each demonstration. Then run:

```bash
cd /scratch/wangpc/DiscreteRTCv2
/home/wangpc/miniconda3/envs/starVLA/bin/python \
  examples/realRobots/Piper/eval_files/piper_live_client.py \
  --host 127.0.0.1 --port 10093 \
  --stats results/Checkpoints/stacking_cups_action_30hz_qwenpi_uniform_left_s2_h30_c18_2gpu_gb32_50k/dataset_statistics.json \
  --start-stats /ABSOLUTE/PATH/TO/stacking_cups_start_statistics.json \
  --task "Stack the cups." \
  --rate-hz 30 --duration 30 \
  --save-viewer-data
```

Do not add `--execute` until camera order, state units, 30 Hz timing, predicted
range, gripper direction, and start pose have all been checked. The code makes
dry-run the default and requires an interactive confirmation for execution at
[`piper_live_client.py#L331-L398`](examples/realRobots/Piper/eval_files/piper_live_client.py#L331-L398)
and refuses another command publisher at
[`piper_live_client.py#L471-L489`](examples/realRobots/Piper/eval_files/piper_live_client.py#L471-L489).

## Raw ttRTC deployment

Start the raw ttRTC server:

```bash
cd /scratch/wangpc/DiscreteRTCv2
STARVLA_PYTHON=/home/wangpc/miniconda3/envs/starVLA/bin/python \
GPU_ID=0 PORT=10093 RTC_INFERENCE_STEPS=4 \
CKPT=/scratch/wangpc/DiscreteRTCv2/results/Checkpoints/stacking_cups_action_30hz_qwenpi_raw_h30_training_rtc_prefix1to15_2gpu_gb32_ft5k/checkpoints/steps_5000_pytorch_model.pt \
examples/realRobots/Piper/eval_files/run_policy_server_training_rtc.sh
```

The server validates the TrainingRTC framework and advertises its trained
maximum delay at
[`piper_training_rtc_server.py#L31-L90`](examples/realRobots/Piper/eval_files/piper_training_rtc_server.py#L31-L90).
The WebSocket dispatcher routes `infer_realtime` into the wrapper at
[`websocket_policy_server.py#L105-L140`](deployment/model_server/tools/websocket_policy_server.py#L105-L140).
For raw actions, the wrapper normalizes the previous robot-coordinate chunk and
calls the learned sampler at
[`policy_wrapper.py#L276-L348`](deployment/model_server/policy_wrapper.py#L276-L348).

Use the async TrainingRTC client, overriding all H=50/50 Hz defaults:

```bash
cd /scratch/wangpc/DiscreteRTCv2
/home/wangpc/miniconda3/envs/starVLA/bin/python \
  examples/realRobots/Piper/eval_files/piper_async_training_rtc_client.py \
  --host 127.0.0.1 --port 10093 \
  --stats results/Checkpoints/stacking_cups_action_30hz_qwenpi_raw_h30_training_rtc_prefix1to15_2gpu_gb32_ft5k/dataset_statistics.json \
  --start-stats /ABSOLUTE/PATH/TO/stacking_cups_start_statistics.json \
  --task "Stack the cups." \
  --rate-hz 30 --duration 30 \
  --initial-delay-steps 15 \
  --min-execution-horizon 15 \
  --save-viewer-data
```

This is also a dry-run without `--execute`. The client clamps a requested
delay to the trained maximum at
[`piper_async_training_rtc_client.py#L27-L78`](examples/realRobots/Piper/eval_files/piper_async_training_rtc_client.py#L27-L78).
Its action-time prefix alignment and continuity checks are at
[`piper_async_common.py#L471-L497`](examples/realRobots/Piper/eval_files/piper_async_common.py#L471-L497).
If measured inference exceeds 15 action steps (0.5 seconds at 30 Hz), stop and
revisit the runtime; do not hide the overflow by increasing a client flag
beyond the checkpoint's trained range.

## B-spline ttRTC gap

Do **not** connect the B-spline ttRTC checkpoint to the raw TrainingRTC async
client above.

The checkpoint was trained with `rtc_conditioning_space:
spline_parameter_rows`. Its prefix is 4--8 normalized rows of shape
`[B,18,7]`, not a prefix of decoded `[B,30,7]` robot actions. The contract and
shape check are implemented at
[`QwenPI_v3_BSpline_TrainingRTC.py#L27-L78`](starVLA/model/framework/VLM4A/QwenPI_v3_BSpline_TrainingRTC.py#L27-L78),
and the framework conditions on parameter rows before decoding at
[`QwenPI_v3_BSpline_TrainingRTC.py#L81-L113`](starVLA/model/framework/VLM4A/QwenPI_v3_BSpline_TrainingRTC.py#L81-L113).

The generic wrapper deliberately rejects this mismatch at
[`policy_wrapper.py#L190-L203`](deployment/model_server/policy_wrapper.py#L190-L203) and
[`policy_wrapper.py#L296-L302`](deployment/model_server/policy_wrapper.py#L296-L302).
That guard is correct and must not be removed.

The next implementation agent should:

1. Make the B-spline framework return both normalized predicted control rows
   and decoded normalized actions. Currently
   [`QwenPI_v3_BSpline_TrainingRTC.py#L106-L113`](starVLA/model/framework/VLM4A/QwenPI_v3_BSpline_TrainingRTC.py#L106-L113)
   discards the predicted rows after decoding.
2. Add a representation-aware wrapper/server method that accepts cached
   **normalized control rows** directly. It must not call
   `PolicyNormProcessor.apply_actions`, which is the decoded robot-action path.
3. Cache the prior prediction's 18 normalized control rows and define the live
   scheduling rule that maps model readiness to a trained `ready_rows` value in
   `[4,8]`. Do not reinterpret 30 Hz action latency as a control-row index.
4. Return the decoded 30x7 robot-coordinate action chunk for execution while
   retaining the new 18x7 parameter tensor for the next request.
5. Advertise and validate `supports_parameter_row_rtc` and
   `rtc_conditioning_space` during the handshake; reject any decoded-action
   prefix sent to this endpoint.
6. Add a contract test covering prefix preservation, output shape, and a hard
   rejection of `[B,30,7]` decoded actions.

Until that adapter exists, the B-spline ttRTC checkpoint may be served through
the **standard** server for ordinary 30-step chunk inference, but that path
does not exercise its learned hard-prefix RTC behavior.

## Offline evaluation and visualization

No open-loop report for these exact four final checkpoints is stored in their
run directories yet. Run a bounded B-spline base evaluation before physical
deployment:

```bash
cd /scratch/wangpc/DiscreteRTCv2
export PYTHONPATH="$PWD:$PWD/BSplineEncoder/src:${PYTHONPATH:-}"
/home/wangpc/miniconda3/envs/starVLA/bin/python \
  examples/realRobots/Piper/eval_files/evaluate_uniform_bspline_open_loop.py \
  --checkpoint results/Checkpoints/stacking_cups_action_30hz_qwenpi_uniform_left_s2_h30_c18_2gpu_gb32_50k/checkpoints/steps_50000_pytorch_model.pt \
  --config examples/realRobots/Piper/train_files/starvla_qwenpi_stacking_cups_action_30hz_uniform_left_s2_h30_c18_2gpu_gb32_50k.yaml \
  --output-dir results/Evaluations/stacking_cups_action_30hz_uniform_left_50k \
  --batch-size 16 --num-workers 8 --max-samples 512
```

The evaluator verifies that the dataset and model basis matrices match exactly
at
[`evaluate_uniform_bspline_open_loop.py#L188-L216`](examples/realRobots/Piper/eval_files/evaluate_uniform_bspline_open_loop.py#L188-L216),
then separates model error from representation error at
[`evaluate_uniform_bspline_open_loop.py#L249-L329`](examples/realRobots/Piper/eval_files/evaluate_uniform_bspline_open_loop.py#L249-L329).

Visualize that subset with:

```bash
/home/wangpc/miniconda3/envs/starVLA/bin/python \
  examples/realRobots/Piper/eval_files/visualize_uniform_bspline_open_loop.py \
  --run-dir results/Evaluations/stacking_cups_action_30hz_uniform_left_50k \
  --dataset data/stacking_cups_action_30hz \
  --controls-dataset data/stacking_cups_action_30hz_UniformBSpline \
  --report-name open_loop_evaluation_subset_512.json \
  --arrays-name open_loop_errors_subset_512.npz \
  --output-prefix step50000
```

This produces a dashboard, trajectory examples, and a control-point/B-spline
figure through
[`visualize_uniform_bspline_open_loop.py#L529-L560`](examples/realRobots/Piper/eval_files/visualize_uniform_bspline_open_loop.py#L529-L560).
The evaluator's ordinary `predict_action_parameters` path does not measure the
conditioned ttRTC behavior; a separate parameter-row RTC evaluation is needed
after the adapter is implemented.

## Hugging Face upload handoff

The existing uploader
[`upload_stacking_cups_checkpoints.py`](deployment/upload/upload_stacking_cups_checkpoints.py)
is hard-coded for the older `stacking_cups_20260911` runs at
[`upload_stacking_cups_checkpoints.py#L15-L31`](deployment/upload/upload_stacking_cups_checkpoints.py#L15-L31).
Do not run it unchanged for these checkpoints.

For the new uploader, reuse its local checkpoint validation and remote LFS
size/SHA-256 verification at
[`upload_stacking_cups_checkpoints.py#L37-L59`](deployment/upload/upload_stacking_cups_checkpoints.py#L37-L59)
and
[`upload_stacking_cups_checkpoints.py#L71-L141`](deployment/upload/upload_stacking_cups_checkpoints.py#L71-L141).
Upload all four numbered checkpoints plus each run's `config.yaml`,
`config.full.yaml`, `dataset_statistics.json`, and `summary.jsonl`. For both
B-spline policies also upload `encoder.json`, `manifest.json`, and
`verification.json` from the encoded dataset. Write a verified receipt beside
each run only after remote size and LFS SHA-256 match.

Suggested remote namespace: the existing dataset repository
`DiscreteRTC/dRTC`, under a new unambiguous prefix containing
`stacking-cups-action-30hz`. Confirm the final remote paths with the owner
before publishing if naming matters to downstream automation.

## Handoff checklist

1. Run the 512-observation B-spline open-loop evaluation and inspect all three
   visualizations.
2. Add an equivalent raw-policy evaluator so raw and B-spline checkpoints are
   compared on identical observations and physical-action targets.
3. Generate and review task-specific stacking-cups start-state statistics.
4. Start the base B-spline server and complete a live dry-run at 30 Hz without
   `--execute`.
5. Validate gripper direction and the server's binary threshold on the real
   robot interface.
6. Only then perform a guarded physical base-policy rollout.
7. Exercise raw ttRTC in dry-run with the explicit H=30 overrides above before
   considering physical execution.
8. Implement and test the parameter-row adapter before claiming B-spline ttRTC
   deployment support.
9. Upload and remotely verify all four checkpoint bundles; retain the generated
   receipts.
