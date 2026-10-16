# Piper demos: data preparation, review and classify_blocks training handoff

Status date: 2026-10-15. Nothing in this document was pushed or published.

| | |
|---|---|
| starVLA commit reviewed | `345c213` on `feature/data-collection` plus the uncommitted working tree (see §8) |
| BSplineEncoder submodule | `be3a559` (bumped from `5aede68`; tests: 382 passed, 2 skipped in env `starVLA`) |
| Raw data source | `huggingface.co/datasets/DiscreteRTC/dRTC`, folder `piper_demos/`, revision `88a1916a887679dc70c01fa268bf5d6f4e604f5e` |
| Python env | conda `starVLA` (Python 3.10, torch 2.7.0+cu128, accelerate 1.5.2, deepspeed 0.16.9, transformers 4.57.0) |

## 1. Ready-to-run commands

Run from the repository root (`/scratch/wangpc/DiscreteRTCv2`).

```bash
PY=/home/wangpc/miniconda3/envs/starVLA/bin/python

# 1. Download + verify (idempotent; pinned revision; never deletes; raw files made read-only)
$PY examples/realRobots/Piper/data_files/download_piper_demos.py --output data/raw
$PY examples/realRobots/Piper/data_files/download_piper_demos.py --output data/raw --verify-only

# 2. Preprocess all three tasks (refuses to overwrite; add --overwrite to rebuild atomically)
$PY examples/realRobots/Piper/data_files/preprocess_piper_demos.py
#    re-check geometry/aliases of existing outputs only:
$PY examples/realRobots/Piper/data_files/preprocess_piper_demos.py --skip-existing

# 3. Validate (alignment, finiteness, reconstruction, timing; --loader also builds starVLA datasets)
$PY examples/realRobots/Piper/data_files/validate_piper_demos.py --loader

# 4. Train classify_blocks (2 GPUs, 10 h budget each). GPUS is mandatory; pick free GPUs.
GPUS=4,5 bash examples/realRobots/Piper/train_files/run_qwenpi_piper_demos_classify_blocks_30hz_raw_h30_2gpu_gb32_30k.sh
GPUS=6,7 bash examples/realRobots/Piper/train_files/run_qwenpi_piper_demos_classify_blocks_30hz_uniform_left_s2_h30_c18_2gpu_gb32_30k.sh
#    resume after an interruption: rerun the SAME command (RESUME=auto picks up the latest checkpoint
#    and the saved optimizer state; the budget ledger carries the time already used).
#    preview without side effects: DRY_RUN=1 GPUS=... bash <launcher>
```

Logs: `results/training_logs/piper_demos/<run_id>_<UTC stamp>.log` (one per launch, with git commit,
GPUs, resume flag, remaining budget and the exact command at the top). Data logs: `results/data_logs/`.

## 2. Task names and aliases

| HF folder (canonical) | Aliases found in the repo / request | Instruction string | Episodes | Frames after cleanup | Split (train / validation) |
|---|---|---|---:|---:|---|
| `classify_blocks` | classification_cube, cubes_classification (`cubes_classification_30hz*` configs) | `Classify the blocks.` | 85 | 186,282 | 0:80 / 80:85 |
| `hanging_mug` | – | `Hang the mug.` | 61 | 26,888 | 0:56 / 56:61 |
| `stacking_cup` | stacking_cups (`stacking_cups_*` configs) | `Stack the cups.` | 61 | 42,018 | 0:56 / 56:61 |

Instructions stay the pipeline's short task names: the trained policies and their open-loop evaluations
on these datasets use them. `preprocess_piper_demos.py --detailed-instructions` (opt-in, not applied) would
switch to the longer sentences in `DETAILED_TASK_INSTRUCTIONS`, which were written from the recorded videos:
- classify_blocks: red and blue cubes, each sorted into the box of its own colour, one at a time.
- hanging_mug: the dark green mug, gripped by its rim and hung by its handle on the left peg of the wooden mug tree.
- stacking_cup: three upside-down cups; teal goes over red, then blue over teal.

`hanging_mug` and `stacking_cup` episodes sit under a dated sub-folder (`20260921/`); the pipeline discovers
episodes recursively. Validation episodes are the last 5 recordings (recording-time order).

## 3. Data layout

```text
data/raw/piper_demos/                       HF layout preserved, read-only files
  SOURCE.json                               repo, resolved SHA, per-task counts, size+sha256 of all 622 files
  classify_blocks/<episode>/{data.pkl,global_image.mp4,hand_image.mp4}
  hanging_mug/20260921/<episode>/...        stacking_cup/20260921/<episode>/...
  convert_piper_to_lerobot.py               (upstream helper, kept as downloaded)
data/processed/piper_demos/
  PREPROCESSING.json  VALIDATION.json
  ActionCommand/State_Joint/<task>_30hz_cleanup/                                   LeRobot v2.1 (both variants)
  ActionCommand/State_Joint/<task>_30hz_cleanup_spline_uniform_left_s2_h30_c18/    B-spline sidecar
```

Each LeRobot dataset keeps the pipeline's provenance: `meta/source_episodes.jsonl` (raw episode path,
removed prefix), `meta/source_alignment/` (raw robot index, raw-clock time, camera source frame for
every row), `meta/cleanup_*`, `meta/validation.json`. Each sidecar keeps `encoder.json` (full geometry),
`manifest.json`, `preprocessing_provenance.json` and one `data/episode_XXXXXX.npz` per episode with
`frame_index`, `observation_index`, `controls[T,18,7]`, `knots[T,22]`, `executable_steps`, `duration_steps`.

## 4. Observation / action definitions (both variants identical except the target)

Preset `joint-command` of the BSplineEncoder pipeline (= `scripts/legacy_piper/preprocess_piper.py
--setup joint-command`, family `ActionCommand/State_Joint`). This is the convention of the earlier
`stacking_cups_action_30hz` policies: their checkpoint statistics show a {0,1} commanded gripper action
next to a continuous measured gripper state.

| Item | Definition |
|---|---|
| Clock | 30 Hz target clock over the overlap of robot, `global` and `hand` streams; every stream resampled by nearest timestamp (ties → earlier) |
| State `observation.state` (7) | measured joints 1–6 [rad] + measured gripper [0,1] at observation time t (modality names `measured_joints`/`measured_gripper`, aliased to `joints`/`gripper`) |
| Action `action` (7) | **absolute** commanded (GELLO) joints 1–6 [rad] + commanded gripper at t (`action_offset=0`), not deltas |
| Gripper | passed through as recorded, range [0,1]; in this data it is **continuous** (classify: 58% < 0.05, 34% > 0.95), not binary as in the old stacking-cups data. Deployment still thresholds on the server side |
| Images | two RGB frames: `video.global` then `video.hand`, both the frame nearest to t (`delta_indices=[0]`), resized to 224×224 by the loader; h264 re-encoded at 30 Hz |
| Language | per-task instruction above |
| Leading idle | removed per episode (first sustained motion of commanded joints; 0.4 s window, ≥50% moving); classify 4,207 / hanging 2,816 / stacking 3,267 frames removed; internal pauses kept |
| Raw target (`piper_abs_joints_30step`) | `action[t : t+30]`, padded past the episode end by repeating the final action (`first_last`) |
| Spline target (`piper_uniform_left_bspline_30step`) | 18 controls fitted to the same 30-action window, same final-action tail padding (+1 endpoint row, `end_padding`) |
| Normalization | starVLA `min_max` to [-1, 1] per dimension, statistics from the **train split only** (`meta/stats_gr00t_train.json`); controls use the same action statistics; affine and unclamped, so decoding commutes with it |

Timing measured from raw `data.pkl` (signed offset of the selected sample relative to t, all tasks):
robot sample −10…+10 ms, global camera −61…+17 ms, hand camera −34…+34 ms, about half after t. With
nearest-sample alignment an observation can therefore contain a hand frame captured up to ~1 control step
(34 ms) after t. This is inherent to the established preset; it is bounded and recorded, not removed (see §9).

## 5. B-spline geometry (asserted by `preprocess_piper_demos.py` and the tests)

`uniform_left`, cubic, 30 Hz, chunk 30, span 2 action steps → **15 executable spans, 3 right-context
spans, 18 controls** (30 actions ≠ 30 controls), knots `[0,0,0,0,2,4,…,36]` steps (22 knots), decode basis
`[30,18]` with partition of unity and S(0)=c₀, `regularization=1e-4`, `end_padding=true`,
`implementation_version=uniform_left_direct_fit_v1`, unquantized, unnormalized controls
(`tokenizer_id 3282a6009a52b9a0`). The model rebuilds the same basis from
`framework.action_representation` (identical array, tested).

## 6. Validation results (`data/processed/piper_demos/VALIDATION.json`)

All three tasks PASS. Checks per task: LeRobot v2.1 metadata, cameras, splits cover all episodes
without overlap, state aliases point at the same columns, sidecar ↔ dataset frame/observation mapping for
every episode, record/episode counts, fixed knots, geometry, refit reproducibility (stored controls equal
an independent refit: max difference 0.0), no non-finite values, train-split provenance, loader build.

Reconstruction of the 30-step windows with the policy's decode basis (physical units: rad / gripper):

| Task | RMSE train | RMSE validation | p99 abs | arm max abs | gripper max abs | step-0 max abs |
|---|---:|---:|---:|---:|---:|---:|
| classify_blocks | 0.00498 | 0.00342 | 0.0128 | 0.045 | 0.298 | 0.010 |
| hanging_mug | 0.00213 | 0.00267 | 0.0087 | 0.068 | 0.138 | 0.004 |
| stacking_cup | 0.00217 | 0.00219 | 0.0083 | 0.038 | 0.148 | 0.004 |

Largest errors are gripper steps (a cubic spline with 2-step knots cannot follow an instantaneous 0→1).

Loader compatibility (`--loader`, real starVLA datasets with `episode_split=train`, 30 samples/task incl.
episode tails): only train episodes are loaded; raw and spline variants share identical normalization
statistics equal to the train-split min/max; both return 2 × 224×224 images, identical (1,7) state and
identical images/language for the same (episode, frame); targets (30,7) raw / (18,7) spline; decoded spline
targets match raw targets in normalized space up to the reconstruction error (max 0.254 classify, 0.153
hanging, 0.164 stacking normalized units, all at gripper steps).

## 7. classify_blocks training configuration

Configs: `examples/realRobots/Piper/train_files/starvla_qwenpi_piper_demos_classify_blocks_30hz_{raw_h30,uniform_left_s2_h30_c18}_2gpu_gb32_30k.yaml`.
They differ only in `framework.name`, `action_model.action_horizon`, `framework.action_representation`,
`data_mix` and `controls_dataset_path` (tested).

| Choice | Value | Source |
|---|---|---|
| Backbone | Qwen3-VL-4B-Instruct (`playground/Pretrained_models/Qwen3-VL-4B-Instruct` → HF cache snapshot), flash-attention-2, `freeze_modules: qwen_vl_interface` | all prior Piper 30 Hz configs |
| Action head | existing `LayerwiseFM` (layer-wise DiT, flow matching), hidden 1024, 4 inference steps, `repeated_diffusion_steps 2`, Beta(1.5,1) time sampling | same |
| Raw head | `QwenPI_v3`, horizon 30 × 7 | `cubes_classification_30hz_raw_h30` / `stacking_cups_action_30hz_raw_h30` |
| Spline head | `QwenPI_v3_UniformBSpline`, horizon 18 × 7, decodes `[30,18]@[B,18,7]` before un-normalization | same, uniform-left variants |
| Loss | mean squared velocity error over **all** target rows and dims, no masking (30 rows raw, 18 rows spline) | `LayerwiseFM_ActionHeader.forward` |
| State | `include_state: true`: 7-D state tokens in the prompt; DiT state encoder unused (`state=None`), unchanged framework behaviour | QwenPI_v3 |
| Batch | 16 per GPU × 2 GPUs × accum 1 = 32; DeepSpeed ZeRO-2 bf16 (`deepspeed_zero2_two_gpu_grad_accum_1.yaml`) | `stacking_cups_action_30hz` 2-GPU recipe |
| Schedule | 30,000 steps, LR action/projector 1e-4, VLM 1e-6, base 1e-5, cosine→1e-6, warmup 500, clip 1.0, AdamW (0.9,0.95) | `cubes_classification_30hz` 30k recipe |
| Checkpoints | every 5,000 steps (`steps_N_pytorch_model.pt`, ~11 GB) + one rotating `checkpoints/training_state/` (optimizer/RNG) + `final_model/` | new budget controls |
| Data | `episode_split: train`; validation episodes 80–84 never loaded; `num_workers: 8` per rank (2 starved the GPUs) | new / measured |

**Default backbone for new runs (from 2026-10-16): Qwen3-VL-2B-Instruct-Action**, StarVLA's checkpoint with
added action tokens. It is linked as `playground/Pretrained_models/Qwen3-VL-2B-Instruct-Action` →
`/scratch/wangpc/starVLA/playground/Pretrained_models/Qwen3-VL-2B-Instruct-Action`. Matched configs and wrappers:
`starvla_qwenpi_qwen3vl2b_piper_demos_classify_blocks_30hz_{raw_h30,uniform_left_s2_h30_c18}_2gpu_gb32_30k.yaml`
and `run_qwenpi_qwen3vl2b_...sh` (ports 29861/29862). They differ from the 4B configs only in `base_vlm` and
`run_id` (tested). QwenPI_v3 derives the head depth from the VLM, giving 28 DiT blocks and 28 projectors, and
the width from the VLM hidden size (2048). A CPU check of both representations (bf16 autocast) built the model:
2,611.7 M params, of which 480.0 M are head + projectors (4B: 5,071 M / 633 M). Forward and backward passes
gave finite loss and gradients, and `predict_action` returned (1, 30, 7). Not yet timed on GPU, so step time
and checkpoint size (~5 GB expected) are unmeasured. The 4B configs remain the record of the 2026-10-15 runs.

Head dimension checks: action_dim 7 = state_dim 7 = dataset action/state width; spline horizon 18 equals
`num_control_points` and `(18-3)·2 = 30`; position embeddings (1024) exceed both horizons; tiny CPU heads
for horizons 30 and 18 produce finite loss and `[B,H,7]` predictions (tests).

## 8. Budget controls (2 GPUs × 10 h per policy)

Expected time: the identical 2-GPU / global-batch-32 recipe ran at 0.97 s/step including eval and saves
(stacking_cups_action_30hz, 10k steps in 2 h 41 min). On classify_blocks the long episodes made the
default 2 dataloader workers the bottleneck (0.3–1.2 s data wait per step, 1.35 s/step); with
`num_workers: 8` per rank the data wait is ~0 and the raw policy runs at **0.81 s/step** (smoke test D).
30k steps ≈ 6.8 h + ≈ 15 min eval/saves/startup ≈ 7.1 h, against a 9.67 h soft limit.

1. **In-process soft limit** (`trainer.max_wall_time_hours: 10`, `wall_time_safety_minutes: 20`;
   `starVLA/training/trainer_utils/run_budget.py`): every optimizer step all ranks all-reduce a stop flag;
   when the remaining budget drops under 20 min, or on SIGTERM/SIGUSR1/SIGINT, the trainer writes a
   checkpoint and the optimizer state and exits 0 without `final_model/`.
2. **Ledger** `<run_dir>/wall_time_ledger.json`: per-session elapsed time, start/last step and end reason;
   resumed launches only get the remainder. The launcher refuses to start with < 25 min left.
3. **Hard limit**: the launcher wraps the job in `timeout --signal=TERM --kill-after=5m <remaining>`.
4. **Explicit GPUs**: `GPUS` is required, must list exactly two ids, and busy GPUs (> 2 GB used) are
   refused unless `ALLOW_BUSY_GPUS=1`. A `flock` prevents two launches on the same run directory.
5. **Resume**: `RESUME=auto` → `is_resume` when `steps_*` exist; weights from the latest `steps_N`, and the
   optimizer/RNG from `training_state/` when its step matches (otherwise a logged weights-only fallback).
   Checkpoint files are written to `*.tmp` and renamed, so an interrupted save cannot become "latest".

## 9. Code changes (all opt-in; defaults preserve existing behaviour)

### 9.1 Integration fixes and glue

| File | Change | Why |
|---|---|---|
| `starVLA/dataloader/gr00t_lerobot/datasets.py` | opt-in `datasets.vla_data.episode_split` (`resolve_split_episode_ids`): trajectories and normalization statistics restricted to that `meta/info.json` split; stats cached in `meta/stats_gr00t_<split>.json` | normalization fitted on training data only; validation episodes never trained on |
| same | step-index cache: split-specific file `meta/steps_data_index_<split>.pkl`, split in the config key, and `_check_cached_steps` (config key + exact trajectory set; legacy caches only if inside the dataset) | **bug**: the cache was returned without any check, so a stale whole-dataset cache would silently re-insert validation episodes (reproduced during validation; regression test added) |
| `examples/realRobots/Piper/train_files/data_registry/data_config.py` | sidecar record-count checks compare against all source episodes when a split is active (`source_record_count`); `piper_demos_<task>_30hz_{raw_h30,uniform_left_bspline}` mixtures for the 3 tasks | split-aware sidecar loading; new datasets |
| `starVLA/training/trainer_utils/run_budget.py` (new), `starVLA/training/train_starvla.py` | wall-time budget + ledger + signal handling (§8), opt-in `trainer.save_training_state` (rotating DeepSpeed optimizer/RNG state, restored on `is_resume`), checkpoint files written via `.tmp` + rename | budget, graceful checkpointing, resume |
| `examples/realRobots/Piper/train_files/deepspeed_zero2_single_gpu.yaml` | restored to the committed accum-8 DeepSpeed config; the uncommitted accum-4 content now lives in `deepspeed_zero2_single_gpu_grad_accum_4.yaml`, used by the two 25 Hz `*_training_rtc_gb32_5k.sh` launchers | the edit had broken `run_piper_train.sh` (asserts accum 8) and its 4 tracked wrappers |
| `pyproject.toml`, `pyrightconfig.json`, `.dockerignore` | exclude `ARCHIVE/` (packages, black, ruff, pyright, Docker); `[tool.pytest.ini_options] testpaths=["tests"]`, `norecursedirs` incl. `ARCHIVE`, `data`, `results`, `BSplineEncoder` | archive excluded from build/discovery/tests |
| `examples/realRobots/Piper/data_files/{download,preprocess,validate}_piper_demos.py` (new) | download/verify, preprocessing driver around the BSplineEncoder pipeline (+ starVLA state-key aliases), validator | data preparation |
| `examples/realRobots/Piper/train_files/launch_budgeted_2gpu_training.sh`, `run_qwenpi_piper_demos_classify_blocks_30hz_*_2gpu_gb32_30k.sh`, `starvla_qwenpi_piper_demos_classify_blocks_30hz_*_2gpu_gb32_30k.yaml` (new) | launchers and configs | training |
| `tests/test_piper_demos_contract.py` (new) | 16 CPU tests: split parsing, current-step-only observations for every Piper robot type, mixture pairing, config parity, model-vs-encoder basis equality and normalization commutation, LayerwiseFM head shapes/loss for horizons 30 and 18, budget ledger/signal/disabled, steps-cache guard, split filtering on real data | regression checks |

Public interfaces preserved: no existing config key, CLI flag, registry name, robot type, checkpoint
format or deployment contract changed. New keys default to off (`episode_split`, `max_wall_time_hours`,
`save_training_state`). The BSplineEncoder submodule was not modified.

Pre-existing uncommitted changes that were already in the tree when this review started (kept):
`strict_pretrained_checkpoint`, `rtc_min_delay_steps`, `QwenPI_v3_UniformBSpline` default
`uniform_left_direct_fit_v1`, the 30-step/adaptive data configs, policy-wrapper variable horizon, and the
untracked adaptive/TrainingRTC frameworks, eval scripts and run files.

### 9.2 Checks run (exact results)

| Check | Result |
|---|---|
| `BSplineEncoder`: `PYTHONPATH=src python -m pytest -q` | 382 passed, 2 skipped |
| `pytest -q --ignore=tests/test_gr00t_zmq_compat_server.py` (repo tests, CPU) | 54 passed (test_gr00t_zmq_compat_server.py cannot be collected: `zmq` is not installed in env `starVLA`; pre-existing) |
| `tests/test_piper_demos_contract.py` | 24 passed (adds training-state barrier regression, 2B config parity, instruction rewrite + short-instruction checks) |
| download rerun / `--verify-only` | 622/622 files verified (size + LFS sha256), nothing re-downloaded, raw files read-only |
| pipeline's own validator (`meta/validation.json`) | passed for 3 tasks: raw values, action alignment, independent idle cutoff, indices, episode stats, source alignment, metadata, 170/122/122 videos |
| `validate_piper_demos.py --loader` | PASS ×3 (§6) |
| `ruff`/`black` | not available in env `starVLA` (not run); `bash -n` on the launcher passes |

Two-GPU smoke tests (GPUs 6,7, real configs, real data, full per-GPU batch 16, W&B disabled; run dirs in
`results/smoke_tests/Checkpoints/`, logs in `results/data_logs/smoke_*.out` and `results/smoke_tests/logs/`):

| Test | What it covered | Result |
|---|---|---|
| A raw, 8 steps | model build (5,071 M params, 633 M trainable), train-split loader, DeepSpeed ZeRO-2 bf16 init on 2 ranks, forward/backward, eval at 4 and 8, `steps_4/8` + `training_state`, `final_model` | exit 0; 6 min 43 s wall incl. a one-off 4.5 min cold dataset setup; weights 11 GB per checkpoint, training state 25 GB, host RSS ≤ 53 GB per process; ledger 375 s |
| B raw resume → 12 | launcher `RESUME=auto`, remaining budget from ledger (35,624 s), weights from `steps_8`, **optimizer/RNG restored** from `training_state` (step 8), steps 9–12, save, final | exit 0; ledger 2 sessions, 466 s total; warm startup ≈ 30 s |
| C spline, budget 0.15 h / margin 3 min | `QwenPI_v3_UniformBSpline` forward/backward, spline eval (`predict_action_parameters`, control-space MSE 0.0096 at step 200), **budget-triggered graceful stop** at step 249 with checkpoint + training state, exit 0; relaunch refused (exit 5, budget exhausted) | pass; 1.35 s/step with 2 workers (data-bound) |
| D raw, 160 steps, `num_workers 8` | throughput | 0.81 s/step, data wait ≈ 0 |
| E spline, 160 steps, final config | throughput with config defaults (`num_workers: 8`) | exit 0; 0.78 s/step with data wait ≈ 0 for steps 20–80; afterwards 1.6–3.2 s/step because another user's 24 processes started on GPUs 6–7 mid-test (contention, not data) |

Smoke resource use: GPUs 6–7 for ≈ 30 min in total (5 launches); 121G of smoke checkpoints under
`results/smoke_tests/` (safe to delete once reviewed; not needed by anything).
Not exercised: a SIGTERM/SIGUSR1 delivered to a live multi-GPU run (the signal path shares the budget
path's all-reduce and is unit-tested; under `timeout`, `torchrun` may kill workers ~30 s after its own
SIGTERM, so the 20 min soft margin, not the signal, is the primary protection), multi-hour stability.

## 10. Archive inventory

Moved into `ARCHIVE/<original path>` (index with reasons: `ARCHIVE/INDEX.md`; nothing deleted):

| Original path | Git | Reason |
|---|---|---|
| `examples/realRobots/Piper/train_files/deepspeed_zero2_two_gpu.yaml` | untracked | unreferenced; superseded by `deepspeed_zero2_two_gpu_grad_accum_1.yaml` |
| `examples/realRobots/Piper/train_files/ds_config_zero2_grad_accum_2.json` | untracked | only used by the archived yaml above |
| `examples/realRobots/Piper/train_files/run_piper_qwenpi_20260818_25hz_uniform_left_bspline_gb32_30k_train.sh` + `starvla_qwenpi_piper_20260818_25hz_uniform_left_bspline_gb32_30k.yaml` | tracked | unreferenced; superseded by the gb32_50k run; no local checkpoint |
| `examples/realRobots/Piper/train_files/run_piper_qwenpi_50hz_h50_train.sh` + `starvla_qwenpi_piper_20260814_50hz_h50.yaml` | tracked | unreferenced; 20260814 data superseded by 20260818 measured_50hz |
| `deployment/upload/finish_stacking_cups_training.py` | untracked | one-off watcher whose job finished (`stacking_cups_20260911_pipeline_state.json` stage=complete, 3 upload receipts) |
| `results/training_logs/stacking_cups_action_30hz/pipeline_resume.pid` | untracked | stale PID file |

Tracked moves appear in `git status` as deletions plus an untracked `ARCHIVE/`; nothing was staged.
`ARCHIVE/` is excluded from packaging, pytest, black/ruff, pyright and Docker; the data-registry scan only
globs `examples/**/train_files/data_registry`, so archived configs are never registered.

Kept on purpose (not obsolete or not certain): all 8 checkpoint runs in `results/Checkpoints/` (326 GB;
every one is referenced by a yaml `pretrained_checkpoint`, the stacking-cups handoff or an upload script; the
20260911 raw ttRTC and all action_30hz runs are not uploaded yet); their 10k–40k intermediate checkpoints
(≈ 44 GB per 50k run) are pruning candidates only with owner approval; cubes_classification_30hz scripts
(reproduce runs named in `upload_cubes_classification_checkpoints.py`); 25 Hz/50 Hz RTC lineage scripts;
`finish_stacking_cups_raw_training_rtc.py` (upload still blocked on HF auth); the three uploaders whose run
dirs are gone (cannot confirm the HF uploads without checking the repo); unreferenced utilities/docs
(`analyze_action_chunks.py`, `downsample_lerobot.py`, RTC docs, `eval_files/analyze_gripper_postprocessing.py`).
`starVLA.egg-info/`, `__pycache__/`, `.pytest_cache/` are regenerable caches (gitignored), left in place.

## 11. Remaining issues / not done

1. **Full classify_blocks runs launched 2026-10-15 17:48 UTC** (raw on GPUs 0,1; spline on GPUs 2,3).
   The raw run crashed at 19:03 while saving `training_state` at step 5,000: `accelerator.save_state` ends
   with each rank writing `random_states_<rank>.pkl`, and rank 0 renamed `training_state.tmp` before rank 1
   had finished (`RuntimeError: Parent directory .../training_state.tmp does not exist`). Fixed with a barrier
   after `save_state` in `VLATrainer._save_training_state` (regression test
   `test_training_state_save_waits_for_all_ranks_before_rename`). The smoke tests missed it because the race
   depends on timing. The raw run resumed at 22:26 UTC from `steps_5000` with the optimizer restored. Rank 1's RNG
   file is missing from that state, and accelerate logs and skips it. The spline run was launched with the
   old code; it survived its 5k–20k saves, and if it crashes at a later save, rerun its §1 command to resume.
2. **Sub-step observation timing**: nearest-sample alignment can pick a hand frame up to +34 ms and a
   robot sample up to +10 ms after t (§4). It is bounded and identical for both variants, but it is a
   train/deploy skew (a live client only has frames ≤ t). A causal ("latest sample ≤ t") alignment option
   would have to be added to the BSplineEncoder pipeline and the data regenerated.
3. **Gripper**: commanded gripper is continuous here but binary in the earlier stacking-cups data; the
   spline cannot follow 0→1 steps exactly (max error 0.30). The deployment server's 0.3 threshold still
   binarizes the output. Consider a binary gripper head or excluding the gripper from the spline if that matters.
4. The pipeline's state-key names (`measured_*`) differ from starVLA's (`joints`/`gripper`); this is bridged by
   alias entries in `meta/modality.json` written by `preprocess_piper_demos.py`. A future pipeline option for
   starVLA key names would remove the post-step.
5. `data` is a generic top-level package name in BSplineEncoder; `preprocess_piper_demos.py` asserts it
   imports the BSplineEncoder package, but other tools run from the repo root could be shadowed.
6. Spline policies cannot yet use learned RTC at deployment (`rtc_conditioning_space` parameter-row adapter
   is still a TODO from the stacking-cups handoff); only plain chunk inference is servable.
7. `test_gr00t_zmq_compat_server.py` needs `pyzmq` in env `starVLA`; `ruff`/`black` are not installed.
8. The first launch on a fresh dataset spent ≈ 4.5 min in dataset setup (cold cache); later launches ≈ 30 s.
   Counted inside the budget.
9. Smoke checkpoints in `results/smoke_tests/` (121G) can be deleted after review.
11. **Shared GPUs**: the launcher only refuses GPUs that are busy at launch time. Another job starting on
    the same GPUs later roughly doubled step time in smoke E; at 1.6 s/step 30k steps would not fit, and the
    budget would stop the run gracefully around step 21k. Use GPUs that stay exclusive for the 10 h.
12. **Upload of classify_blocks checkpoints**: `deployment/upload/upload_piper_demos_starvla_checkpoints.py`
    (started 2026-10-15 22:38 UTC, log `results/training_logs/piper_demos/upload_starvla_checkpoints.log`)
    waits until both runs have `steps_30000` and `final_model/`. It then uploads one commit per run to
    `DiscreteRTC/dRTC` (dataset repo) under `StarVLA/<run_name>/`: `checkpoints/steps_30000_pytorch_model.pt`,
    `config.yaml`, `config.full.yaml`, `dataset_statistics.json`, `summary.jsonl` and `wall_time_ledger.json`.
    That is the layout `from_pretrained` loads. The script verifies the remote size and LFS SHA-256 and
    writes `<run_dir>/checkpoint_upload.json`. If the uploader stops, rerun it; it waits again and re-commits.
10. Not uploaded: 20260911 raw ttRTC (HF auth) and the four stacking_cups_action_30hz runs (no uploader).
