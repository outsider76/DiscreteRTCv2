# Piper demos: data preparation, review and classify_blocks training handoff

Status date: 2026-10-16 (review written 2026-10-15). See §0 for the current state of runs, uploads and code.

| | |
|---|---|
| starVLA commit reviewed | `345c213` on `feature/data-collection` plus the then-uncommitted working tree (see §9); now committed (§0) |
| BSplineEncoder submodule | `be3a559` (bumped from `5aede68`; tests: 382 passed, 2 skipped in env `starVLA`) |
| Raw data source | `huggingface.co/datasets/DiscreteRTC/dRTC`, folder `piper_demos/`, revision `88a1916a887679dc70c01fa268bf5d6f4e604f5e` |
| Python env | conda `starVLA` (Python 3.10, torch 2.7.0+cu128, accelerate 1.5.2, deepspeed 0.16.9, transformers 4.57.0) |

## 0. Current status (2026-10-22)

### Code
- **Branch:** `feature/data-collection`. All Piper work, including what was uncommitted before this
  review, is in the commits that follow `345c213`.
- **Contract tests:** `tests/test_piper_demos_contract.py`, 38 tests passing (2026-10-22). They cover:
  - split filtering and the steps-cache key;
  - budget ledger and signal handling;
  - the training-state save barrier;
  - raw vs B-spline config parity, and 4B vs 2B config parity;
  - encoder geometry;
  - the instruction-rewrite helper;
  - short instructions kept in the datasets;
  - delta-joint (rel) configs, stats-cache isolation, sidecar mode checks and the open-loop offset;
  - per-step delta-joint (`StateDeltaJoint`) configs, mixtures, preset paths, stored targets and the
    builder's delta/rel inverse;
  - the 30k delta-joint configs differing from the 10k ones only in `run_id` / `max_train_steps`.
- **Default backbone for new Piper runs:** Qwen3-VL-2B-Instruct-Action (§7). The 4B configs remain the
  record of the 2026-10-15 runs.
- **Dataset instructions:** kept short ("Classify the blocks." / "Hang the mug." / "Stack the cups."),
  because the trained policies and their open-loop evaluations use them. The detailed sentences are opt-in
  only (`preprocess_piper_demos.py --detailed-instructions`).
- **Fixed during the runs:** a rank race in `VLATrainer._save_training_state`. Rank 0 renamed
  `training_state.tmp` while rank 1 was still writing its RNG file, which crashed the raw run at step 5,000.
  A barrier now follows `save_state`, and a regression test covers it.

### classify_blocks policies (Qwen3-VL-4B-Instruct, LayerwiseFM, 2 GPUs, global batch 32)

| Run | GPUs | Outcome | Wall time | Uploaded checkpoint |
|---|---|---|---|---|
| `piper_demos_classify_blocks_30hz_qwenpi_uniform_left_s2_h30_c18_2gpu_gb32_30k` (B-spline) | 2,3 | completed 30,000 steps, exit 0 (2026-10-16 00:29 UTC) | 6.67 h | step 30,000, HF commit `64e4b30f` |
| `piper_demos_classify_blocks_30hz_qwenpi_raw_h30_2gpu_gb32_30k` (raw) | 0,1 | crashed at the step-5k save (race above); resumed from step 5,000; **stopped on request** with SIGTERM at step 21,342 (graceful save, exit 0, 2026-10-16 02:00 UTC) | 4.75 h counted (1.21 h + 3.54 h; the time it sat crashed is not counted) | step 20,000 (`--stopped-early`), HF commit `e8409148` |

- **Upload location:** `DiscreteRTC/dRTC` (dataset repo), `StarVLA/<run_id>/`. Each folder holds:
  - `checkpoints/steps_N_pytorch_model.pt` (10.9 GB, full bf16 state dict);
  - `config.yaml`, `config.full.yaml`, `dataset_statistics.json`, `summary.jsonl`, `wall_time_ledger.json`.

  This is the layout `from_pretrained` loads directly.
- **Upload checks:** remote size and LFS SHA-256 match the local file. Each run directory has a
  `checkpoint_upload.json` record.
- **Local copies:**
  - Raw: weights at steps 5k–20k and 21,342, plus a resumable `training_state` at 21,342. No `final_model`,
    because the run was stopped early. Rerunning its launcher with free GPUs would resume to 30k
    (5.25 h of the 10 h budget left; ≈ 1.9 h needed for the remaining 8,658 steps).
  - B-spline: weights at steps 5k–30k plus `final_model`.
- **Not done yet:** open-loop evaluation of the two classify_blocks policies.

### classify_blocks PER-STEP delta-joint policies (Qwen3-VL-2B-Instruct-Action, 10k steps, from 2026-10-22 01:39 UTC)

**This is the per-step delta-joint representation asked for on 2026-10-22.** It uses BSplineEncoder's own
preset `state-delta-joint` (`BSplineEncoder/src/data/config.py`, `transforms/spaces.py:next_step_joint_actions`),
built from the same raw recordings as the absolute dataset:

    action[t, joints]  = measured q[t+1] - measured q[t]     (6, rad; one step at 30 Hz)
    action[t, gripper] = measured gripper[t+1]               (absolute)
    chunk target k     = action[t+k]   ->  joint targets  q[t+k+1] = q[t] + sum_{j<=k} d[j]

- **Data:** `data/processed/piper_demos/StateDeltaJoint/classify_blocks_30hz_cleanup` (85 episodes, 185,998
  frames, splits train 0:80 / validation 80:85, "Classify the blocks."), plus the sidecar
  `..._spline_uniform_left_s2_h30_c18` (same h30/c18/s2 geometry, regularization 1e-4). Built by
  `preprocess_piper_demos.py --tasks classify_blocks --preset state-delta-joint` (summary in
  `StateDeltaJoint/PREPROCESSING.json`). It has 284 fewer frames than the command dataset because the
  last row of each episode (no next sample) is dropped and the idle cleanup runs on measured motion.
- **Loader:** the deltas are stored in the dataset, so the configs keep `action_mode: abs`. A starVLA
  `delta`/`rel` transform on top would convert twice. `modality.json` marks `action.joints` non-absolute,
  so raw chunks past an episode end pad with **zero** deltas; the gripper repeats its last value.
- **Configs:** `starvla_qwenpi_qwen3vl2b_..._delta_joint_{raw_h30,uniform_left_s2_h30_c18}_2gpu_gb32_10k.yaml`
  (wrappers `run_qwenpi_qwen3vl2b_..._delta_joint_*_10k.sh`, ports 29865/29866). They are identical to the
  absolute 2B configs except for `run_id`, `data_mix` (`piper_demos_classify_blocks_30hz_delta_joint_*`),
  `controls_dataset_path`, the `action_type` label, `max_train_steps: 10000` and `save_interval: 2500`.
  Tests enforce this.
- **Verification (2026-10-22, before launch):**
  - Stored actions vs `observation.state`: `action[t] == (q[t+1]-q[t], g[t+1])` over all 185,998 rows,
    max error ≤ 2e-9 rad (float32 rounding of the stored delta).
  - Observations are identical to the command dataset's after per-episode alignment; the idle-trim
    offset differs by -16…11 frames.
  - Sidecar decode vs 30-step delta windows: max 0.015–0.027 rad per joint.
  - Integrated `q[t] + cumsum(decoded)` vs measured `q[t+k+1]`: RMSE 0.6–1.2 mrad, max 0.008–0.014 rad.
  - Through the real training `__getitem__` (transforms, float16 packing, un-normalized with the run's
    stats; 400 samples each):
    - raw targets equal the independent per-step target within 2e-5 rad, including zero tail padding;
    - B-spline controls decode to it within the fit error (≤0.02 rad).
  - Action stats (saved to each run's `dataset_statistics.json`): joint min/max -0.078…0.096 rad;
    gripper 0.004…1.
  - **Known tail difference:** only episodes 33 and 36 end on a non-zero delta (0.004 / 0.009 rad). Past
    the end of those two episodes the sidecar repeats that delta, while raw chunks use 0. This touches
    at most 58 of 174,087 training windows.
- **Deployment / eval:** a policy outputs 30 per-step joint deltas. Commanded joints are
  `q[t] + cumsum(pred[:, :6])`; the gripper is absolute. Neither the live/RTC servers nor
  `evaluate_uniform_bspline_open_loop.py` integrate yet.
- Server note: `PolicyServerWrapper(use_bf16=True)` makes raw `QwenPI_v3.predict_action` fail with
  `TypeError: Got unsupported ScalarType BFloat16` at `.numpy()`; load it in fp32.

| Run | GPUs | Outcome |
|---|---|---|
| `piper_demos_classify_blocks_30hz_qwenpi_qwen3vl2b_delta_joint_raw_h30_2gpu_gb32_10k` (raw) | 0,1 | **completed 10,000 steps**, exit 0 at 05:44 UTC 2026-10-22 (4.07 h, ≈1.46 s/step); `final_model/pytorch_model.pt` + `steps_{2500,5000,7500,10000}` |
| `piper_demos_classify_blocks_30hz_qwenpi_qwen3vl2b_delta_joint_uniform_left_s2_h30_c18_2gpu_gb32_10k` (B-spline) | 2,3 | **completed 10,000 steps**, exit 0 at 05:40 UTC (4.00 h, ≈1.44 s/step); same checkpoint set |

- **Checkpoint check at step 2,500** (256 held-out observations, validation episodes 80–84, fp32; predicted
  chunks un-normalized with the checkpoint's own stats). Both models output **per-step deltas**.
  *Caveat (found 2026-10-22): this check and the step-10,000 one below fed the **validation-split**
  normalized state (see the open-loop section); the open-loop numbers below supersede them.*

  | | raw | B-spline (decoded) | ground truth |
  |---|---|---|---|
  | max predicted joint delta, rad | 0.022–0.073 | 0.023–0.063 | 0.033–0.067 |
  | delta std, rad | 0.004–0.014 | 0.003–0.012 | 0.005–0.014 |
  | corr(pred, GT), joints 1–6 | .58 .74 .79 .31 .47 .18 | .62 .76 .82 .27 .49 .26 | |
  | q[t+30] RMSE after q[t]+cumsum, joints 2/3, rad | 0.179 / 0.221 | 0.165 / 0.184 | hold-position 0.291 / 0.381 |

  Absolute-angle outputs would be on the scale of 1 rad. This early, joints 4–6 are not yet better than
  zero-delta / hold-position.

- **Checkpoint check at step 10,000** (same 256 held-out observations and procedure, same caveat;
  `verify_delta_joint_checkpoint.py <run_dir> 10000 256`). Both still output **per-step deltas** and now
  beat hold-position on joints 1–3 and 5:

  | | raw | B-spline (decoded) | ground truth / baseline |
  |---|---|---|---|
  | max predicted joint delta, rad | 0.024–0.066 | 0.033–0.063 | 0.033–0.066 |
  | delta std, rad | 0.004–0.013 | 0.004–0.013 | 0.005–0.014 |
  | delta RMSE, joints 1–6, ×1e-3 rad | 4.7 5.9 7.1 5.1 7.2 8.2 | 4.9 6.1 7.1 4.8 7.3 8.4 | zero-delta 6.4 11.1 13.8 4.5 8.8 8.0 |
  | corr(pred, GT), joints 1–6 | .72 .86 .86 .29 .59 .32 | .70 .85 .86 .38 .59 .36 | |
  | q[t+30] RMSE after q[t]+cumsum, joints 1–6, rad | .109 .119 .158 .109 .148 .198 | .117 .123 .155 .102 .153 .210 | hold .171 .291 .381 .099 .206 .200 |
  | gripper RMSE (absolute) | 0.159 | 0.153 | |

  Joints 4 (wrist) and 6 are still at or slightly worse than zero-delta / hold-position. Raw and B-spline
  are about equal at 10k. This is an offline check only; no closed-loop evaluation has been run.

#### Open-loop evaluation on the held-out episodes (2026-10-22)

`eval_files/evaluate_delta_joint_open_loop.py --run-dir <run_dir> --step N [--stride S]` (new). It evaluates
every frame of validation episodes 80–84 (11,911 observations) from the recorded images and state; predictions
are never fed back. Before any metric it **aborts unless all of these hold** (all passed for both runs, every
checkpoint):

| Check | raw | B-spline |
|---|---|---|
| loader `action_mode: abs`, no apply keys; dataset is `StateDeltaJoint` (and the sidecar is built from it) | ok | ok |
| stored `action[t] == (q[t+1]-q[t], g[t+1])` on the validation episodes | max 1.9e-9 rad | same |
| checkpoint un-normalization stats == `meta/stats_gr00t_train.json` (joint range −0.078…0.097 rad) | ok | ok |
| loader sample is `observation.state[t]` | 2.4e-4 (float16) | same |
| loader training target == per-step delta chunk / sidecar controls | 9.4e-5 rad | 8.2e-4 rel. |
| model decode basis (30×18) == sidecar `encoder.json` basis | – | exact |
| integrating the ground-truth chunk through the eval code reproduces measured `q[t+k+1]` | 2.4e-10 rad (float16-packed targets: 6.6e-5) | 1.4e-3 rad (fit error) |

**Normalization pitfall found here:** with `episode_split: validation` the starVLA loader normalizes with the
*validation-split* statistics (`meta/stats_gr00t_validation.json`), not the train statistics the policy was
trained and is deployed with (state ranges differ by up to 0.36 rad; up to 0.44 in normalized units). The
evaluator therefore feeds the state normalized with the checkpoint statistics, exactly as
`piper_live_client.py` does. Training is unaffected (train split + train stats). Any other open-loop script
that takes `state` from a validation-split loader has the same problem.

Results at step 10,000 (all 11,911 observations; rad):

| | raw | B-spline (decoded) | baseline |
|---|---|---|---|
| per-step delta RMSE, joints 1–6 (×1e-3) | 4.6 6.0 7.2 5.0 7.5 8.4 | 4.6 6.1 7.2 4.9 7.4 8.6 | zero-delta 6.5 11.2 13.8 4.9 8.7 8.6 |
| corr(pred, GT) | .72 .86 .86 .33 .57 .35 | .73 .86 .86 .35 .58 .35 | |
| q[t+10] RMSE after q[t]+cumsum | .040 .053 .064 .045 .067 .076 | .040 .054 .063 .045 .066 .078 | hold .062 .109 .134 .045 .081 .082 |
| q[t+30] RMSE | .106 .124 .161 .099 .155 .205 | .105 .129 .160 .098 .153 .213 | hold .170 .298 .376 .106 .204 .219 |
| replay (replan every 30 steps, whole episodes) | .065 .080 .102 .061 .100 .122 | .065 .084 .104 .065 .099 .132 | hold .104 .183 .228 .065 .128 .134 |
| gripper RMSE (absolute, 0–1) | 0.147 | 0.141 | |

Replay joint RMSE across training (stride 3 for 2.5k–7.5k):

| step | raw, joints 1–6 | B-spline, joints 1–6 |
|---|---|---|
| 2,500 | .112 .109 .129 .072 .133 .140 | .082 .114 .129 .073 .128 .135 |
| 5,000 | .075 .104 .113 .069 .114 .135 | .070 .090 .113 .072 .126 .158 |
| 7,500 | .067 .086 .106 .062 .102 .122 | .069 .091 .112 .067 .108 .134 |
| 10,000 | .065 .080 .102 .061 .100 .122 | .065 .084 .104 .065 .099 .132 |

Both representations are correct and equivalent at 10k: joints 1–3 and 5 are clearly better than hold-position
(30–60 % lower error); joint 4 (wrist roll, little motion) and joint 6 are about at hold-position. Reports and
per-episode replay plots: `<run_dir>/open_loop_delta_joint/steps_<N>/` (`metrics.json`,
`replay_episode_0NN.png`; plots for step 10,000).

#### 30k continuation (from 2026-10-22 07:56 UTC)

New run dirs, seeded with a **copy** (byte-identical) of each 10k run's `steps_10000_pytorch_model.pt` and
`training_state` (optimizer/RNG at step 10,000); the 10k runs are untouched. Configs/wrappers
`..._delta_joint_{raw_h30,uniform_left_s2_h30_c18}_2gpu_gb32_30k.{yaml,sh}` equal the 10k ones except `run_id`
and `max_train_steps: 30000` (tested). The LR scheduler is rebuilt for 30k and advanced to step 10,000, so the
action-model LR goes from the 10k run's final 1e-5 to 7.9e-5 and then follows the 30k cosine. Loss continued
without a jump (mean of the last 10 logs: raw 0.041 → 0.041, B-spline 0.074 → 0.073).

| Run | GPUs | Outcome |
|---|---|---|
| `piper_demos_classify_blocks_30hz_qwenpi_qwen3vl2b_delta_joint_raw_h30_2gpu_gb32_30k` | 0,1 | **completed 30,000 steps**, exit 0 at 14:30 UTC 2026-10-22 (6.56 h for 20k steps, ≈1.18 s/step); `final_model` + `steps_{10000..30000}` every 2,500; log `results/training_logs/piper_demos/nohup_2b_raw_delta_joint_30k.out` |
| `piper_demos_classify_blocks_30hz_qwenpi_qwen3vl2b_delta_joint_uniform_left_s2_h30_c18_2gpu_gb32_30k` | 2,3 | **completed 30,000 steps**, exit 0 at 14:28 UTC (6.51 h); same checkpoint set |

Open-loop at step 30,000 (same evaluator, all 11,911 held-out observations; every self-check passed; rad, joints 1–6):

| | raw | B-spline (decoded) | baseline |
|---|---|---|---|
| per-step delta RMSE (×1e-3) | 4.5 5.9 7.0 4.9 7.4 8.2 | 4.4 5.8 6.9 4.8 7.5 8.0 | zero-delta 6.5 11.2 13.8 4.9 8.7 8.6 |
| corr(pred, GT) | .74 .87 .87 .36 .60 .37 | .75 .87 .87 .36 .60 .42 | |
| q[t+10] RMSE | .039 .052 .061 .044 .066 .074 | .038 .051 .060 .044 .068 .072 | hold .062 .109 .134 .045 .081 .082 |
| q[t+30] RMSE | .101 .117 .153 .095 .153 .200 | .098 .117 .151 .096 .157 .195 | hold .170 .298 .376 .106 .204 .219 |
| replay (replan every 30) | .063 .075 .100 .061 .100 .123 | .059 .076 .100 .065 .103 .120 | hold .104 .183 .228 .065 .128 .134 |
| gripper RMSE | 0.136 (replay 0.127) | 0.133 (replay 0.124) | |

Replay at step 20,000 (stride 3): raw .068 .090 .104 .067 .099 .129, B-spline .063 .087 .102 .062 .107 .122.
**Uploaded (2026-10-22):** step 30,000 of both runs to `DiscreteRTC/dRTC/StarVLA/<run_id>/` (same layout as the
earlier uploads: `checkpoints/steps_30000_pytorch_model.pt` + `config.yaml`, `config.full.yaml`,
`dataset_statistics.json`, `summary.jsonl`, `wall_time_ledger.json`). 5.85 GB each; local SHA-256 raw `cf0476f9…`,
B-spline `e52ac4e8…`; HF commits raw `86b86c78`, B-spline `aa222434`. Remote size and LFS SHA-256 match; recorded
in each run's `checkpoint_upload.json`. Command: `upload_piper_demos_starvla_checkpoints.py --runs <both 30k run_ids>
--step 30000`. Code at `b579523`.

30k vs 10k: a small, consistent improvement (q[t+30] RMSE −2…−8 %, correlations up 0.01–0.07); the two
representations stay within noise of each other (B-spline marginally ahead on joints 1 and 6). Joint 4 is still
only at hold-position level. Reports/plots: `<30k run_dir>/open_loop_delta_joint/steps_30000/`.

An unused alternative: `data_files/build_relative_spline_sidecar.py --action-mode delta` builds a sidecar
for starVLA's own `delta` convention on the *commanded* joints (`a[t]-q[t]`, then `a[t+k]-a[t+k-1]`).
Its output is `..._c18_delta_joints` next to the command dataset, built 2026-10-22; no run uses it.

### classify_blocks delta-joint policies (Qwen3-VL-2B-Instruct-Action, 2026-10-19 03:36–07:28 UTC) — SUPERSEDED: state-relative, not per-step

**"Delta joint angle" here means state-relative targets** (π0/openpi-style delta actions). It is
starVLA `action_mode: rel` with `action_mode_apply_keys: [action.joints]`:

    target[k, joints] = command_joints[t+k] - measured_joints[t]    (radians, k = 0..29)
    target[k, gripper] = command_gripper[t+k]                        (absolute, unchanged)

- **Not per-step.** The per-step representation the user wanted is the `StateDeltaJoint` section above.
- **Deployment:** commanded joints = predicted joints + measured joints at the observation.
  `evaluate_uniform_bspline_open_loop.py` adds the offset back automatically for rel configs.
  The live, RTC and policy servers do **not** do this yet.

| Run | GPUs | Config | Outcome |
|---|---|---|---|
| `piper_demos_classify_blocks_30hz_qwenpi_qwen3vl2b_raw_h30_reljoints_2gpu_gb32_30k` (raw) | 0,1 | `starvla_qwenpi_qwen3vl2b_..._raw_h30_reljoints_2gpu_gb32_30k.yaml`, port 29863 | **stopped on request** (SIGTERM, graceful save, exit 0) at step 17,904; 3.86 h used. Step 15,000 uploaded. |
| `piper_demos_classify_blocks_30hz_qwenpi_qwen3vl2b_uniform_left_s2_h30_c18_reljoints_2gpu_gb32_30k` (B-spline) | 2,3 | `starvla_qwenpi_qwen3vl2b_..._uniform_left_s2_h30_c18_reljoints_2gpu_gb32_30k.yaml`, port 29864 | **stopped on request** at step 15,970; 3.86 h used. Step 15,000 uploaded. |

- **Why they stopped:** the user asked to stop at 15k and upload. The learning-rate schedule (cosine over 30k
  steps) was therefore not annealed: at step 15k the action-head LR was 5.6e-5, ≈ 56 % of its 1e-4 peak.
- **Local copies:** weights at steps 5k, 10k and 15k, plus the stop step (17,904 / 15,970), plus a
  resumable `training_state` at the stop step. No `final_model`. Rerunning a wrapper resumes toward 30k.
- **Upload:** `DiscreteRTC/dRTC/StarVLA/<run_id>/` with the same layout as the 4B uploads.
  - 5.85 GB weights; local SHA-256: raw `78ede7f7…`, B-spline `cc16be7a…`.
  - HF commits: raw `6dc90420`, B-spline `1faf065a`. Remote size and LFS SHA-256 match the local files;
    recorded in each run's `checkpoint_upload.json`.
  - The upload command was `upload_piper_demos_starvla_checkpoints.py --runs <run_id>… --step 15000 --stopped-early`.

- **Config parity:** each config differs from its absolute 2B counterpart only in `run_id`, `action_type`,
  `action_mode` and `action_mode_apply_keys`. The B-spline config also points `controls_dataset_path` at
  the relative sidecar. Tests enforce this.
- **B-spline targets:** `..._spline_uniform_left_s2_h30_c18_rel_joints`, built by
  `data_files/build_relative_spline_sidecar.py` from the same encoder.
  - The controls are refitted on the relative windows rather than offset, because the ridge term is not
    translation-invariant.
  - Results: 186,282 records; joint RMSE 1.0–2.4 mrad; max joint error 0.045 rad; the gripper is unchanged.
  - The manifest records `action_mode`/`action_mode_apply_keys`. The loader rejects an absolute sidecar
    under a rel config, and the reverse.
- **Normalization:** min-max, as in the absolute runs. Rel stats live in
  `meta/stats_gr00t_train_rel_a2c0914c.json`, shared by both runs; non-abs caches are now keyed by
  horizon, keys and state map. The B-spline dataset computes them over the 30-step physical horizon,
  not the 18 control rows.
  - Rel joint ranges are min/max −1.70…1.72 rad, q01/q99 about ±0.7 rad; std 0.08–0.24 rad, against
    0.21–0.51 rad absolute.
- **Throughput:** 0.62–0.69 s/step on a quiet node, against 0.81 s/step for 4B. It rose to about
  1.15 s/step while another user saturated the CPUs (load average ≈ 350 of 384 cores).
  - From about 06:40 UTC another user's jobs also shared GPUs 2,3, and the B-spline run dropped to
    2–7 s/step.

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

1. **Full classify_blocks runs launched 2026-10-15 17:48 UTC** (raw on GPUs 0,1; spline on GPUs 2,3); outcome in §0.
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
12. **Upload of classify_blocks checkpoints** (done; see §0, which supersedes the plan below): `deployment/upload/upload_piper_demos_starvla_checkpoints.py`
    (started 2026-10-15 22:38 UTC, log `results/training_logs/piper_demos/upload_starvla_checkpoints.log`)
    waits until both runs have `steps_30000` and `final_model/`. It then uploads one commit per run to
    `DiscreteRTC/dRTC` (dataset repo) under `StarVLA/<run_name>/`: `checkpoints/steps_30000_pytorch_model.pt`,
    `config.yaml`, `config.full.yaml`, `dataset_statistics.json`, `summary.jsonl` and `wall_time_ledger.json`.
    That is the layout `from_pretrained` loads. The script verifies the remote size and LFS SHA-256 and
    writes `<run_dir>/checkpoint_upload.json`. If the uploader stops, rerun it; it waits again and re-commits.
10. Not uploaded: 20260911 raw ttRTC (HF auth) and the four stacking_cups_action_30hz runs (no uploader).
