"""Contracts for the piper_demos raw-action / uniform-left B-spline training glue.

CPU-only.  Dataset-backed checks skip when ``data/processed/piper_demos`` has
not been generated (see examples/realRobots/Piper/data_files/).
"""

from __future__ import annotations

import json
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILES = REPO_ROOT / "examples/realRobots/Piper/train_files"
DATA_FILES = REPO_ROOT / "examples/realRobots/Piper/data_files"
RAW_YAML = TRAIN_FILES / "starvla_qwenpi_piper_demos_classify_blocks_30hz_raw_h30_2gpu_gb32_30k.yaml"
SPLINE_YAML = TRAIN_FILES / "starvla_qwenpi_piper_demos_classify_blocks_30hz_uniform_left_s2_h30_c18_2gpu_gb32_30k.yaml"
PROCESSED = REPO_ROOT / "data/processed/piper_demos/ActionCommand/State_Joint"
sys.path.insert(0, str(DATA_FILES))

from starVLA.dataloader.gr00t_lerobot.datasets import resolve_split_episode_ids  # noqa: E402
from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES, ROBOT_TYPE_CONFIG_MAP  # noqa: E402
from starVLA.training.trainer_utils import run_budget  # noqa: E402


def test_split_ranges_parse_lerobot_info():
    info = {"splits": {"train": "0:80", "validation": "80:85"}}
    assert resolve_split_episode_ids(info, "train") == set(range(80))
    assert resolve_split_episode_ids(info, "validation") == set(range(80, 85))
    with pytest.raises(KeyError):
        resolve_split_episode_ids(info, "test")


@pytest.mark.parametrize("robot_type", sorted(k for k in ROBOT_TYPE_CONFIG_MAP if k.startswith("piper")))
def test_piper_observations_use_current_step_only(robot_type):
    """Two images = global then hand at step t; state at t.  No past/future frames."""
    config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    modality = config.modality_config()
    assert modality["video"].modality_keys == ["video.global", "video.hand"]
    assert modality["video"].delta_indices == [0]
    assert modality["state"].delta_indices == [0]
    assert modality["state"].modality_keys == ["state.joints", "state.gripper"]
    assert modality["action"].delta_indices[0] == 0


def test_piper_demos_mixtures_pair_the_same_dataset():
    for task in ("classify_blocks", "hanging_mug", "stacking_cup"):
        (raw_name, _, raw_type), = DATASET_NAMED_MIXTURES[f"piper_demos_{task}_30hz_raw_h30"]
        (spline_name, _, spline_type), = DATASET_NAMED_MIXTURES[f"piper_demos_{task}_30hz_uniform_left_bspline"]
        assert raw_name == spline_name == f"processed/piper_demos/ActionCommand/State_Joint/{task}_30hz_cleanup"
        assert raw_type == "piper_abs_joints_30step" and spline_type == "piper_uniform_left_bspline_30step"
        assert len(ROBOT_TYPE_CONFIG_MAP[raw_type].action_indices) == 30
        assert len(ROBOT_TYPE_CONFIG_MAP[spline_type].action_indices) == 18


def _strip_variant(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg.pop("run_id")
    cfg["framework"].pop("name")
    cfg["framework"]["action_model"].pop("action_horizon")
    cfg["framework"].pop("action_representation", None)
    cfg["datasets"]["vla_data"].pop("data_mix")
    cfg["datasets"]["vla_data"].pop("controls_dataset_path", None)
    return cfg


QWEN2B_RAW_YAML = TRAIN_FILES / "starvla_qwenpi_qwen3vl2b_piper_demos_classify_blocks_30hz_raw_h30_2gpu_gb32_30k.yaml"
QWEN2B_SPLINE_YAML = (
    TRAIN_FILES / "starvla_qwenpi_qwen3vl2b_piper_demos_classify_blocks_30hz_uniform_left_s2_h30_c18_2gpu_gb32_30k.yaml"
)


@pytest.mark.parametrize("raw_yaml,spline_yaml", [(RAW_YAML, SPLINE_YAML), (QWEN2B_RAW_YAML, QWEN2B_SPLINE_YAML)])
def test_classify_blocks_configs_differ_only_in_action_representation(raw_yaml, spline_yaml):
    raw, spline = OmegaConf.load(raw_yaml), OmegaConf.load(spline_yaml)
    assert _strip_variant(raw) == _strip_variant(spline)
    assert raw.framework.name == "QwenPI_v3" and spline.framework.name == "QwenPI_v3_UniformBSpline"
    assert raw.framework.action_model.action_model_type == spline.framework.action_model.action_model_type == "LayerwiseFM"
    assert raw.framework.action_model.action_horizon == 30
    assert spline.framework.action_model.action_horizon == spline.framework.action_representation.num_control_points == 18
    rep = spline.framework.action_representation
    assert (rep.mode, rep.degree, rep.span_length_steps, rep.executable_horizon, rep.frequency_hz) == (
        "uniform_left", 3, 2, 30, 30.0)
    assert (rep.num_control_points - rep.degree) * rep.span_length_steps == rep.executable_horizon
    data = raw.datasets.vla_data
    assert data.episode_split == "train" and data.include_state is True
    trainer = raw.trainer
    assert trainer.max_wall_time_hours == 10.0 and trainer.save_training_state is True
    assert data.per_device_batch_size * 2 * trainer.gradient_accumulation_steps == 32


@pytest.mark.parametrize("qwen4b_yaml,qwen2b_yaml", [(RAW_YAML, QWEN2B_RAW_YAML), (SPLINE_YAML, QWEN2B_SPLINE_YAML)])
def test_qwen3vl2b_configs_only_swap_the_backbone(qwen4b_yaml, qwen2b_yaml):
    big, small = OmegaConf.load(qwen4b_yaml), OmegaConf.load(qwen2b_yaml)
    assert small.framework.qwenvl.base_vlm == "./playground/Pretrained_models/Qwen3-VL-2B-Instruct-Action"
    assert small.run_id != big.run_id
    for cfg in (big, small):
        cfg.run_id = None
        cfg.framework.qwenvl.base_vlm = None
    assert big == small


def test_model_decode_basis_matches_encoder_geometry():
    """The basis QwenPI_v3_UniformBSpline builds from the YAML is the sidecar encoder basis."""
    from preprocess_piper_demos import EXPECTED_ENCODER
    from spline_encoder import UniformLeftBSplineConfig, create_encoder, preset  # noqa: F401

    rep = OmegaConf.load(SPLINE_YAML).framework.action_representation
    model_side = create_encoder(UniformLeftBSplineConfig(
        action_dim=rep.action_dim, chunk_size=rep.executable_horizon, frequency_hz=rep.frequency_hz,
        degree=rep.degree, num_basis=rep.num_control_points, span_length_steps=rep.span_length_steps,
        regularization=rep.regularization, end_padding=rep.end_padding,
        implementation_version=rep.implementation_version))
    data_side = create_encoder(UniformLeftBSplineConfig(
        **{k: v for k, v in EXPECTED_ENCODER.items() if k != "mode"}))
    assert model_side.basis.shape == (30, 18)
    np.testing.assert_array_equal(model_side.basis, data_side.basis)
    np.testing.assert_allclose(model_side.basis.sum(axis=1), 1.0, atol=1e-12)
    # Partition of unity makes decoding commute with the affine min_max normalization.
    controls = np.random.default_rng(0).normal(size=(18, 7))
    low, high = -np.arange(1, 8.0), np.arange(1, 8.0) * 2
    normalize = lambda x: 2 * (x - low) / (high - low) - 1  # noqa: E731
    np.testing.assert_allclose(model_side.basis @ normalize(controls), normalize(model_side.basis @ controls), atol=1e-12)


@pytest.mark.parametrize("horizon", [30, 18])
def test_layerwise_fm_head_shapes_and_unmasked_loss(horizon):
    from starVLA.model.framework.share_tools import populate_layerwise_dit_cfg
    from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import get_action_model

    cfg = OmegaConf.load(RAW_YAML)
    cfg.framework.action_model.action_horizon = horizon
    populate_layerwise_dit_cfg(cfg, dit_hidden_dim=64, num_dit_layers=2)
    torch.manual_seed(0)
    head = get_action_model(config=cfg).eval()
    batch = 3
    vl = [torch.randn(batch, 5, 64) for _ in range(2)]
    actions = torch.randn(batch, horizon, 7)
    state = torch.randn(batch, 1, 7)
    loss = head(vl, actions, state)
    assert loss.ndim == 0 and torch.isfinite(loss)
    with torch.no_grad():
        out = head.predict_action(vl, state)
    assert out.shape == (batch, horizon, 7)


def test_wall_time_budget_ledger_and_signal(tmp_path, monkeypatch):
    cfg = SimpleNamespace(max_wall_time_hours=1.0, wall_time_safety_minutes=10.0)
    budget = run_budget.WallTimeBudget(cfg, tmp_path, is_main_process=True, start_step=0)
    try:
        assert budget.enabled and not budget.should_stop("cpu")
        monkeypatch.setattr(budget, "session_seconds", lambda: 3600.0 - 599.0)  # 599 s left < 600 s margin
        assert budget.should_stop("cpu") and budget.stop_reason == "wall_time_budget"
        monkeypatch.setattr(budget, "session_seconds", lambda: 1000.0)
        budget.write_ledger(123, end_reason="signal")
        ledger = json.loads((tmp_path / run_budget.LEDGER_NAME).read_text())
        assert ledger["total_elapsed_seconds"] == pytest.approx(1000.0)
        assert ledger["sessions"][-1]["last_step"] == 123

        # A resumed launch only gets what earlier sessions left.
        resumed = run_budget.WallTimeBudget(cfg, tmp_path, is_main_process=True, start_step=123)
        assert resumed.previous_seconds == pytest.approx(1000.0)
        monkeypatch.setattr(resumed, "session_seconds", lambda: 1999.0)
        assert not resumed.should_stop("cpu")
        monkeypatch.setattr(resumed, "session_seconds", lambda: 2001.0)
        assert resumed.should_stop("cpu")
        resumed.stop_reason = None
        monkeypatch.setattr(resumed, "session_seconds", lambda: 1.0)
        resumed._on_signal(signal.SIGTERM, None)
        assert resumed.should_stop("cpu") and resumed.stop_reason == "signal"
    finally:
        for sig in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
            signal.signal(sig, signal.SIG_DFL if sig != signal.SIGINT else signal.default_int_handler)


def test_budget_disabled_by_default(tmp_path):
    budget = run_budget.WallTimeBudget(SimpleNamespace(), tmp_path, is_main_process=True, start_step=0)
    assert not budget.enabled and not budget.should_stop("cpu")
    budget.write_ledger(1)
    assert not (tmp_path / run_budget.LEDGER_NAME).exists()


@pytest.mark.skipif(not (PROCESSED / "hanging_mug_30hz_cleanup").exists(), reason="processed data not generated")
def test_episode_split_filters_trajectories_and_statistics():
    from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset

    data_root = REPO_ROOT / "data"
    name = "processed/piper_demos/ActionCommand/State_Joint/hanging_mug_30hz_cleanup"
    sidecar = PROCESSED / "hanging_mug_30hz_cleanup_spline_uniform_left_s2_h30_c18"
    cfg = OmegaConf.merge(OmegaConf.load(SPLINE_YAML).datasets.vla_data,
                          {"controls_dataset_path": str(sidecar), "data_root_dir": str(data_root)})
    info = json.loads((data_root / name / "meta/info.json").read_text())
    train = resolve_split_episode_ids(info, "train")
    spline = make_LeRobotSingleDataset(data_root, name, "piper_uniform_left_bspline_30step", data_cfg=cfg)
    assert set(map(int, spline.trajectory_ids)) == train
    assert (data_root / name / "meta/stats_gr00t_train.json").exists()
    full_cfg = OmegaConf.merge(cfg, {"episode_split": None})
    full = make_LeRobotSingleDataset(data_root, name, "piper_abs_joints_30step", data_cfg=full_cfg)
    assert len(full.trajectory_ids) == info["total_episodes"] > len(train)


def test_steps_cache_rejects_other_trajectory_sets():
    """A cached step index from another split must never be reused (validation leakage)."""
    from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset

    check = LeRobotSingleDataset._check_cached_steps
    steps = [(0, 0), (0, 1), (1, 0)]
    check({"config_key": "k", "steps": steps, "trajectory_ids": [0, 1]}, "k", {0, 1})
    with pytest.raises(ValueError):
        check({"config_key": "k", "steps": steps, "trajectory_ids": [0, 1, 2]}, "k", {0, 1})
    with pytest.raises(ValueError):
        check({"config_key": "other", "steps": steps, "trajectory_ids": [0, 1]}, "k", {0, 1})
    # Legacy caches without trajectory_ids: accepted only if they stay inside the dataset/split.
    check({"config_key": "k", "steps": steps}, "k", {0, 1, 2})
    with pytest.raises(ValueError):
        check({"config_key": "k", "steps": steps + [(5, 0)]}, "k", {0, 1})


def test_training_state_save_waits_for_all_ranks_before_rename(tmp_path, monkeypatch):
    """Regression: rank 0 renamed training_state.tmp while rank 1 still wrote its RNG state."""
    from starVLA.training import train_starvla

    events = []

    class FakeAccelerator:
        is_main_process = True

        def wait_for_everyone(self):
            events.append("barrier")

        def save_state(self, path):
            Path(path).mkdir(parents=True)
            events.append("save_state")

        def print(self, *args, **kwargs):
            pass

    real_replace = train_starvla.os.replace

    def recording_replace(src, dst):
        events.append("rename")
        real_replace(src, dst)

    monkeypatch.setattr(train_starvla.os, "replace", recording_replace)
    trainer = SimpleNamespace(
        config=SimpleNamespace(trainer=SimpleNamespace(save_training_state=True)),
        checkpoint_dir=str(tmp_path),
        accelerator=FakeAccelerator(),
        completed_steps=7,
    )
    trainer._training_state_dir = lambda: tmp_path / "training_state"
    train_starvla.VLATrainer._save_training_state(trainer)

    save_idx = events.index("save_state")
    assert "barrier" in events[save_idx + 1 : events.index("rename")]
    assert json.loads((tmp_path / "training_state" / "training_state_meta.json").read_text()) == {"steps": 7}


def test_set_task_instruction_rewrites_only_task_text(tmp_path):
    from preprocess_piper_demos import DETAILED_TASK_INSTRUCTIONS, TASKS, set_task_instruction

    assert set(DETAILED_TASK_INSTRUCTIONS) == set(TASKS)
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "Stack the cups."}) + "\n")
    episodes = [{"episode_index": i, "tasks": ["Stack the cups."], "length": 10 + i} for i in range(3)]
    (meta / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in episodes))

    new = DETAILED_TASK_INSTRUCTIONS["stacking_cup"]
    assert set_task_instruction(tmp_path, new) == "Stack the cups."
    assert set_task_instruction(tmp_path, new) is None  # idempotent
    assert [json.loads(line) for line in (meta / "tasks.jsonl").read_text().splitlines()] == [
        {"task_index": 0, "task": new}
    ]
    rows = [json.loads(line) for line in (meta / "episodes.jsonl").read_text().splitlines()]
    assert rows == [{**row, "tasks": [new]} for row in episodes]


@pytest.mark.parametrize(
    "task,instruction",
    [("classify_blocks", "Classify the blocks."), ("hanging_mug", "Hang the mug."), ("stacking_cup", "Stack the cups.")],
)
def test_processed_datasets_keep_short_instructions(task, instruction):
    """Trained policies and their open-loop evals on these datasets use the short task names."""
    meta = PROCESSED / f"{task}_30hz_cleanup" / "meta"
    if not meta.exists():
        pytest.skip("processed data not present")
    tasks = [json.loads(line) for line in (meta / "tasks.jsonl").read_text().splitlines()]
    assert tasks == [{"task_index": 0, "task": instruction}]
    episodes = [json.loads(line) for line in (meta / "episodes.jsonl").read_text().splitlines()]
    assert all(row["tasks"] == [instruction] for row in episodes)
