import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from lumbar_ct.training import choose_device, validate_full_source

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cohort():
    rows = []
    for split, count in (("train", 1082), ("validation", 57), ("test", 89)):
        rows.extend(
            {
                "case_id": f"{split}_{number:04d}",
                "dataset": "totalsegmentator_v201",
                "split": split,
            }
            for number in range(count)
        )
    train = []
    validation = []
    for row in rows:
        if row["dataset"] != "totalsegmentator_v201":
            continue
        if row["split"] == "train":
            train.append(row["case_id"])
        elif row["split"] == "validation":
            validation.append(row["case_id"])
    cases = []
    for role, ids in (("train", train), ("validation", validation)):
        for case_id in ids:
            cases.append(
                {
                    "case_id": case_id,
                    "role": role,
                    "status": "prepared",
                    "dataset": "totalsegmentator_v201",
                }
            )
    manifest = {
        "train_ids": train,
        "validation_ids": validation,
        "preparation_exclusions": [],
        "cases": cases,
    }
    return manifest, rows


def test_full_split_accounts_for_rejected_training_cases(cohort):
    manifest, rows = cohort
    rejected = manifest["train_ids"].pop()
    manifest["preparation_exclusions"].append({"case_id": rejected, "error": "overlap"})
    manifest["cases"] = [row for row in manifest["cases"] if row["case_id"] != rejected]
    validate_full_source(manifest, rows)


def test_full_split_rejects_silent_omission(cohort):
    manifest, rows = cohort
    manifest["train_ids"].pop()
    with pytest.raises(ValueError, match="accounted for"):
        validate_full_source(manifest, rows)


def test_full_split_rejects_test_case(cohort):
    manifest, rows = cohort
    manifest["train_ids"][0] = next(
        row["case_id"]
        for row in rows
        if row["dataset"] == "totalsegmentator_v201" and row["split"] == "test"
    )
    with pytest.raises(ValueError, match="accounted for"):
        validate_full_source(manifest, rows)


def test_full_split_keeps_all_validation(cohort):
    manifest, rows = cohort
    manifest["validation_ids"].pop()
    with pytest.raises(ValueError, match="57"):
        validate_full_source(manifest, rows)


def test_auto_uses_cpu_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert choose_device("auto") == torch.device("cpu")
    with pytest.raises(ValueError, match="CUDA is unavailable"):
        choose_device("cuda")


def test_unsupported_mps_stops_before_preparation(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    def unsupported(*args, **kwargs):
        raise NotImplementedError("AvgPool3d is unavailable")

    monkeypatch.setattr(torch, "ones", unsupported)
    with pytest.raises(ValueError, match="3D operations"):
        choose_device("mps")


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_full_training", PROJECT / "scripts/run_full_training.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launcher_selects_on_validation_and_preserves_packaged_weights(tmp_path, monkeypatch):
    runner = load_runner()
    project = tmp_path / "project"
    selected = project / "outputs/nnunet/model.pt"
    selected.parent.mkdir(parents=True)
    selected.write_bytes(b"original packaged weights")
    data = tmp_path / "data"
    data.mkdir()
    (data / "meta.csv").write_text("metadata")
    work = project / "training/new_run"
    monkeypatch.setattr(runner, "PROJECT", project)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_full_training.py",
            "--device",
            "cpu",
            "--data-dir",
            str(data),
            "--work-dir",
            str(work),
        ],
    )
    calls = []

    def stage(script, *arguments):
        calls.append(script)
        arguments = [str(value) for value in arguments]
        if script == "export_nnunet.py":
            checkpoint = Path(arguments[arguments.index("--output") + 1])
            checkpoint.write_bytes(checkpoint.stem.encode())
        if script == "validate_model.py":
            checkpoint = Path(arguments[arguments.index("--checkpoint") + 1])
            output = Path(arguments[arguments.index("--output") + 1])
            output.mkdir()
            macro = 0.7 if checkpoint.stem == "midway" else 0.8
            runner.write_json(
                output / "summary.json",
                {
                    "checkpoint_sha256": runner.file_hash(checkpoint),
                    "summary": {
                        "attempted_cases": 57,
                        "macro": macro,
                        "levels": {"L1": {"mean": macro}},
                    },
                },
            )

    monkeypatch.setattr(runner, "run", stage)
    runner.main()
    assert calls == [
        "prepare_full_source.py",
        "prepare_nnunet.py",
        "train_nnunet.py",
        "export_nnunet.py",
        "validate_model.py",
        "export_nnunet.py",
        "validate_model.py",
    ]
    assert (work / "model.pt").read_bytes() == b"final"
    assert selected.read_bytes() == b"original packaged weights"
    with pytest.raises(ValueError, match="existing run"):
        runner.check_work_directory(work, data)
    with pytest.raises(ValueError, match="overlap"):
        runner.check_work_directory(selected.parent, data)
