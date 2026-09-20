import json

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.rl.train import main
import pytest

from .test_dataset import TAG, modalities, write_lerobot


def test_real_dataset_cli_train_resume_and_metadata_guard(tmp_path, monkeypatch):
    data, output = tmp_path / "data", tmp_path / "run"
    write_lerobot(data)
    monkeypatch.setitem(MODALITY_CONFIGS, TAG.value, modalities())
    args = [
        "--dataset-path",
        str(data),
        "--output-dir",
        str(output),
        "--reward-column",
        "reward",
        "--terminated-column",
        "terminated",
        "--last-row-is-observation",
        "--horizon",
        "2",
        "--batch-size",
        "2",
        "--hidden-dim",
        "8",
        "--hidden-layers",
        "1",
        "--flow-steps",
        "2",
        "--candidates",
        "2",
        "--cpu-threads",
        "2",
    ]
    main([*args, "--steps", "2"])
    checkpoint = output / "checkpoints/step-2.pt"
    assert checkpoint.exists()
    main([*args, "--steps", "3", "--resume", str(checkpoint)])
    lines = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert [line["step"] for line in lines] == [1, 2, 3]
    with pytest.raises(ValueError, match="metadata"):
        main([*args, "--steps", "3", "--gamma", "0.5", "--resume", str(checkpoint)])
    with pytest.raises(FileExistsError):
        main([*args, "--steps", "4"])
    with pytest.raises(ValueError, match="ahead of checkpoint"):
        main([*args, "--steps", "4", "--resume", str(checkpoint)])
    manifest = output / "run.json"
    previous = json.loads(manifest.read_text())
    previous["metadata"]["rl_signal_sha256"] = "different run"
    manifest.write_text(json.dumps(previous))
    with pytest.raises(ValueError, match="Output run metadata"):
        main([*args, "--steps", "4", "--resume", str(output / "checkpoints/step-3.pt")])


def test_missing_reward_column_fails_before_creating_run(tmp_path, monkeypatch):
    data, output = tmp_path / "data", tmp_path / "run"
    write_lerobot(data)
    monkeypatch.setitem(MODALITY_CONFIGS, TAG.value, modalities())
    with pytest.raises(ValueError, match="explicit RL columns"):
        main(
            [
                "--dataset-path",
                str(data),
                "--output-dir",
                str(output),
                "--reward-column",
                "missing",
                "--steps",
                "1",
            ]
        )
    assert not output.exists()
