import json

from lumbar_ct.cli import main


def test_cli_reports_missing_input_as_failure(tmp_path, capsys):
    result = main(["audit", str(tmp_path / "missing.nii.gz"), "--output", str(tmp_path / "audit.json")])
    assert result == 1
    assert json.loads(capsys.readouterr().err)["status"] == "failed"
    assert not (tmp_path / "audit.json").exists()


def test_cli_reports_invalid_nifti_as_failure(tmp_path, capsys):
    path = tmp_path / "invalid.nii.gz"
    path.write_text("not a NIfTI")
    result = main(["audit", str(path), "--output", str(tmp_path / "audit.json")])
    assert result == 1
    assert json.loads(capsys.readouterr().err)["status"] == "failed"
