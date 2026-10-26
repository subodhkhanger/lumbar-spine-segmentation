"""Check the one-command launcher without downloading packages."""

import os
from pathlib import Path
import shutil
import subprocess


def test_launcher_installs_once_and_reuses_its_environment(tmp_path):
    source = Path(__file__).resolve().parents[1]
    project = tmp_path / "package"
    project.mkdir()
    shutil.copy2(source / "run_inference.sh", project / "run_inference.sh")
    (project / "requirements-cpu.lock").write_text("numpy==2.2.6\n")
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    python = fake_bin / "python3.11"
    python.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == '-m' && \"$2\" == 'venv' ]]; then\n"
        "  mkdir -p \"$3/bin\"\n"
        "  cp \"$0\" \"$3/bin/python\"\n"
        "  exit 0\n"
        "fi\n"
        "if [[ \"$1\" == '-m' && \"$2\" == 'pip' ]]; then\n"
        "  echo install >> \"$TEST_PROJECT/install_calls\"\n"
        "  touch \"$TEST_PROJECT/installed\"\n"
        "  exit 0\n"
        "fi\n"
        "if [[ \"$1\" == '-' && \"$2\" == \"$TEST_PROJECT/requirements-cpu.lock\" ]]; then\n"
        "  [[ -f \"$TEST_PROJECT/installed\" ]]\n"
        "  exit $?\n"
        "fi\n"
        "echo inference >> \"$TEST_PROJECT/inference_calls\"\n"
    )
    python.chmod(0o755)
    environment = os.environ.copy()
    environment.pop("LUMBAR_PYTHON", None)
    environment["PATH"] = str(fake_bin) + os.pathsep + environment["PATH"]
    environment["TEST_PROJECT"] = str(project)
    command = [str(project / "run_inference.sh"), str(input_dir), str(tmp_path / "output")]
    subprocess.run(command, check=True, env=environment)
    subprocess.run(command, check=True, env=environment)
    assert (project / "install_calls").read_text().splitlines() == ["install"]
    assert len((project / "inference_calls").read_text().splitlines()) == 2
