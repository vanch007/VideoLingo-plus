from pathlib import Path
import subprocess

import pytest
import install


def test_existing_native_library_is_not_replaced(tmp_path, monkeypatch):
    source = tmp_path / "brew"
    source.mkdir()
    (source / "libavutil.59.dylib").write_text("new library")
    prefix = tmp_path / "environment"
    (prefix / "lib").mkdir(parents=True)
    existing = prefix / "lib/libavutil.59.dylib"
    existing.write_text("environment-owned library")
    monkeypatch.setattr(install.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(install.sys, "prefix", str(prefix))
    original = install.Path
    monkeypatch.setattr(install, "Path", lambda value: source if str(value).endswith("opt/ffmpeg@7/lib") else original(value))
    assert install.configure_ffmpeg_libraries() == []
    assert existing.read_text() == "environment-owned library"


def test_dependency_install_failure_stops_verification(monkeypatch):
    def failed_install(*args):
        raise subprocess.CalledProcessError(1, "pip install")
    monkeypatch.setattr(install, "install_package", failed_install)
    monkeypatch.setattr(install, "configure_ffmpeg_libraries", lambda: pytest.fail("must stop after install failure"))
    with pytest.raises(subprocess.CalledProcessError):
        install.install_runtime_dependencies()


def test_broken_dependency_graph_stops_native_validation(monkeypatch):
    monkeypatch.setattr(install, "install_package", lambda *args: None)
    monkeypatch.setattr(install, "configure_ffmpeg_libraries", lambda: [])
    calls = []
    def run(cmd):
        calls.append(cmd)
        raise subprocess.CalledProcessError(1, cmd)
    monkeypatch.setattr(install.subprocess, "check_call", run)
    with pytest.raises(subprocess.CalledProcessError):
        install.install_runtime_dependencies()
    assert len(calls) == 1 and calls[0][-2:] == ["pip", "check"]
