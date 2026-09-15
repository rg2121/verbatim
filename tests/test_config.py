import subprocess
import sys
from pathlib import Path


def run_config(env: dict[str, str], cwd: Path) -> dict[str, str]:
    """Import config in a fresh interpreter, since it reads the environment once."""
    script = (
        "from verbatim import config; "
        "print(config.CHUNK_SIZE, config.OLLAMA_MODEL, config.OLLAMA_TIMEOUT, sep='|')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, check=True, cwd=cwd,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(Path(__file__).parents[1] / "src"), **env},
    )
    size, model, timeout = result.stdout.strip().split("|")
    return {"chunk_size": size, "ollama_model": model, "timeout": timeout}


def test_env_file_supplies_defaults(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "VERBATIM_CHUNK_SIZE=777\nOLLAMA_MODEL=from-file\nOLLAMA_TIMEOUT=45\n"
    )

    values = run_config({}, tmp_path)

    assert values == {"chunk_size": "777", "ollama_model": "from-file", "timeout": "45.0"}


def test_an_exported_variable_beats_the_env_file(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("VERBATIM_CHUNK_SIZE=777\nOLLAMA_MODEL=from-file\n")

    values = run_config({"VERBATIM_CHUNK_SIZE": "42", "OLLAMA_MODEL": "from-export"}, tmp_path)

    assert values["chunk_size"] == "42"
    assert values["ollama_model"] == "from-export"


def test_defaults_apply_with_no_env_file(tmp_path: Path) -> None:
    values = run_config({}, tmp_path)

    assert values == {"chunk_size": "1200", "ollama_model": "None", "timeout": "600.0"}


def test_a_non_numeric_setting_is_rejected_with_its_name(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("VERBATIM_CHUNK_SIZE=not-a-number\n")

    result = subprocess.run(
        [sys.executable, "-c", "from verbatim import config"],
        capture_output=True, text=True, cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(Path(__file__).parents[1] / "src")},
    )

    assert result.returncode != 0
    assert "VERBATIM_CHUNK_SIZE must be an integer" in result.stderr


def test_the_extraction_timeout_setting_reaches_the_chunker(tmp_path: Path) -> None:
    """It was read into config, documented in .env.example, and never used.

    Chunking hard-coded 180 seconds, so setting it changed nothing.
    """
    result = subprocess.run(
        [sys.executable, "-c",
         "from verbatim import chunking; print(chunking.EXTRACTION_TIMEOUT_SECONDS)"],
        capture_output=True, text=True, check=True, cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(Path(__file__).parents[1] / "src"),
             "VERBATIM_EXTRACTION_TIMEOUT": "7"},
    )

    assert result.stdout.strip() == "7.0"
