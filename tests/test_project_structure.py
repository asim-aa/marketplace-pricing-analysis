from pathlib import Path


def test_required_phase_one_documents_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    required_paths = [
        root / "docs" / "project_definition.md",
        root / "docs" / "data_dictionary.md",
        root / "data" / "raw",
        root / "src" / "data",
        root / "tests",
    ]

    missing = [str(path) for path in required_paths if not path.exists()]
    assert not missing, f"Missing required project paths: {missing}"
