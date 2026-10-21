from pathlib import Path

import pytest

from grblr import settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("GRBLR_DB", "VC_DB", "GRBLR_CORPUS_ROOT", "GRBLR_DATA"):
        monkeypatch.delenv(k, raising=False)


def test_db_path_prefers_grblr_over_vc(monkeypatch):
    monkeypatch.setenv("VC_DB", "old.db")
    monkeypatch.setenv("GRBLR_DB", "new.db")
    assert settings.db_path() == Path("new.db")


def test_db_path_falls_back_to_vc(monkeypatch):
    monkeypatch.setenv("VC_DB", "old.db")
    assert settings.db_path() == Path("old.db")


def test_db_path_none_when_unset():
    assert settings.db_path() is None
    assert settings.corpus_root() is None


def test_data_dir_defaults_under_home_and_honours_env(monkeypatch, tmp_path):
    assert settings.data_dir("renders") == Path.home() / ".grblr" / "renders"
    monkeypatch.setenv("GRBLR_DATA", str(tmp_path))
    assert settings.data_dir("renders") == tmp_path / "renders"


def test_cli_without_db_exits_with_message(capsys):
    from grblr.cli import main

    with pytest.raises(SystemExit):
        main(["stats"])
    assert "GRBLR_DB" in capsys.readouterr().err
