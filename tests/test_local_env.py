import os

from research0.local_env import load_local_env


def test_loads_local_env_without_overriding_process(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("# comment\nDATABASE_URL='postgresql://local/test'\nOPENAI_API_KEY=local-key\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "inherited-key")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    load_local_env(path)
    assert os.environ["DATABASE_URL"] == "postgresql://local/test"
    assert os.environ["OPENAI_API_KEY"] == "inherited-key"
