import pytest


@pytest.fixture(autouse=True)
def _db_terpisah(tmp_path, monkeypatch):
    """Setiap tes memakai DB ledger/cache sendiri (cegah cache lintas-tes dan file ledger.sqlite di repo)."""
    monkeypatch.setenv("LEDGER_DB_URL", f"sqlite:///{tmp_path}/auto_ledger.db")
