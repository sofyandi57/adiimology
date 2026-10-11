import datetime as dt
from zoneinfo import ZoneInfo

import pytest
import requests

import adimology as ad
import apicache as ac
import ledger as lg

WIB = ZoneInfo("Asia/Jakarta")


class Resp:
    def __init__(self, code, j=None):
        self.status_code, self.j = code, j

    def json(self):
        return self.j

    def raise_for_status(self):
        pass


@pytest.fixture
def eng(tmp_path):
    return ac.siapkan(lg.engine(f"sqlite:///{tmp_path}/c.db"))


@pytest.fixture
def api(monkeypatch):
    calls = []

    def fake(url, headers=None, params=None, timeout=None):
        calls.append((url, dict(params or {})))
        if params and params.get("limit") == "99":
            return Resp(422)
        return Resp(200, {"url": url, "n": len(calls)})
    monkeypatch.setattr(requests, "get", fake)
    return calls


def test_ttl_kebijakan():
    now = dt.datetime(2026, 10, 12, 10, 0, tzinfo=WIB)
    sore = dt.datetime(2026, 10, 12, 19, 0, tzinfo=WIB)
    T = ac.ttl_detik
    assert T("/analysis/order-book/BBCA", (), now) is None
    assert T("/analysis/summary/stock/BBCA", (("from", "2026-10-09"), ("to", "2026-10-09")), now) is None
    assert T("/analysis/financial-statement/BBCA", (("limit", 8),), now) == 7 * 86400
    assert T("/analysis/shareholder/number/BBCA", (), now) == 86400
    assert T("/analysis/chart/stock/BBCA", (("from", "2026-01-01"), ("to", "2026-10-09")), now) == 30 * 86400
    assert T("/analysis/chart/stock/BBCA", (("from", "2026-01-01"), ("to", "2026-10-12")), now) == 600
    assert T("/analysis/chart/stock/BBCA", (("from", "2026-01-01"), ("to", "2026-10-12")), sore) == 12 * 3600


def test_hit_menghemat_dan_dihitung(eng, api):
    p = (("from", "2026-01-01"), ("to", "2026-10-09"))
    a = ac.ambil(eng, "/analysis/chart/stock/BBCA", p, "k")
    b = ac.ambil(eng, "/analysis/chart/stock/BBCA", p, "k")
    assert a == b and len(api) == 1
    u = ac.pemakaian(eng)
    assert u["req"] == 1 and u["hit"] == 1 and u["entri"] == 1 and u["kuota"] == 30000


def test_realtime_tidak_disimpan(eng, api):
    for _ in range(2):
        ac.ambil(eng, "/analysis/order-book/BBCA", (("a", 1),), "k")
    assert len(api) == 2 and ac.pemakaian(eng)["entri"] == 0


def test_422_disimpan_negatif(eng, api):
    p = (("limit", "99"), ("statement", "IS"))
    for _ in range(2):
        with pytest.raises(ad.ParamDitolak):
            ac.ambil(eng, "/analysis/financial-statement/KETR", p, "k")
    assert len(api) == 1


def test_kedaluwarsa_dan_bersihkan(eng, api):
    p = (("x", 1),)
    t0 = dt.datetime(2026, 10, 12, 10, 0, tzinfo=WIB)
    ac.ambil(eng, "/analysis/calendar", p, "k", now=t0)
    ac.ambil(eng, "/analysis/calendar", p, "k", now=t0 + dt.timedelta(hours=2))
    assert len(api) == 1
    ac.ambil(eng, "/analysis/calendar", p, "k", now=t0 + dt.timedelta(days=2))
    assert len(api) == 2
    ac.bersihkan(eng, now=t0 + dt.timedelta(days=10))
    assert ac.pemakaian(eng, now=t0)["entri"] == 0


def test_db_mati_jatuh_ke_api(api):
    class Rusak:
        dialect = type("D", (), {"name": "x"})()

        def connect(self):
            raise RuntimeError("db mati")
    r = ac.ambil(Rusak(), "/analysis/chart/stock/BBCA", (("to", "2020-01-01"),), "k")
    assert r["n"] == 1


def test_ditolak_menurut_nama_tahan_reload(eng, monkeypatch):
    import importlib
    old = ad.ParamDitolak
    importlib.reload(ad)                       # kelas baru != kelas lama (seperti pemuatan ulang modul oleh Streamlit)
    assert ad.ParamDitolak is not old
    assert ac.ditolak(old("x")) and ac.ditolak(ad.ParamDitolak("x"))

    def fake(url, headers=None, params=None, timeout=None):
        return Resp(422)
    monkeypatch.setattr(requests, "get", fake)
    with pytest.raises(Exception) as ei:
        ac.ambil(eng, "/analysis/financial-statement/X", (("limit", "8"),), "k")
    assert ac.ditolak(ei.value)
    with pytest.raises(Exception) as ei2:                    # kedua kali dari cache negatif
        ac.ambil(eng, "/analysis/financial-statement/X", (("limit", "8"),), "k")
    assert ac.ditolak(ei2.value)
