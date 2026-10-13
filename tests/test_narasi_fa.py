import math

import narasi_fundamental as nf
from test_laporan import _inputs
from test_skor import f_


def test_narasi_dasar_dan_skenario():
    r, rk, f = _inputs()
    n = nf.susun(f, r.harga, "TEST")
    assert n["label"][0][0] == "GROWTH + MURAH" and n["grid"] is not None and list(n["grid"].columns)[:2] == ["Pertumbuhan EPS", "PER 6x"]
    sk = [x for x in n["valuasi"] if x.startswith("[EST] Skenario")][0]
    assert "konservatif" in sk and "optimis" in sk
    assert all(x["url"].startswith("https://") for x in n["rujukan"]) and "NARASI FUNDAMENTAL" in "\n".join(nf.baris(n))


def test_label_sektor_dan_kelemahan():
    batu = dict(f_(sektor="Energi", subsektor="Batu Bara", m=dict(eps=100.0, mcap=1e12, periode=["Q2 2026"], div_ttm=-1.2e11, cfo_kuartal=[-1.0, -2.0, 3.0, 4.0], gearing=1.5)))
    n = nf.susun(batu, 500.0, "COAL")
    tags = [a for a, _ in n["label"]]
    assert "KOMODITAS SIKLIKAL" in tags and "DIVIDEND PLAY" in tags                     # yield 12%
    assert any("tradingeconomics.com/commodity/coal" in x["url"] for x in n["rujukan"]) and any("minerba.esdm.go.id" in x["url"] for x in n["rujukan"])
    assert "Catatan keras" in n["tesis"] and "gearing 150%" in n["tesis"]
    narasi = nf.susun(dict(f_(m=dict(eps=2.0, mcap=1e12, periode=["Q2 2026"]))), 200.0, "MAHAL")
    assert narasi["label"][0][0] == "NARASI"                                            # PER 100x
    bank = nf.susun(dict(f_(bank=True, sektor="Keuangan", subsektor="Bank", m=dict(eps=100.0, mcap=1e12, periode=["Q2 2026"]))), 800.0, "BANK")
    assert any("money-supply-m2" in x["url"] for x in bank["rujukan"]) and any("ojk.go.id" in x["url"] for x in bank["rujukan"])
    rugi = nf.susun(dict(f_(m=dict(eps=-5.0, mcap=1e12, periode=["Q2 2026"]))), 200.0, "RUGI")
    assert rugi["grid"] is None and any(a == "RUGI / TURNAROUND" for a, _ in rugi["label"])
    assert nf.susun(None, 100.0) is None
