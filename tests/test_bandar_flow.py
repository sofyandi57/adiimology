import datetime as dt

import numpy as np
import pandas as pd

import bandar_flow as bf

TGL = [d.date() for d in pd.bdate_range(end="2026-10-09", periods=60)]


def _data(nonritel_net, ritel_net, harga, vol=100_000.0, hari_vol=None):
    net = pd.DataFrame({"AK": nonritel_net, "BK": [0.0] * len(TGL), "XL": ritel_net}, index=TGL)
    v = [vol * 100] * len(TGL)
    for i in (hari_vol or []):
        v[i] = vol * 100 * 3
    o = pd.DataFrame({"tgl": TGL, "open": harga, "high": [h * 1.01 for h in harga], "low": [h * 0.99 for h in harga], "close": harga, "volume": v})
    return dict(net=net, volume=pd.Series([x / 100 for x in v], index=TGL), sumber="uji", lengkap_broker=True, crossing_pct=0.0), o


def test_akumulasi_terkonfirmasi_vs_kandidat():
    naik = list(np.linspace(1000, 1100, 60))
    d, o = _data([3000.0] * 60, [-3000.0] * 60, naik)
    r = bf.analisa(d, o)
    assert r["label"] == bf.ACC_K and r["observed"]["jendela"].tolist()[:2] == ["1D", "3D"] and r["invalidasi"]
    turun = list(np.linspace(1100, 1000, 60))
    d, o = _data([3000.0] * 60, [-3000.0] * 60, turun)
    r = bf.analisa(d, o)
    assert r["label"] == bf.ACC_C and "belum efektif" in r["harga"]["respons"]            # net beli tetapi harga turun: BUKAN akumulasi bullish


def test_distribusi_dan_netral_dan_insufficient():
    turun = list(np.linspace(1100, 1000, 60))
    d, o = _data([-3000.0] * 60, [3000.0] * 60, turun)
    r = bf.analisa(d, o)
    assert r["label"] == bf.DIS_K and r["evidence"] in ("Medium", "High")
    naik = list(np.linspace(1000, 1100, 60))
    d, o = _data([-3000.0] * 60, [3000.0] * 60, naik)
    r = bf.analisa(d, o)
    assert r["label"] in (bf.DIS_C, bf.DIS_K) and "rapuh" in r["harga"]["respons"]
    acak = [1000.0] * 60
    d, o = _data([100.0 if i % 2 else -100.0 for i in range(60)], [0.0] * 60, acak)
    assert bf.analisa(d, o)["label"] in (bf.NETRAL, bf.DIS_C)
    d, o = _data([3000.0] * 60, [-3000.0] * 60, acak)
    d["net"] = d["net"].iloc[-10:]
    assert bf.analisa(d, o)["label"] == bf.INSUF
    assert bf.analisa(None, None)["label"] == bf.INSUF


def test_laporan_tiga_lapisan():
    naik = list(np.linspace(1000, 1100, 60))
    d, o = _data([3000.0] * 60, [-3000.0] * 60, naik)
    L = "\n".join(bf.baris(bf.analisa(d, o)))
    assert "Lapisan 1: observed flow" in L and "Lapisan 2: price confirmation" in L and "Lapisan 3: interpretation" in L and "BUKAN probabilitas" in L
