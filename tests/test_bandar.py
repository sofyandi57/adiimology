import dataclasses

import numpy as np
import pandas as pd

import adimology as ad
import bandar as bd
from fakes import setup_case
from test_teknikal import ohlc


def _r(**kw):
    r = ad.analyze(setup_case(), ad.Params(kode="TEST"))
    r = dataclasses.replace(r, konsentrasi=dict(r.konsentrasi, status="AKUMULASI TERKONSENTRASI"))
    return dataclasses.replace(r, **kw) if kw else r


def _aliran(sama=4, asing=5e9, nego="NEGO KECIL", bnet=0.0):
    kc = pd.DataFrame({"window": ["1 hari", "3 hari", "7 hari", "20 hari"], "kode": ["CC"] * 4,
                       "sama_dengan_bandar": [True] * sama + [False] * (4 - sama)})
    return {"konsistensi": kc, "asing": {"net_value": asing, "status": "NET BUY ASING" if asing > 0 else "NET SELL ASING"},
            "nego": {"status": nego, "bandar_net_lot": bnet, "vol_lot": 100.0, "porsi": 0.01, "top": []}, "afiliasi": None}


def _sideways():
    x = np.arange(200)
    return {"ind": ohlc(1000 + 30 * np.sin(x * 2 * np.pi / 15))}


def test_fase_patterns():
    assert bd.fase(ohlc(np.linspace(2000, 1000, 200)))["fase"] == "markdown"
    assert bd.fase(ohlc(np.r_[np.full(100, 1000.0), np.linspace(1000, 1500, 100)]))["fase"] == "distribusi?"
    assert bd.fase(ohlc(np.r_[np.full(100, 1000.0), np.linspace(1000, 1200, 100)]))["fase"] == "markup"
    assert bd.fase(_sideways()["ind"])["fase"] == "akumulasi?"
    assert bd.fase(ohlc(np.linspace(100, 110, 30)))["fase"] == "tidak diukur"


def test_strong_requires_converging_evidence():
    p = bd.penilaian(_r(), _aliran(), _sideways())
    assert p["kategori"] == "AKUMULASI KUAT" and not p["jebakan"], (p["skor"], p["kriteria"])
    weak = bd.penilaian(_r(), _aliran(sama=2, asing=-1e9), _sideways())
    assert weak["kategori"] != "AKUMULASI KUAT"


def test_without_flow_data_never_strong():
    p = bd.penilaian(_r(), None, _sideways())
    assert p["kategori"] != "AKUMULASI KUAT" and not p["lengkap"]
    assert "tidak diukur" in set(bd.tabel(p)["Status"].str.split(" ", n=1).str[1])


def test_traps():
    nego = bd.penilaian(_r(), _aliran(nego="BANDAR JUAL DI NEGO", bnet=-5000), _sideways())
    assert nego["kategori"].startswith("WASPADA") and any("negosiasi" in j for j in nego["jebakan"])
    kejar = bd.penilaian(_r(floating=0.3), _aliran(), _sideways())
    assert kejar["kategori"].startswith("WASPADA")
    # distribusi terselubung: harga naik > 5% di window, 3 penjual besar >= 15 pembeli
    r = _r()
    ch = r.chart.copy().astype({"open": float, "close": float})
    ch.loc[ch.index[-1], "close"] = ch["open"].iloc[0] * 1.10
    bs = r.bs.copy()
    extra = pd.DataFrame([{"code": f"S{i}", "name": "X", "net_volume": -10_000_000.0, "buy_volume": 1.0} for i in range(3)])
    p = bd.penilaian(dataclasses.replace(r, chart=ch, bs=pd.concat([bs, extra], ignore_index=True)), _aliran(), _sideways())
    assert any("Distribusi terselubung" in j for j in p["jebakan"])
