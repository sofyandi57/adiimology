import dataclasses
import datetime as dt

import numpy as np
import pandas as pd
import pytest

import adimology as ad
import bandar as bd
import hengky as hk
import teknikal as tk
from fakes import setup_case
from test_bandar import _aliran
from test_teknikal import ohlc


def _series(close, vol=None):
    df = ohlc(close)
    if vol is not None:
        df["volume"] = vol
    return df


def test_klasifikasi_pv():
    k = hk.klasifikasi_pv
    assert k(0.03, 3.0) == "SEHAT" and k(0.03, 0.4) == "NAIK_SEPI"
    assert k(-0.03, 0.4) == "KOREKSI_SEPI" and k(-0.03, 3.0) == "DISTRIBUSI"
    assert k(0.001, 3.0) == "NETRAL" and k(0.03, 1.0) == "NETRAL"


def test_pv_baca_last_days_and_crossing():
    close = np.full(60, 1000.0)
    vol = np.full(60, 1_000_000.0)
    close[-1], vol[-1] = 1050.0, 4_000_000.0              # hari terakhir: +5% volume 4x
    pv = hk.pv_baca(_series(close, vol))
    assert pv["harian"]["label"].iloc[-1] == "SEHAT"
    c2 = np.full(60, 1000.0); v2 = np.full(60, 1_000_000.0)
    v2[-3] = 5_000_000.0                                   # candle datar (spread 1%) + volume 5x
    pv2 = hk.pv_baca(_series(c2, v2))
    assert len(pv2["crossing"]) == 1
    assert hk.pv_baca(ohlc(np.full(20, 1000.0))) is None


def test_selling_climax_detected_and_low_held():
    close = np.r_[np.full(30, 1000.0), np.linspace(1000, 760, 19), [700.0], np.linspace(715, 740, 10)]
    vol = np.full(len(close), 1_000_000.0); vol[49] = 5_000_000.0
    df = ohlc(close); df["volume"] = vol
    pv = hk.pv_baca(df)
    cl = pv["climax"]
    assert cl and cl["rvol"] >= 3 and cl["low_bertahan"] is True
    close2 = close.copy(); close2[-3:] = [650.0, 640.0, 645.0]    # low climax ditembus
    df2 = ohlc(close2); df2["volume"] = vol
    assert hk.pv_baca(df2)["climax"]["low_bertahan"] is False


def _ob(bids, offers, bf=None, of=None):
    return dict(bids=bids, offers=offers, bids_freq=bf or {}, offers_freq=of or {}, total_bid=sum(bids.values()),
                total_offer=sum(offers.values()), n_levels=len(bids) + len(offers))


def test_baca_orderbook_lpa_cantik_dinding_tutup():
    ob = _ob({100: 5000, 99: 5000, 98: 5000, 97: 4000}, {101: 1000, 102: 1200, 103: 1100, 104: 90000},
             {100: 10, 99: 20, 98: 10, 97: 8}, {101: 100, 102: 120, 103: 110, 104: 5})
    b = hk.baca_orderbook(ob)
    assert b["bid"]["lpa"] == pytest.approx(19000 / 48) and b["freq_tersedia"] and b["likuid_3_papan"]
    assert len(b["bid"]["cantik"]) >= 3                    # 5000 berulang 3x = angka cantik
    assert b["offer"]["dinding"][0]["harga"] == 104.0 and b["offer"]["dinding"][0]["kali"] > 5
    assert hk.baca_orderbook(_ob({100: 10}, {101: 10}))["tutup"] and hk.baca_orderbook(None) is None


def _rk(harga=1000.0, sup=950.0, res=(1100.0, 1200.0)):
    x = np.arange(320)
    rk = tk.rekomendasi(ohlc(np.linspace(800, 1200, 320) + 60 * np.sin(x * 2 * np.pi / 30)), {"status": "AKUMULASI"})
    return dict(rk, harga=harga, sup=[{"harga": sup, "n": 3}], res=[{"harga": h, "n": 2} for h in res],
                entry_lo=sup, entry_hi=sup + 20, zona_valid=True, target=res[0], target_nama="TP1", atr=15.0)


def test_rencana_stop_one_tick_and_breakout():
    rn = hk.rencana(_rk(), 0.0015, 0.0025)
    assert rn["support"]["stop_tick"] == 945.0              # fraksi 5 di 950
    assert rn["breakout"]["entry"] == 1105.0 and rn["breakout"]["stop_tick"] == 1095.0
    assert rn["breakout"]["target"] == 1200.0 and rn["breakout"]["rr"] == pytest.approx(95 / 10)
    assert rn["impas"] == pytest.approx(1000 * 1.0015 / 0.9975) and rn["jarak_res"] == pytest.approx(0.10)
    assert hk.rencana(None, 0.0015, 0.0025) is None


class SID(ad.InvezgoClient):
    def __init__(self, vals, prices):
        super().__init__("x")
        self.vals, self.prices = vals, prices

    def get(self, path, params=None):
        return [{"code": "T", "date": f"2026-{m:02d}-28T00:00:00Z", "value": v, "price": str(p)}
                for m, (v, p) in enumerate(zip(self.vals, self.prices), 5)]


def test_sid_trend_and_pra_open():
    assert hk.sid_trend(SID([1000, 980, 960, 900], [500, 505, 495, 500]), "T")["status"] == "SID TURUN SAAT SIDEWAYS"
    assert hk.sid_trend(SID([1000, 1020, 1050, 1100], [500, 480, 450, 440]), "T")["status"] == "SID NAIK SAAT HARGA TURUN"
    assert hk.sid_trend(SID([1000, 1001], [5, 5]), "T")["status"] == "TIDAK DIUKUR"

    class P(ad.InvezgoClient):
        def __init__(self):
            super().__init__("x")

        def get(self, path, params=None):
            return {"iep": 1050, "iev": 1_000_000, "prev": 1000, "open": 0, "close": 0, "bid_price": 1040, "bid_lot": 10,
                    "offer_price": 1060, "offer_lot": 20}
    po = hk.pra_open(P(), "T")
    assert po["valid"] and po["gap"] == pytest.approx(0.05) and po["iev_lot"] == 10_000


def _r():
    r = ad.analyze(setup_case(), ad.Params(kode="TEST"))
    return dataclasses.replace(r, konsentrasi=dict(r.konsentrasi, status="AKUMULASI TERKONSENTRASI"))


def test_checklist_blocks_downtrend_and_passes_ideal():
    r = _r()
    x = np.arange(320)
    down = tk.rekomendasi(ohlc(np.linspace(2000, 1000, 320)), {"status": "DISTRIBUSI"})
    nilai = bd.penilaian(r, _aliran(), down)
    a = hk.analisa(None, r, down, nilai, min_nilai_m=0.0)
    cl = a["checklist"]
    assert not cl["lolos_inti"] and cl["hasil"].startswith("TAHAN") and a["profil"]["profil"] == "BUKAN SETUP TRADING"
    assert dict(zip([k["kriteria"][:6] for k in cl["kriteria"]], [k["status"] for k in cl["kriteria"]]))["Uptren"] == hk.TIDAK
    # uptrend + di area support + likuid -> aturan inti terpenuhi
    up = tk.rekomendasi(ohlc(np.linspace(800, 1200, 320) + 25 * np.sin(x * 2 * np.pi / 40)), {"status": "AKUMULASI"})
    up = dict(up, harga=up["entry_lo"] + 1, sup=[{"harga": up["entry_lo"], "n": 3}], res=[{"harga": up["tp1"], "n": 2}],
              zona_valid=True)
    up["ma"] = {**up["ma"], 50: up["harga"] * 0.95}
    s = up["ind"]["ma50"].copy(); s.iloc[-11:] = np.linspace(up["harga"] * 0.9, up["harga"] * 0.95, 11)
    up["ind"] = up["ind"].assign(ma50=s)
    a2 = hk.analisa(None, r, up, bd.penilaian(r, _aliran(), up), min_nilai_m=0.0)
    assert a2["checklist"]["lolos_inti"], hk.tabel(a2["checklist"])
    assert len(hk.tabel(a2["checklist"])) >= 7
