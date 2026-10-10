import dataclasses
import datetime as dt

import numpy as np
import pandas as pd
import pytest

import aliran as al
import hengky as hk
from fakes import _row


def _bs(rows):
    df = pd.DataFrame(rows)
    for c in ["net_volume", "buy_volume", "sell_volume", "buy_avg", "sell_avg"]:
        df[c] = pd.to_numeric(df[c])
    return df


def test_ritel_flow_net_sell_and_buy():
    jual = _bs([_row("XL", -2_000_000, 100.0), _row("PD", -1_000_000, 101.0), _row("AA", 3_000_000, 99.0), _row("ZZ", 1000, 99.0)])
    f = al.ritel_flow(jual, al.RITEL_ANALISA, harga=95.0)
    assert f["status"] == "RITEL NET JUAL (dikeringkan)" and f["net_lot"] == pytest.approx(-30_000)
    assert [x["code"] for x in f["per_broker"]] == ["XL", "PD"] and f["net_pct"] < -0.02
    beli = _bs([_row("XL", 2_000_000, 100.0), _row("AA", -2_000_000, 99.0)])
    fb = al.ritel_flow(beli, ("XL",), harga=110.0)
    assert fb["status"] == "RITEL NET BELI (akumulasi ritel)" and fb["floating"] == pytest.approx(0.10, rel=0.02)
    assert al.ritel_flow(_bs([_row("AA", 5, 1.0)]), ("XL",))["status"] == "TIDAK ADA TRANSAKSI RITEL"
    assert al.ritel_flow(None, ("XL",))["status"] == "TIDAK DIUKUR" and al.ritel_flow(jual, ())["status"] == "TIDAK DIUKUR"
    netral = _bs([_row("XL", 100, 100.0), _row("AA", 5_000_000, 99.0)])
    assert al.ritel_flow(netral, ("XL",))["status"] == "RITEL NETRAL"


class SID(hk.ad.InvezgoClient):
    def __init__(self, vals, prices):
        super().__init__("x")
        self.vals, self.prices = vals, prices

    def get(self, path, params=None):
        return [{"code": "T", "date": f"2026-{m:02d}-28T00:00:00Z", "value": v, "price": str(p)}
                for m, (v, p) in enumerate(zip(self.vals, self.prices), 4)]


def test_sid_arah_and_series():
    s = hk.sid_trend(SID([1000, 990, 970, 940, 900], [500, 505, 495, 500, 498]), "T")
    assert s["arah"] == "TURUN" and s["seri"][-1][1] == 900.0 and len(s["seri"]) == 5 and s["chg_1b"] < 0
    assert hk.sid_trend(SID([1000, 1001, 999, 1000], [5, 5, 5, 5]), "T")["arah"] == "STABIL"
    assert hk.sid_trend(SID([900, 950, 1000, 1100], [5, 5, 5, 5]), "T")["arah"] == "NAIK"


def _sid(arah):
    base = dict(status="SID TURUN SAAT SIDEWAYS", arah=arah, sid_awal=1000.0, sid_akhir=900.0, chg=-0.1, chg_1b=-0.03,
                dari=dt.date(2026, 6, 28), sampai=dt.date(2026, 9, 28), price_chg=0.0, seri=[])
    return base


def _fw(status):
    return dict(status=status, codes=["XL"], net_lot=-1000.0 if "JUAL" in status else 1000.0 if "BELI" in status else 0.0,
                net_pct=-0.05 if "JUAL" in status else 0.05 if "BELI" in status else 0.0)


def test_minat_ritel_combinations():
    m = al.minat_ritel(_sid("TURUN"), _fw("RITEL NET JUAL (dikeringkan)"), None, ret20=-0.03)
    assert m["status"] == "RITEL KELUAR / DIKERINGKAN" and m["minat"].startswith("MENURUN") and "tidak menarik" in m["ringkas"]
    assert "akumulasi pihak besar" in m["ringkas"]
    up = al.minat_ritel(_sid("TURUN"), _fw("RITEL NET JUAL (dikeringkan)"), None, ret20=0.25)
    assert "distribusi" in up["ringkas"]                                   # ritel menjual ke kenaikan
    assert al.minat_ritel(_sid("NAIK"), _fw("RITEL NET BELI (akumulasi ritel)"))["status"] == "RITEL MASUK / AKUMULASI RITEL"
    assert al.minat_ritel(_sid("TURUN"), _fw("RITEL NET BELI (akumulasi ritel)"))["status"].startswith("CAMPURAN")
    assert al.minat_ritel(None, _fw("RITEL NET JUAL (dikeringkan)"))["status"] == "RITEL DIKERINGKAN (tanpa data SID)"
    assert al.minat_ritel(None, None)["status"] == "TIDAK DIUKUR"
    assert al.minat_ritel({"status": "TIDAK DIUKUR"}, None)["minat"] == "TIDAK DIUKUR"
