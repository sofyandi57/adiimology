import datetime as dt

import numpy as np
import pandas as pd
import pytest

import adimology as ad
import bandar as bd
from fakes import _row
from test_teknikal import ohlc


class C(ad.InvezgoClient):
    def __init__(self, rows):
        super().__init__("x")
        self.rows, self.calls = rows, []

    def get(self, path, params=None):
        self.calls.append(params)
        return self.rows


TGL = [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(260)]


def test_volume_rata_lot_and_value():
    df = ohlc(np.full(210, 1000.0))                       # volume 1.000.000 lembar/hari = 10.000 lot
    v = bd.volume_rata(df)
    assert v[20]["lot_rata"] == pytest.approx(10_000) and v[200]["hari"] == 200
    assert v[20]["nilai_rata"] == pytest.approx(1000 * 1_000_000) and v["hari_terakhir"]["rvol"] == pytest.approx(1.0)


def test_bandar_nyangkut_vs_untung():
    rows = [_row("AA", 3_000_000, 1000.0), _row("BB", 1_000_000, 1100.0), _row("SS", -2_000_000, 1200.0), _row("XL", 9_000_000, 999.0)]
    for r in rows:
        r["net_value"] = str(int(r["net_volume"]) * 1000)
    c = C(rows)
    p = bd.bandar_panjang(c, "TEST", TGL, harga=900.0, excl=("XL",))
    assert len(c.calls) == 1 and p["hari"] == 200 and p["nyangkut"] == "NYANGKUT"
    assert p["modal_top3"] == pytest.approx(1025.0, rel=1e-3)        # tertimbang volume beli (rows pakai buy_volume = 2*|net|+1000)
    assert p["akumulator"][0]["code"] == "AA" and p["akumulator"][0]["floating"] < 0
    assert p["status"] == "AKUMULASI"
    untung = bd.bandar_panjang(C(rows), "TEST", TGL, harga=1300.0, excl=("XL",))
    assert untung["nyangkut"] == "TIDAK NYANGKUT"
    jual = bd.bandar_panjang(C([_row("SS", -2_000_000, 1200.0)]), "TEST", TGL, harga=900.0)
    assert jual["nyangkut"] == "TIDAK ADA AKUMULATOR BERSIH"
    with pytest.raises(ad.AdimologyError):
        bd.bandar_panjang(C(rows), "TEST", TGL[:10], harga=900.0)


def test_bandar_panjang_includes_ritel_flow():
    rows = [_row("XL", -3_000_000, 1000.0), _row("AA", 4_000_000, 1000.0), _row("YP", -1_000_000, 1001.0)]
    for r in rows:
        r["net_value"] = str(int(r["net_volume"]) * 1000)
    p = bd.bandar_panjang(C(rows), "TEST", TGL, harga=1000.0, excl=("XL", "YP"), ritel_codes=("XL", "YP"))
    assert p["ritel"]["status"] == "RITEL NET JUAL (dikeringkan)" and p["ritel"]["net_lot"] == pytest.approx(-40_000)
    assert bd.bandar_panjang(C(rows), "TEST", TGL, harga=1000.0, excl=("XL",))["ritel"] is None
