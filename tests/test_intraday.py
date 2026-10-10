import datetime as dt

import numpy as np
import pandas as pd
import pytest

import adimology as ad
import intraday as idt

JAM = ["08:30", "09:00", "09:30", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30", "15:00", "15:30", "16:00"]


def bars(n_hari=60, pola=None):
    """Hari genap: volume sore 3x dan besok pagi naik 3%; hari ganjil: sore normal, besok pagi flat."""
    rows, harga, d = [], 1000.0, dt.date(2026, 3, 2)
    for i in range(n_hari):
        while d.weekday() >= 5:
            d += dt.timedelta(days=1)
        sore_tinggi = i % 2 == 0
        prev_tinggi = i > 0 and (i - 1) % 2 == 0
        for j in JAM:
            o = harga
            h = harga * (1.03 if prev_tinggi and j == "09:00" else 1.002)
            vol = 1_000_000 * (3 if sore_tinggi and j >= "15:00" else 1)
            rows.append({"date": f"{d}T{j}:00.000Z", "open": o, "high": h, "low": o * 0.998, "close": o, "volume": vol})
        d += dt.timedelta(days=1)
    return rows


class C(ad.InvezgoClient):
    def __init__(self, raw):
        super().__init__("x")
        self.raw = raw

    def get(self, path, params=None):
        assert "/chart/multi-time/" in path and params["timeframe"] == "30"
        return self.raw


def test_bars_parse_wib_and_sessions():
    b = idt.get_bars(C(bars(30)), "TEST", None)
    assert b["jam"].iloc[0] == "08:30" and b["jam"].iloc[-1] == "16:00"   # jam tetap WIB
    d = idt.per_hari(b)
    assert len(d) == 30 and d["vol_sore"].iloc[0] == 9_000_000 and d["vol_buka"].iloc[0] == 2_000_000


def test_high_afternoon_volume_predicts_next_morning():
    st = idt.statistik(idt.get_bars(C(bars(60)), "TEST", None), fee=0.004)
    tinggi = st["bsjp"]["Volume tinggi (≥1.5x)"]
    normal = st["bsjp"]["Volume normal"]
    assert tinggi["n"] > 10 and normal["n"] > 10
    assert tinggi["p_2"] == pytest.approx(1.0) and normal["p_1"] == pytest.approx(0.0)
    assert tinggi["jam_puncak"] == "09:00"
    assert st["ambang"]["sore"] > 0
    tab = idt.tabel(st["bsjp"], bsjp=True)
    assert "Peluang gap naik besok" in tab.index and list(tab.columns)[0] == "Semua hari"


def test_signal_today_uses_matching_group_and_tick_target():
    raw = bars(61)                         # hari terakhir (indeks 60, genap) = volume sore tinggi
    st = idt.statistik(idt.get_bars(C(raw), "TEST", None), fee=0.004)
    sg = st["sinyal"]["bsjp"]
    assert sg["kelompok"].startswith("Volume tinggi") and sg["layak"]
    assert sg["target"] % ad.fraksi(sg["target"]) == 0 and sg["target"] > sg["ref"]
    assert sg["jam_jual"] == "09:00"


def test_too_little_data():
    with pytest.raises(ad.AdimologyError):
        idt.statistik(idt.get_bars(C(bars(10)), "TEST", None))
