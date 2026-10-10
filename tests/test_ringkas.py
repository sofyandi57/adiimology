import numpy as np
import pandas as pd
import pytest

import adimology as ad
import bandar as bd
import hengky as hk
import intraday as idt
import ringkas as rg
import teknikal as tk
from fakes import setup_case
from test_bandar import _aliran
from test_laporan import _inputs
from test_intraday import C, bars
from test_teknikal import ohlc


def test_statistik_harga_base_rates():
    x = np.arange(400)
    close = 1000 + 5 * x                                    # tren naik mulus: selalu naik, tidak pernah -5%
    df = tk.indikator(ohlc(close, spread=0.002))
    st = rg.statistik_harga(df, 0.05, -0.05)
    g = st["per_horizon"][10]["semua"]
    assert g["p_naik"] == 1.0 and g["p_m5"] == 0.0 and g["p_sl"] == 0.0
    g20 = st["per_horizon"][20]["semua"]
    assert 0.3 < g20["p_tp"] <= 1.0 and g20["p_sl"] == 0.0 and g20["p_tp"] >= g["p_tp"]   # +5% makin jauh saat harga naik
    assert st["per_horizon"][10]["mirip"]["n"] > 0           # kondisi kini (di atas MA50, RSI>70) ada padanannya
    turun = rg.statistik_harga(tk.indikator(ohlc(2000 - 3 * x, spread=0.002)), 0.05, -0.05)
    assert turun["per_horizon"][10]["semua"]["p_naik"] == 0.0 and turun["per_horizon"][20]["semua"]["p_sl"] > 0.2
    assert turun["per_horizon"][20]["semua"]["p_tp"] == 0.0
    assert rg.statistik_harga(ohlc(np.full(40, 100.0)), 0.05, -0.05) is None


def test_txt_is_compact_and_complete():
    r, rk, f = _inputs()
    ovn = idt.statistik(idt.get_bars(C(bars(61)), "TEST", None))
    hkr = hk.analisa(None, r, rk, bd.penilaian(r, _aliran(), rk), 0.0)
    b200 = dict(dari="2026-01-01", sampai="2026-10-09", hari=200, status="AKUMULASI", net_acc5_value=9e9, net_dist5_value=-4e9,
                modal_top3=300.0, net_top3_lot=5000.0, floating_top3=-0.12, nyangkut="NYANGKUT", keterangan="x",
                akumulator=[{"code": "AA", "broker": "X", "net_lot": 5000.0, "avg": 300.0, "floating": -0.12, "net_window_lot": 1.0}],
                distributor=[{"code": "BB", "broker": "Y", "net_lot": -3000.0, "avg": 320.0, "floating": -0.1, "net_window_lot": 1.0}],
                konsentrasi={"status": "TERSEBAR"})
    import aliran as al_mod
    import datetime as dt_
    sid = dict(status="SID TURUN SAAT SIDEWAYS", arah="TURUN", sid_awal=1000.0, sid_akhir=900.0, chg=-0.1, chg_1b=-0.03,
               dari=dt_.date(2026, 6, 28), sampai=dt_.date(2026, 9, 28), price_chg=0.0,
               seri=[(dt_.date(2026, 6, 28), 1000.0, 500.0), (dt_.date(2026, 9, 28), 900.0, 498.0)])
    fw = al_mod.ritel_flow(pd.DataFrame([{"code": "XL", "net_volume": -2e6, "buy_volume": 1e6, "sell_volume": 3e6, "buy_avg": 100.0, "sell_avg": 101.0},
                                         {"code": "AA", "net_volume": 2e6, "buy_volume": 3e6, "sell_volume": 1e6, "buy_avg": 99.0, "sell_avg": 100.0}]), ("XL",), 98.0)
    ritel = al_mod.minat_ritel(sid, fw, None, -0.02)
    out = rg.txt_bytes(r, rk, f, "2026-10-11 02:00 WIB", ovn, _aliran(), b200, hkr, None, bd.volume_rata(rk["ind"]), ritel).decode()
    for h in ("# TEST", "## KEPUTUSAN APP", "## HARGA", "## TEKNIKAL", "## RENCANA APP", "## STATISTIK HISTORIS HARGA",
              "## GAYA TRADING", "## BANDAR (window)", "## BANDAR 200H", "## VOLUME", "## BPJP/BSJP", "## FUNDAMENTAL", "## CATATAN SUMBER"):
        assert h in out, h
    assert "## MINAT RITEL" in out and "arah=TURUN" in out and "status=RITEL KELUAR / DIKERINGKAN" in out and "SID turun = minat ritel menurun" in out
    assert "nyangkut=NYANGKUT" in out and "P(naik)" in out and "area_beli=" in out and "UNTUK AI:" in out
    assert len(out.encode()) < 14_000, len(out.encode())
    assert out.count("\n") < 140
    assert all(ord(ch) < 0x2500 for ch in out)               # tanpa emoji/dekorasi


def test_txt_minimal_inputs():
    r, _, _ = _inputs()
    out = rg.txt_bytes(r, None, None, "x").decode()
    assert "## KEPUTUSAN APP" in out and "## STATISTIK" not in out
