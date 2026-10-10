import datetime as dt

import numpy as np
import pandas as pd
import pytest

import teknikal as tk


def ohlc(close, spread=0.01):
    close = np.asarray(close, dtype=float)
    d0 = dt.date(2025, 1, 1)
    return pd.DataFrame({"tgl": [d0 + dt.timedelta(days=i) for i in range(len(close))],
                         "open": close, "high": close * (1 + spread), "low": close * (1 - spread),
                         "close": close, "volume": 1_000_000})


def test_rsi_extremes_and_macd_flat():
    up = pd.Series(np.arange(100, 160, dtype=float))
    assert tk.rsi(up).iloc[-1] == pytest.approx(100)
    assert tk.rsi(up[::-1].reset_index(drop=True)).iloc[-1] == pytest.approx(0)
    m = tk.macd(pd.Series([500.0] * 60))
    assert m.abs().max().max() == pytest.approx(0)


def test_ma_needs_enough_bars():
    ind = tk.indikator(ohlc(np.linspace(1000, 1200, 150)))
    assert pd.isna(ind["ma200"].iloc[-1]) and not pd.isna(ind["ma100"].iloc[-1])
    assert ind["ma5"].iloc[-1] == pytest.approx(ind["close"].tail(5).mean())


def test_support_resistance_from_oscillation():
    # harga berayun 900-1100, sekarang di tengah (1000)
    x = np.arange(300)
    close = 1000 + 100 * np.sin(x * 2 * np.pi / 40)
    df = ohlc(close, spread=0.0)
    df.loc[df.index[-1], ["close", "high", "low", "open"]] = 1000
    sup, res = tk.support_resistance(df, 1000)
    assert sup and res
    assert sup[0]["harga"] == pytest.approx(900, rel=0.02)
    assert res[0]["harga"] == pytest.approx(1100, rel=0.02)


def test_rekomendasi_levels_are_ordered():
    x = np.arange(320)
    close = np.linspace(800, 1200, 320) + 60 * np.sin(x * 2 * np.pi / 30)
    rk = tk.rekomendasi(ohlc(close), {"status": "AKUMULASI"})
    assert rk["sl"] < rk["entry_lo"] < rk["tp1"] <= rk["tp2"]
    if rk["zona_valid"]:
        assert rk["entry_lo"] <= rk["entry_hi"] <= rk["entry_maks"]
        assert rk["rr"] >= 1.5 - 1e-9
    assert rk["n_bar"] == 320 and rk["ma"][200] is not None
    assert rk["verdict"] in {"BELI DI AREA ENTRY", "TUNGGU KE AREA ENTRY", "TUNGGU", "HINDARI"}  # data cukup


def test_downtrend_with_distribution_is_avoided():
    close = np.linspace(2000, 1000, 300)
    rk = tk.rekomendasi(ohlc(close), {"status": "DISTRIBUSI"})
    assert rk["verdict"] == "HINDARI" and rk["skor"] < 0


def test_bandar_value():
    bs = pd.DataFrame({"code": list("ABCDEFG"), "net_value": [5e9, 3e9, 1e9, -2e9, -4e9, 0, 7e9]})
    bv = tk.bandar_value(bs, "A", excl=("G",))
    assert bv["bandar"] == 5e9 and bv["top5_beli"] == 9e9 and bv["top5_jual"] == -6e9
    assert bv["status"] == "AKUMULASI"
    assert tk.bandar_value(pd.DataFrame(), None)["status"] == "TIDAK TERUKUR"


def test_short_history_is_not_measured():
    rk = tk.rekomendasi(ohlc(np.linspace(100, 120, 20)), {"status": "AKUMULASI"})
    assert rk["verdict"] == "TIDAK TERUKUR"


def test_entry_maks_gives_min_rr():
    # tiap harga di area beli harus memberi R/R >= 1,5 ke target utama
    x = np.arange(320)
    for amp in (20, 60, 120):
        rk = tk.rekomendasi(ohlc(np.linspace(800, 1200, 320) + amp * np.sin(x * 2 * np.pi / 30)), {"status": "AKUMULASI"})
        if rk["zona_valid"]:
            e = rk["entry_hi"]
            assert (rk["target"] - e) / (e - rk["sl"]) >= 1.5 - 1e-9
        else:
            assert rk["verdict"] in {"TUNGGU", "HINDARI", "TIDAK TERUKUR"}


def test_stock_split_is_adjusted():
    # pola DSSA: ~67.000 lalu split 1:25 -> ~2.700
    close = np.r_[np.linspace(60000, 67000, 150), np.linspace(2700, 2900, 150)]
    df = ohlc(close)
    adj, ev = tk.sesuaikan_split(df)
    assert len(ev) == 1 and ev[0]["faktor"] == 25 and ev[0]["jenis"] == "stock split 1:25"
    assert adj["close"].iloc[149] == pytest.approx(67000 / 25)
    assert adj["close"].pct_change().abs().max() < 0.2          # tidak ada lompatan tersisa
    # gerak harian normal (<35%) tidak disentuh
    assert tk.sesuaikan_split(ohlc(np.linspace(100, 300, 200)))[1] == []


def test_dssa_real_split_maps_to_1_25():
    # data Invezgo asli DSSA: close 8 Apr 2026 = 67.000, 9 Apr 2026 open 2.600 close 3.120
    df = pd.DataFrame({"tgl": [dt.date(2026, 4, 8), dt.date(2026, 4, 9), dt.date(2026, 4, 10)],
                       "open": [66000.0, 2600, 3150], "high": [68000.0, 3220, 3300], "low": [64400.0, 2600, 3040],
                       "close": [67000.0, 3120, 3110], "volume": [2613500.0, 109066800, 72192100]})
    adj, ev = tk.sesuaikan_split(df)
    assert ev[0]["jenis"] == "stock split 1:25" and ev[0]["tgl"] == dt.date(2026, 4, 9)
    assert adj["close"].iloc[0] == pytest.approx(2680)
