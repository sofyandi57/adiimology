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
    assert rk["sl"] < rk["entry_lo"] <= rk["entry_hi"] < rk["tp1"] <= rk["tp2"]
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
