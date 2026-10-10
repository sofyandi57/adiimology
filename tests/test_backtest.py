import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

import adimology as ad
from backtest import run_backtest as bt
from fakes import BacktestFake, mcas, setup_case

ARGS = ["--tickers", "AAAA,BBBB", "--start", "2026-01-05", "--end", "2026-06-30",
        "--windows", "5,10", "--horizons", "5,10", "--n-boot", "200"]


def _cfg(tmp_path, **kw):
    return bt.Config(tickers=["AAAA", "BBBB"], start=dt.date(2026, 1, 5), end=dt.date(2026, 6, 30),
                     windows=(5,), horizons=(5, 10), n_boot=200, cache_dir=tmp_path / "cache",
                     results_dir=tmp_path / "res", **kw)


def _args(tmp_path, *extra):
    return ARGS + ["--cache-dir", str(tmp_path / "cache"), "--results-dir", str(tmp_path / "res"), *extra]


@pytest.mark.parametrize("make", [mcas, setup_case])
def test_signal_matches_ad_analyze(make, tmp_path):
    """Level & verdict backtest identik dengan ad.analyze (rumus tidak diubah)."""
    client = make()
    r = ad.analyze(client, ad.Params(kode="MCAS", window_hari=5))
    chart = bt.load_chart(client, "MCAS", _cfg(tmp_path, retail=("XL",)))
    s = bt.signal_at(client, "MCAS", chart, len(chart) - 1, 5, _cfg(tmp_path, retail=("XL",)))
    assert s["bandar"] == r.bandar["code"] and s["avg"] == pytest.approx(r.bandar["avg"])
    assert s["lpp_sumber"] in ("orderbook", "proxy")
    if s["lpp_sumber"] == "orderbook":
        assert s["lpp"] == pytest.approx(r.lot_per_papan)
        assert (s["t_real_r"], s["t_max_r"]) == (r.calc["t_real_r"], r.calc["t_max_r"])
    assert s["setup_ok"] == r.setup_ok and s["distribusi"] == r.distribusi and s["sl"] == r.plan["sl"]


def test_no_look_ahead(tmp_path):
    """Semua request broker/orderbook untuk tanggal t tidak melewati t, dan hasil sama dengan chart terpotong."""
    fake, cfg = BacktestFake(), _cfg(tmp_path)
    chart = bt.load_chart(fake, "AAAA", cfg)
    for i in bt.eval_indices(chart, cfg.start, cfg.end)[:12]:
        t = chart["tgl"].iloc[i]
        fake.calls.clear()
        full = bt.signal_at(fake, "AAAA", chart, i, 5, cfg)
        for path, p in fake.calls:
            assert (p.get("to") or p.get("date")) <= t.isoformat(), (path, p)
        cut = bt.signal_at(fake, "AAAA", chart.iloc[: i + 1], i, 5, cfg)
        assert {k: v for k, v in full.items() if k != "lpp_orderbook"} == \
               {k: v for k, v in cut.items() if k != "lpp_orderbook"}


def test_eval_dates_are_week_last_trading_day(tmp_path):
    fake, cfg = BacktestFake(), _cfg(tmp_path)
    chart = bt.load_chart(fake, "AAAA", cfg)
    days = [chart["tgl"].iloc[i] for i in bt.eval_indices(chart, cfg.start, cfg.end)]
    assert days and all(d.weekday() == 4 for d in days)


def _bars(rows):
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"])


def test_simulate_rules():
    f = dict(fee_beli=0.0015, fee_jual=0.0025)
    tp = bt.simulate(_bars([(100, 103, 99, 102), (102, 111, 101, 108)]), 110, 95, **f)
    assert tp["hasil"] == "tp" and tp["exit"] == 110 and tp["hari_pegang"] == 2
    assert tp["net_ret"] == pytest.approx(110 * 0.9975 / (100 * 1.0015) - 1)
    both = bt.simulate(_bars([(100, 112, 94, 100)]), 110, 95, **f)          # TP & SL sehari: SL dulu
    assert both["hasil"] == "sl" and both["exit"] == 95
    gap = bt.simulate(_bars([(100, 101, 99, 100), (90, 92, 88, 91)]), 110, 95, **f)
    assert gap["hasil"] == "sl" and gap["exit"] == 90
    t = bt.simulate(_bars([(100, 104, 97, 101), (101, 105, 98, 103)]), 110, 95, **f)
    assert t["hasil"] == "waktu" and t["exit"] == 103
    assert t["mfe"] == pytest.approx(0.05) and t["mae"] == pytest.approx(-0.03)


def test_lpp_proxy():
    hist = pd.DataFrame({"low": [100, 100, 100], "high": [110, 104, 100]})     # papan 10, 4, 0->1
    assert bt.lpp_proxy(hist, total_lot=3000) == pytest.approx(1000 / 4)


def test_estimate_without_yes_makes_no_calls(tmp_path, capsys):
    fake = BacktestFake()
    assert bt.main(_args(tmp_path), client=fake) == 2
    assert fake.calls == [] and not (tmp_path / "res").exists()
    assert "Estimasi panggilan API" in capsys.readouterr().out


def test_end_to_end_and_cache(tmp_path, capsys):
    fake = BacktestFake()
    assert bt.main(_args(tmp_path, "--yes"), client=fake) == 0
    out = capsys.readouterr().out
    assert "Ringkasan" in out and "Adimology vs baseline" in out
    sig = pd.read_csv(next((tmp_path / "res").glob("sinyal_*.csv")))
    summ = pd.read_csv(next((tmp_path / "res").glob("ringkasan_*.csv")))
    comp = pd.read_csv(next((tmp_path / "res").glob("perbandingan_*.csv")))
    assert set(sig["strategi"]) == {"adimology", "akumulasi", "target_tetap", "acak"}
    assert {"proxy", "orderbook"} <= set(sig["regime"])
    assert set(sig.loc[sig["strategi"] == "adimology", "lpp_sumber"]) >= {"proxy", "orderbook"}
    assert set(summ["window"]) == {5, 10} and set(summ["H"]) == {5, 10}
    assert set(comp["baseline"]) == {"akumulasi", "target_tetap", "acak"}
    assert sig.loc[sig["strategi"] == "adimology", "bandar"].eq("CC").all()     # XL (ritel) dikecualikan

    n_live = len(fake.calls)
    est = bt.estimate_calls(bt.parse_args(_args(tmp_path))[0])
    assert 0 < n_live <= est["total"]                    # estimasi = batas atas

    fake2 = BacktestFake()                               # rerun: semuanya dari cache
    assert bt.main(_args(tmp_path, "--yes"), client=fake2) == 0
    assert fake2.calls == []
    assert bt.main(_args(tmp_path, "--offline")) == 0    # offline tanpa key


def test_max_calls_stops_and_resumes(tmp_path):
    fake = BacktestFake()
    assert bt.main(_args(tmp_path, "--yes", "--max-calls", "5"), client=fake) == 4
    assert len(fake.calls) == 5
    assert bt.main(_args(tmp_path, "--yes"), client=BacktestFake()) == 0


def test_cache_404_and_422(tmp_path):
    class C(ad.InvezgoClient):
        n = 0

        def get(self, path, params=None):
            C.n += 1
            raise ad.ParamDitolak("422") if "x" in path else ad.AdimologyError("Data tidak ditemukan (y)")

    cc = bt.CachingClient(C("k"), tmp_path)
    for path, exc in (("/a/x", ad.ParamDitolak), ("/a/y", ad.AdimologyError)):
        for _ in range(2):
            with pytest.raises(exc):
                cc.get(path, {"q": 1})
    assert C.n == 2


def test_api_key_only_in_header(tmp_path, monkeypatch, capsys):
    """Jalur produksi (requests.get asli diganti): key hanya di header, tidak di cache, hasil, atau log."""
    import requests
    secret = "SECRET-KEY-123"
    fake, seen = BacktestFake(), []

    class Resp:
        status_code = 200

        def __init__(self, data):
            self._d = data

        def json(self):
            return self._d

        def raise_for_status(self):
            pass

    def fake_get(url, headers=None, params=None, timeout=None):
        seen.append(headers)
        return Resp(fake.get(url.replace(ad.BASE_URL, ""), params))

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setenv("INVEZGO_API_KEY", secret)
    args = ["--tickers", "AAAA", "--start", "2026-03-02", "--end", "2026-05-29", "--n-boot", "50",
            "--horizons", "5", "--cache-dir", str(tmp_path / "cache"), "--results-dir", str(tmp_path / "res"), "--yes"]
    assert bt.main(args) == 0
    assert seen and all(h == {"Authorization": f"Bearer {secret}"} for h in seen)
    assert secret not in capsys.readouterr().out
    for f in tmp_path.rglob("*"):
        if f.is_file():
            assert secret not in f.read_text()


def test_missing_key(tmp_path, monkeypatch):
    monkeypatch.delenv("INVEZGO_API_KEY", raising=False)
    assert bt.main(_args(tmp_path, "--yes")) == 3


def test_bootstrap_identical_groups_contains_zero():
    rng = np.random.default_rng(0)
    g = pd.DataFrame({"t": np.repeat(np.arange(30), 4), "v": rng.normal(0, 0.05, 120)})
    b = bt._boot_mean([g, g.copy()], "v", 500, 1)
    assert np.allclose(b[:, 0], b[:, 1])
    lo, hi = bt._ci(bt._boot_mean([g], "v", 500, 1)[:, 0])
    assert lo < g["v"].mean() < hi
    assert bt._verdict(0.01, 0.02) == "Adimology LEBIH BAIK" and bt._verdict(-0.01, 0.02) == "tidak berbeda nyata"


def test_split_signals_excluded_from_summary(tmp_path):
    fake = BacktestFake()
    bars = fake.bars["AAAA"]
    keys = sorted(bars)
    for d in keys[: keys.index("2026-02-02")]:        # harga lama 5x (setelah itu "split 1:5")
        o, h, l, c, v = bars[d]
        bars[d] = (o * 5, h * 5, l * 5, c * 5, v)
    cfg = _cfg(tmp_path)
    chart = bt.load_chart(fake, "AAAA", cfg)
    assert dt.date(2026, 2, 2) in bt.split_dates(chart)
    df = bt.run(bt.CachingClient(fake, tmp_path / "c"), cfg, log=lambda *_: None)
    flagged = df[df["split_dalam_rentang"].astype(bool)]
    assert len(flagged) and (flagged["t"] <= dt.date(2026, 2, 2)).all()
