import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

import adimology as ad
from backtest import run_backtest as bt
from fakes import mcas, setup_case


def _row(code, net, avg, name="X", buy_volume=None):
    return dict(code=code, buy_freq="1", buy_volume=str(buy_volume if buy_volume is not None else abs(net) * 2 + 1000),
                buy_value="1", sell_freq="1", sell_volume="1", sell_value="1", buy_avg=str(avg), sell_avg="290",
                net_value="1", net_volume=str(net), net_freq="1", name=name)


class BacktestFake(ad.InvezgoClient):
    """Data sintetis multi-bulan (melintasi OB_START 2026-03-25 agar kedua regime ada).

    AAAA ~100 (fraksi 1), BBBB ~650 (fraksi 5). Akumulator: CC (terbesar) dan AK; XL ritel; YU penjual.
    Rasio offer/bid orderbook berputar 0,4..1,6 per hari. Semua request dicatat di self.calls.
    """

    def __init__(self, start="2025-11-03", end="2026-06-30"):
        super().__init__("x")
        import math
        self.calls, self.days, d = [], [], dt.date.fromisoformat(start)
        while d <= dt.date.fromisoformat(end):
            if d.weekday() < 5:
                self.days.append(d)
            d += dt.timedelta(days=1)
        self.pos = {x.isoformat(): k for k, x in enumerate(self.days)}
        self.bars = {}
        for kode, base, amp, per, fr in (("AAAA", 110, 25, 9, 1), ("BBBB", 650, 90, 7, 5)):
            rows, prev = {}, base
            for k, d in enumerate(self.days):
                c = round((base + amp * math.sin(k / per) + 0.05 * k) / fr) * fr
                o = prev
                rows[d.isoformat()] = (o, max(o, c) + 2 * fr, min(o, c) - 2 * fr, c, 1_000_000 + 10_000 * (k % 7))
                prev = c
            self.bars[kode] = rows

    def get(self, path, params=None):
        import math
        self.calls.append((path, dict(params or {})))
        bars = self.bars.get(path.rstrip("/").split("/")[-1])
        if bars is None:
            raise ad.AdimologyError("Data tidak ditemukan (fake).")
        if "/chart/stock/" in path:
            return [dict(date=d + "T00:00:00.000Z", open=o, high=h, low=l, close=c, volume=str(v))
                    for d, (o, h, l, c, v) in bars.items() if params["from"] <= d <= params["to"]]
        if "/summary/stock/" in path:
            f, t = params["from"], params["to"]
            sel = [b for d, b in bars.items() if f <= d <= t]
            vol, avg = sum(b[4] for b in sel), sum(b[3] for b in sel) / len(sel) * 0.99
            if f == t:                                   # harian: CC distribusi tiap hari ke-4
                net = -400_000 if self.pos[f] % 4 == 0 else 20_000
                return [_row("CC", net, avg, "MANDIRI", vol), _row("AK", 5_000, avg * 1.01, "UBS")]
            net = 300_000 if math.sin(self.pos[f] / 5) > -0.3 else -200_000
            return [_row("CC", net, avg, "MANDIRI", vol // 3), _row("AK", 100_000, avg * 1.02, "UBS", vol // 6),
                    _row("XL", 900_000, avg * 1.03, "STOCKBIT", vol // 3), _row("YU", -50_000, avg, "CGS")]
        if "/order-book/" in path:
            d = params.get("date") or max(bars)          # tanpa tanggal: snapshot "sekarang"
            c = bars[d][3]
            fr, offer = ad.fraksi(c), round(1500 * (0.4 + 0.3 * (self.pos[d] % 5)))
            return {"bid": [{f"bid{j}price": c - j * fr, f"bid{j}lot": 1500} for j in range(1, 6)],
                    "offer": [{f"offer{j}price": c + j * fr, f"offer{j}lot": offer} for j in range(1, 6)]}
        raise AssertionError(path)

ARGS = ["--tickers", "AAAA,BBBB", "--start", "2026-01-05", "--end", "2026-06-30",
        "--windows", "5,10", "--modes", "top1,top3", "--horizons", "5,10", "--n-boot", "200"]


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
    assert "A. Backtest target Adimology" in out and "Adimology vs baseline" in out
    sig = pd.read_csv(next((tmp_path / "res").glob("sinyal_*.csv")))
    summ = pd.read_csv(next((tmp_path / "res").glob("ringkasan_*.csv")))
    comp = pd.read_csv(next((tmp_path / "res").glob("perbandingan_*.csv")))
    assert set(sig["strategi"]) == {"adimology", "akumulasi", "target_tetap", "acak"}
    assert {"proxy", "orderbook"} <= set(sig["regime"])
    assert set(sig.loc[sig["strategi"] == "adimology", "lpp_sumber"]) >= {"proxy", "orderbook"}
    assert set(summ["window"]) == {5, 10} and set(summ["H"]) == {5, 10}
    assert set(summ["mode_bandar"]) == {"top1", "top3"}
    assert set(comp["baseline"]) == {"akumulasi", "target_tetap", "acak"}
    a = sig[sig["strategi"] == "adimology"]
    assert set(a.loc[a["mode_bandar"] == "top1", "bandar"]) == {"CC", "AK"}    # XL (ritel) tidak pernah terpilih
    assert set(a.loc[a["mode_bandar"] == "top3", "bandar"]) <= {"CC+AK", "AK"}
    assert "B. Studi imbalance" in out
    isumm = pd.read_csv(next((tmp_path / "res").glob("imbalance_ringkasan_*.csv")))
    assert "semua hari (baseline)" in set(isumm["bucket"]) and set(isumm["H"]) == {1, 5}

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


def test_top3_matches_ad_analyze(tmp_path):
    fake, t = BacktestFake(), dt.date(2026, 2, 13)
    r = ad.analyze(fake, ad.Params(kode="AAAA", tanggal_akhir=t, mode_bandar="top3", lot_per_papan_override=2000,
                                   broker_dikecualikan=bt.RETAIL_DEFAULT))
    chart = bt.load_chart(fake, "AAAA", _cfg(tmp_path))
    i = int(chart.index[chart["tgl"] == t][0])
    s = bt.signal_at(fake, "AAAA", chart, i, 5, _cfg(tmp_path), "top3")
    assert s["bandar"] == r.bandar["code"] == "CC+AK"
    assert s["avg"] == pytest.approx(r.bandar["avg"]) and s["barang_lot"] == pytest.approx(r.bandar["net_lot"])
    assert s["net_terakhir_lot"] == pytest.approx(r.net_terakhir) and s["distribusi"] == r.distribusi


def test_imbalance_rows_and_summary(tmp_path):
    fake, cfg = BacktestFake(), _cfg(tmp_path)
    chart = bt.load_chart(fake, "AAAA", cfg)
    rows = pd.DataFrame(bt.imbalance_rows(fake, "AAAA", chart, cfg))
    assert rows["t"].min() >= bt.OB_START
    assert all(p.get("time") == "14:30" for path, p in fake.calls if "order-book" in path)
    r0 = rows.iloc[0]
    k = fake.pos[r0["t"].isoformat()]
    assert r0["rasio_offer_bid"] == pytest.approx(round(1500 * (0.4 + 0.3 * (k % 5))) / 1500)
    i = int(chart.index[chart["tgl"] == r0["t"]][0])
    assert r0["ret_1"] == pytest.approx(chart["close"].iloc[i + 1] / chart["close"].iloc[i] - 1)
    assert rows["ret_5"].iloc[-1] != rows["ret_5"].iloc[-1]          # NaN di ujung data
    summ, comp = bt.summarize_imbalance(rows, cfg)
    assert set(summ["bucket"]) == {"semua hari (baseline)"} | {lbl for *_, lbl in bt.IMB_BUCKETS}
    base = summ[(summ["H"] == 1) & (summ["bucket"] == "semua hari (baseline)")].iloc[0]
    assert base["n"] == rows["ret_1"].notna().sum()
    assert {"offer >= bid"} <= set(comp["kelompok"]) and set(comp["metrik"]) == {"rata_return", "pct_naik"}


def test_imbalance_only_study_uses_chart_and_orderbook_only(tmp_path):
    fake = BacktestFake()
    assert bt.main(_args(tmp_path, "--study", "imbalance", "--yes"), client=fake) == 0
    paths = {p.split("/")[2] for p, _ in fake.calls}
    assert paths == {"chart", "order-book"}
    est = bt.estimate_calls(bt.parse_args(_args(tmp_path, "--study", "imbalance"))[0])
    assert est["window_summary"] == 0 and len(fake.calls) <= est["total"]
    assert not list((tmp_path / "res").glob("sinyal_*.csv")) and list((tmp_path / "res").glob("imbalance_*.csv"))
