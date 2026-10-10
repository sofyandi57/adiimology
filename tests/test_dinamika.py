import datetime as dt

import numpy as np
import pandas as pd
import pytest

import dinamika as dn

N = 60


def _data(price_path, flows, vol=None, name=None):
    """flows = {broker: harian net lembar}; cum = cumsum. price_path = close harian."""
    d0 = dt.date(2026, 7, 1)
    dates = [d0 + dt.timedelta(days=i) for i in range(len(price_path))]
    close = np.asarray(price_path, dtype=float)
    vol = np.full(len(close), 1_000_000.0) if vol is None else np.asarray(vol, dtype=float)
    price = pd.DataFrame({"date": dates, "open": close, "high": close * 1.005, "low": close * 0.995, "close": close, "volume": vol})
    cum = pd.DataFrame({k: np.cumsum(v) for k, v in flows.items()}, index=dates)
    return dict(price=price, cum=cum, names={k: k for k in flows})


def _summ(avg, codes=("AA",), net=3_000_000):
    rows = [dict(code=c, name=c, buy_avg=avg, buy_volume=net * 2, net_volume=net) for c in codes]
    return pd.DataFrame(rows)


def test_parse_inventori_cumulative_and_group_selection():
    raw = {"price": [{"code": "X", "date": f"2026-10-0{i}", "open": 10, "high": 11, "low": 9, "close": 10, "volume": 1000} for i in range(1, 6)],
           "broker": [{"broker": "AA", "name": "Alfa", "data": [{"date": f"2026-10-0{i}", "value": 100 * i} for i in range(1, 6)]},
                      {"broker": "XL", "data": [{"date": f"2026-10-0{i}", "value": 900 * i} for i in range(1, 6)]},
                      {"broker": "BB", "data": [{"date": f"2026-10-0{i}", "value": -50 * i} for i in range(1, 6)]}]}
    d = dn.parse_inventori(raw)
    assert list(d["cum"].columns) == ["AA", "XL", "BB"] and d["names"]["AA"] == "Alfa" and d["cum"]["AA"].iloc[-1] == 500
    acc, dist = dn.pilih_grup(d["cum"], excl=("XL",))
    assert acc == ["AA"] and dist == ["BB"]                       # XL (ritel) dikeluarkan
    net = dn._harian(d["cum"])
    assert list(net["AA"]) == [100, 100, 100, 100, 100]
    with pytest.raises(Exception):
        dn.parse_inventori({"price": [], "broker": []})


def test_confirmed_accumulation_with_price_response():
    rng = np.random.default_rng(1)
    price = 1000 * np.cumprod(1 + 0.0016 + rng.normal(0, 0.003, N))              # naik ~10% / 60 hari
    flows = {"AA": np.full(N, 30_000.0) + rng.normal(0, 3000, N), "BB": np.full(N, 10_000.0), "SS": np.full(N, -8_000.0), "XL": np.full(N, -50_000.0)}
    d = dn.analisa(_data(price, flows), harga=float(price[-1]), excl=("XL",), konsentrasi={"rasio": 2.0}, pv={"label_n": "SEHAT", "chg_n": 0.03, "rvol_n": 1.8},
                   summaries={20: _summ(float(price[-20])), 60: _summ(float(price[0]))})
    assert d["kategori"] == dn.K_TERKONFIRMASI and d["tingkat"] == "Konfirmasi perilaku harga" and d["keyakinan"] == "Sedang"
    assert [x["dimensi"] for x in d["dimensi"]] == list(dn.BOBOT) and sum(x["bobot"] for x in d["dimensi"]) == pytest.approx(1.0)
    assert d["grup"][0] == "AA" and "XL" not in d["grup"] and d["pembatalan"] and d["mendukung"]
    t = dict(dn.tujuh_pertanyaan(d, "pantau"))
    assert t["5 Probabilitas & payoff tervalidasi"].startswith("BELUM ADA") and len(dn.tujuh_pertanyaan(d, "x")) == 8


def test_flow_without_price_response_is_only_indicated():
    price = 1000 + np.sin(np.arange(N) / 6) * 3 - np.arange(N) * 0.2                # harga datar/turun tipis
    flows = {"AA": np.full(N, 30_000.0), "BB": np.full(N, 10_000.0), "SS": np.full(N, -8_000.0)}
    d = dn.analisa(_data(price, flows), harga=float(price[-1]), excl=(), konsentrasi={"rasio": 2.0}, pv=None,
                   summaries={20: _summ(1000.0), 60: _summ(1000.0)})
    assert d["kategori"] == dn.K_TERINDIKASI and d["tingkat"] == "Indikasi awal"
    assert any("unconfirmed flow" in x for x in d["menentang"])


def test_distribution_days_and_failed_breakout():
    rng = np.random.default_rng(3)
    price = np.r_[np.linspace(1000, 1150, 40), np.linspace(1150, 1040, 20)]
    vol = np.full(N, 1_000_000.0)
    for i in (46, 50, 54, 58):                                                      # hari jatuh -2,5% dengan volume 3x
        vol[i] = 3_000_000.0
        price[i:] = price[i:] * 0.975
    aa = np.r_[np.full(40, 40_000.0), np.full(20, -25_000.0)]                       # kelompok berbalik jual
    flows = {"AA": aa, "BB": np.full(N, 5_000.0), "SS": np.r_[np.full(40, -2000.0), np.full(20, -20_000.0)]}
    d = dn.analisa(_data(price, flows, vol), harga=float(price[-1]), excl=(), konsentrasi={"rasio": 1.2}, pv=None, summaries={20: _summ(1100.0), 60: _summ(1050.0)})
    assert d["kategori"] == dn.K_DISTRIBUSI and d["supply"]["n_hari_distribusi"] >= 2
    assert any("distribusi" in x for x in d["menentang"] + d["mendukung"])


def test_absorption_of_supply():
    price = np.full(N, 1000.0) + np.sin(np.arange(N) / 3) * 2                       # harga bertahan
    ss = np.zeros(N); ss[::3] = -90_000.0                                           # penjualan besar berkala
    flows = {"AA": np.full(N, 8_000.0), "SS": ss, "BB": np.full(N, 1_000.0)}
    d = dn.analisa(_data(price, flows), harga=float(price[-1]), excl=(), konsentrasi={"rasio": 1.0}, pv=None, summaries={20: _summ(1000.0), 60: _summ(1000.0)})
    assert d["supply"]["n_jual_besar"] >= 5 and d["supply"]["frac_absorpsi"] >= 0.5 and d["kategori"] == dn.K_ABSORPSI


def test_inconclusive_when_data_short_or_no_group():
    price = np.linspace(1000, 1050, 15)
    d = dn.analisa(_data(price, {"AA": np.full(15, 1000.0)}), harga=1050.0, excl=(), konsentrasi={"rasio": 1.0}, pv=None)
    assert d["kategori"] == dn.K_TIDAK and d["tingkat"] == "Tidak konklusif" and d["keyakinan"] == "Rendah"
    d2 = dn.analisa(_data(np.linspace(1000, 1050, N), {"XL": np.full(N, 9000.0)}), harga=1050.0, excl=("XL",), konsentrasi={"rasio": 1.0}, pv=None)
    assert d2["kategori"] == dn.K_TIDAK and not d2["dimensi"]                       # hanya broker ritel: tidak ada kelompok
    assert dn.K_TIDAK in dict(dn.tujuh_pertanyaan(d2, "tunggu"))["2 Hipotesis"]


def test_cost_estimate_range_and_pressed_position_rule():
    mo = dn.estimasi_modal({20: _summ(100.0), 60: _summ(104.0), 200: _summ(98.0)}, ["AA"], harga=90.0)
    assert mo["pusat"] == 100.0 and (mo["rendah"], mo["tinggi"]) == (98.0, 104.0) and mo["harga_vs_rentang"] == "di bawah rentang"
    assert not dn.estimasi_modal({20: _summ(100.0, net=-5)}, ["AA"], 90.0)["ada"]
    base = dict(modal=mo, tabel=pd.DataFrame([dict(jendela=20, ret_harga=-0.06, net_lot=500.0)]), konsistensi=dict(frac_pos=0.7), dimensi=[1])
    assert dn.posisi_tertekan(base)[0] == "INDIKASI POSISI TERTEKAN"
    assert dn.posisi_tertekan(dict(base, konsistensi=dict(frac_pos=0.3)))[0] == "TIDAK DAPAT DISIMPULKAN"       # waktu tak konsisten
    dekat = dict(base, modal=dn.estimasi_modal({20: _summ(100.0)}, ["AA"], harga=99.0))
    assert dn.posisi_tertekan(dekat)[0] == "TIDAK TERINDIKASI"
    assert dn.posisi_tertekan(dict(base, modal={"ada": False, "alasan": "x"}))[0] == "TIDAK DAPAT DISIMPULKAN"


def test_ambil_makes_four_calls_and_handles_optional_failures():
    class C:
        def __init__(self):
            self.calls = []

        def get(self, path, params=None):
            self.calls.append((path, dict(params or {})))
            if "inventory-chart" in path:
                if params.get("filter"):
                    raise RuntimeError("boom")                                       # inventory 200h opsional
                n = 60
                return {"price": [{"code": "X", "date": str(dt.date(2026, 7, 1) + dt.timedelta(days=i)), "open": 10, "high": 11, "low": 9, "close": 10,
                                   "volume": 1000} for i in range(n)],
                        "broker": [{"broker": "AA", "data": [{"date": str(dt.date(2026, 7, 1) + dt.timedelta(days=i)), "value": 100 * (i + 1)} for i in range(n)]}]}
            return [dict(code="AA", name="A", buy_volume="10", sell_volume="1", buy_avg="100", sell_avg="99", net_volume="9", net_value="1")]
    c = C()
    tgl = [dt.date(2026, 3, 1) + dt.timedelta(days=i) for i in range(210)]
    r = dn.ambil(c, "TEST", tgl, excl=())
    assert len(c.calls) == 4 and r["inv200"] is None and set(r["summaries"]) == {20, 60}
    with pytest.raises(Exception):
        dn.ambil(C(), "TEST", tgl[:10], excl=())
