import numpy as np
import pytest

import fundamental as fd
from fakes import panel


T = 1e12
# angka ASII asli dari Invezgo (YTD, Rp): Q1 2025, Q2 2025, Q3 2025, Q1 2026, Q2 2026
ASII_REV = {(2025, 1): 83.361 * T, (2025, 2): 162.857 * T, (2025, 3): 243.608 * T,
            (2026, 1): 78.668 * T, (2026, 2): 157.913 * T}


def test_ytd_to_quarter_matches_invezgo():
    q = fd.per_kuartal(fd.parse_panel(panel({"Penjualan dan pendapatan usaha": ASII_REV})).iloc[0])
    assert q[(2026, 2)] == pytest.approx(79.245 * T)          # = keystat "Revenue (Quarter)" Invezgo
    assert q[(2026, 1)] == pytest.approx(78.668 * T)
    assert q[(2025, 3)] == pytest.approx(80.751 * T)


def test_ttm_needs_four_consecutive_quarters():
    ytd = fd.parse_panel(panel({"x": ASII_REV})).iloc[0]
    v, basis = fd.ttm(fd.per_kuartal(ytd), ytd)                # Q4 2025 tidak ada -> disetahunkan
    assert basis == "YTD disetahunkan" and v == pytest.approx(157.913 * T * 2)
    full = {(2025, 1): 10, (2025, 2): 21, (2025, 3): 33, (2025, 4): 46, (2026, 1): 12}
    ytd2 = fd.parse_panel(panel({"x": full})).iloc[0]
    v2, basis2 = fd.ttm(fd.per_kuartal(ytd2), ytd2)
    assert basis2 == "TTM" and v2 == pytest.approx(11 + 12 + 13 + 12)


def _company(liab_scale=0.5, rev_growth=0.1):
    ks = [(2025, q) for q in (1, 2, 3, 4)] + [(2026, q) for q in (1, 2)]
    rev_q = {k: 100e9 * (1 + rev_growth) ** i for i, k in enumerate(ks)}
    def ytd(d):
        out, acc, yr = {}, 0, None
        for k in ks:
            acc = d[k] if k[0] != yr else acc + d[k]
            yr = k[0]
            out[k] = acc
        return out
    is_ = panel({"Penjualan dan pendapatan usaha": ytd(rev_q), "Jumlah laba bruto": ytd({k: v * .3 for k, v in rev_q.items()}),
                 "Laba (rugi) yang dapat diatribusikan ke entitas induk": ytd({k: v * .1 for k, v in rev_q.items()})})
    aset = {k: 1000e9 for k in ks}
    liab = {k: 1000e9 * liab_scale for k in ks}
    bs = panel({"Jumlah aset": aset, "Jumlah liabilitas": liab,
                "Jumlah ekuitas": {k: aset[k] - liab[k] for k in ks},
                "Jumlah aset lancar": {k: 400e9 for k in ks}, "Jumlah liabilitas jangka pendek": {k: 300e9 for k in ks}})
    cf = panel({"Jumlah arus kas bersih yang diperoleh dari (digunakan untuk) aktivitas operasi": ytd({k: 15e9 for k in ks}),
                "Pembayaran untuk perolehan aset tetap": ytd({k: 5e9 for k in ks})})
    ks_raw = panel({"Saham": {k: 1e9 for k in ks}})
    return dict(info={"name": "PT Contoh", "sector": "Industri"}, is_raw=is_, bs_raw=bs, cf_raw=cf, keystat_raw=ks_raw)


def test_growth_marks_green_and_valuation():
    f = fd.analisa(harga=500, **_company())
    assert f["periode"] == "Q2 2026" and f["kolom"][-1] == "Q2 2026"
    assert f["mark"][("pnl", "Pendapatan", "Q2 2026")] == fd.HIJAU          # naik YoY
    assert f["mark"][("pnl", "Pendapatan QoQ", "Q2 2026")] == fd.HIJAU
    assert any("naik" in s and "YoY" in s for s in f["sorotan"])
    val = dict(zip(f["val"]["Metrik"], f["val"]["Nilai"]))
    assert val["Kapitalisasi pasar"] == "Rp500.0 M"
    assert not any("LEBIH BESAR" in p for p in f["peringatan"])


def test_liabilities_above_assets_is_yellow_warning():
    f = fd.analisa(harga=500, **_company(liab_scale=1.3))
    assert f["mark"][("bs", "Total liabilitas", "Q2 2026")] == fd.KUNING
    assert f["mark"][("bs", "Total ekuitas", "Q2 2026")] == fd.KUNING
    assert any("LEBIH BESAR dari total aset" in p for p in f["peringatan"])


def test_bank_uses_interest_income_and_skips_current_ratio():
    c = _company()
    c["info"] = {"name": "Bank X", "sector": "Keuangan", "subsector": "Bank"}
    c["is_raw"] = panel({"Pendapatan bunga": {(2026, 1): 50e9, (2026, 2): 101e9},
                         "Laba (rugi) yang dapat diatribusikan ke entitas induk": {(2026, 1): 20e9, (2026, 2): 41e9}})
    f = fd.analisa(harga=1000, **c)
    assert f["bank"] and "Pendapatan bunga" in f["pnl"].index
    assert "Current ratio" not in f["bs"].index
    assert f["pnl"].loc["Pendapatan bunga", "Q2 2026"] == "51"


def test_missing_shares_warns():
    c = _company()
    c["keystat_raw"] = {"rows": []}
    f = fd.analisa(harga=500, **c)
    assert any("Jumlah saham tidak tersedia" in p for p in f["peringatan"])
    assert np.isnan(fd._shares({"rows": []}))
