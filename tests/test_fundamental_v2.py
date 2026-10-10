import numpy as np
import pandas as pd
import pytest

import fundamental as fd
from fakes import panel

M = 1e9


def panel_fy(rows):
    """{akun: {(tahun, 'FY'): nilai}} -> bentuk respons Invezgo type=FY."""
    return {"rows": [{"name": n, "level": 0, "values": [{"col": f"FY {y}", "year": y, "amount": v, "period": "FY"}
                                                         for (y, _), v in sorted(d.items(), reverse=True)]} for n, d in rows.items()]}


REV, NI, CFO = "Penjualan dan pendapatan usaha", "Laba (rugi) yang dapat diatribusikan ke entitas induk", \
    "Jumlah arus kas bersih yang diperoleh dari (digunakan untuk) aktivitas operasi"


def _meja():
    """Hanya Q2 (kumulatif 6 bulan) + FY 2025, seperti MEJA."""
    is_ = panel({REV: {(2024, 2): 22.7 * M, (2025, 2): 31.1 * M, (2026, 2): 6.4 * M},
                 NI: {(2024, 2): 2.4 * M, (2025, 2): 1.0 * M, (2026, 2): -8.9 * M}})
    cf = panel({CFO: {(2024, 2): -9.1 * M, (2025, 2): 12.8 * M, (2026, 2): -27.1 * M}})
    bs = panel({"Jumlah aset": {(2024, 2): 103 * M, (2025, 2): 107 * M, (2026, 2): 118 * M},
                "Jumlah liabilitas": {(2024, 2): 20 * M, (2025, 2): 22 * M, (2026, 2): 27 * M},
                "Jumlah ekuitas": {(2024, 2): 84 * M, (2025, 2): 85 * M, (2026, 2): 91 * M}})
    fy = {"IS": panel_fy({REV: {(2025, "FY"): 60 * M}, NI: {(2025, "FY"): 3.0 * M}}),
          "CF": panel_fy({CFO: {(2025, "FY"): 10 * M}})}
    return is_, bs, cf, fy


def test_semester_only_reporter_shows_cumulative_not_empty():
    is_, bs, cf, fy = _meja()
    f = fd.analisa({"name": "PT Meja", "sector": "Industri"}, is_, bs, cf, panel({"Saham": {(2026, 2): 1.9e9}}), 159.0, fy=fy)
    assert f["pnl"].loc["Pendapatan", "Q2 2026"] == "6*" and "Q2 2026" in f["kualitas"]["ytd_kolom"]      # bukan kosong lagi
    assert f["cf"].loc["Arus kas operasi (CFO)", "Q2 2026"].startswith("-27")
    assert any("KUMULATIF" in w for w in f["peringatan"])
    rev, basis = f["ltm"]["rev"]
    assert rev == pytest.approx(60 * M - 31.1 * M + 6.4 * M) and basis.startswith("LTM")                  # FY25 - H1'25 + H1'26
    assert f["pnl"].loc["Pendapatan YoY", "Q2 2026"].startswith("-79")                                    # sesama kumulatif
    assert f["ltm"]["ni"][0] == pytest.approx(3.0 * M - 1.0 * M - 8.9 * M)
    assert f["tesis"]["kesimpulan"] in {"TESIS LEMAH", "CAMPURAN: BUTUH RISET TAMBAHAN"} and f["tesis"]["melemahkan"]


def test_q4_derived_from_fy_and_ltm_is_four_quarters():
    ks = [(2025, 1), (2025, 2), (2025, 3), (2026, 1), (2026, 2)]
    ytd = {(2025, 1): 10 * M, (2025, 2): 21 * M, (2025, 3): 33 * M, (2026, 1): 12 * M, (2026, 2): 25 * M}
    is_ = panel({REV: ytd, NI: {k: v / 10 for k, v in ytd.items()}})
    fy = {"IS": panel_fy({REV: {(2025, "FY"): 46 * M}, NI: {(2025, "FY"): 4.6 * M}})}
    bs = panel({"Jumlah aset": {k: 100 * M for k in ks}, "Jumlah liabilitas": {k: 40 * M for k in ks}, "Jumlah ekuitas": {k: 60 * M for k in ks}})
    f = fd.analisa({}, is_, bs, {"rows": []}, panel({"Saham": {k: 1e9 for k in ks}}), 100.0, fy=fy)
    assert "IS" in f["kualitas"]["fy"] and "Q4 2025" in f["kolom"]
    assert f["pnl"].loc["Pendapatan", "Q4 2025"] == "13"                          # 46 - 33
    assert f["ltm"]["rev"][0] == pytest.approx(13 * M + 12 * M + 13 * M + 11 * M - 0) or f["ltm"]["rev"][1].startswith("LTM")
    assert f["ltm"]["rev"][0] == pytest.approx(46 * M - 21 * M + 25 * M)          # LTM = FY25 - H1'25 + H1'26 = 50


def test_one_off_quarter_is_normalised_and_flagged():
    ks = [(2025, 1), (2025, 2), (2025, 3), (2025, 4), (2026, 1), (2026, 2)]
    q = [100, 110, 105, 120, 4500, -1300]            # Q1'26 laba satu kali, Q2'26 rugi satu kali
    ytd, acc, yr = {}, 0, None
    for k, v in zip(ks, q):
        acc = v if k[0] != yr else acc + v
        yr = k[0]; ytd[k] = acc * M
    is_ = panel({REV: {k: (1000 + i * 10) * M * (1 if k[1] == 1 else k[1]) for i, k in enumerate(ks)}, NI: ytd})
    bs = panel({"Jumlah aset": {k: 20000 * M for k in ks}, "Jumlah liabilitas": {k: 5000 * M for k in ks}, "Jumlah ekuitas": {k: 15000 * M for k in ks}})
    f = fd.analisa({}, is_, bs, {"rows": []}, panel({"Saham": {k: 1e9 for k in ks}}), 500.0)
    n = f["ltm"]["norm"]
    assert n["ok"] and set(n["outliers"]) == {"Q1 2026", "Q2 2026"} and n["asli"] > 2 * n["nilai"]
    assert any("tidak andal" in w for w in f["peringatan"]) and f["vmark"]["PER"] == fd.KUNING
    val = dict(zip(f["val"]["Metrik"], f["val"]["Nilai"]))
    assert val["PER (laba normalisasi)"] != "-"
    assert f["tesis"]["kesimpulan"] == "TIDAK DAPAT DINILAI"                       # tanpa arus kas: tidak diisi asumsi diam-diam
    d = fd._kesimpulan_tesis(dict(hilang=[], ni=5e12, ni_norm=3e11, cfo=2e11, fcf=1e11, net_debt=-1e12, ekuitas=7e12,
                                  cash_conv=0.04, bank=False, g_rev=0.4, outliers=["Q1 2026"]))
    assert any("ekstrem" in x for x in d["melemahkan"]) and any("kas bersih" in x for x in d["mendukung"])


def test_growth_not_meaningful_and_like_for_like():
    m = fd.peta(pd.Series({(2025, 1): 1.0, (2026, 1): 30.0}), None)
    assert fd.tumbuh(m, (2026, 1), 4) == fd.NM                                     # +2900% -> n/m
    m2 = fd.peta(pd.Series({(2025, 2): 100.0}), pd.Series({(2026, 2): 150.0}))
    assert np.isnan(fd.tumbuh(m2, (2026, 2), 4))                                   # kuartal vs YTD tidak dibandingkan


def test_tesis_cannot_be_assessed_without_core_data():
    d = fd._kesimpulan_tesis(dict(hilang=["laba bersih"], ni=np.nan, ni_norm=np.nan, cfo=np.nan, fcf=np.nan, net_debt=np.nan,
                                  ekuitas=np.nan, cash_conv=np.nan, bank=False, g_rev=np.nan, outliers=[]))
    assert d["kesimpulan"] == "TIDAK DAPAT DINILAI" and "laba bersih" in d["alasan"] and d["belum_diketahui"]
