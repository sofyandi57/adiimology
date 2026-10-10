import datetime as dt
import pytest
import adimology as ad
from fakes import mcas, setup_case


def test_fraksi_tiers():
    assert [ad.fraksi(p) for p in (50, 199, 200, 499, 500, 1999, 2000, 4999, 5000)] == [1, 1, 2, 2, 5, 5, 10, 10, 25]


def test_count_papan_soci_cross_tier():
    assert ad.count_papan(446, 640) == pytest.approx(55)      # 27 papan fraksi 2 + 28 papan fraksi 5
    assert ad.count_papan(264, 266) == pytest.approx(1)
    assert ad.count_papan(300, 300) == 0


def test_step_up_crosses_tier():
    assert ad.step_up(498, 1) == pytest.approx(500)           # 1 tick fraksi 2
    assert ad.step_up(498, 2) == pytest.approx(505)           # +1 tick fraksi 5
    assert ad.step_up(526.05, 2.701 / 2) == pytest.approx(526.05 + 1.3505 * 5)


def test_gzco_slide_numbers():
    base = 380 * 1.05                                          # 399
    p = 820000 / 35000                                         # 23.4 papan
    assert ad.step_up(base, p) == pytest.approx(445.86, abs=0.01)    # slide: 446
    assert ad.step_up(base, p / 2) == pytest.approx(422.43, abs=0.01)  # slide: 424


def test_mcas_matches_notebook_run():
    r = ad.analyze(mcas(), ad.Params(kode="mcas"))
    assert r.bandar["code"] == "CC"                            # XL dikecualikan
    assert r.bandar["avg"] == pytest.approx(274.0888)
    assert r.lot_per_papan == pytest.approx(3406)
    assert r.calc["t_real_r"] == 292 and r.calc["t_max_r"] == 296   # sama dengan output Colab Anda
    assert r.distribusi and r.verdict == "TIDAK ADA SINYAL" and not r.setup_ok
    assert r.plan["sl"] == 220


def test_setup_case_ok():
    r = ad.analyze(setup_case(), ad.Params(kode="TEST", window_hari=5))
    assert r.verdict == "SETUP TERPENUHI" and r.setup_ok
    assert r.calc["t_real_r"] > r.harga
    assert r.plan["lot"] > 0 and r.plan["laba_target"] > 0 > r.plan["rugi_sl"]


def test_manual_lot_per_papan_and_window():
    r = ad.analyze(mcas(), ad.Params(kode="MCAS", lot_per_papan_override=5000))
    assert r.lpp_sumber == "manual" and r.calc["t_max_r"] == 294


def test_empty_orderbook_requires_override():
    c = mcas(); c.ob = {"code": "MCAS", "bid": [], "offer": []}
    with pytest.raises(ad.AdimologyError):
        ad.analyze(c, ad.Params(kode="MCAS"))
    assert ad.analyze(c, ad.Params(kode="MCAS", lot_per_papan_override=3000)).lpp_sumber == "manual"


def test_bad_code_and_unknown_broker():
    with pytest.raises(ad.AdimologyError):
        ad.analyze(mcas(), ad.Params(kode="TOOLONG"))
    with pytest.raises(ad.AdimologyError):
        ad.analyze(mcas(), ad.Params(kode="MCAS", broker_manual="ZZ"))
