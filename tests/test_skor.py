import numpy as np
import pandas as pd
import pytest

import skor as sk
import laporan as lp

NAN = float("nan")


class R:
    verdict = "TIDAK ADA SINYAL"

    class params:
        kode = "TEST"


def rk_(**kw):
    ind = pd.DataFrame({"close": np.linspace(900, 1000, 30)})
    base = dict(harga=1000.0, tgl=None, ma={5: 990, 20: 960, 50: 930, 100: 900, 200: 850}, rsi=55.0, macd=2.0, signal=1.0, verdict="BELI DI AREA ENTRY",
                alasan="harga di area beli", entry_lo=980.0, entry_hi=1010.0, entry_maks=1010.0, zona_valid=True, sl=940.0, tp1=1100.0, tp2=1200.0,
                target=1100.0, target_nama="TP1", rr=2.0, rr_harga=2.0, n_bar=300, ind=ind, sup=[{"harga": 980}], res=[{"harga": 1100}])
    base.update(kw)
    return base


def f_(**kw):
    m = dict(der=0.8, der_prev=0.8, cr=1.8, cr_prev=1.8, ll=100.0, ll_prev=100.0, utang=50.0, utang_prev=50.0, g_rev_yoy=0.3, g_rev_qoq=0.05, g_np_yoy=0.4, g_np_qoq=0.1,
             gpm=0.4, npm=0.15, roe=0.14, roa=0.07, pbv=1.5, per=12.0, per_norm=12.0, rev_ttm=1e12, np_ttm=1.5e11, cfo_ttm=2e11, fcf_ttm=1e11, capex_ttm=1e11,
             ekuitas=1e12, aset=2e12, liab=1e12, rugi_kuartal=0, n_np_kuartal=4, rev_kuartal=[1, 2, 3, 4], np_kuartal=[1.0, 2.0, 3.0, 4.0])
    m.update(kw.pop("m", {}))
    d = dict(metrik=m, bank=False, sektor="Industri", subsektor="Manufaktur", ltm=dict(norm={"ok": False}, ni_norm=NAN), kualitas=dict(hilang=[]))
    d.update(kw)
    return d


def nilai_(kat="AKUMULASI TERINDIKASI", jebakan=()):
    return dict(kategori=kat, skor=3, skor_maks=4, jebakan=list(jebakan), kriteria=[dict(kriteria="k", status="ya", bukti="b")], info=[])


def ledger_(skor=8.0, conf=0.8, status="ACCUMULATING", zona=()):
    return dict(bandar_score=skor, confidence_score=conf, episode_status=status, supporting_evidence=["[OBS] a"], contradicting_evidence=["[OBS] b"],
                score_components=[dict(dimensi="Persistensi", poin=2, maks=2, detail="x")], confidence_reason=["ok"], data_completeness=dict(sessions_loaded=200),
                distribusi_ritel=[], zona_pasokan=list(zona))


def test_teknikal_skor_dan_batas():
    t = sk.skor_teknikal(rk_())
    assert t["skor"] >= 9 and t["andal"] and t["label"] == "KUAT"
    jauh = sk.skor_teknikal(rk_(harga=1200.0, rr_harga=0.4, verdict="TUNGGU KE AREA ENTRY", ma={5: 1190, 20: 1100, 50: 1000, 100: 900, 200: 850}))
    assert jauh["skor"] < t["skor"] and any("di atas" in x for x in jauh["lawan"])
    hind = sk.skor_teknikal(rk_(verdict="HINDARI"))
    assert hind["skor"] <= 3.0
    assert sk.skor_teknikal(None)["skor"] is None and sk.skor_teknikal(rk_(verdict="TIDAK TERUKUR", n_bar=30))["skor"] is None
    # kejar harga: naik >=4% sehari
    ind = pd.DataFrame({"close": [900.0] * 28 + [960.0, 1000.0]})
    kejar = sk.skor_teknikal(rk_(ind=ind))
    assert any("mengejar" in c["detail"] for c in kejar["komponen"])
    # target di zona pasokan
    zp = sk.skor_teknikal(rk_(), ledger_(zona=[dict(lo=1080.0, hi=1120.0, lot=500000.0, pct=0.4)]))
    assert zp["skor"] < t["skor"] and any("zona pasokan" in c["detail"] for c in zp["komponen"])


def test_fundamental_sehat_fatal_dan_rampup():
    sehat = sk.skor_fundamental(f_())
    assert sehat["skor"] >= 8 and sehat["andal"]
    fatal = sk.skor_fundamental(f_(m=dict(liab=3e12, aset=2e12)))
    assert fatal["skor"] <= 2.5 and "SOLVABILITAS FATAL" in fatal["flags"]
    # tambang ramp-up: rugi tetapi pendapatan melonjak dan margin kotor positif -> BUKAN fatal
    ramp = sk.skor_fundamental(f_(sektor="Basic Materials", subsektor="Gold Mining", m=dict(np_ttm=-5e10, npm=-0.2, g_rev_qoq=10.0, g_rev_yoy=NAN, gpm=0.46, per=-246.0, per_norm=NAN,
                                                                                      g_np_yoy=NAN, g_np_qoq=NAN, np_kuartal=[-1.0, -3.0, -2.0, -1.0], fcf_ttm=-4e11)))
    assert ramp["skor"] >= 4.5 and "ramp-up" in ramp["catatan"].lower()
    assert any("ramp-up" in c["detail"] for c in ramp["komponen"])
    rugi = sk.skor_fundamental(f_(m=dict(np_ttm=-5e10, npm=-0.2, g_rev_qoq=-0.1, g_rev_yoy=-0.2, gpm=0.1, per=-5.0, per_norm=NAN, g_np_qoq=-0.5, g_np_yoy=NAN)))
    assert rugi["skor"] < ramp["skor"]                                            # rugi biasa lebih buruk daripada rugi saat ramp-up
    # risiko neraca: current ratio < 0,5 + lonjakan liabilitas jangka pendek -> bendera
    risk = sk.skor_fundamental(f_(m=dict(cr=0.4, ll=550.0, ll_prev=110.0, der=1.7)))
    assert "RISIKO DILUSI / REFINANCING" in risk["flags"] and risk["skor"] < sehat["skor"]
    assert sk.skor_fundamental(None)["skor"] is None
    bank = sk.skor_fundamental(f_(bank=True))
    assert bank["skor"] is not None


def test_bandar_ledger_utama_dan_fallback_tidak_andal():
    led = sk.skor_bandar(nilai_(), ledger_(8.0, 0.8))
    assert led["skor"] >= 8 and led["andal"] and any("Ledger" in x for x in led["lensa"])
    rendah = sk.skor_bandar(nilai_(), ledger_(8.0, 0.4))
    assert not rendah["andal"]
    fb = sk.skor_bandar(nilai_("AKUMULASI TERKONFIRMASI (perilaku harga)"), None)
    assert fb["skor"] <= 7.0 and not fb["andal"] and "pegangan" in fb["catatan"]       # jendela pendek dibatasi dan TIDAK andal
    dist = sk.skor_bandar(nilai_("DISTRIBUSI TERINDIKASI", ["Bandar menjual 5.000 lot di pasar negosiasi"]), None)
    assert dist["skor"] <= 3.0 and "DISTRIBUSI/PENJUALAN BANDAR" in dist["flags"]
    konflik = sk.skor_bandar(nilai_("AKUMULASI KUAT"), ledger_(3.0, 0.8, "DISTRIBUTING"))
    assert any("ledger dipakai" in x for x in konflik["lawan"]) and konflik["skor"] < 5
    assert sk.skor_bandar(None, None)["skor"] is None


def test_vonis_bertingkat_dan_batas():
    r = R()
    ctx = dict(mode="lengkap", ledger=ledger_(8.0, 0.8), konteks=None)
    ok = sk.hitung(r, rk_(), f_(), None, nilai_(), None, ctx)
    assert ok["vonis"] == "BELI" and ok["rata"] >= 7 and ok["n_andal"] == 3
    # skor kuat tetapi harga belum di area beli -> TUNGGU (bukan HINDARI)
    jauh = sk.hitung(r, rk_(verdict="TUNGGU KE AREA ENTRY", harga=1040.0, rr_harga=1.4), f_(), None, nilai_(), None, ctx)
    assert jauh["vonis"] == "TUNGGU"
    # rata-rata menengah -> TUNGGU, tidak HINDARI
    sedang = sk.hitung(r, rk_(verdict="TUNGGU", harga=1200.0, rr_harga=0.4, ma={5: 1190, 20: 1100, 50: 1000, 100: 900, 200: 850}), f_(m=dict(np_ttm=-1e10, per=-3.0, per_norm=NAN, g_np_qoq=-1)),
                       None, nilai_("TIDAK KONKLUSIF"), None, dict(mode="lengkap", ledger=ledger_(5.0, 0.8, "MIXED_OR_UNCLEAR")))
    assert sedang["vonis"] == "TUNGGU" and 4 <= sedang["rata"] < 7
    # semua buruk -> HINDARI
    buruk = sk.hitung(r, rk_(verdict="HINDARI", harga=900.0), f_(m=dict(liab=3e12, aset=2e12)), None, nilai_("DISTRIBUSI TERINDIKASI"), None, dict(mode="lengkap", ledger=ledger_(2.0, 0.8, "MARKDOWN")))
    assert buruk["vonis"] == "HINDARI / JUAL" and buruk["rata"] < 4
    # mode cepat tidak boleh BELI; notasi khusus menahan; Hengky belum lolos menahan
    assert sk.hitung(r, rk_(), f_(), None, nilai_(), None, dict(mode="cepat", ledger=ledger_()))["vonis"] == "TUNGGU"
    notasi = dict(ihsg=dict(status="na"), notasi=dict(status="ok", ada=True, teks=["X"]))
    assert sk.hitung(r, rk_(), f_(), None, nilai_(), None, dict(mode="lengkap", ledger=ledger_(), konteks=notasi))["vonis"] == "TUNGGU"
    hk = dict(checklist=dict(lolos_inti=False, hasil="belum"))
    assert sk.hitung(r, rk_(), f_(), None, nilai_(), hk, ctx)["vonis"] == "TUNGGU"
    # bandar tidak andal -> tidak dipakai untuk vonis bila ada >= 2 dimensi andal lain
    tanpa = sk.hitung(r, rk_(), f_(), None, nilai_("DISTRIBUSI TERINDIKASI", ["Bandar menjual 1 lot di nego"]), None, dict(mode="lengkap"))
    assert tanpa["n_andal"] == 2 and tanpa["rata"] == tanpa["rata_andal"] and tanpa["rata_semua"] < tanpa["rata"]
    # data kurang (hanya 1 dimensi) -> TUNGGU
    assert sk.hitung(r, rk_(), None, None, None, None, dict(mode="lengkap"))["vonis"] == "TUNGGU"


def test_posisi_dan_laporan():
    r = R()
    ctx = dict(mode="lengkap", ledger=ledger_())
    sc = sk.hitung(r, rk_(), f_(), None, nilai_(), None, ctx)
    assert "TAHAN" in sc["posisi"]["pemegang"] and "cicil" in sc["posisi"]["calon_pembeli"].lower()
    patah = sk.hitung(r, rk_(harga=930.0, verdict="HINDARI"), f_(), None, nilai_(), None, ctx)
    assert "KURANGI/KELUAR" in patah["posisi"]["pemegang"] and patah["posisi"]["calon_pembeli"] == "Jangan masuk." or patah["vonis"] != "HINDARI / JUAL"
    L = "\n".join(sk.baris_laporan(sc))
    assert "## SKOR 3 DIMENSI" in L and "Komponen Teknikal" in L and "[DEC] Pemegang:" in L
    kep, al = lp.keputusan(r, rk_(), f_(), None, nilai_(), None, ctx)
    assert kep == sc["vonis"] and al[0].startswith("RINGKAS: ")
    import keputusan_ui as ku
    assert ku.alasan_ringkas(kep, al).startswith("B ") or "rata" in ku.alasan_ringkas(kep, al)
