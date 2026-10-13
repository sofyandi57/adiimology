import numpy as np
import pandas as pd

import laporan as lp
import skor as sk

NAN = float("nan")


class R:
    verdict = "TIDAK ADA SINYAL"

    class params:
        kode = "TEST"


def rk_(**kw):
    ind = pd.DataFrame({"close": np.linspace(900, 1000, 30), "volume": [10_000_000.0] * 30})
    base = dict(harga=1000.0, tgl=None, ma={5: 990, 20: 960, 50: 930, 100: 900, 200: 850}, rsi=55.0, macd=2.0, signal=1.0, verdict="BELI DI AREA ENTRY",
                alasan="harga di area beli", entry_lo=980.0, entry_hi=1010.0, entry_maks=1010.0, zona_valid=True, sl=940.0, tp1=1100.0, tp2=1200.0,
                target=1100.0, target_nama="TP1", rr=2.0, rr_harga=2.0, n_bar=300, ind=ind, sup=[{"harga": 980}], res=[{"harga": 1100}])
    base.update(kw)
    return base


def f_(**kw):
    m = dict(der=0.8, der_prev=0.8, cr=1.8, cr_prev=1.8, ll=100.0, ll_prev=100.0, utang=50.0, utang_prev=50.0, g_rev_yoy=0.3, g_rev_qoq=0.05, g_np_yoy=0.4, g_np_qoq=0.1,
             gpm=0.4, npm=0.15, roe=0.14, roa=0.07, pbv=1.5, per=12.0, per_norm=12.0, rev_ttm=1e12, np_ttm=1.5e11, cfo_ttm=2e11, fcf_ttm=1e11, capex_ttm=1e11,
             ekuitas=1e12, aset=2e12, liab=1e12, rugi_kuartal=0, n_np_kuartal=4, rev_kuartal=[1, 2, 3, 4], np_kuartal=[1.0, 2.0, 3.0, 4.0], cfo_kuartal=[5.0, 5.0, 5.0, 5.0])
    m.update(kw.pop("m", {}))
    d = dict(metrik=m, bank=False, sektor="Industri", subsektor="Manufaktur", ltm=dict(norm={"ok": False}, ni_norm=NAN), kualitas=dict(hilang=[]))
    d.update(kw)
    return d


def flow_(label="ACCUMULATION_CONFIRMED", ev="High", nilai20=10e9):
    return dict(label=label, evidence=ev, net20=5000.0, partisipasi=0.05, mendukung=["[OBS] a"], melawan=["[OBS] b"], invalidasi=["x"],
                kualitas=dict(sumber="ledger", nilai_transaksi_median_20d=nilai20), harga=dict(respons="flow beli diikuti ketahanan/kenaikan harga"), cfg=dict(min_nilai_miliar=5.0))


def ktx_(**kw):
    k = dict(ihsg=dict(status="na"), notasi=dict(status="ok", ada=False, teks=[]), disclosure=dict(status="ok", n=1, items=[dict(tgl="x", judul="Pembagian dividen tunai", risiko=[])]),
             news=dict(status="ok", n=0, items=[]), insider=dict(status="ok", n=0, nilai_beli=0, nilai_jual=0, pengendali_jual=False))
    k.update(kw)
    return k


def test_diskrit_dan_null():
    assert [sk.diskrit(x) for x in (0.0, 0.1, 0.4, 0.62, 0.9, 1.2)] == [0.0, 0.0, 0.5, 0.5, 1.0, 1.0]
    assert sk.diskrit(None) is None
    assert sk.skor_teknikal(None)["nilai"] is None and sk.skor_fundamental(None)["nilai"] is None
    assert sk.skor_ba(None)["nilai"] is None and sk.skor_ca(None)["nilai"] is None


def test_ta_batas():
    assert sk.skor_teknikal(rk_())["nilai"] == 1.0
    assert sk.skor_teknikal(rk_(verdict="HINDARI"))["nilai"] <= 0.25
    jauh = sk.skor_teknikal(rk_(harga=1200.0, rr_harga=0.4, ma={5: 1190, 20: 1100, 50: 1000, 100: 900, 200: 850}))
    assert jauh["nilai"] < 1.0


def test_fa_cap_arus_kas_dan_utang():
    assert sk.skor_fundamental(f_())["nilai"] >= 0.75
    fatal = sk.skor_fundamental(f_(m=dict(liab=3e12, aset=2e12)))
    assert fatal["nilai"] == 0.0 and fatal["gates"][0]["berat"] == "AVOID"
    berat = sk.skor_fundamental(f_(m=dict(cfo_kuartal=[-1.0, -2.0, 3.0, -1.0], cfo_ttm=-1e10, fcf_ttm=-5e10, der=2.5)))
    assert berat["nilai"] == 0.0 and any(g["gate"] == "FUNDAMENTAL_RED_FLAG" for g in berat["gates"])
    ulang = sk.skor_fundamental(f_(m=dict(cfo_kuartal=[-1.0, -2.0, 3.0, 4.0], fcf_ttm=-5e10)))
    assert ulang["nilai"] <= 0.25 and ulang["gates"][0]["berat"] == "WATCH"
    lev = sk.skor_fundamental(f_(m=dict(der=2.6, cfo_ttm=-1e9)))
    assert lev["nilai"] <= 0.25
    sekali = sk.skor_fundamental(f_(m=dict(cfo_kuartal=[-1.0, 5.0, 5.0, 5.0])))
    assert sekali["nilai"] >= 0.75 and not sekali["gates"]                       # CFO negatif satu periode: tidak otomatis distress
    properti = sk.skor_fundamental(f_(sektor="Properti", m=dict(der=2.3)))
    assert not any("ambang sektor 2" == x for x in properti["lawan"]) and properti["nilai"] >= 0.5   # ambang DER properti 2,5
    null = sk.skor_fundamental(f_(m=dict(cfo_ttm=NAN, der=NAN)))
    assert null["nilai"] is None                                                   # data arus kas/utang tidak ada -> NULL, bukan 0
    ramp = sk.skor_fundamental(f_(sektor="Basic Materials", subsektor="Gold Mining", m=dict(np_ttm=-5e10, npm=-0.2, g_rev_qoq=10.0, g_rev_yoy=NAN, gpm=0.46, per=-246.0,
                                                                                      per_norm=NAN, g_np_yoy=NAN, g_np_qoq=NAN, cfo_kuartal=[-1.0, -1.0, 2.0, -1.0], fcf_ttm=-4e11)))
    assert ramp["nilai"] <= 0.5 and "ramp-up" in ramp["catatan"].lower()


def test_ba_dari_label_aliran():
    assert sk.skor_ba(flow_())["nilai"] == 1.0
    assert sk.skor_ba(flow_("ACCUMULATION_CANDIDATE", "Medium"))["nilai"] == 0.5
    d = sk.skor_ba(flow_("DISTRIBUTION_CONFIRMED", "High"))
    assert d["nilai"] == 0.0 and d["gates"][0]["gate"] == "DISTRIBUTION_WARNING" and d["gates"][0]["berat"] == "AVOID"
    assert sk.skor_ba(dict(label="INSUFFICIENT_DATA", alasan_data="x"))["nilai"] is None


def test_ca_kata_kunci():
    assert sk.skor_ca(ktx_())["nilai"] == 0.75
    # "dividend" mengandung "dividen": tetap SATU kategori (bug lama menghitungnya dua -> 1,0)
    satu = sk.skor_ca(ktx_(disclosure=dict(status="ok", items=[dict(judul="Jadwal pembagian dividen", risiko=[]), dict(judul="Cash dividend schedule", risiko=[])])))
    assert satu["nilai"] == 0.75 and "dividen" in satu["bukti"][0]
    dua = sk.skor_ca(ktx_(disclosure=dict(status="ok", items=[dict(judul="Jadwal pembagian dividen", risiko=[]), dict(judul="Rencana buyback saham", risiko=[])])))
    assert dua["nilai"] == 1.0
    dil = sk.skor_ca(ktx_(disclosure=dict(status="ok", items=[dict(judul="Rencana Penambahan Modal dengan HMETD", risiko=[])])))
    assert dil["nilai"] == 0.25
    berat = sk.skor_ca(ktx_(news=dict(status="ok", items=[dict(judul="Permohonan PKPU terhadap emiten", risiko=["pkpu"])])))
    assert berat["nilai"] == 0.0 and berat["gates"][0]["berat"] == "AVOID"
    nt = sk.skor_ca(ktx_(notasi=dict(status="ok", ada=True, teks=["X"])))
    assert nt["nilai"] <= 0.25 and nt["gates"][0]["berat"] == "WATCH"


def test_total_coverage_signal_dan_gate():
    r = R()
    ctx = dict(mode="lengkap", flow=flow_(), konteks=ktx_())
    ok = sk.hitung(r, rk_(), f_(), None, None, None, ctx)
    assert ok["coverage"] == 1.0 and ok["total"] >= 0.7 and ok["vonis"] == "BELI"
    # total tinggi tetapi FA red flag berat -> gate membatasi (rumus Total tidak berubah)
    merah = sk.hitung(r, rk_(), f_(m=dict(cfo_kuartal=[-1.0, -2.0, 3.0, -1.0], cfo_ttm=-1e10, fcf_ttm=-5e10, der=2.5)), None, None, None, ctx)
    assert merah["vonis"] == "HINDARI / JUAL" and any(g["gate"] == "FUNDAMENTAL_RED_FLAG" for g in merah["gates"]) and merah["total"] >= 0.6
    # likuiditas rendah -> paling tinggi TUNGGU
    tipis = sk.hitung(r, rk_(), f_(), None, None, None, dict(ctx, flow=flow_(nilai20=1e9)))
    assert tipis["vonis"] == "TUNGGU" and any(g["gate"] == "LIQUIDITY_RED_FLAG" for g in tipis["gates"])
    # coverage 50% (BA & CA NULL) -> DATA_QUALITY_WARNING, tidak BELI
    sebagian = sk.hitung(r, rk_(), f_(), None, None, None, dict(mode="lengkap"))
    assert sebagian["coverage"] == 0.5 and sebagian["vonis"] == "TUNGGU" and any(g["gate"] == "DATA_QUALITY_WARNING" for g in sebagian["gates"])
    # total rendah -> HINDARI
    buruk = sk.hitung(r, rk_(verdict="HINDARI", harga=900.0), f_(m=dict(liab=3e12, aset=2e12)), None, None, None, dict(mode="lengkap", flow=flow_("DISTRIBUTION_CONFIRMED", "High"), konteks=ktx_()))
    assert buruk["vonis"] == "HINDARI / JUAL"
    # cepat / Hengky menahan BELI
    assert sk.hitung(r, rk_(), f_(), None, None, None, dict(ctx, mode="cepat"))["vonis"] == "TUNGGU"
    assert sk.hitung(r, rk_(), f_(), None, None, {"checklist": {"lolos_inti": False, "hasil": "x"}}, ctx)["vonis"] == "TUNGGU"
    assert ok["invalidasi"] and ok["data_quality"].startswith("coverage")


def test_posisi_dan_laporan():
    r = R()
    ctx = dict(mode="lengkap", flow=flow_(), konteks=ktx_())
    sc = sk.hitung(r, rk_(), f_(), None, None, None, ctx)
    assert "TAHAN" in sc["posisi"]["pemegang"]
    L = "\n".join(sk.baris_laporan(sc))
    assert "SKOR TA / FA / BA / CA" in L and "Invalidasi" in L and "[DEC] Pemegang:" in L
    kep, al = lp.keputusan(r, rk_(), f_(), None, dict(kategori="x", jebakan=[], kriteria=[], skor=0, skor_maks=0), None, ctx)
    assert kep == sc["vonis"] and al[0].startswith("RINGKAS: ")
