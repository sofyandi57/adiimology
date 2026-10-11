"""Skor tiga dimensi 1-10 (Bandarmologi, Teknikal, Fundamental) + rata-rata, dan vonis bertingkat dari skor.

Prinsip (belajar dari kritik analisa dan dari skill 'bedah'):
  - Setiap poin dapat ditelusuri: komponen, bukti pendukung, bukti berlawanan. Tidak ada angka tanpa alasan tertulis.
  - Data tidak cukup => skor dinyatakan TIDAK ANDAL (atau N/A), bukan tebakan. Rata-rata untuk vonis memakai dimensi yang andal.
  - Bandarmologi hanya andal bila ledger >= 60 sesi (lensa utama). Jendela pendek (<=20 hari) hanya fallback dan dibatasi (maks 7, tidak andal).
  - Rugi bersih pada fase ramp-up (pendapatan melonjak, margin kotor positif) bukan sinyal fatal; risiko neraca (current ratio, lonjakan utang
    jangka pendek) justru diberi bobot. Solvabilitas fatal (liabilitas > aset / ekuitas negatif) membatasi skor fundamental.
  - Skor = heuristik internal yang BELUM terbukti (belum ada uji out-of-sample). Rata-rata hanya alat ringkas; angka per dimensi yang penting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VERSI = "skor-2026.10.2"
KATA_TAMBANG = ("tambang", "mining", "emas", "gold", "batubara", "coal", "nikel", "nickel", "tembaga", "copper", "mineral", "logam", "metal", "basic materials", "energy")


def _ok(x) -> bool:
    return x is not None and not (isinstance(x, float) and np.isnan(x)) and not pd.isna(x)


def _fm(x, d: int = 0) -> str:
    return f"{x:,.{d}f}" if _ok(x) else "N/A"


def label(s: float | None) -> str:
    if s is None:
        return "N/A"
    return "KUAT" if s >= 8 else "BAIK" if s >= 6.5 else "SEDANG" if s >= 5 else "LEMAH" if s >= 3.5 else "BURUK"


def _skala(total: float, maks: float) -> float | None:
    """total/maks -> skala 1-10."""
    if maks <= 0:
        return None
    return round(float(min(10.0, max(1.0, 1 + 9 * total / maks))), 1)


def _dim(nama, skor, komponen, bukti, lawan, andal, catatan="", flags=None) -> dict:
    return dict(nama=nama, skor=skor, label=label(skor), komponen=komponen, bukti=bukti, lawan=lawan, andal=bool(andal) and skor is not None, catatan=catatan, flags=flags or [])


# =============================================================== TEKNIKAL
def skor_teknikal(rk: dict | None, ledger: dict | None = None) -> dict:
    if not rk:
        return _dim("Teknikal", None, [], [], ["analisa teknikal tidak tersedia (data gagal diambil)"], False, "N/A")
    if rk.get("verdict") == "TIDAK TERUKUR":
        return _dim("Teknikal", None, [], [], [f"data harga kurang dari 60 hari bursa ({rk.get('n_bar', 0)}): indikator tidak dapat diandalkan"], False, "N/A")
    h, ma = rk["harga"], rk["ma"]
    komp, sup, lawan = [], [], []
    # --- tren (maks 3; MA200 dihitung hanya bila datanya ada)
    pts, mk = 0.0, 0.0
    for nm, ok, w in (("harga di atas MA20", _ok(ma.get(20)) and h > ma[20], 1.0), ("MA20 di atas MA50", _ok(ma.get(20)) and _ok(ma.get(50)) and ma[20] > ma[50], 1.0),
                      ("harga di atas MA50", _ok(ma.get(50)) and h > ma[50], 0.5)):
        mk += w
        pts += w if ok else 0
        (sup if ok else lawan).append(f"[OBS] {nm}: {'ya' if ok else 'tidak'}")
    if _ok(ma.get(200)):
        mk += 0.5
        ok = h > ma[200]
        pts += 0.5 if ok else 0
        (sup if ok else lawan).append(f"[OBS] harga di atas MA200: {'ya' if ok else 'tidak'}")
    komp.append(dict(nama="Tren (MA)", poin=round(pts, 2), maks=mk, detail=f"MA20 {_fm(ma.get(20))} · MA50 {_fm(ma.get(50))}"))
    tren = pts / mk * 3 if mk else 0.0
    # --- momentum (maks 2)
    rsi = rk["rsi"]
    pm = 0.0
    if _ok(rsi):
        pm += 1.0 if 45 <= rsi <= 68 else 0.5 if (30 <= rsi < 45 or 68 < rsi <= 78) else 0.0
        (sup if 45 <= rsi <= 68 else lawan).append(f"[OBS] RSI {rsi:.0f} ({'zona sehat' if 45 <= rsi <= 68 else 'lemah/jenuh'})")
    if _ok(rk.get("macd")) and _ok(rk.get("signal")):
        ok = rk["macd"] > rk["signal"]
        pm += 1.0 if ok else 0.0
        (sup if ok else lawan).append(f"[OBS] MACD {'di atas' if ok else 'di bawah'} sinyal")
    komp.append(dict(nama="Momentum (RSI, MACD)", poin=pm, maks=2, detail=f"RSI {rsi:.0f}" if _ok(rsi) else "RSI N/A"))
    # --- lokasi terhadap area beli (maks 3)
    ind = rk.get("ind")
    ret1 = float(ind["close"].iloc[-1] / ind["close"].iloc[-2] - 1) if ind is not None and len(ind) > 2 else 0.0
    if h <= rk["sl"]:
        pl, dtl = 0.0, f"harga {h:,.0f} sudah di bawah stop loss {rk['sl']:,.0f}"
    elif rk["zona_valid"] and rk["entry_lo"] <= h <= rk["entry_hi"]:
        pl, dtl = 3.0, f"harga di dalam area beli {rk['entry_lo']:,.0f}-{rk['entry_hi']:,.0f}"
    elif h < rk["entry_lo"]:
        pl, dtl = 2.0, f"harga di bawah area beli (support {rk['entry_lo']:,.0f}); belum menembus SL"
    else:
        ref = rk["entry_hi"] if rk["zona_valid"] else rk["entry_lo"]
        j = h / ref - 1
        pl, dtl = (2.0 if j <= 0.02 else 1.0 if j <= 0.05 else 0.0), f"harga {j:+.1%} di atas {'entry maksimal' if rk['zona_valid'] else 'support'} {ref:,.0f}"
    if ret1 >= 0.04 and h >= rk["entry_lo"]:
        pl = max(0.0, pl - 1.0)
        dtl += f"; naik {ret1:+.1%} dalam sehari: area beli ditetapkan setelah lonjakan (mengejar)"
    komp.append(dict(nama="Lokasi harga vs area beli", poin=pl, maks=3, detail=dtl))
    (sup if pl >= 2 else lawan).append("[OBS] " + dtl)
    # --- R/R di harga sekarang (maks 2)
    rr = rk.get("rr_harga")
    pr = 0.0 if not _ok(rr) else 2.0 if rr >= 2 else 1.5 if rr >= 1.5 else 1.0 if rr >= 1 else 0.5 if rr >= 0.5 else 0.0
    dr = f"R/R bila beli di harga sekarang {rr:.2f} ke {rk['target_nama']} {rk['target']:,.0f}" if _ok(rr) else "R/R tidak terhitung"
    zp = (ledger or {}).get("zona_pasokan") or []
    kena = [z for z in zp if z["lo"] <= rk["target"] <= z["hi"] * 1.02]
    if kena:
        pr = max(0.0, pr - 1.0)
        dr += f"; target berada di zona pasokan {kena[0]['lo']:,.0f}-{kena[0]['hi']:,.0f} (ritel net beli {kena[0]['lot']:,.0f} lot: kemungkinan jualan pemegang rugi)"
    if rk["tp1"] / h - 1 < 0.03:
        dr += f"; TP1 hanya {rk['tp1'] / h - 1:+.1%} dari harga (praktis sama dengan resistance, bukan target)"
    komp.append(dict(nama="R/R", poin=pr, maks=2, detail=dr))
    (sup if pr >= 1.5 else lawan).append("[OBS] " + dr)
    total = tren + pm + pl + pr
    skor = _skala(total, 10)
    if rk["verdict"] == "HINDARI" and skor is not None:
        skor = min(skor, 3.0)
        lawan.append("[OBS] verdict teknikal HINDARI (harga tembus SL atau indikator mayoritas melawan): skor dibatasi 3,0")
    return _dim("Teknikal", skor, komp, sup, lawan, rk.get("n_bar", 0) >= 60, "")


# ============================================================ FUNDAMENTAL
def _tambang(f: dict) -> bool:
    t = " ".join(str(f.get(k) or "") for k in ("sektor", "subsektor")).lower()
    return any(k in t for k in KATA_TAMBANG)


def skor_fundamental(f: dict | None, konteks: dict | None = None) -> dict:
    if not f or not f.get("metrik"):
        return _dim("Fundamental", None, [], [], ["laporan keuangan belum dihitung / gagal diambil"], False, "N/A")
    m = f["metrik"]
    bank = f.get("bank", False)
    komp, sup, lawan, flags = [], [], [], []
    np_ttm = m["np_ttm"]
    ramp = bool(_ok(np_ttm) and np_ttm < 0 and ((_ok(m["g_rev_qoq"]) and m["g_rev_qoq"] > 0.5) or (_ok(m["g_rev_yoy"]) and m["g_rev_yoy"] > 1.0)) and _ok(m["gpm"]) and m["gpm"] > 0)
    tambang = _tambang(f)
    avail_max, total = 0.0, 0.0
    # A profitabilitas & trajektori (maks 3)
    if _ok(np_ttm):
        nn = m["np_ttm"] if not (f["ltm"].get("norm", {}).get("ok")) else f["ltm"]["ni_norm"]
        pa = 0.0
        if _ok(nn) and nn > 0:
            pa += 1.5
            pa += 0.75 if _ok(m["g_np_yoy"]) and m["g_np_yoy"] > 0 else 0
            pa += 0.75 if _ok(m["npm"]) and m["npm"] >= 0.10 else 0
            d = f"laba LTM positif; NPM {m['npm']:.1%}" if _ok(m["npm"]) else "laba LTM positif"
        elif ramp:
            pa += 1.0
            pa += 0.5 if m["gpm"] >= 0.3 else 0
            npk = [x for x in m["np_kuartal"] if x is not None]
            pa += 0.5 if len(npk) >= 2 and npk[-1] > npk[-2] else 0
            d = (f"RUGI pada fase ramp-up (pendapatan naik {m['g_rev_qoq']:+.0%} QoQ, margin kotor {m['gpm']:.0%}): rugi bukan sinyal fatal"
                 + (" untuk emiten tambang pra/awal produksi; nilai dengan jadwal produksi, AISC, NAV, runway pendanaan" if tambang else ""))
        else:
            pa += 0.5 if _ok(m["g_np_qoq"]) and m["g_np_qoq"] > 0 else 0
            d = f"rugi bersih LTM tanpa tanda ramp-up; rugi {'menyempit' if pa else 'belum menyempit'}"
        komp.append(dict(nama="Profitabilitas & trajektori", poin=round(pa, 2), maks=3, detail=d))
        (sup if pa >= 1.5 else lawan).append("[OBS] " + d)
        avail_max += 3
        total += pa
    # B pertumbuhan pendapatan (maks 2)
    g = m["g_rev_yoy"] if _ok(m["g_rev_yoy"]) else m["g_rev_qoq"]
    if _ok(g):
        pb = 2.0 if g > 0.2 else 1.5 if g > 0.05 else 1.0 if g > 0 else 0.0
        d = f"pendapatan {'YoY' if _ok(m['g_rev_yoy']) else 'QoQ'} {g:+.1%}"
        komp.append(dict(nama="Pertumbuhan pendapatan", poin=pb, maks=2, detail=d))
        (sup if pb >= 1.5 else lawan).append("[OBS] " + d)
        avail_max += 2
        total += pb
    # C neraca & solvabilitas (maks 3)
    fatal = bool((_ok(m["liab"]) and _ok(m["aset"]) and m["liab"] > m["aset"]) or (_ok(m["ekuitas"]) and m["ekuitas"] < 0))
    if _ok(m["ekuitas"]) or _ok(m["aset"]):
        if fatal:
            pc, d = 0.0, "FATAL: liabilitas lebih besar dari aset atau ekuitas negatif (risiko solvabilitas)"
            flags.append("SOLVABILITAS FATAL")
        elif bank:
            pc, d = 3.0, "bank: neraca diperiksa lewat ekuitas positif (DER/current ratio tidak berlaku)"
        else:
            pc, notes = 3.0, []
            if _ok(m["der"]):
                if m["der"] > 2:
                    pc -= 1.0; notes.append(f"DER {m['der']:.2f}x (>2)")
                elif m["der"] > 1.5:
                    pc -= 0.5; notes.append(f"DER {m['der']:.2f}x (>1,5)")
                else:
                    notes.append(f"DER {m['der']:.2f}x")
            if _ok(m["cr"]):
                if m["cr"] < 0.5:
                    pc -= 1.0; notes.append(f"current ratio {m['cr']:.2f} (<0,5)")
                elif m["cr"] < 1:
                    pc -= 0.5; notes.append(f"current ratio {m['cr']:.2f} (<1)")
                else:
                    notes.append(f"current ratio {m['cr']:.2f}")
            sp_ll = _ok(m["ll"]) and _ok(m["ll_prev"]) and m["ll_prev"] > 0 and m["ll"] / m["ll_prev"] >= 2
            sp_ut = _ok(m["utang"]) and _ok(m["utang_prev"]) and m["utang_prev"] > 0 and m["utang"] / m["utang_prev"] >= 1.5
            if sp_ll:
                pc -= 0.5; notes.append(f"liabilitas jangka pendek naik {m['ll'] / m['ll_prev']:.1f}x dalam 2 kuartal")
            if sp_ut:
                pc -= 0.5; notes.append(f"utang berbunga naik {m['utang'] / m['utang_prev']:.1f}x dalam 2 kuartal")
            if _ok(m["cr"]) and m["cr"] < 0.5 and (sp_ll or sp_ut):
                flags.append("RISIKO DILUSI / REFINANCING")
                kb = (konteks or {}).get("disclosure") or {}
                if kb.get("status") == "ok" and any("afiliasi" in i["judul"].lower() for i in kb.get("items", [])):
                    notes.append("keterbukaan memuat transaksi afiliasi: periksa tujuan pinjaman/jaminan")
            pc = max(0.0, pc)
            d = "; ".join(notes) or "rasio neraca tidak tersedia"
        komp.append(dict(nama="Neraca & solvabilitas", poin=round(pc, 2), maks=3, detail=d))
        (sup if pc >= 2 else lawan).append("[OBS] " + d)
        avail_max += 3
        total += pc
    # D arus kas (maks 1)
    if not bank and _ok(m["cfo_ttm"]):
        pd_ = 0.6 if m["cfo_ttm"] > 0 else 0.0
        if _ok(m["fcf_ttm"]):
            pd_ += 0.4 if m["fcf_ttm"] > 0 else (0.2 if (ramp and m["cfo_ttm"] > 0) else 0.0)
        d = f"CFO LTM {m['cfo_ttm'] / 1e9:,.0f} M; FCF {m['fcf_ttm'] / 1e9:,.0f} M" if _ok(m["fcf_ttm"]) else f"CFO LTM {m['cfo_ttm'] / 1e9:,.0f} M"
        komp.append(dict(nama="Arus kas", poin=round(pd_, 2), maks=1, detail=d))
        (sup if pd_ >= 0.6 else lawan).append("[OBS] " + d)
        avail_max += 1
        total += pd_
    # E valuasi (maks 1)
    per = m["per_norm"] if _ok(m["per_norm"]) and m["per_norm"] > 0 else m["per"]
    if _ok(per) and per > 0:
        pe = 1.0 if per <= 15 else 0.7 if per <= 30 else 0.4 if per <= 50 else 0.1
        d = f"PER {per:.1f}x"
        avail_max += 1
        total += pe
        komp.append(dict(nama="Valuasi", poin=pe, maks=1, detail=d))
        (sup if pe >= 0.7 else lawan).append("[OBS] " + d)
    elif ramp:
        komp.append(dict(nama="Valuasi", poin=0.5, maks=1, detail="PER/ROE tidak bermakna pada fase ramp-up (laba negatif): netral; pakai NAV/EV per ons cadangan"))
        avail_max += 1
        total += 0.5
    elif _ok(per) or _ok(np_ttm):
        komp.append(dict(nama="Valuasi", poin=0.0, maks=1, detail="PER tidak bermakna (laba negatif)"))
        lawan.append("[OBS] PER tidak bermakna (laba negatif)")
        avail_max += 1
    skor = _skala(total, avail_max)
    if fatal and skor is not None:
        skor = min(skor, 2.5)
        lawan.append("[OBS] solvabilitas fatal: skor fundamental dibatasi 2,5")
    andal = avail_max >= 6 and not f["kualitas"].get("hilang")
    cat = ""
    if ramp:
        cat = "Fase ramp-up terdeteksi: laba/PER/ROE bukan ukuran utama; LTM dari YTD yang disetahunkan dapat menyesatkan ke bawah."
    return _dim("Fundamental", skor, komp, sup, lawan, andal, cat, flags)


# =============================================================== BANDAR
_BASE_KAT = {"AKUMULASI TERKONFIRMASI (perilaku harga)": 8.0, "AKUMULASI KUAT": 8.0, "AKUMULASI TERINDIKASI": 6.5, "ABSORPSI SUPPLY": 6.0,
             "AKUMULASI LEMAH / BELUM SEARAH": 5.0, "TIDAK KONKLUSIF": 5.0, "TIDAK ADA BUKTI AKUMULASI": 3.5, "DISTRIBUSI TERINDIKASI": 2.5}


def _jebakan_berat(j: str) -> bool:
    t = j.lower()
    return "menjual" in t or "distribusi" in t


def skor_bandar(nilai: dict | None, ledger: dict | None) -> dict:
    lensa = []
    if ledger and ledger.get("bandar_score") is not None:
        s = _skala(ledger["bandar_score"], 10)
        andal = ledger.get("confidence_score", 0) >= 0.6
        komp = [dict(nama=c["dimensi"], poin=c["poin"], maks=c["maks"], detail=c["detail"]) for c in ledger["score_components"]]
        lensa.append(f"Ledger (lensa utama, {ledger['data_completeness']['sessions_loaded']} sesi): skor {ledger['bandar_score']}/10, status {ledger['episode_status']}, confidence {ledger['confidence_score']}")
        lawan = list(ledger["contradicting_evidence"])
        if nilai:
            lensa.append(f"Jendela pendek (konfirmasi saja): {nilai['kategori']}")
            if ledger["episode_status"] in ("DISTRIBUTING", "MARKDOWN") and nilai["kategori"].startswith("AKUMULASI"):
                lawan.append("[INF] lensa jendela pendek berkata akumulasi tetapi ledger jangka panjang berkata distribusi: ledger dipakai (jendela pendek rentan artefak tanggal mulai)")
        cat = "; ".join(ledger["confidence_reason"]) if not andal else ""
        fl = ["DISTRIBUSI KE RITEL"] if any(x.get("turun") for x in ledger.get("distribusi_ritel", [])) else []
        d = _dim("Bandarmologi", s, komp, list(ledger["supporting_evidence"]), lawan, andal, cat, fl)
        d["lensa"] = lensa
        d["status_episode"] = ledger["episode_status"]
        return d
    if nilai is None:
        d = _dim("Bandarmologi", None, [], [], ["data broker tidak tersedia"], False, "N/A")
        d["lensa"] = []
        return d
    kat = nilai["kategori"]
    base = next((v for k, v in _BASE_KAT.items() if kat.startswith(k) or kat == k), 3.0 if kat.startswith("WASPADA") else 5.0)
    jb = nilai.get("jebakan", [])
    berat = [j for j in jb if _jebakan_berat(j)]
    ringan = [j for j in jb if not _jebakan_berat(j)]
    s = base - min(3.0, 1.5 * len(berat)) - min(1.0, 0.5 * len(ringan))
    s = round(float(min(7.0, max(1.0, s))), 1)            # fallback jendela pendek dibatasi 7
    komp = [dict(nama="Kategori bandar (jendela pendek)", poin=base, maks=10, detail=kat)]
    sup = [f"[OBS] {x['kriteria']}: {x['bukti']}" for x in nilai["kriteria"] if x["status"] == "ya"]
    lawan = [f"[OBS] {x['kriteria']}: {x['bukti']}" for x in nilai["kriteria"] if x["status"] == "tidak"] + [f"[OBS] jebakan: {j}" for j in jb]
    lensa.append(f"Jendela pendek: {kat}")
    d = _dim("Bandarmologi", s, komp, sup, lawan, False,
             "Belum bisa jadi pegangan: hanya jendela pendek (net positif dalam jendela bisa artefak tanggal mulai). Isi Ledger >= 60 sesi (idealnya 200) agar skor andal.",
             ["DISTRIBUSI/PENJUALAN BANDAR"] if berat else [])
    d["lensa"] = lensa
    d["status_episode"] = None
    return d


# ================================================================ GABUNG
def _rata(dims):
    v = [d["skor"] for d in dims if d["skor"] is not None]
    return round(float(np.mean(v)), 1) if v else None


def hitung(r, rk, f, alr, nilai, hk_, ctx: dict | None = None) -> dict:
    ctx = ctx or {}
    mode = ctx.get("mode", "lengkap")
    ledger = ctx.get("ledger")
    konteks = ctx.get("konteks")
    dims = [skor_bandar(nilai, ledger), skor_teknikal(rk, ledger), skor_fundamental(f, konteks)]
    ada = [d for d in dims if d["skor"] is not None]
    andal = [d for d in ada if d["andal"]]
    base = andal if len(andal) >= 2 else ada
    rata_semua, rata_andal = _rata(ada), _rata(andal)
    rata = _rata(base)
    mn = min((d["skor"] for d in base), default=None)
    kep, ket = "TUNGGU", []
    if len(base) < 2:
        ket.append("data belum cukup untuk vonis (kurang dari 2 dimensi yang bisa dinilai)")
    elif rata < 4.0 or (rk and rk["verdict"] == "HINDARI" and rata < 5.0):
        kep = "HINDARI / JUAL"
        ket.append(f"rata-rata skor {rata} < {'5,0 dan teknikal HINDARI' if rata >= 4.0 else '4,0'}")
    elif rata >= 7.0 and mn >= 5.0 and rk and rk["verdict"] == "BELI DI AREA ENTRY":
        kep = "BELI"
        ket.append(f"rata-rata {rata} >= 7 dan tidak ada dimensi di bawah 5 (terendah {mn}); harga di area beli")
    elif rata >= 7.0 and mn >= 5.0:
        ket.append(f"skor kuat (rata {rata}) tetapi harga belum di area beli: tunggu pullback/entry maksimal")
    elif rata >= 5.0:
        ket.append(f"rata-rata {rata}: sedang; tidak ada alasan kuat untuk BELI maupun HINDARI")
    else:
        ket.append(f"rata-rata {rata}: lemah (batas HINDARI di bawah 4,0)")
    if kep == "BELI":
        if mode == "cepat":
            kep = "TUNGGU"; ket.append("BELI hanya dapat ditetapkan pada analisis LENGKAP (fundamental, dinamika bandar & aturan Hengky belum dihitung)")
        elif hk_ is not None and not hk_["checklist"]["lolos_inti"]:
            kep = "TUNGGU"; ket.append("BELI ditahan: aturan inti Hengky belum terpenuhi (uptrend, likuid, level, level pembatal)")
        elif ((konteks or {}).get("notasi") or {}).get("ada"):
            kep = "TUNGGU"; ket.append("BELI ditahan: saham memiliki notasi khusus BEI")
    flags = [x for d in dims for x in d["flags"]]
    terlemah = min(ada, key=lambda d: d["skor"]) if ada else None
    ringkas = " · ".join(f"{d['nama'][:1]} {d['skor']}{'' if d['andal'] else '*'}" if d["skor"] is not None else f"{d['nama'][:1]} N/A" for d in dims)
    ringkas += f" → rata {rata if rata is not None else 'N/A'}" + (f" (terlemah: {terlemah['nama']} {terlemah['skor']})" if terlemah else "")
    alasan = [f"RINGKAS: {ringkas}"]
    for d in dims:
        alasan.append(f"Skor {d['nama']}: {d['skor'] if d['skor'] is not None else 'N/A'}/10 ({d['label']}{'' if d['andal'] or d['skor'] is None else ', belum andal'})" +
                      (f" - {d['catatan']}" if d["catatan"] else ""))
    alasan += [f"Vonis dari skor: {'; '.join(ket)}"]
    alasan += [f"Bendera: {x}" for x in flags]
    if rk:
        alasan.append(f"Teknikal: {rk['verdict']} ({rk['alasan']})")
    alasan.append(f"Skenario Adimology: {r.verdict} (bukan dasar keputusan; belum terbukti di backtest)")
    return dict(dims=dims, rata=rata, rata_semua=rata_semua, rata_andal=rata_andal, min=mn, vonis=kep, ket=ket, flags=flags, ringkas=ringkas, alasan=alasan,
                n_andal=len(andal), versi=VERSI, posisi=panduan_posisi(rata, rk, kep))


def panduan_posisi(rata, rk, kep) -> dict:
    """Vonis dipisah: calon pembeli vs pemegang (kritik: satu vonis 'JUAL' tidak adil bagi pemegang dengan tren yang masih utuh)."""
    if kep == "BELI":
        cp = "Boleh mencicil di area beli; ukuran posisi dari anggaran risiko; stop loss wajib."
    elif kep == "HINDARI / JUAL":
        cp = "Jangan masuk."
    else:
        cp = "Tunggu: " + ("harga masuk area beli atau skor membaik." if rata is not None and rata >= 5 else "skor membaik dulu.")
    if not rk:
        return dict(calon_pembeli=cp, pemegang="Tidak ada rencana angka (teknikal tidak tersedia).")
    h, ma = rk["harga"], rk["ma"]
    patah = h <= rk["sl"] or (_ok(ma.get(50)) and h < ma[50])
    utuh = _ok(ma.get(20)) and _ok(ma.get(50)) and h > ma[20] > ma[50]
    if patah:
        pm = f"KURANGI/KELUAR: harga {h:,.0f} di bawah SL {rk['sl']:,.0f} atau MA50 (struktur tren patah)."
    elif rata is not None and rata < 4:
        pm = f"KURANGI bertahap: skor rendah ({rata}) walau tren belum patah; SL {rk['sl']:,.0f}."
    else:
        pm = (f"TAHAN selama close di atas SL {rk['sl']:,.0f}; ambil sebagian di {rk['tp1']:,.0f} / {rk['tp2']:,.0f}; jangan average up." if utuh else
              f"TAHAN dengan disiplin SL {rk['sl']:,.0f}; tren belum sepenuhnya naik (harga/MA20/MA50), jangan menambah.")
    return dict(calon_pembeli=cp, pemegang=pm)


# ============================================================== laporan
def baris_laporan(sc: dict | None) -> list[str]:
    L = ["## SKOR 3 DIMENSI (1-10; heuristik internal, BELUM terbukti secara statistik)"]
    if not sc:
        L.append("N/A")
        return L
    L.append(f"[EST] {sc['ringkas']}  (* = belum andal) · versi {sc['versi']}")
    L.append(f"[DEC] vonis dari skor = {sc['vonis']}: {'; '.join(sc['ket'])}")
    L.append("Dimensi | Skor | Label | Andal? | Catatan")
    for d in sc["dims"]:
        L.append(f"{d['nama']} | {d['skor'] if d['skor'] is not None else 'N/A'} | {d['label']} | {'ya' if d['andal'] else 'tidak'} | {d['catatan'] or '-'}")
    for d in sc["dims"]:
        if d["komponen"]:
            L.append(f"### Komponen {d['nama']}")
            L.append("Komponen | Poin | Maks | Detail")
            for c in d["komponen"]:
                L.append(f"{c['nama']} | {c['poin']} | {c['maks']} | {str(c['detail'])[:160]}")
        for t, lst in (("mendukung", d["bukti"]), ("berlawanan", d["lawan"])):
            for x in lst[:6]:
                L.append(f"- ({d['nama']} {t}) {x}")
        for ln in d.get("lensa", []):
            L.append(f"- (lensa bandar) {ln}")
    if sc["flags"]:
        L.append("Bendera: " + "; ".join(sc["flags"]))
    L.append(f"[DEC] Calon pembeli: {sc['posisi']['calon_pembeli']}")
    L.append(f"[DEC] Pemegang: {sc['posisi']['pemegang']}")
    return L
