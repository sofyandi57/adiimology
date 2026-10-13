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

VERSI = "skor-2026.10.3 (TA/FA/BA/CA diskrit 0-1)"
NILAI = (0.0, 0.25, 0.5, 0.75, 1.0)
ARTI = {0.0: "sangat tidak mendukung", 0.25: "dukungan lemah / ada risiko utama", 0.5: "campuran / netral", 0.75: "dukungan kuat", 1.0: "dukungan sangat kuat"}
# ambang leverage per sektor (DER liabilitas/ekuitas); bank & pembiayaan tidak memakai DER
DER_SEKTOR = ((("properti", "real estate", "konstruksi", "construction", "infrastruktur", "infrastructure"), 2.5), (("utilitas", "utilities", "energi listrik"), 3.0))
DER_DEFAULT = 2.0
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


def diskrit(frac: float | None) -> float | None:
    """Pecahan 0-1 -> skor diskrit 0 / 0,25 / 0,5 / 0,75 / 1."""
    if frac is None or (isinstance(frac, float) and np.isnan(frac)):
        return None
    return float(min(1.0, max(0.0, round(frac * 4) / 4)))


def _dim(nama, skor, komponen, bukti, lawan, andal, catatan="", flags=None, frac=None, kode="") -> dict:
    nilai = diskrit(frac) if frac is not None else (diskrit((skor - 1) / 9) if skor is not None else None)
    return dict(nama=nama, kode=kode, skor=skor, nilai=nilai, arti=ARTI.get(nilai, "belum dapat dinilai (NULL)") if nilai is not None else "belum dapat dinilai (NULL)",
                label=label(skor), komponen=komponen, bukti=bukti, lawan=lawan, andal=bool(andal) and skor is not None, catatan=catatan, flags=flags or [], gates=[])


# =============================================================== TEKNIKAL
def skor_teknikal(rk: dict | None, ledger: dict | None = None) -> dict:
    if not rk:
        return _dim("Teknikal", None, [], [], ["analisa teknikal tidak tersedia (data gagal diambil)"], False, "N/A", kode="TA")
    if rk.get("verdict") == "TIDAK TERUKUR":
        return _dim("Teknikal", None, [], [], [f"data harga kurang dari 60 hari bursa ({rk.get('n_bar', 0)}): indikator tidak dapat diandalkan"], False, "N/A", kode="TA")
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
    frac = total / 10
    if rk["verdict"] == "HINDARI":
        frac = min(frac, 0.25)
        lawan.append("[OBS] verdict teknikal HINDARI (harga tembus SL atau indikator mayoritas melawan): TA dibatasi 0,25")
    skor = _skala(frac * 10, 10)
    return _dim("Teknikal", skor, komp, sup, lawan, rk.get("n_bar", 0) >= 60, "", frac=frac, kode="TA")


# ============================================================ FUNDAMENTAL
def _tambang(f: dict) -> bool:
    t = " ".join(str(f.get(k) or "") for k in ("sektor", "subsektor")).lower()
    return any(k in t for k in KATA_TAMBANG)


def _der_max(f: dict) -> float:
    t = " ".join(str(f.get(k) or "") for k in ("sektor", "subsektor")).lower()
    for kata, batas in DER_SEKTOR:
        if any(k in t for k in kata):
            return batas
    return DER_DEFAULT


def skor_fundamental(f: dict | None, konteks: dict | None = None) -> dict:
    if not f or not f.get("metrik"):
        return _dim("Fundamental", None, [], [], ["laporan keuangan belum dihitung / gagal diambil"], False, "N/A", kode="FA")
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
            der_max = _der_max(f)
            if _ok(m["der"]):
                if m["der"] > der_max:
                    pc -= 1.0; notes.append(f"DER {m['der']:.2f}x (> ambang sektor {der_max:g})")
                elif m["der"] > 0.75 * der_max:
                    pc -= 0.5; notes.append(f"DER {m['der']:.2f}x (> {0.75 * der_max:.2f})")
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
    if not bank and not _ok(m["cfo_ttm"]) and not _ok(m["der"]):
        return _dim("Fundamental", None, komp, sup, lawan + ["[OBS] arus kas dan utang tidak tersedia: FA = NULL (bukan 0)"], False, "data arus kas/utang tidak tersedia", flags, kode="FA")
    frac = total / avail_max if avail_max else None
    # ---- batas maksimum (cap) red flag arus kas & utang: aturan terdokumentasi, terpisah dari komponen
    cfo_q = [x for x in m.get("cfo_kuartal", []) if x is not None]
    cfo_neg_ulang = sum(1 for x in cfo_q if x < 0) >= 2
    fcf_neg = _ok(m["fcf_ttm"]) and m["fcf_ttm"] < 0
    lev_tinggi = (not bank) and _ok(m["der"]) and m["der"] > _der_max(f)
    cfo_lemah = _ok(m["cfo_ttm"]) and m["cfo_ttm"] <= 0
    refin = "RISIKO DILUSI / REFINANCING" in flags
    cap, berat, alasan_cap = None, None, None
    if fatal:
        cap, berat, alasan_cap = 0.0, "AVOID", "liabilitas > aset / ekuitas negatif"
    elif cfo_neg_ulang and fcf_neg and lev_tinggi:
        cap, berat, alasan_cap = 0.0, "AVOID", "CFO negatif berulang + FCF negatif + leverage tinggi"
    elif lev_tinggi and (cfo_lemah or refin):
        cap, berat, alasan_cap = 0.25, "WATCH", "utang berlebihan disertai CFO lemah atau risiko refinancing"
    elif cfo_neg_ulang and fcf_neg:
        if ramp:
            cap, berat, alasan_cap = 0.5, "WATCH", "CFO negatif berulang + FCF negatif pada fase ramp-up (pendanaan belum diverifikasi)"
        else:
            cap, berat, alasan_cap = 0.25, "WATCH", "CFO negatif berulang + FCF negatif"
    elif refin:
        cap, berat, alasan_cap = 0.5, "WATCH", "current ratio < 0,5 + lonjakan utang jangka pendek"
    gates = []
    if cap is not None and frac is not None and diskrit(frac) > cap:
        lawan.append(f"[OBS] FA dibatasi maksimum {cap:.2f}: {alasan_cap}")
        frac = cap
    if cap is not None:
        gates.append(dict(gate="FUNDAMENTAL_RED_FLAG", berat=berat, alasan=alasan_cap))
    if cfo_q and sum(1 for x in cfo_q if x < 0) == 1 and not cfo_neg_ulang:
        sup.append("[INF] CFO negatif hanya satu kuartal: bisa modal kerja/musiman, tidak otomatis distress")
    skor = _skala(frac * 10, 10) if frac is not None else None
    andal = avail_max >= 6 and not f["kualitas"].get("hilang")
    cat = ""
    if ramp:
        cat = "Fase ramp-up terdeteksi: laba/PER/ROE bukan ukuran utama; LTM dari YTD yang disetahunkan dapat menyesatkan ke bawah."
    cat += (" Interest coverage & jadwal jatuh tempo utang tidak tersedia di data: leverage dinilai dari DER, current ratio, CFO." if not bank else "")
    d = _dim("Fundamental", skor, komp, sup, lawan, andal, cat.strip(), flags, frac=frac, kode="FA")
    d["gates"] = gates
    return d


# =============================================================== BA (aliran transaksi)
BA_NILAI = {"ACCUMULATION_CONFIRMED": {"High": 1.0, "Medium": 0.75, "Low": 0.75}, "ACCUMULATION_CANDIDATE": {"High": 0.75, "Medium": 0.5, "Low": 0.5},
            "NEUTRAL": {"High": 0.5, "Medium": 0.5, "Low": 0.5}, "DISTRIBUTION_CANDIDATE": {"High": 0.25, "Medium": 0.25, "Low": 0.25},
            "DISTRIBUTION_CONFIRMED": {"High": 0.0, "Medium": 0.0, "Low": 0.25}}


def skor_ba(flow: dict | None, ledger: dict | None = None) -> dict:
    """BA dari tiga lapisan aliran (bandar_flow). INSUFFICIENT_DATA -> NULL (bukan netral)."""
    if not flow or flow.get("label") in (None, "INSUFFICIENT_DATA"):
        alasan = (flow or {}).get("alasan_data") or "tidak ada data broker harian multi-sesi"
        d = _dim("Bandarmologi (aliran)", None, [], [], [f"[OBS] INSUFFICIENT_DATA: {alasan}. Isi ledger (tab Bandar & Ledger) atau jalankan Hitung lengkap."], False,
                 "BA = NULL (data tidak cukup), tidak dianggap nol", kode="BA")
        d["flow_label"] = "INSUFFICIENT_DATA"
        return d
    lab, ev = flow["label"], flow["evidence"]
    nilai = BA_NILAI[lab][ev]
    komp = [dict(nama="Label aliran (lapisan 3)", poin=nilai, maks=1, detail=f"{lab} · evidence {ev} (tingkat bukti, bukan probabilitas)"),
            dict(nama="Lapisan 1: aliran teramati", poin=None, maks=None, detail=f"non-ritel net {flow['net20']:+,.0f} lot 20 sesi ({flow['partisipasi']:+.1%} volume); sumber {flow['kualitas']['sumber']}"),
            dict(nama="Lapisan 2: konfirmasi harga", poin=None, maks=None, detail=flow["harga"]["respons"])]
    lawan = list(flow["melawan"])
    gates = []
    if lab == "DISTRIBUTION_CONFIRMED":
        gates.append(dict(gate="DISTRIBUTION_WARNING", berat="AVOID" if ev == "High" else "WATCH", alasan=f"{lab} (evidence {ev})"))
    elif lab == "DISTRIBUTION_CANDIDATE":
        gates.append(dict(gate="DISTRIBUTION_WARNING", berat="WATCH", alasan=f"{lab} (evidence {ev})"))
    if ledger and ledger.get("episode_status") in ("DISTRIBUTING", "MARKDOWN") and ledger.get("confidence_score", 0) >= 0.6:
        gates.append(dict(gate="DISTRIBUTION_WARNING", berat="WATCH", alasan=f"ledger kelompok kunci {ledger['episode_status']} (confidence {ledger['confidence_score']})"))
        lawan.append(f"[OBS] ledger: {ledger['episode_status']}, Bandar Score {ledger.get('bandar_score')}/10")
    d = _dim("Bandarmologi (aliran)", _skala(nilai * 10, 10), komp, list(flow["mendukung"]), lawan, ev != "Low",
             "Label ritel = proksi; kode broker = banyak nasabah; bukan identitas/niat 'bandar'.", [], frac=nilai, kode="BA")
    d["gates"] = gates
    d["flow_label"] = lab
    d["invalidasi"] = list(flow.get("invalidasi", []))
    return d


# =============================================================== CA (aksi korporasi & keterbukaan)
CA_BERAT = ("pkpu", "pailit", "delisting", "suspen", "gagal bayar", "penghentian sementara")
CA_DILUSI = ("hmetd", "right issue", "rights issue", "private placement", "penambahan modal", "obligasi wajib konversi")
CA_POSITIF = {"buyback": ("buyback", "pembelian kembali"), "dividen": ("dividen", "dividend")}   # dihitung per KATEGORI (dividen & dividend = satu kategori)


def skor_ca(konteks: dict | None, f: dict | None = None, situasi: dict | None = None) -> dict:
    """CA dari keterbukaan, berita, insider, notasi khusus, kalender aksi (agenda RUPS dibaca UTUH; hanya aksi mendatang / <= 180 hari).
    NULL bila tidak ada sumber yang terbaca. Situasi khusus (backdoor) TIDAK dinilai baik/buruk di sini: ia memicu gate EVENT_DRIVEN_WARNING."""
    import datetime as _dt
    k = konteks or {}
    teks, sumber = [], []
    for nm in ("disclosure", "news"):
        x = k.get(nm) or {}
        if x.get("status") == "ok":
            sumber.append(nm)
            teks += [i["judul"].lower() for i in x.get("items", [])]
    aksi = (f or {}).get("aksi")
    today = _dt.date.today()
    if aksi:
        sumber.append("kalender aksi")
        for a in aksi:
            tg = a.get("tgl")
            if tg is not None and (today - tg).days > 180:
                continue                                     # aksi lama (mis. rights issue bertahun lalu) tidak relevan
            teks.append((a.get("type", "") + " " + (a.get("teks") or a.get("ringkas", ""))).lower())
    ins = k.get("insider") or {}
    if ins.get("status") == "ok":
        sumber.append("insider")
    nt = k.get("notasi") or {}
    if nt.get("status") == "ok":
        sumber.append("notasi")
    if not sumber:
        return _dim("Aksi korporasi & keterbukaan", None, [], [], ["[OBS] keterbukaan/berita/insider/kalender tidak diambil (Hitung lengkap) atau gagal: CA = NULL"], False, "N/A", kode="CA")
    import re as _re
    nilai, sup, lawan, gates = 0.5, [], [], []
    sx_aktif = bool(situasi and situasi.get("aktif"))
    berat = sorted({w for t in teks for w in CA_BERAT if w in t})
    pmt = [t for t in teks if _re.search(r"pmthmetd|tanpa hak memesan|private placement", t)]
    hm = [t for t in teks if _re.search(r"\bpmhmetd\b|(?<!tanpa )hak memesan efek terlebih dahulu|\bhmetd\b|right[s]? issue|obligasi wajib konversi", t)]
    pos = sorted({kat for t in teks for kat, kata in CA_POSITIF.items() if any(w in t for w in kata)})
    if "buyback" in pos and sx_aktif:
        pos.remove("buyback")
        lawan.append("[INF] rencana buyback di tengah situasi khusus: tidak dihitung positif (kelayakan dana dan tujuan belum diverifikasi)")
    if pos:
        nilai = 0.75 if len(pos) == 1 else 1.0
        contoh = [t for t in teks if any(w in t for kat in pos for w in CA_POSITIF[kat])][:3]
        sup.append("[OBS] aksi positif (" + ", ".join(pos) + "): " + " | ".join(c[:90] for c in contoh))
    if ins.get("status") == "ok" and ins.get("nilai_beli", 0) > ins.get("nilai_jual", 0) and ins.get("n", 0):
        nilai = min(1.0, nilai + 0.25)
        sup.append(f"[OBS] insider net beli (Rp{ins['nilai_beli']:,.0f} vs jual Rp{ins['nilai_jual']:,.0f})")
    if ins.get("status") == "ok" and ins.get("pengendali_jual"):
        nilai = min(nilai, 0.25)
        lawan.append("[OBS] pemegang saham pengendali menjual dalam 120 hari")
    if pmt:
        nilai = min(nilai, 0.25)
        lawan.append("[OBS] penambahan modal TANPA HMETD / private placement (investor publik tidak bisa ikut): " + " | ".join(t[:100] for t in pmt[:2]))
    if hm:
        nilai = min(nilai, 0.25)
        lawan.append("[OBS] rights issue / HMETD (dilusi bagi yang tidak menebus; besaran belum terbaca dari data): " + " | ".join(t[:100] for t in hm[:2]))
    dw = (situasi or {}).get("dilusi_waran")
    if dw is not None and dw >= 0.10:
        nilai = min(nilai, 0.5)
        lawan.append(f"[OBS] sisa waran ≈ {dw:.0%} dari saham beredar: dilusi bertahap")
    if nt.get("ada"):
        nilai = min(nilai, 0.25)
        lawan.append("[OBS] notasi khusus BEI: " + "; ".join(nt.get("teks", [])[:2]))
        gates.append(dict(gate="FUNDAMENTAL_RED_FLAG", berat="WATCH", alasan="notasi khusus BEI"))
    if berat:
        nilai = 0.0
        lawan.append("[OBS] peristiwa berat: " + ", ".join(berat) + " | " + " | ".join(t[:90] for t in teks if any(w in t for w in berat))[:250])
        gates.append(dict(gate="FUNDAMENTAL_RED_FLAG", berat="AVOID", alasan="peristiwa berat: " + ", ".join(berat)))
    if sx_aktif:
        gates.append(dict(gate="EVENT_DRIVEN_WARNING", berat="WATCH",
                          alasan=f"{situasi['jenis']} (tahap: {situasi['tahap_teks']}); spekulatif, posisi maksimal {situasi['posisi']['maks_pct']:.1%} modal"))
        lawan += [f"[OBS] {b_}" for b_ in situasi.get("flags", [])]
    if not (pos or pmt or hm or berat or nt.get("ada") or ins.get("pengendali_jual") or sx_aktif):
        sup.append("[OBS] tidak ada aksi korporasi/keterbukaan bermakna yang terdeteksi (netral)")
    aturan = ("mulai 0,50 · 1 kategori positif (buyback ATAU dividen) = 0,75 · keduanya = 1,00 · insider net beli +0,25 · "
              "rights issue/HMETD, PMTHMETD/private placement, pengendali jual, notasi khusus = maks 0,25 · sisa waran >= 10% = maks 0,50 · "
              "PKPU/pailit/delisting/suspensi = 0 · buyback saat situasi khusus tidak dihitung positif · aksi kalender > 180 hari diabaikan")
    d = _dim("Aksi korporasi & keterbukaan", _skala(nilai * 10, 10), [dict(nama="Sumber terbaca", poin=nilai, maks=1, detail=", ".join(sumber) + f" ({len(teks)} judul/aksi relevan)"),
                                                                       dict(nama="Aturan CA", poin=None, maks=None, detail=aturan)], sup, lawan,
             len(sumber) >= 2, "Deteksi kata kunci pada judul keterbukaan/berita dan teks agenda kalender; baca dokumen aslinya sebelum memutuskan.", [], frac=nilai, kode="CA")
    d["gates"] = gates
    return d


# ================================================================ GABUNG
def _gate_likuiditas(rk, flow, min_nilai_miliar=5.0) -> dict | None:
    nilai = None
    if flow and (flow.get("kualitas") or {}).get("nilai_transaksi_median_20d") is not None:
        nilai = flow["kualitas"]["nilai_transaksi_median_20d"]
    elif rk is not None and rk.get("ind") is not None:
        d = rk["ind"]
        nilai = float((d["close"] * d["volume"]).tail(20).median())
    if nilai is not None and _ok(nilai) and nilai < min_nilai_miliar * 1e9:
        return dict(gate="LIQUIDITY_RED_FLAG", berat="WATCH", alasan=f"nilai transaksi median 20D Rp{nilai / 1e9:,.1f} M < Rp{min_nilai_miliar:g} M (risiko slippage/kapasitas)")
    return None


def hitung(r, rk, f, alr, nilai, hk_, ctx: dict | None = None) -> dict:
    """TA, FA, BA, CA (diskrit 0-1, NULL bila tidak dapat dinilai) -> Total (rata-rata komponen valid) + Score_Coverage -> Signal, lalu hard gates.
    Rumus Total TIDAK diubah oleh gate: gate adalah lapisan keputusan terpisah yang transparan."""
    ctx = ctx or {}
    mode = ctx.get("mode", "lengkap")
    ledger, konteks, flow = ctx.get("ledger"), ctx.get("konteks"), ctx.get("flow")
    dims = [skor_teknikal(rk, ledger), skor_fundamental(f, konteks), skor_ba(flow, ledger), skor_ca(konteks, f, ctx.get("situasi"))]
    valid = [d for d in dims if d["nilai"] is not None]
    total = round(float(np.mean([d["nilai"] for d in valid])), 3) if valid else None
    coverage = len(valid) / 4
    mn = min((d["nilai"] for d in valid), default=None)
    ket = []
    if coverage < 0.5 or total is None:
        sinyal = "TUNGGU"; ket.append(f"Score_Coverage {coverage:.0%}: data belum cukup untuk vonis")
    elif total < 0.40:
        sinyal = "HINDARI / JUAL"; ket.append(f"Total {total:.2f} < 0,40")
    elif total >= 0.70 and coverage >= 0.75 and mn >= 0.5 and rk and rk["verdict"] == "BELI DI AREA ENTRY":
        sinyal = "BELI"; ket.append(f"Total {total:.2f} ≥ 0,70, coverage {coverage:.0%}, tidak ada komponen < 0,50, harga di area beli")
    elif total >= 0.70:
        sinyal = "TUNGGU"; ket.append(f"Total {total:.2f} kuat, tetapi " + ("coverage < 75%" if coverage < 0.75 else "ada komponen < 0,50" if mn < 0.5 else "harga belum di area beli"))
    else:
        sinyal = "TUNGGU"; ket.append(f"Total {total:.2f}: campuran (HINDARI di bawah 0,40; BELI mulai 0,70)")
    # ---- hard gates (terpisah dari Total)
    gates = [g for d in dims for g in d.get("gates", [])]
    gl = _gate_likuiditas(rk, flow, (flow or {}).get("cfg", {}).get("min_nilai_miliar", 5.0) if flow else 5.0)
    if gl:
        gates.append(gl)
    if coverage < 0.75:
        gates.append(dict(gate="DATA_QUALITY_WARNING", berat="WATCH", alasan=f"Score_Coverage {coverage:.0%} (komponen NULL: {', '.join(d['kode'] for d in dims if d['nilai'] is None)})"))
    if ctx.get("validasi") == "GAGAL":
        gates.append(dict(gate="DATA_QUALITY_WARNING", berat="WATCH", alasan="validasi angka laporan GAGAL"))
    sinyal_awal = sinyal
    if any(g["berat"] == "AVOID" for g in gates) and sinyal != "HINDARI / JUAL":
        sinyal = "HINDARI / JUAL"
    elif any(g["berat"] == "WATCH" for g in gates) and sinyal == "BELI":
        sinyal = "TUNGGU"
    if sinyal != sinyal_awal:
        ket.append(f"dibatasi hard gate: {sinyal_awal} → {sinyal}")
    if sinyal == "BELI":
        if mode == "cepat":
            sinyal = "TUNGGU"; ket.append("BELI hanya dapat ditetapkan pada analisis LENGKAP")
        elif hk_ is not None and not hk_["checklist"]["lolos_inti"]:
            sinyal = "TUNGGU"; ket.append("BELI ditahan: aturan inti Hengky belum terpenuhi")
    # ---- invalidasi keputusan
    inval = []
    if rk:
        inval.append(f"close di bawah stop loss {rk['sl']:,.0f}")
    for d in dims:
        inval += d.get("invalidasi", [])
    flags = [x for d in dims for x in d["flags"]]
    teks_k = " · ".join(f"{d['kode']} {d['nilai']:.2f}" if d["nilai"] is not None else f"{d['kode']} NULL" for d in dims)
    ringkas = f"{teks_k} → Total {total:.2f} (coverage {coverage:.0%})" if total is not None else f"{teks_k} → Total N/A"
    alasan = [f"RINGKAS: {ringkas}"]
    for d in dims:
        alasan.append(f"Skor {d['nama']} ({d['kode']}): {('%.2f' % d['nilai']) if d['nilai'] is not None else 'NULL'} ({d['arti']})" + (f" - {d['catatan']}" if d["catatan"] else ""))
    alasan.append(f"Vonis dari skor: {'; '.join(ket)}")
    alasan += [f"Gate: {g['gate']} ({g['berat']}): {g['alasan']}" for g in gates]
    alasan += [f"Bendera: {x}" for x in flags]
    if rk:
        alasan.append(f"Teknikal: {rk['verdict']} ({rk['alasan']})")
    alasan.append(f"Skenario Adimology: {r.verdict} (bukan dasar keputusan; belum terbukti di backtest)")
    dq = f"coverage {coverage:.0%}; BA {next(d.get('flow_label') for d in dims if d['kode'] == 'BA')}" + (f"; flow evidence {flow['evidence']}" if flow and flow.get("label") != "INSUFFICIENT_DATA" else "")
    rata10 = round(total * 10, 1) if total is not None else None
    sx = ctx.get("situasi")
    posisi = panduan_posisi(rata10, rk, sinyal)
    if sx and sx.get("aktif"):
        ps = sx["posisi"]
        posisi["calon_pembeli"] = (f"SPEKULATIF ({sx['jenis']}): bila tetap masuk, MAKSIMAL {ps['maks_pct']:.1%} modal ≈ {ps['lot']:,} lot; anggap seluruh posisi bisa hilang "
                                   "(gap/batal transaksi tidak tertahan stop loss). " + posisi["calon_pembeli"])
        posisi["pemegang"] += f" Situasi khusus: jangan menambah melebihi {ps['maks_pct']:.1%} modal; pantau tanggal kunci dan kondisi batal."
    return dict(dims=dims, total=total, coverage=coverage, rata=rata10, rata_semua=rata10, rata_andal=rata10, min=mn, vonis=sinyal, ket=ket, gates=gates, flags=flags,
                ringkas=ringkas, alasan=alasan, n_andal=sum(1 for d in dims if d["andal"]), versi=VERSI, invalidasi=inval, data_quality=dq,
                posisi=posisi, situasi=sx)


def panduan_posisi(rata, rk, kep) -> dict:
    """Vonis dipisah: calon pembeli vs pemegang (satu vonis 'JUAL' tidak adil bagi pemegang dengan tren yang masih utuh). rata dalam skala 0-10."""
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
        pm = f"KURANGI bertahap: skor rendah walau tren belum patah; SL {rk['sl']:,.0f}."
    else:
        pm = (f"TAHAN selama close di atas SL {rk['sl']:,.0f}; ambil sebagian di {rk['tp1']:,.0f} / {rk['tp2']:,.0f}; jangan average up." if utuh else
              f"TAHAN dengan disiplin SL {rk['sl']:,.0f}; tren belum sepenuhnya naik (harga/MA20/MA50), jangan menambah.")
    return dict(calon_pembeli=cp, pemegang=pm)


# ============================================================== laporan
def baris_laporan(sc: dict | None) -> list[str]:
    L = ["## SKOR TA / FA / BA / CA (diskrit 0 / 0,25 / 0,5 / 0,75 / 1; NULL = belum dapat dinilai; heuristik BELUM terbukti)"]
    if not sc:
        L.append("N/A")
        return L
    L.append(f"[EST] {sc['ringkas']} · versi {sc['versi']}")
    L.append(f"[DEC] Signal = {sc['vonis']}: {'; '.join(sc['ket'])}")
    L.append("Komponen | Nilai | Arti | Catatan")
    for d in sc["dims"]:
        L.append(f"{d['kode']} {d['nama']} | {('%.2f' % d['nilai']) if d['nilai'] is not None else 'NULL'} | {d['arti']} | {d['catatan'] or '-'}")
    for g in sc["gates"]:
        L.append(f"[DEC] hard gate {g['gate']} ({g['berat']}): {g['alasan']}")
    for d in sc["dims"]:
        if d["komponen"]:
            L.append(f"### Rincian {d['kode']}")
            L.append("Komponen | Poin | Maks | Detail")
            for c in d["komponen"]:
                L.append(f"{c['nama']} | {'-' if c['poin'] is None else c['poin']} | {'-' if c['maks'] is None else c['maks']} | {str(c['detail'])[:160]}")
        for t, lst in (("mendukung", d["bukti"]), ("berlawanan", d["lawan"])):
            for x in lst[:6]:
                L.append(f"- ({d['kode']} {t}) {x}")
    if sc["flags"]:
        L.append("Bendera: " + "; ".join(sc["flags"]))
    L.append("Invalidasi: " + ("; ".join(sc["invalidasi"]) or "-"))
    L.append(f"[DEC] Calon pembeli: {sc['posisi']['calon_pembeli']}")
    L.append(f"[DEC] Pemegang: {sc['posisi']['pemegang']}")
    return L
