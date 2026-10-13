"""Narasi fundamental: membaca laporan keuangan seperti investor fundamental yang disiplin (kerangka value-growth).

Kerangka (dirangkum dari cara berpikir investor fundamental, terinspirasi metode 'Filbert Gameplan'; teks di sini ditulis ulang, bukan kutipan
dan BUKAN pendapat penulis tersebut):
  1. Mesin laba: pendapatan tumbuh dari VOLUME atau HARGA JUAL? Bila pendapatan naik tetapi laba stagnan, cari penyebabnya (margin kotor
     menyusut, beban operasional/bunga naik lebih cepat).
  2. Dua mesin kenaikan harga: earnings expansion (EPS naik) dan PE expansion (pasar mau membayar kelipatan lebih tinggi).
     Harga wajar = EPS tahun depan x PER yang masuk akal; tampilkan beberapa skenario, pilih yang konservatif.
  3. Ambang: cari laba tumbuh >= 10%/tahun; bila laba flat, harus ada diskon besar DAN dividen yield >= 8%.
  4. Kelemahan yang wajib disebut: arus kas operasi yang sering negatif, gearing (utang berbunga/ekuitas) tinggi, rugi.
  5. PER sangat tinggi (> 50x) = harga digerakkan narasi, bukan kinerja. Holding: pakai NAV & diskon terhadap NAV.
     Komoditas: laba mengikuti harga komoditas (data eksternal) -> masuk watchlist saat murah, tunggu siklus.
  6. Review berkala: tesis harus dicek ulang tiap kuartal; 'jangka pendek' ±3-6 bulan, 'jangka panjang' > 1 tahun.
Data eksternal (harga komoditas, M2, suku bunga, penjualan mobil) TIDAK diambil aplikasi; narasi merujuk ke sumber yang tepat.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

VERSI = "narasi-fa-2026.10.1"

TE = "https://tradingeconomics.com"
R = {
    "coal": ("Harga batu bara Newcastle (acuan ekspor)", f"{TE}/commodity/coal"),
    "hba": ("HBA / harga acuan minerba (Kementerian ESDM)", "https://www.minerba.esdm.go.id/harga_acuan"),
    "wti": ("Minyak WTI", f"{TE}/commodity/crude-oil"), "brent": ("Minyak Brent", f"{TE}/commodity/brent-crude-oil"),
    "gas": ("Gas alam", f"{TE}/commodity/natural-gas"), "nickel": ("Nikel LME", f"{TE}/commodity/nickel"),
    "hpm": ("HPM nikel / harga acuan mineral (Kementerian ESDM)", "https://www.minerba.esdm.go.id/harga_acuan"),
    "gold": ("Emas", f"{TE}/commodity/gold"), "copper": ("Tembaga", f"{TE}/commodity/copper"), "tin": ("Timah", f"{TE}/commodity/tin"),
    "alu": ("Aluminium", f"{TE}/commodity/aluminum"), "cpo": ("CPO / palm oil", f"{TE}/commodity/palm-oil"), "rubber": ("Karet", f"{TE}/commodity/rubber"),
    "sugar": ("Gula", f"{TE}/commodity/sugar"), "wheat": ("Gandum", f"{TE}/commodity/wheat"),
    "m2": ("Uang beredar M2 Indonesia", f"{TE}/indonesia/money-supply-m2"),
    "seki": ("Bank Indonesia SEKI (DPK, kredit, M2)", "https://www.bi.go.id/id/statistik/ekonomi-keuangan/seki/Default.aspx"),
    "birate": ("BI-Rate (Bank Indonesia)", "https://www.bi.go.id/id/statistik/indikator/bi-rate.aspx"),
    "ojk": ("OJK Statistik Perbankan Indonesia (LDR, NPL, NIM)", "https://www.ojk.go.id/id/kanal/perbankan/data-dan-statistik/statistik-perbankan-indonesia/Default.aspx"),
    "gaikindo": ("GAIKINDO (penjualan mobil domestik)", "https://www.gaikindo.or.id/"), "car": ("Penjualan mobil Indonesia", f"{TE}/indonesia/car-sales"),
    "cpi": ("Inflasi Indonesia", f"{TE}/indonesia/inflation-cpi"), "ikk": ("Keyakinan konsumen Indonesia", f"{TE}/indonesia/consumer-confidence"),
    "bps": ("BPS (inflasi, PDB, penjualan eceran)", "https://www.bps.go.id/"), "gdp": ("Pertumbuhan PDB Indonesia", f"{TE}/indonesia/gdp-growth-annual"),
    "dxy": ("Dollar Index (DXY)", f"{TE}/united-states/currency"), "idr": ("Kurs USD/IDR", f"{TE}/indonesia/currency"),
    "fed": ("Suku bunga The Fed", f"{TE}/united-states/interest-rate"), "bi_te": ("Suku bunga BI", f"{TE}/indonesia/interest-rate"),
}
# (kata kunci sektor/subsektor, nama, kode rujukan, pertanyaan volume, pertanyaan harga)
SEKTOR = [
    (("batu bara", "batubara", "coal"), "Batu bara", ("coal", "hba"), "produksi dan stripping ratio naik/turun?", "ASP mengikuti Newcastle/HBA dengan jeda ±1 kuartal"),
    (("nikel", "nickel"), "Nikel", ("nickel", "hpm"), "produksi bijih/olahan (NPI, matte, MHP) bertambah?", "harga jual mengikuti LME dan HPM nikel"),
    (("emas", "gold"), "Emas", ("gold",), "produksi (ons) dan cadangan bertambah?", "harga jual mengikuti harga emas dunia"),
    (("tembaga", "copper"), "Tembaga", ("copper", "gold"), "produksi konsentrat naik?", "harga mengikuti LME tembaga"),
    (("timah", "tin"), "Timah", ("tin",), "produksi logam timah naik?", "harga mengikuti LME timah"),
    (("aluminium", "aluminum", "bauksit"), "Aluminium/bauksit", ("alu",), "kapasitas smelter terpakai?", "harga mengikuti LME aluminium"),
    (("minyak", "oil", "gas"), "Minyak & gas", ("wti", "brent", "gas"), "lifting/produksi barel naik?", "harga mengikuti WTI/Brent"),
    (("perkebunan", "sawit", "plantation", "palm", "cpo", "agri"), "CPO / perkebunan", ("cpo", "rubber"), "produksi TBS/CPO dan usia tanaman?", "harga mengikuti CPO dunia (dan pungutan ekspor)"),
    (("bank",), "Perbankan", ("m2", "seki", "ojk", "birate"), "kredit dan DPK tumbuh? (likuiditas M2 longgar atau kering)", "NIM: cost of fund naik/turun mengikuti BI-Rate dan perebutan DPK"),
    (("otomotif", "automotive", "kendaraan"), "Otomotif", ("gaikindo", "car", "birate"), "unit terjual naik? (data GAIKINDO bulanan)", "harga/mix produk dan diskon"),
    (("properti", "real estat", "real estate"), "Properti", ("birate", "bi_te"), "marketing sales (unit) naik?", "harga jual per m² dan suku bunga KPR"),
    (("gula", "sugar"), "Gula", ("sugar",), "kuota impor & volume giling?", "harga gula"),
    (("gandum", "tepung", "wheat"), "Tepung/gandum", ("wheat", "idr"), "volume penjualan?", "harga gandum impor dan kurs"),
    (("konsumen", "consumer", "makanan", "minuman", "ritel", "retail", "rokok"), "Konsumen/ritel", ("cpi", "ikk", "bps"), "volume/same store sales tumbuh saat daya beli lemah?",
     "bisa menaikkan harga tanpa kehilangan volume?"),
]
GLOBAL = ("dxy", "idr", "fed", "birate")


def _ok(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x)) and not pd.isna(x)


def _p(x) -> str:
    return f"{x:+.1%}" if _ok(x) else "N/A"


def _last(seri):
    v = [x for x in (seri or []) if _ok(x)]
    return v[-1] if v else float("nan")


def _first(seri):
    v = [x for x in (seri or []) if _ok(x)]
    return v[0] if v else float("nan")


def sektor(f: dict) -> tuple:
    t = " ".join(str(f.get(k) or "") for k in ("sektor", "subsektor")).lower()
    if f.get("bank"):
        t += " bank"
    for kata, nama, kode, qv, qh in SEKTOR:
        if any(k in t for k in kata):
            return nama, kode, qv, qh
    return None, (), "volume penjualan bisa tumbuh (atau minimal bertahan) saat daya beli lemah?", "harga jual bisa naik (atau minimal bertahan)?"


def skenario(eps: float, per_now: float, harga: float) -> pd.DataFrame:
    """Matriks harga = EPS x (1+g) x PER. Baris = pertumbuhan EPS, kolom = PER."""
    gs, pers = (-0.10, 0.0, 0.10, 0.20), (6, 8, 10, 12, 14)
    rows = []
    for g in gs:
        row = {"Pertumbuhan EPS": f"{g:+.0%}"}
        for p_ in pers:
            row[f"PER {p_}x"] = eps * (1 + g) * p_
        rows.append(row)
    return pd.DataFrame(rows)


def susun(f: dict | None, harga: float, kode: str = "") -> dict | None:
    if not f or not f.get("metrik"):
        return None
    m = f["metrik"]
    nama_sek, kode_ref, q_vol, q_harga = sektor(f)
    eps, bvps = m.get("eps"), m.get("bvps")
    per_now = harga / eps if _ok(eps) and eps > 0 else float("nan")
    pbv_now = harga / bvps if _ok(bvps) and bvps > 0 else float("nan")
    g_rev, g_np = _last(m.get("rev_yoy_seri")), _last(m.get("np_yoy_seri"))
    if not _ok(g_rev):
        g_rev = m.get("g_rev_yoy")
    if not _ok(g_np):
        g_np = m.get("g_np_yoy")
    gpm0, gpm1 = _first(m.get("gpm_seri")), _last(m.get("gpm_seri"))
    npm0, npm1 = _first(m.get("npm_seri")), _last(m.get("npm_seri"))
    div = abs(m["div_ttm"]) if _ok(m.get("div_ttm")) else float("nan")
    dy = div / m["mcap"] if _ok(div) and _ok(m.get("mcap")) and m["mcap"] > 0 else float("nan")
    cfo_q = [x for x in m.get("cfo_kuartal", []) if x is not None]
    cfo_neg = sum(1 for x in cfo_q if x < 0)
    gear = m.get("gearing")
    holding = any(k in " ".join(str(f.get(x) or "") for x in ("sektor", "subsektor")).lower() for k in ("holding", "investasi", "investment"))
    komoditas = nama_sek in ("Batu bara", "Nikel", "Emas", "Tembaga", "Timah", "Aluminium/bauksit", "Minyak & gas", "CPO / perkebunan")
    rugi = not (_ok(eps) and eps > 0)
    per_tag = []
    # ---------------- 1. mesin laba
    mesin = [f"[OBS] Pendapatan {_p(g_rev)} YoY, laba bersih {_p(g_np)} YoY (periode terakhir {m['periode'][-1] if m.get('periode') else '-'})."]
    if _ok(g_rev) and _ok(g_np) and g_rev > 0.05 and g_np < max(g_rev / 2, 0.0):
        if _ok(gpm0) and _ok(gpm1) and gpm1 < gpm0 - 0.02:
            mesin.append(f"[INF] Pendapatan tumbuh tetapi laba tertinggal: margin kotor menyusut dari {gpm0:.1%} ke {gpm1:.1%}. Pertanyaannya: tekanan harga jual atau biaya bahan baku?")
        else:
            mesin.append("[INF] Pendapatan tumbuh tetapi laba tertinggal walau margin kotor relatif stabil: kemungkinan beban operasional (gaji, sewa, promosi) atau beban bunga naik lebih cepat. Cek rincian beban di laporan.")
    elif _ok(g_np) and g_np >= 0.10:
        mesin.append(f"[INF] Laba tumbuh {g_np:.0%}: ada katalis earnings expansion. Pastikan pertumbuhan berulang, bukan laba satu kali.")
    elif _ok(g_np) and g_np < 0:
        mesin.append("[INF] Laba turun: tesisnya menjadi 'recovery'. Jawab dulu: mengapa tahun depan bisa pulih, dan apa pemicunya?")
    mesin.append(f"[TANYA] Volume: {q_vol}  ·  Harga jual: {q_harga}")
    if _ok(npm0) and _ok(npm1):
        mesin.append(f"[OBS] Margin bersih {npm0:.1%} → {npm1:.1%} dalam {len(m.get('npm_seri') or [])} kuartal terakhir.")
    # ---------------- 2. kualitas laba & neraca
    kual = []
    if cfo_q:
        kual.append(f"[OBS] Arus kas operasi negatif {cfo_neg} dari {len(cfo_q)} kuartal terakhir." + (" Laba yang tidak berubah menjadi kas adalah kelemahan utama." if cfo_neg >= 2 else ""))
    if _ok(gear):
        kual.append(f"[OBS] Gearing (utang berbunga / ekuitas) {gear:.0%}." + (" Tinggi: kenaikan suku bunga langsung menggerus laba." if gear > 1.0 else ""))
    if _ok(m.get("der")):
        kual.append(f"[OBS] DER {m['der']:.2f}x" + (f", current ratio {m['cr']:.2f}" if _ok(m.get("cr")) else "") + ".")
    if _ok(m.get("fcf_ttm")) and m["fcf_ttm"] < 0:
        kual.append("[OBS] Free cash flow 12 bulan negatif: ekspansi dibiayai utang/ekuitas, bukan dari kas operasi.")
    # ---------------- 3. valuasi & skenario
    val = []
    grid = None
    if not rugi:
        val.append(f"[OBS] EPS LTM {eps:,.1f} ({m.get('np_basis', 'LTM')}) → PER {per_now:.1f}x pada harga {harga:,.0f}" + (f"; PBV {pbv_now:.2f}x" if _ok(pbv_now) else "") + ".")
        grid = skenario(eps, per_now, harga)
        g_base = 0.0 if not _ok(g_np) else max(min(g_np * 0.7, 0.20), -0.10)
        per_opt = per_now * 1.25 if per_now < 12 else per_now          # PE expansion moderat (mis. 8x -> 10x), tidak melewati ~12x
        sk_ = [("konservatif", min(g_base, 0.0), per_now), ("dasar (hanya earnings expansion)", g_base, per_now), ("optimis (+ PE expansion moderat)", g_base, min(per_opt, max(12, per_now)))]
        val.append("[EST] Skenario 12 bulan: " + "; ".join(f"{nm} EPS {g:+.0%} × PER {pe:.1f}x ≈ {eps * (1 + g) * pe:,.0f} ({eps * (1 + g) * pe / harga - 1:+.0%})" for nm, g, pe in sk_) + ".")
        val.append("[INF] Kenaikan harga datang dari dua mesin: earnings expansion (EPS naik) dan PE expansion (PER naik). Tanpa salah satunya, harga sulit naik berkelanjutan.")
    else:
        val.append("[OBS] Laba 12 bulan negatif: PER tidak bermakna." + (f" PBV {pbv_now:.2f}x." if _ok(pbv_now) else ""))
    if _ok(dy):
        val.append(f"[OBS] Dividen 12 bulan ≈ yield {dy:.1%}" + (" (≥ 8%: kandidat dividend play)." if dy >= 0.08 else "."))
    # ---------------- 4. label
    if _ok(per_now) and per_now > 50:
        per_tag.append(("NARASI", "PER > 50x: harga digerakkan narasi, bukan kinerja. Bila ikut, tetap rasional dan pantau tesis non-fundamentalnya."))
    if not rugi and _ok(g_np) and g_np >= 0.10 and _ok(per_now) and per_now <= 15:
        per_tag.append(("GROWTH + MURAH", f"laba tumbuh {g_np:.0%} dengan PER {per_now:.1f}x: dua mesin (EPS & PER) berpotensi bekerja bersamaan."))
    elif not rugi and _ok(g_np) and g_np >= 0.10 and _ok(per_now) and per_now <= 50:
        per_tag.append(("GROWTH, VALUASI PENUH", f"laba tumbuh {g_np:.0%} tetapi PER {per_now:.1f}x sudah memasukkan sebagian harapan."))
    if not rugi and _ok(g_np) and g_np < 0.10 and _ok(per_now) and per_now <= 10:
        per_tag.append(("MURAH, MENUNGGU RECOVERY", "kinerja sedang tertekan/flat dan valuasi murah: menarik HANYA bila ada alasan kuat kinerja pulih; butuh kesabaran ekstra."))
    if _ok(dy) and dy >= 0.08:
        per_tag.append(("DIVIDEND PLAY", f"yield {dy:.1%}: imbal hasil tetap ada walau laba flat."))
    if komoditas:
        per_tag.append(("KOMODITAS SIKLIKAL", f"laba mengikuti harga {nama_sek.lower()}: beli saat murah dan sabar menunggu siklus; jangan menilai dari satu kuartal."))
    if holding:
        per_tag.append(("HOLDING: PAKAI NAV", "nilai wajar = NAV × (1 − diskon NAV); NAV dihitung dari harga pasar anak usaha (tidak dihitung aplikasi)."))
    if rugi:
        per_tag.append(("RUGI / TURNAROUND", "belum ada laba: yang dinilai adalah jalur menuju laba (ramp-up, efisiensi) dan pendanaannya."))
    if not per_tag:
        per_tag.append(("BELUM MENARIK", "tidak memenuhi ambang pertumbuhan ≥ 10% maupun diskon + dividen ≥ 8%."))
    # ---------------- 5. tesis satu paragraf
    utama = per_tag[0]
    kalimat = [f"{kode or 'Saham ini'} masuk kategori **{utama[0]}**: {utama[1]}"]
    if cfo_neg >= 2 or (_ok(gear) and gear > 1.0):
        kalimat.append("Catatan keras: " + ("arus kas operasi sering negatif" if cfo_neg >= 2 else "") + (" dan " if cfo_neg >= 2 and _ok(gear) and gear > 1.0 else "")
                       + (f"gearing {gear:.0%}" if _ok(gear) and gear > 1.0 else "") + ".")
    if nama_sek:
        kalimat.append(f"Sebagai emiten {nama_sek.lower()}, pantau data eksternal di bawah karena laba bergantung padanya.")
    kalimat.append("Tesis wajib direview tiap rilis laporan kuartalan.")
    # ---------------- 6. pantauan & rujukan
    rujuk = [dict(nama=R[k][0], url=R[k][1], untuk=nama_sek or "makro") for k in kode_ref] + [dict(nama=R[k][0], url=R[k][1], untuk="makro global/domestik") for k in GLOBAL]
    pantau = [f"Laporan kuartal berikutnya: apakah pendapatan {_p(g_rev)} YoY berlanjut dan margin kotor bertahan di atas {gpm1:.1%}?" if _ok(gpm1) else "Laporan kuartal berikutnya: pertumbuhan pendapatan dan margin.",
              "Arus kas operasi kembali positif dan stabil?" if cfo_neg else "Arus kas operasi tetap positif?",
              "Rencana dividen / buyback dan penggunaan dana (keterbukaan informasi IDX)."]
    if komoditas or nama_sek in ("Perbankan", "Otomotif", "Properti", "Konsumen/ritel"):
        pantau.append(f"Data eksternal {nama_sek.lower()}: " + ", ".join(r_["nama"] for r_ in rujuk if r_["untuk"] == nama_sek))
    return dict(versi=VERSI, sektor=nama_sek, tesis=" ".join(kalimat), label=per_tag, mesin=mesin, kualitas=kual, valuasi=val, grid=grid, per_now=per_now,
                pantau=pantau, rujukan=rujuk, horizon="jangka pendek ±3–6 bulan; jangka panjang > 1 tahun",
                catatan="Narasi otomatis dari laporan keuangan (bukan kutipan dan bukan pendapat pihak mana pun). Skenario harga = EPS × PER, bukan target. Data eksternal tidak diambil aplikasi: buka tautan rujukan.")


def baris(n: dict | None) -> list[str]:
    L = ["## NARASI FUNDAMENTAL (kerangka value-growth; skenario, bukan target)"]
    if not n:
        L.append("N/A: laporan keuangan belum dihitung (Hitung lengkap).")
        return L
    L.append(f"[INF] tesis: {n['tesis'].replace('**', '')}")
    L.append("label: " + "; ".join(f"{a} ({b})" for a, b in n["label"]))
    L += ["### Mesin laba"] + [f"- {x}" for x in n["mesin"]]
    if n["kualitas"]:
        L += ["### Kualitas laba & neraca"] + [f"- {x}" for x in n["kualitas"]]
    L += ["### Valuasi & skenario"] + [f"- {x}" for x in n["valuasi"]]
    L += ["### Pantau (review tiap kuartal)"] + [f"- {x}" for x in n["pantau"]]
    L.append("rujukan eksternal: " + "; ".join(f"{r_['nama']} {r_['url']}" for r_ in n["rujukan"]))
    L.append(f"horizon: {n['horizon']}. Skenario = EPS x PER, bukan target; data eksternal tidak diambil aplikasi.")
    return L
