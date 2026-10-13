"""Situasi khusus (event-driven): backdoor listing, perubahan pengendali, restrukturisasi modal.

Backdoor listing = pihak baru mengambil alih emiten yang sudah tercatat (sering emiten kecil dengan bisnis lemah), lalu memasukkan
bisnis/aset baru (penambahan modal dengan/tanpa HMETD, perubahan kegiatan usaha, akuisisi). Harga digerakkan oleh NASIB TRANSAKSI,
bukan oleh laba, chart, atau aliran broker. Karena itu:
  - tidak dinilai 'bagus/jelek' lewat rata-rata skor; tampil sebagai kartu Situasi Khusus + gate EVENT_DRIVEN_WARNING (membatasi BELI);
  - selalu ada PERINGATAN dan POSISI MAKSIMAL (% modal) karena risiko gap / batal transaksi tidak dapat dibatasi stop loss.
Deteksi hanya dari data yang tersedia: data insider (perubahan pengendali, harga transaksi), kalender aksi korporasi (agenda RUPS, rights,
waran) dan catatan manual pengguna (term sheet, tenggat, rekam jejak pihak) karena keterbukaan/berita Invezgo sering kosong.
"""
from __future__ import annotations

import datetime as dt
import math
import re

import pandas as pd

VERSI = "situasi-2026.10.1"
KATEGORI = {
    "PMTHMETD / private placement": r"pmthmetd|tanpa hak memesan|private placement",
    "rights issue (HMETD)": r"\bpmhmetd\b|(?<!tanpa )hak memesan efek terlebih dahulu|\bhmetd\b|right[s]? issue",
    "peningkatan modal dasar": r"modal dasar",
    "perubahan / penghentian kegiatan usaha": r"kegiatan usaha|discontinue|penghentian.{0,40}(usaha|operasional)|segmen usaha",
    "akuisisi / pengambilalihan / merger": r"akuisisi|pengambilalihan|penggabungan|peleburan|merger|transaksi material",
    "buyback": r"buyback|pembelian kembali",
    "perubahan pengurus": r"direksi|komisaris",
}
RESTRUKTURISASI = ("PMTHMETD / private placement", "rights issue (HMETD)", "peningkatan modal dasar", "perubahan / penghentian kegiatan usaha",
                   "akuisisi / pengambilalihan / merger")
JENIS_MANUAL = ("Backdoor / perubahan pengendali", "Term sheet / MoU akuisisi", "Rights issue / private placement", "Rekam jejak pihak (tata kelola)", "Lainnya")
TAHAP = {1: "rumor / term sheet (catatan manual)", 2: "perubahan pengendali", 3: "harga transaksi jauh dari pasar", 4: "agenda restrukturisasi di RUPS",
         5: "eksekusi (rights/waran/konversi)"}


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _d(x):
    try:
        return pd.to_datetime(x, utc=True).tz_convert("Asia/Jakarta").date()
    except Exception:
        return None


def _close_pada(ohlc: pd.DataFrame | None, tgl: dt.date | None) -> float:
    if ohlc is None or ohlc.empty or tgl is None:
        return float("nan")
    o = ohlc[pd.to_datetime(ohlc["tgl"]).dt.date <= tgl]
    return float(o["close"].iloc[-1]) if len(o) else float("nan")


def perubahan_pengendali(rows: list | None, ohlc: pd.DataFrame | None = None, today: dt.date | None = None) -> dict:
    """Data insider 365 hari -> pengendali keluar, pihak baru melewati 20-25%, harga transaksi vs pasar, pola beli murah-jual mahal."""
    today = today or dt.date.today()
    keluar, masuk, flip = [], [], []
    trans = {}
    for x in rows or []:
        if not isinstance(x, dict):
            continue
        tg = _d(x.get("date"))
        if tg is None or (today - tg).days > 365:
            continue
        nm, badge = str(x.get("name", ""))[:40], str(x.get("badge", "")).upper()
        prev, nxt = _f(x.get("prev_percent")), _f(x.get("next_percent"))
        sub = [dict(tgl=_d(s.get("date")), harga=_f(s.get("price")), status=str(s.get("status", "")).lower(), lembar=_f(s.get("value"))) for s in (x.get("subrow") or [])]
        for s in sub:
            trans.setdefault(nm, []).append(s)
        harga_tx = next((s["harga"] for s in sub if not math.isnan(s["harga"])), float("nan"))
        pasar = _close_pada(ohlc, sub[0]["tgl"] if sub else tg)
        rec = dict(tgl=str(tg), nama=nm, peran=badge.strip("{}") or "-", dari_pct=prev, ke_pct=nxt, harga_transaksi=harga_tx, harga_pasar=pasar,
                   selisih=(harga_tx / pasar - 1) if pasar and not math.isnan(pasar) and not math.isnan(harga_tx) else float("nan"))
        if "PENGENDALI" in badge and prev >= 5 and nxt < 1:
            keluar.append(rec)
        elif not math.isnan(nxt) and nxt >= 20 and (math.isnan(prev) or prev < 20 or (nxt - prev) >= 10):
            masuk.append(rec)
    for nm, tx in trans.items():
        beli = [t for t in tx if t["status"].startswith("b") and t["tgl"]]
        jual = [t for t in tx if t["status"].startswith("s") and t["tgl"]]
        for b in beli:
            for j in jual:
                if 0 <= (j["tgl"] - b["tgl"]).days <= 45 and b["harga"] > 0 and j["harga"] >= 1.5 * b["harga"]:
                    flip.append(dict(nama=nm, beli_tgl=str(b["tgl"]), beli_harga=b["harga"], jual_tgl=str(j["tgl"]), jual_harga=j["harga"], lembar=abs(j["lembar"])))
                    break
    return dict(ada=bool(keluar or masuk), keluar=keluar, masuk=masuk, flip=flip)


def agenda(aksi: list | None, today: dt.date | None = None, lalu_hari: int = 180) -> list[dict]:
    """Agenda/aksi kalender relevan (mendatang atau <= lalu_hari terakhir) beserta kategori restrukturisasi yang terdeteksi."""
    today = today or dt.date.today()
    out = []
    for a in aksi or []:
        tg = a.get("tgl")
        if tg is not None and (today - tg).days > lalu_hari:
            continue
        teks = str(a.get("teks") or a.get("ringkas") or "").lower()
        kat = [k for k, pola in KATEGORI.items() if re.search(pola, teks)]
        if "PMTHMETD / private placement" in kat and "rights issue (HMETD)" in kat and not re.search(r"\bpmhmetd\b|(?<!tanpa )hak memesan efek terlebih dahulu", teks):
            kat.remove("rights issue (HMETD)")
        out.append(dict(tgl=str(tg) if tg else "-", type=a.get("type", ""), kategori=kat, mendatang=bool(tg and tg >= today), teks=teks[:220]))
    return out


def dilusi_waran(aksi: list | None, today: dt.date | None = None) -> float | None:
    """Sisa waran / total saham dari konversi waran TERBARU di kalender (payload: 'Sisa Waran N', TotalShare)."""
    kand = []
    for a in aksi or []:
        if str(a.get("type", "")).upper() != "CONVERTION":
            continue
        teks = str(a.get("teks", ""))
        m = re.search(r"sisa waran\s*([\d,\.]+)", teks.lower())
        tot = [float(t) for t in teks.split() if t.isdigit() and len(t) >= 8]
        if m and tot:
            sisa = float(m.group(1).replace(",", "").replace(".", ""))
            kand.append((a.get("tgl") or dt.date.min, sisa / tot[-1]))
    return max(kand)[1] if kand else None


def deteksi(konteks: dict | None, aksi: list | None, ohlc: pd.DataFrame | None, catatan: list | None, harga: float, modal: float,
            maks_pct: float = 0.02, fee_beli: float = 0.0015, today: dt.date | None = None) -> dict:
    today = today or dt.date.today()
    k = konteks or {}
    pp = perubahan_pengendali(k.get("insider_raw"), ohlc, today)
    ag = agenda(aksi, today)
    kat_all = sorted({c for a in ag for c in a["kategori"]})
    restr = [c for c in kat_all if c in RESTRUKTURISASI]
    eksekusi = [a for a in ag if str(a["type"]).upper() in ("RIGHT", "WARRANT", "CONVERTION")]
    manual = [c for c in (catatan or []) if c.get("jenis") in JENIS_MANUAL[:3]]
    tahap = []
    if manual:
        tahap.append(1)
    if pp["ada"]:
        tahap.append(2)
    if any(not math.isnan(x["selisih"]) and x["selisih"] <= -0.5 for x in pp["keluar"] + pp["masuk"]):
        tahap.append(3)
    if len(restr) >= 2 or (restr and pp["ada"]):
        tahap.append(4)
    if eksekusi and (pp["ada"] or restr):
        tahap.append(5)
    aktif = bool(tahap)
    bukti, flags, inval, tgl_kunci = [], [], [], []
    for x in pp["keluar"]:
        bukti.append(f"[OBS] {x['tgl']}: pengendali {x['nama']} {x['dari_pct']:.2f}% -> {x['ke_pct']:.2f}% di harga {x['harga_transaksi']:,.0f}"
                     + (f" (harga pasar {x['harga_pasar']:,.0f}; {x['selisih']:+.0%})" if not math.isnan(x["selisih"]) else ""))
    for x in pp["masuk"]:
        bukti.append(f"[OBS] {x['tgl']}: {x['nama']} naik {x['dari_pct']:.2f}% -> {x['ke_pct']:.2f}% di harga {x['harga_transaksi']:,.0f}"
                     + (f" (harga pasar {x['harga_pasar']:,.0f}; {x['selisih']:+.0%})" if not math.isnan(x["selisih"]) else ""))
    for f_ in pp["flip"]:
        flags.append(f"TATA KELOLA: {f_['nama']} membeli di {f_['beli_harga']:,.0f} ({f_['beli_tgl']}) lalu menjual {f_['lembar']:,.0f} lembar di {f_['jual_harga']:,.0f} ({f_['jual_tgl']})")
    for a in ag:
        if a["kategori"]:
            bukti.append(f"[OBS] kalender {a['type']} {a['tgl']}: " + ", ".join(a["kategori"]))
            if a["mendatang"]:
                tgl_kunci.append(f"{a['tgl']} {a['type']}: " + ", ".join(a["kategori"]))
                if any(c in RESTRUKTURISASI for c in a["kategori"]):
                    inval.append(f"RUPS {a['tgl']} menolak/menunda agenda restrukturisasi")
    dw = dilusi_waran(aksi, today)
    if dw is not None:
        bukti.append(f"[OBS] sisa waran ≈ {dw:.1%} dari saham beredar (dilusi bertahap bila dikonversi)")
    for c in manual + [c for c in (catatan or []) if c not in manual]:
        bukti.append(f"[MANUAL] {c.get('jenis')}: {c.get('teks')}" + (f" (tenggat {c['tenggat']})" if c.get("tenggat") else ""))
        if c.get("tenggat"):
            tgl_kunci.append(f"{c['tenggat']} tenggat: {c.get('jenis')}")
            inval.append(f"tenggat {c['tenggat']} lewat tanpa kelanjutan transaksi (risiko batal; harga dapat turun tajam)")
        if c.get("jenis") == "Rekam jejak pihak (tata kelola)":
            flags.append("TATA KELOLA (manual): " + str(c.get("teks"))[:160])
    if aktif:
        inval += ["transaksi pengambilalihan/akuisisi diumumkan batal atau tertunda tanpa kepastian",
                  "penambahan modal diumumkan jauh di bawah harga pasar tanpa kejelasan aset/bisnis yang masuk"]
    lot = int(math.floor(modal * maks_pct / (100 * harga * (1 + fee_beli)))) if harga and harga > 0 else 0
    posisi = dict(maks_pct=maks_pct, lot=lot, nilai=lot * 100 * harga * (1 + fee_beli),
                  aturan="ukur posisi dari kerugian TOTAL yang siap ditanggung (anggap seluruh posisi bisa hilang/gap), bukan dari jarak stop loss")
    tt = max(tahap) if tahap else None
    return dict(aktif=aktif, versi=VERSI, jenis="BACKDOOR / PERUBAHAN PENGENDALI" if (pp["ada"] or manual) else ("RESTRUKTURISASI MODAL" if restr else None),
                tahap=sorted(tahap), tahap_teks="; ".join(TAHAP[t] for t in sorted(tahap)), tahap_tertinggi=tt, kategori=kat_all, bukti=bukti, flags=flags,
                tanggal_kunci=tgl_kunci, invalidasi=list(dict.fromkeys(inval)), posisi=posisi, dilusi_waran=dw, pengendali=pp, agenda=ag,
                catatan_data="Keterbukaan & berita Invezgo sering kosong; term sheet/MoU dan rekam jejak pihak hanya terbaca bila Anda mencatatnya manual.")


def baris(sx: dict | None) -> list[str]:
    L = ["## SITUASI KHUSUS (event-driven: backdoor / perubahan pengendali / restrukturisasi modal)"]
    if not sx or not sx.get("aktif"):
        L.append("[OBS] tidak terdeteksi dari data insider 365 hari, kalender aksi, dan catatan manual." + (f" {sx['catatan_data']}" if sx else ""))
        return L
    L.append(f"[DEC] {sx['jenis']} · tahap: {sx['tahap_teks']} · SPEKULATIF (harga digerakkan nasib transaksi; skor biasa kurang relevan)")
    L.append(f"[DEC] posisi maksimal {sx['posisi']['maks_pct']:.1%} modal ≈ {sx['posisi']['lot']:,} lot (Rp{sx['posisi']['nilai']:,.0f}); {sx['posisi']['aturan']}")
    L += [f"- {x}" for x in sx["bukti"]]
    L += [f"- (bendera) {x}" for x in sx["flags"]]
    L += [f"- (tanggal kunci) {x}" for x in sx["tanggal_kunci"]]
    L += [f"- (invalidasi) {x}" for x in sx["invalidasi"]]
    L.append(f"catatan data: {sx['catatan_data']}")
    return L
