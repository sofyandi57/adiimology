"""Satu sumber angka risiko, validasi laporan, metadata kualitas data, dan daftar batasan endpoint.

Tujuan: setiap angka di laporan TXT bisa ditelusuri dan direkonsiliasi; bila tidak cocok, laporan ditandai GAGAL.
Edukasi, bukan ajakan transaksi.
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd

LOT = 100
FEE_BELI, FEE_JUAL = 0.0015, 0.0025
RUMUS_POSISI = ("lot = floor(min(modal*risiko / (100*(entry*(1+fee_beli) - stop*(1-fee_jual))), modal / (100*entry*(1+fee_beli)))); "
                "rugi_di_stop = lot*100*(entry*(1+fee_beli) - stop*(1-fee_jual)); laba_di_target = lot*100*(target*(1-fee_jual) - entry*(1+fee_beli)); "
                "gap = stop dilewati dan keluar di stop*(1-5%)")


# ------------------------------------------------------------------- posisi
def hitung_posisi(modal: float, risiko_pct: float, entry: float, stop: float, target: float | None,
                  fee_b: float = FEE_BELI, fee_j: float = FEE_JUAL, gap: float = 0.05) -> dict:
    """Ukuran posisi & risiko nominal. SATU-SATUNYA fungsi yang dipakai laporan untuk lot / rugi / laba."""
    base = dict(entry=entry, stop=stop, target=target, modal=modal, risiko_pct=risiko_pct, fee_b=fee_b, fee_j=fee_j)
    if not (entry and stop and entry > stop > 0):
        return dict(base, lot=0, alasan="stop tidak di bawah entry")
    per_lot = LOT * (entry * (1 + fee_b) - stop * (1 - fee_j))
    anggaran = modal * risiko_pct
    lot_risiko = int(math.floor(anggaran / per_lot))
    lot_modal = int(math.floor(modal / (LOT * entry * (1 + fee_b))))
    lot = max(min(lot_risiko, lot_modal), 0)
    rugi = lot * per_lot
    laba = lot * LOT * (target * (1 - fee_j) - entry * (1 + fee_b)) if target else float("nan")
    return dict(base, lot=lot, lot_risiko=lot_risiko, lot_modal=lot_modal, anggaran_risiko=anggaran,
                modal_pakai=lot * LOT * entry * (1 + fee_b), rugi_stop=rugi, rugi_stop_pct_modal=rugi / modal if modal else float("nan"),
                laba_target=laba, rr_bersih=(laba / rugi) if rugi > 0 and not math.isnan(laba) else float("nan"),
                rugi_gap=lot * LOT * (entry * (1 + fee_b) - stop * (1 - gap) * (1 - fee_j)), per_lot_rugi=per_lot,
                pengikat="risiko" if lot_risiko <= lot_modal else "modal", alasan="")


def skenario_posisi(r, rk: dict | None) -> list[dict]:
    """Dua skenario entry dengan stop yang SAMA dengan yang ditampilkan (SL teknikal), bukan SL window Adimology."""
    if not rk:
        return []
    p, out = r.params, []
    if rk["zona_valid"]:
        out.append(dict(nama="entry di area beli (batas atas)", **hitung_posisi(p.modal, p.risiko_pct, rk["entry_hi"], rk["sl"], rk["target"], p.fee_beli, p.fee_jual)))
    out.append(dict(nama="entry di harga sekarang", **hitung_posisi(p.modal, p.risiko_pct, rk["harga"], rk["sl"], rk["target"], p.fee_beli, p.fee_jual)))
    return out


# ----------------------------------------------------------------- validasi
def _c(id_, status, detail):
    return dict(id=id_, status=status, detail=detail)


def validasi(r, rk: dict | None, f: dict | None, posisi: list[dict], waktu: dt.datetime | None = None) -> dict:
    """Pemeriksaan otomatis. GAGAL = angka tidak konsisten/tak masuk akal (laporan tidak boleh dipakai)."""
    waktu = waktu or dt.datetime.now()
    cek = []
    if rk and r.harga and abs(rk["harga"] / r.harga - 1) > 0.01:
        cek.append(_c("harga_konsisten", "PERINGATAN", f"harga Adimology {r.harga:,.0f} vs harga teknikal {rk['harga']:,.0f}: sumber/tanggal berbeda"))
    if rk:
        e = (rk["entry_lo"] + rk["entry_hi"]) / 2 if rk["zona_valid"] else rk["harga"]
        urut = rk["sl"] < min(e, rk["harga"]) and e < rk["tp1"] <= rk["tp2"]
        cek.append(_c("urutan_level", "OK" if urut else "GAGAL" if not (rk["sl"] < e) else "PERINGATAN",
                      f"SL {rk['sl']:,.0f} < entry {e:,.0f} < TP1 {rk['tp1']:,.0f} <= TP2 {rk['tp2']:,.0f}"
                      + ("" if urut else "; harga kini di atas TP1/entry di luar urutan")))
        rr = (rk["target"] - e) / max(e - rk["sl"], 1e-9)
        cek.append(_c("rr_rekonsiliasi", "OK" if abs(rr - rk["rr"]) < 0.01 else "GAGAL", f"RR dihitung ulang {rr:.2f} vs laporan {rk['rr']:.2f}"))
        tgl = rk["tgl"]
        umur = (waktu.date() - tgl).days if isinstance(tgl, dt.date) else None
        cek.append(_c("kesegaran_data", "OK" if umur is not None and umur <= 4 else "PERINGATAN",
                      f"harga terakhir {tgl} ({umur} hari sebelum snapshot)" + ("" if umur is not None and umur <= 4 else "; data mungkin usang")))
        ma_ok = all((v is None) or v > 0 for v in rk["ma"].values()) and rk["atr"] > 0 and rk["harga"] > 0
        cek.append(_c("satuan_teknikal", "OK" if ma_ok else "GAGAL", f"MA/ATR/harga positif; ATR {rk['atr'] / rk['harga']:.1%} dari harga"))
    for ps in posisi:
        if not ps.get("lot"):
            cek.append(_c(f"posisi[{ps['nama']}]", "PERINGATAN", f"lot 0: {ps.get('alasan') or 'risiko/modal tidak cukup'}"))
            continue
        rugi_manual = ps["lot"] * LOT * (ps["entry"] - ps["stop"]) + ps["lot"] * LOT * (ps["entry"] * ps["fee_b"] + ps["stop"] * ps["fee_j"])
        cocok = abs(rugi_manual - ps["rugi_stop"]) < 1.0
        dalam = ps["rugi_stop"] <= ps["anggaran_risiko"] + 1e-6
        stop_sama = rk is None or abs(ps["stop"] - rk["sl"]) < 1e-9
        cek.append(_c(f"posisi[{ps['nama']}]", "OK" if cocok and dalam and stop_sama else "GAGAL",
                      f"{ps['lot']} lot: rugi di stop Rp{ps['rugi_stop']:,.0f} (selisih harga {ps['lot'] * LOT * (ps['entry'] - ps['stop']):,.0f} + fee "
                      f"{rugi_manual - ps['lot'] * LOT * (ps['entry'] - ps['stop']):,.0f}) <= anggaran Rp{ps['anggaran_risiko']:,.0f}; stop sama dengan SL laporan: {stop_sama}"))
    if f is not None:
        hil = f["kualitas"]["hilang"]
        cek.append(_c("fundamental_lengkap", "OK" if not hil else "PERINGATAN", "data inti lengkap" if not hil else f"hilang: {', '.join(hil)}"))
        basis = f["ltm"]["ni"][1]
        cek.append(_c("basis_laba", "OK" if basis != "YTD disetahunkan" else "PERINGATAN", f"laba bersih basis: {basis}"))
        nm = f["ltm"]["norm"]
        if nm.get("ok") and nm["outliers"]:
            cek.append(_c("laba_satu_kali", "PERINGATAN", f"kuartal ekstrem {', '.join(nm['outliers'])}: PER/ROE reported tidak andal"))
    cek.append(_c("lot_per_papan", "OK" if r.lot_per_papan and r.lot_per_papan > 0 else "GAGAL", f"lot per papan {r.lot_per_papan:,.0f} ({r.lpp_sumber[:24]})"))
    gagal = [c for c in cek if c["status"] == "GAGAL"]
    perh = [c for c in cek if c["status"] == "PERINGATAN"]
    status = "VALID" if not gagal and not perh else "VALID DENGAN PERINGATAN" if not gagal else "GAGAL"
    return dict(status=status, cek=cek, n_gagal=len(gagal), n_peringatan=len(perh))


# ------------------------------------------------------------ aksi korporasi
def parse_aksi(raw) -> list[dict]:
    data = raw.get("data", []) if isinstance(raw, dict) else (raw or [])
    out = []
    for x in data:
        pl = x.get("payload") if isinstance(x, dict) else None
        ring = ", ".join(f"{k}={v}" for k, v in list((pl or {}).items())[:6])[:140]
        out.append(dict(type=str(x.get("type", "")), ringkas=ring))
    return out


# ------------------------------------------------------------------ batasan
STRUKTURAL = [
    "[STRUKTURAL] Tidak tersedia di endpoint: proyeksi/konsensus analis, multiple pembanding sektor, free float, ruang kepemilikan asing, "
    "riwayat dividen terstruktur (hanya kalender aksi korporasi), status margin/suspensi realtime di luar notasi khusus. Nilai wajar DCF/NAV tidak dihitung.",
    "[STRUKTURAL] Keystat Invezgo ditandai 'sedang diaktualisasi' oleh penyedia: hanya jumlah saham dipakai; valuasi dihitung sendiri dari laporan keuangan.",
    "[STRUKTURAL] Broker summary EOD (update 18:00 WIB): hari berjalan belum termasuk; satu kode broker = banyak nasabah, pihak sebenarnya tidak terlihat; "
    "pasar negosiasi hanya agregat per broker.",
    "[STRUKTURAL] Orderbook historis hanya sejak 2026-03-25 dan berupa snapshot: antrean palsu/di-reload tidak terbaca. Aliran intraday broker tidak tersedia.",
    "[STRUKTURAL] SID dari KSEI bulanan dan terlambat; harga tidak disesuaikan untuk dividen/rights (hanya split yang terdeteksi otomatis).",
    "[STRUKTURAL] Biaya, slippage, dan pajak aktual tidak diketahui: fee diasumsikan beli 0,15% dan jual 0,25%; likuiditas keluar di harga stop tidak dijamin (gap).",
    "[STRUKTURAL] Inventory broker = net kumulatif harian hanya untuk top-15 broker: broker di luar daftar tidak terlihat; kelompok broker adalah PROKSI, bukan pihak tunggal; "
    "tidak ada data per nasabah, posisi aktual yang masih dipegang, maupun transaksi antar-nasabah satu broker (cross).",
    "[STRUKTURAL] Kinerja prediktif per broker/pola (broker informatif vs sekadar besar) dan nilai tambah bandarmologi dibanding harga-volume saja BELUM diuji (butuh backtest out-of-sample).",
    "[STRUKTURAL] Backtest: 4 saham, satu periode, tanpa walk-forward/out-of-sample; kerangka bukti bandarmologi dan aturan Hengky belum diuji.",
]


def batasan(status: dict, r, rk, f, ovn, alr, b200, hk_, ritel) -> list[str]:
    """Kekurangan data/endpoint PADA SNAPSHOT INI + batasan struktural. status[k] = ('ok'|'err', payload) dari proses Hitung."""
    out = []
    nama = {"tek": "analisa teknikal", "fund": "laporan keuangan", "ovn": "statistik BPJP/BSJP (candle 30 menit)", "aliran": "aliran asing/nego/konsistensi/afiliasi",
            "b200": "bandar jangka panjang", "hengky": "aturan Hengky", "dinamis": "kerangka bukti bandarmologi (inventory broker)"}
    for k, label in nama.items():
        v = status.get(k)
        if v is not None and v[0] == "err":
            out.append(f"[GAGAL] {label}: {str(v[1])[:160]}")
        elif v is None and k in ("aliran", "b200"):
            out.append(f"[DINONAKTIFKAN] {label} tidak dijalankan (opsi sidebar).")
    if rk is not None:
        if rk["ma"].get(200) is None:
            out.append("[TERBATAS] Data harga < 200 hari bursa: MA200 dan konteks tren besar tidak tersedia (saham baru/data pendek).")
        if rk.get("split"):
            out.append("[KOREKSI] Harga disesuaikan untuk: " + "; ".join(f"{e['jenis']} ({e['tgl']})" for e in rk["split"]))
    if r.ob is None or r.ob.get("n_levels", 0) <= 4:
        out.append("[TERBATAS] Orderbook hanya 1-2 level (pasar tutup/tipis): lot per papan dan bacaan bid-offer tidak representatif.")
    if f is not None:
        q = f["kualitas"]
        for h in q["hilang"]:
            out.append(f"[HILANG] Laporan keuangan: {h} tidak tersedia/tidak terbaca; metrik turunannya tidak dihitung.")
        for nm in ("IS", "BS", "CF"):
            if nm not in q["fy"]:
                out.append(f"[TERBATAS] Data tahunan (FY) {nm} tidak tersedia: Q4 tidak dapat diturunkan, LTM memakai pendekatan cadangan.")
        if q["ytd_kolom"]:
            out.append(f"[TERBATAS] Kuartal sebelumnya tidak dilaporkan untuk {', '.join(q['ytd_kolom'])}: nilai kumulatif (*), bukan kuartal murni.")
        if not f.get("profil"):
            out.append("[HILANG] Profil emiten (papan, notasi khusus, administrasi) tidak terambil.")
        if f.get("aksi_status", "").startswith("TIDAK DIUKUR"):
            out.append("[HILANG] Kalender aksi korporasi gagal diambil: dividen/rights/RUPS mendatang tidak diketahui.")
    elif status.get("fund") is None:
        out.append("[HILANG] Laporan keuangan tidak tersedia.")
    if hk_ is not None and not hk_.get("sid"):
        out.append("[HILANG] SID (jumlah investor) tidak tersedia/ tidak cukup riwayat.")
    if alr is not None and alr.get("afiliasi") is None:
        out.append("[HILANG] Afiliasi pemegang saham >=5% tidak tersedia (endpoint gagal atau tidak ada di paket API).")
    if ovn is None and status.get("ovn") is not None and status["ovn"][0] == "ok":
        out.append("[TERBATAS] Statistik intraday kosong.")
    return out + STRUKTURAL
