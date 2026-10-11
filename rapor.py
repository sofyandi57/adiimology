"""Rapor keputusan: hasil nyata dari setiap keputusan yang pernah dicatat aplikasi (tanpa mengandalkan klaim heuristik).

Aturan simulasi (harian, konservatif, SATU-satunya definisi):
  - entry hanya bila zona beli valid; terisi bila low <= entry_hi dalam `hari_isi` hari bursa sesudah tanggal data; harga isi = min(open, entry_hi)
  - setelah terisi: TP1 bila high >= tp1, SL bila low <= sl; bila keduanya di bar yang sama dihitung SL lebih dulu (konservatif)
  - belum selesai dalam `hari_tahan` hari bursa -> BERJALAN
  - R = (keluar - isi) / (isi - sl) dikurangi biaya bolak-balik 0,4% dari harga isi (dalam satuan R)
Ini evaluasi keputusan seandainya dieksekusi mekanis, bukan catatan transaksi Anda.
"""
from __future__ import annotations

import datetime as dt
import math

import pandas as pd

BIAYA = 0.004


def evaluasi(rec: dict, bars: pd.DataFrame, hari_isi: int = 5, hari_tahan: int = 15) -> dict:
    """rec: entry_lo, entry_hi, zona_valid, sl, tp1, tgl_data (date). bars: kolom tgl, open, high, low, close terurut naik."""
    if not rec.get("zona_valid") or not rec.get("entry_hi") or not rec.get("sl") or not rec.get("tp1"):
        return dict(status="TANPA_ZONA")
    if bars is None or bars.empty:
        return dict(status="BELUM_ADA_DATA")
    b = bars[bars["tgl"] > rec["tgl_data"]].reset_index(drop=True)
    if b.empty:
        return dict(status="BELUM_ADA_DATA")
    eh, sl, tp = float(rec["entry_hi"]), float(rec["sl"]), float(rec["tp1"])
    if not (sl < eh < tp):
        return dict(status="TANPA_ZONA")
    isi = None
    for i in range(min(hari_isi, len(b))):
        if b["low"].iloc[i] <= eh:
            isi, i0, harga_isi = i, i, min(float(b["open"].iloc[i]), eh)
            break
    if isi is None:
        return dict(status="TIDAK_TERISI" if len(b) >= hari_isi else "MENUNGGU_ISI")
    risk = harga_isi - sl
    if risk <= 0:
        return dict(status="TANPA_ZONA")
    for j in range(i0, min(i0 + hari_tahan, len(b))):
        lo, hi = float(b["low"].iloc[j]), float(b["high"].iloc[j])
        if lo <= sl:                                   # SL lebih dulu bila satu bar menyentuh keduanya (dan bar isi: low<=SL)
            return dict(status="SL", r=(sl - harga_isi) / risk - BIAYA * harga_isi / risk, hari=j - i0 + 1, isi=harga_isi)
        if hi >= tp:
            return dict(status="TP1", r=(tp - harga_isi) / risk - BIAYA * harga_isi / risk, hari=j - i0 + 1, isi=harga_isi)
    sisa = len(b) - i0
    return dict(status="BERJALAN" if sisa < hari_tahan else "KEDALUWARSA", isi=harga_isi,
                r=(float(b["close"].iloc[-1]) - harga_isi) / risk - BIAYA * harga_isi / risk)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def ringkas(hasil: list[dict]) -> pd.DataFrame:
    """hasil: [{keputusan, status, r}] -> tabel per keputusan."""
    df = pd.DataFrame(hasil)
    if df.empty:
        return pd.DataFrame()
    rows = []
    for kep, g in df.groupby("keputusan"):
        tp, sl = int((g["status"] == "TP1").sum()), int((g["status"] == "SL").sum())
        n = tp + sl
        lo, hi = wilson(tp, n)
        rs = g.loc[g["status"].isin(["TP1", "SL"]), "r"]
        rows.append({"Keputusan": kep, "Dicatat": len(g), "Terisi": int(g["status"].isin(["TP1", "SL", "BERJALAN", "KEDALUWARSA"]).sum()),
                     "TP1": tp, "SL": sl, "Menang % (TP1/(TP1+SL))": (tp / n if n else float("nan")), "CI95 bawah": lo, "CI95 atas": hi,
                     "Rata-rata R": float(rs.mean()) if len(rs) else float("nan"),
                     "Cukup data?": "ya (n>=30)" if n >= 30 else f"belum (n={n}, butuh >=30)"})
    return pd.DataFrame(rows)
