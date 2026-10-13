"""Konteks pasar & risiko di luar harga/broker saham itu sendiri: IHSG, notasi khusus, keterbukaan/berita, insider, KSEI, peringkat pasar.

Setiap bagian gagal-aman: bila endpoint kosong/berubah bentuk, hasilnya 'na' dengan alasan (tidak mematikan analisis, tidak diisi tebakan).
Bentuk respons ditulis dari contoh nyata Invezgo (chart indeks, insider, KSEI, top/*); notasi khusus, disclosure, news belum punya contoh
respons terverifikasi, sehingga diparse defensif dan ditampilkan apa adanya.
"""
from __future__ import annotations

import datetime as dt
import json
import re

import numpy as np
import pandas as pd

KATA_RISIKO = ("pkpu", "pailit", "suspen", "delisting", "gagal bayar", "penghentian sementara", "hmetd", "right issue", "rights issue",
               "private placement", "go private", "tender offer", "penawaran tender", "buyback", "pembelian kembali", "restrukturisasi", "unusual market")


def _na(alasan: str) -> dict:
    return dict(status="na", alasan=alasan)


def _ok(**kw) -> dict:
    return dict(status="ok", **kw)


# ------------------------------------------------------------------ IHSG
def ihsg(raw) -> dict:
    df = pd.DataFrame(raw if isinstance(raw, list) else [])
    if df.empty or "close" not in df:
        return _na("data indeks kosong")
    c = pd.to_numeric(df["close"], errors="coerce").dropna().reset_index(drop=True)
    if len(c) < 25:
        return _na(f"data indeks hanya {len(c)} bar")
    ma20 = float(c.tail(20).mean())
    ma50 = float(c.tail(50).mean()) if len(c) >= 50 else None
    last = float(c.iloc[-1])
    ret20 = last / float(c.iloc[-21]) - 1
    if last < ma20 and (ma50 is None or ma20 < ma50) and ret20 < 0:
        st_ = "LEMAH"
    elif last > ma20 and (ma50 is None or ma20 > ma50) and ret20 > 0:
        st_ = "KONDUSIF"
    else:
        st_ = "NETRAL"
    return _ok(kondisi=st_, last=last, ma20=ma20, ma50=ma50, ret20=ret20, tgl=str(df["date"].iloc[-1])[:10] if "date" in df else None)


# ------------------------------------------------------------ notasi khusus
def notasi(raw, kode: str) -> dict:
    """Cari entri milik kode di respons /analysis/notation (bentuk belum terverifikasi: diparse defensif)."""
    if not raw:
        return _na("respons notasi kosong")
    hit: list = []

    def walk(x):
        if isinstance(x, dict):
            c = str(x.get("code") or x.get("kode") or x.get("stock") or "").upper()
            if c == kode:
                hit.append({k: v for k, v in x.items() if k not in ("logo",)})
                return
            for k, v in x.items():
                if str(k).upper() == kode:
                    hit.append({"data": v})
                else:
                    walk(v)
        elif isinstance(x, list):
            for i in x:
                walk(i)
    walk(raw)
    if not hit:
        return _ok(ada=False, teks=[], catatan="kode tidak tercantum di daftar notasi khusus BEI")
    return _ok(ada=True, teks=[json.dumps(h, ensure_ascii=False, default=str)[:220] for h in hit[:4]])


# ---------------------------------------------------- keterbukaan & berita
def _judul(x: dict) -> str:
    for k in ("title", "judul", "headline", "subject", "name", "h4"):
        if x.get(k):
            return re.sub(r"\s+", " ", str(x[k]))[:140]
    return re.sub(r"\s+", " ", json.dumps(x, ensure_ascii=False, default=str))[:140]


def _tgl(x: dict):
    for k in ("date", "tanggal", "published_at", "created_at", "publishedAt"):
        if x.get(k):
            t = pd.to_datetime(x[k], utc=True, errors="coerce")
            if pd.notna(t):
                return t.tz_convert("Asia/Jakarta").date()
    return None


def kabar(raw, hari: int = 14, today: dt.date | None = None) -> dict:
    data = raw.get("data", []) if isinstance(raw, dict) else (raw or [])
    if not isinstance(data, list):
        return _na("bentuk respons tidak dikenali")
    today = today or dt.date.today()
    items = []
    for x in data:
        if not isinstance(x, dict):
            continue
        t = _tgl(x)
        if t is None or (today - t).days <= hari:
            j = _judul(x)
            items.append(dict(tgl=str(t) if t else "-", judul=j, risiko=[k for k in KATA_RISIKO if k in j.lower()]))
    return _ok(n=len(items), items=items[:8], ada_kata_risiko=any(i["risiko"] for i in items))


# ------------------------------------------------------------------ insider
def insider(raw, hari: int = 120, today: dt.date | None = None) -> dict:
    data = raw.get("data", []) if isinstance(raw, dict) else (raw or [])
    if not isinstance(data, list):
        return _na("bentuk respons tidak dikenali")
    today = today or dt.date.today()
    rows, beli, jual = [], 0.0, 0.0
    for x in data:
        if not isinstance(x, dict):
            continue
        t = _tgl(x)
        if t is not None and (today - t).days > hari:
            continue
        val = 0.0
        for s in x.get("subrow") or []:
            try:
                val += float(s.get("value", 0))        # nilai sudah bertanda (Sell negatif)
            except (TypeError, ValueError):
                pass
        ch = x.get("change")
        arah = "BELI" if (isinstance(ch, (int, float)) and ch > 0) else "JUAL" if (isinstance(ch, (int, float)) and ch < 0) else "-"
        if val > 0:
            beli += val
        elif val < 0:
            jual += -val
        rows.append(dict(tgl=str(t) if t else "-", nama=str(x.get("name", ""))[:40], peran=str(x.get("badge", "")).strip("{}"), arah=arah,
                         ubah_pp=ch, tujuan=str(x.get("purpose", ""))[:20]))
    pengendali_jual = [r for r in rows if r["arah"] == "JUAL" and ("PENGENDALI" in r["peran"].upper())]
    return _ok(n=len(rows), rows=rows[:6], nilai_beli=beli, nilai_jual=jual, pengendali_jual=bool(pengendali_jual), hari=hari)


# --------------------------------------------------------------------- KSEI
def ksei(raw) -> dict:
    data = raw.get("data", []) if isinstance(raw, dict) else (raw or [])
    rows = []
    for x in data if isinstance(data, list) else []:
        try:
            f_ = sum(float(v) for k, v in x.items() if k.startswith("foreign_"))
            l_ = sum(float(v) for k, v in x.items() if k.startswith("local_"))
        except (TypeError, ValueError, AttributeError):
            continue
        if f_ + l_ > 0:
            rows.append(dict(tgl=str(x.get("date", ""))[:10], asing_pct=f_ / (f_ + l_), asing=f_, lokal=l_))
    if len(rows) < 2:
        return _na("data KSEI kurang dari 2 bulan")
    d = rows[-1]["asing_pct"] - rows[0]["asing_pct"]
    return _ok(rows=rows, arah="NAIK" if d > 0.002 else "TURUN" if d < -0.002 else "DATAR", ubah_pp=d * 100)


# ------------------------------------------------------------ peringkat pasar
def peringkat(raw_by_name: dict, kode: str) -> dict:
    out = []
    for nama, raw in raw_by_name.items():
        if not isinstance(raw, dict):
            continue
        for sisi, lst in raw.items():
            if isinstance(lst, list):
                for i, it in enumerate(lst):
                    if isinstance(it, dict) and str(it.get("code", "")).upper() == kode:
                        out.append(dict(daftar=nama, sisi=sisi, urut=i + 1, change=it.get("change")))
    return _ok(muncul=out, n_daftar=len(raw_by_name))


# ------------------------------------------------------------- pengambilan
def ambil(client, kode: str, tgl_akhir: dt.date, penuh: bool = True) -> dict:
    """Cepat: IHSG + notasi (keduanya bersama semua saham, cache harian). Penuh: + keterbukaan, berita, insider, KSEI, peringkat pasar."""
    out: dict = {}

    def coba(nama, fn):
        try:
            out[nama] = fn()
        except Exception as e:  # satu bagian gagal tidak boleh mematikan yang lain
            out[nama] = _na(f"{type(e).__name__}: {str(e)[:100]}")

    coba("ihsg", lambda: ihsg(client.get("/analysis/chart/index/COMPOSITE",
                                         {"from": (tgl_akhir - dt.timedelta(days=120)).isoformat(), "to": tgl_akhir.isoformat()})))
    coba("notasi", lambda: notasi(client.get("/analysis/notation", {}), kode))
    if penuh:
        coba("disclosure", lambda: kabar(client.get("/analysis/disclosure", {"code": kode, "page": 1, "limit": 10}), today=tgl_akhir))
        coba("news", lambda: kabar(client.get("/analysis/news", {"code": kode, "page": 1, "limit": 10}), today=tgl_akhir))
        try:     # satu panggilan 365 hari: ringkasan insider 120 hari + data mentah untuk deteksi perubahan pengendali
            raw_ins = client.get("/analysis/shareholder-insider", {"code": kode, "from": (tgl_akhir - dt.timedelta(days=365)).isoformat(),
                                                                   "to": tgl_akhir.isoformat(), "page": 1, "limit": 30})
            out["insider"] = insider(raw_ins, today=tgl_akhir)
            out["insider_raw"] = raw_ins.get("data", []) if isinstance(raw_ins, dict) else (raw_ins or [])
        except Exception as e:
            out["insider"] = _na(f"{type(e).__name__}: {str(e)[:100]}")
        coba("ksei", lambda: ksei(client.get(f"/analysis/shareholder/ksei/{kode}", {"range": 6})))
        coba("peringkat", lambda: peringkat({n: client.get(f"/analysis/top/{p}", {"date": tgl_akhir.isoformat()})
                                             for n, p in (("BDM/bandarmologi", "accumulation"), ("asing", "foreign"), ("ritel", "ritel"))}, kode))
    return out


# ------------------------------------------------------------- pengaruh keputusan
def catatan_risiko(k: dict | None) -> list[str]:
    """Daftar peringatan dari konteks (dipakai kartu keputusan & laporan)."""
    if not k:
        return []
    w = []
    n = k.get("notasi") or {}
    if n.get("status") == "ok" and n.get("ada"):
        w.append("Saham memiliki NOTASI KHUSUS BEI: " + "; ".join(n["teks"][:2]))
    ih = k.get("ihsg") or {}
    if ih.get("status") == "ok" and ih["kondisi"] == "LEMAH":
        w.append(f"IHSG lemah (di bawah MA20, MA20<MA50, return 20h {ih['ret20']:+.1%}): confidence setup beli turun")
    for nm in ("disclosure", "news"):
        kb = k.get(nm) or {}
        if kb.get("status") == "ok" and kb.get("ada_kata_risiko"):
            w.append(f"{'Keterbukaan' if nm == 'disclosure' else 'Berita'} 14 hari memuat kata risiko: " + ", ".join(sorted({x for i in kb["items"] for x in i["risiko"]})))
    ins = k.get("insider") or {}
    if ins.get("status") == "ok" and ins.get("pengendali_jual"):
        w.append("Pemegang saham pengendali tercatat menjual (divestasi) dalam 120 hari terakhir")
    return w


def terapkan(kep: str, alasan: list[str], k: dict | None, mode: str = "lengkap") -> tuple[str, list[str]]:
    """Batas keputusan: mode cepat tidak boleh BELI; notasi khusus menahan BELI. Konteks lain hanya peringatan (tidak membalik vonis)."""
    alasan = list(alasan)
    if kep == "BELI" and mode == "cepat":
        kep = "TUNGGU"
        alasan.append("BELI hanya dapat ditetapkan pada analisis LENGKAP (dinamika bandar & aturan Hengky belum dihitung)")
    n = (k or {}).get("notasi") or {}
    if kep == "BELI" and n.get("status") == "ok" and n.get("ada"):
        kep = "TUNGGU"
        alasan.append("BELI ditahan: saham memiliki notasi khusus BEI (risiko papan pemantauan/ekuitas/laba)")
    for w in catatan_risiko(k):
        if "NOTASI" not in w:
            alasan.append("Konteks: " + w)
    return kep, alasan


# ------------------------------------------------------------------ laporan
def baris_laporan(k: dict | None) -> list[str]:
    L = ["## KONTEKS PASAR & RISIKO (di luar data saham itu sendiri)"]
    if not k:
        L.append("N/A: konteks tidak diambil/gagal. IHSG, notasi khusus, keterbukaan, insider, KSEI, peringkat pasar TIDAK dinilai.")
        return L
    ih = k.get("ihsg", {})
    L.append("[OBS] IHSG: " + (f"{ih['kondisi']} (last {ih['last']:,.0f}; MA20 {ih['ma20']:,.0f}; return 20h {ih['ret20']:+.1%}; data {ih.get('tgl')})" if ih.get("status") == "ok"
                               else f"N/A ({ih.get('alasan')})"))
    n = k.get("notasi", {})
    L.append("[OBS] Notasi khusus: " + (("ADA: " + "; ".join(n["teks"][:2])) if n.get("status") == "ok" and n.get("ada") else
                                       "tidak ada" if n.get("status") == "ok" else f"N/A ({n.get('alasan')})"))
    for nm, lbl in (("disclosure", "Keterbukaan informasi 14h"), ("news", "Berita 14h")):
        kb = k.get(nm)
        if kb is None:
            continue
        if kb.get("status") != "ok":
            L.append(f"[OBS] {lbl}: N/A ({kb.get('alasan')})")
        else:
            L.append(f"[OBS] {lbl}: {kb['n']} item" + (" · kata risiko: " + ", ".join(sorted({x for i in kb['items'] for x in i['risiko']})) if kb["ada_kata_risiko"] else ""))
            for i in kb["items"][:4]:
                L.append(f"- {i['tgl']} {i['judul']}")
    ins = k.get("insider")
    if ins is not None:
        if ins.get("status") != "ok":
            L.append(f"[OBS] Insider: N/A ({ins.get('alasan')})")
        else:
            L.append(f"[OBS] Insider {ins['hari']}h: {ins['n']} perubahan; nilai beli Rp{ins['nilai_beli']:,.0f}, jual Rp{ins['nilai_jual']:,.0f}" + (" · pengendali menjual" if ins["pengendali_jual"] else ""))
            for r in ins["rows"][:4]:
                L.append(f"- {r['tgl']} {r['nama']} ({r['peran'] or 'pemegang'}) {r['arah']} {r['ubah_pp']} pp, {r['tujuan']}")
    ks = k.get("ksei")
    if ks is not None:
        L.append("[OBS] KSEI kepemilikan asing: " + (f"{ks['rows'][0]['asing_pct']:.1%} -> {ks['rows'][-1]['asing_pct']:.1%} ({ks['arah']}, {ks['ubah_pp']:+.2f} pp, {len(ks['rows'])} bulan; data bulanan & terlambat)"
                                                    if ks.get("status") == "ok" else f"N/A ({ks.get('alasan')})"))
    pr = k.get("peringkat")
    if pr is not None:
        m = pr.get("muncul") or []
        L.append("[OBS] Peringkat pasar hari terakhir (top-5 per sisi): " + ("; ".join(f"{x['daftar']} {x['sisi']} #{x['urut']}" for x in m) if m else "tidak masuk top-5 daftar mana pun"))
    cw = catatan_risiko(k)
    L.append("[INF] Catatan risiko konteks: " + (" | ".join(cw) if cw else "tidak ada tanda khusus"))
    return L
