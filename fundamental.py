"""Analisa fundamental untuk tab Fundamental: PnL, neraca, arus kas, valuasi.

Data dari Invezgo /analysis/financial-statement (IS/BS/CF, kuartalan) dan /analysis/information.
Laba rugi & arus kas Invezgo bersifat kumulatif tahun berjalan (YTD), jadi diubah ke per kuartal.
Valuasi dihitung sendiri dari laporan + harga terakhir; dari /keystat hanya diambil jumlah saham
(endpoint keystat menurut dokumentasi Invezgo sedang diaktualisasi). Edukasi, bukan ajakan transaksi.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

import adimology as ad

N_KUARTAL = 12          # diambil dari API (3 tahun, cukup untuk YoY + TTM)
N_TAMPIL = 8            # kolom yang ditampilkan

HIJAU = "background-color: #dcfce7; color: #14532d"
KUNING = "background-color: #fef9c3; color: #713f12"


# ----------------------------------------------------------------------- parsing
def parse_panel(raw) -> pd.DataFrame:
    """{"rows": [{name, values: [{year, period, amount}]}]} -> DataFrame (akun x (tahun, kuartal))."""
    if isinstance(raw, dict) and "rows" not in raw and isinstance(raw.get("data"), dict):
        raw = raw["data"]
    rows = raw.get("rows", []) if isinstance(raw, dict) else []
    rec: dict = {}
    for r in rows:
        name = str(r.get("name", "")).strip()
        if not name or name in rec:
            continue
        vals = {}
        for v in r.get("values") or []:
            per = str(v.get("period", ""))
            if not re.fullmatch(r"Q[1-4]|FY", per):
                continue
            try:
                vals[(int(v.get("year")), 4 if per == "FY" else int(per[1]))] = float(v.get("amount"))
            except (TypeError, ValueError):
                continue
        rec[name] = vals
    df = pd.DataFrame(rec).T
    if df.empty:
        return df
    return df[sorted(df.columns)]


def label(k: tuple) -> str:
    return f"Q{k[1]} {k[0]}"


def _pick(df: pd.DataFrame, exact: list[str], pola: str | None = None) -> pd.Series | None:
    """Ambil akun pertama yang cocok (nama persis, lalu regex)."""
    if df is None or df.empty:
        return None
    low = {n.lower(): n for n in df.index}
    for e in exact:
        if e.lower() in low:
            return df.loc[low[e.lower()]]
    if pola:
        for n in df.index:
            if re.search(pola, n, flags=re.I):
                return df.loc[n]
    return None


def _first(*series):
    for x in series:
        if x is not None:
            return x
    return None


def _sum_pola(df: pd.DataFrame, pola: str, kecuali: str | None = None) -> pd.Series | None:
    if df is None or df.empty:
        return None
    names = [n for n in df.index if re.search(pola, n, flags=re.I) and not (kecuali and re.search(kecuali, n, flags=re.I))]
    return df.loc[names].sum(min_count=1) if names else None


# --------------------------------------------------------------- olah periode
def per_kuartal(ytd: pd.Series | None) -> pd.Series | None:
    """YTD -> nilai per kuartal (Q1 apa adanya; Qn = YTD Qn - YTD Qn-1 bila tersedia)."""
    if ytd is None:
        return None
    out = {}
    for (y, q), v in ytd.items():
        if q == 1:
            out[(y, q)] = v
        else:
            prev = ytd.get((y, q - 1))
            out[(y, q)] = v - prev if prev is not None and pd.notna(prev) and pd.notna(v) else np.nan
    return pd.Series(out, dtype=float).sort_index()


def _prev_q(k: tuple, n: int = 1) -> tuple:
    y, q = k
    for _ in range(n):
        y, q = (y, q - 1) if q > 1 else (y - 1, 4)
    return (y, q)


def ttm(q: pd.Series | None, ytd: pd.Series | None) -> tuple[float, str]:
    """Jumlah 4 kuartal terakhir; cadangan = YTD terakhir disetahunkan."""
    if q is None or q.dropna().empty:
        return np.nan, "-"
    last = q.dropna().index[-1]
    keys = [_prev_q(last, i) for i in range(4)]
    if all(k in q.index and pd.notna(q[k]) for k in keys):
        return float(sum(q[k] for k in keys)), "TTM"
    if ytd is not None and last in ytd.index and pd.notna(ytd[last]):
        return float(ytd[last]) * 4 / last[1], "YTD disetahunkan"
    return np.nan, "-"


def growth(s: pd.Series | None, k: tuple, lag: int) -> float:
    """Pertumbuhan s[k] terhadap s[k - lag kuartal]; tak terdefinisi bila pembanding <= 0."""
    if s is None:
        return np.nan
    b = _prev_q(k, lag)
    if k not in s.index or b not in s.index:
        return np.nan
    a, base = s[k], s[b]
    if pd.isna(a) or pd.isna(base) or base <= 0:
        return np.nan
    return a / base - 1


def _ratio(a, b):
    return a / b if pd.notna(a) and pd.notna(b) and b != 0 else np.nan


def _shares(keystat) -> float:
    """Jumlah saham: median nilai non-nol terakhir (keystat kadang meleset di satu periode)."""
    ks = parse_panel(keystat)
    s = _pick(ks, ["Saham"])
    if s is None:
        return np.nan
    s = s[s > 0].dropna()
    return float(s.tail(8).median()) if not s.empty else np.nan


# ----------------------------------------------------- FY, LTM, normalisasi, tesis
NM = float("inf")        # penanda 'n/m' (pertumbuhan tidak bermakna: |g| > 300%)


def gabung(a: pd.DataFrame, b: pd.DataFrame | None) -> pd.DataFrame:
    """Gabungkan panel kuartalan dengan panel FY (FY menjadi kolom (tahun, 4); Q4 diturunkan dari FY - YTD Q3)."""
    if b is None or b.empty:
        return a
    if a is None or a.empty:
        return b
    out = a.combine_first(b)
    return out[sorted(out.columns)]


def ltm_value(ytd: pd.Series | None, q: pd.Series | None = None) -> tuple[float, str]:
    """12 bulan terakhir: FY, atau FY(t-1) - YTD(t-1) + YTD(t), atau 4 kuartal berurutan, atau YTD disetahunkan."""
    if ytd is not None:
        s = ytd.dropna()
        if not s.empty:
            y, qq = s.index[-1]
            if qq == 4:
                return float(s.iloc[-1]), f"FY {y}"
            fy_, sama = (y - 1, 4), (y - 1, qq)
            if fy_ in s.index and sama in s.index:
                return float(s[fy_] - s[sama] + s.iloc[-1]), f"LTM (FY{y - 1} - {qq * 3}B{y - 1} + {qq * 3}B{y})"
    v, b = ttm(q, ytd)
    return v, ("TTM 4 kuartal" if b == "TTM" else "YTD disetahunkan" if b != "-" else "-")


def gabung_tampil(q: pd.Series | None, ytd: pd.Series | None, keys: list) -> tuple[list, list]:
    """Nilai per kolom: kuartal bila bisa dihitung, jika tidak YTD kumulatif (ditandai True)."""
    vals, flags = [], []
    for k in keys:
        v = q.get(k, np.nan) if q is not None else np.nan
        if pd.isna(v) and ytd is not None:
            y = ytd.get(k, np.nan)
            if pd.notna(y):
                vals.append(float(y)); flags.append(True); continue
        vals.append(np.nan if pd.isna(v) else float(v)); flags.append(False)
    return vals, flags


def peta(q: pd.Series | None, ytd: pd.Series | None) -> dict:
    keys = set()
    for s_ in (q, ytd):
        if s_ is not None:
            keys |= set(s_.index)
    m = {}
    for k in keys:
        v, f_ = gabung_tampil(q, ytd, [k])
        if pd.notna(v[0]):
            m[k] = (v[0], f_[0])
    return m


def tumbuh(m: dict, k: tuple, lag: int) -> float:
    """Pertumbuhan like-for-like (kuartal vs kuartal, atau YTD vs YTD); NM bila |g| > 300%; nan bila tak terdefinisi."""
    kp = _prev_q(k, lag)
    if k not in m or kp not in m:
        return np.nan
    (a, fa), (b, fb) = m[k], m[kp]
    if fa != fb or (lag == 1 and fa) or b <= 0:
        return np.nan
    g = a / b - 1
    return NM if abs(g) > 3.0 else g


def _pos(v) -> bool:
    return pd.notna(v) and v != NM and v > 0


def normalisasi(npf: pd.Series | None, n_min: int = 5) -> dict:
    """Laba bersih LTM dinormalisasi: kuartal ekstrem (|selisih dari median| > 3 MAD) diganti median kuartal lain.

    Heuristik untuk laba satu kali (mis. keuntungan pelepasan anak usaha); bukan pengganti membaca CALK.
    """
    if npf is None:
        return dict(ok=False, alasan="laba bersih kuartalan tidak tersedia")
    s = npf.dropna().iloc[-8:]
    if len(s) < n_min:
        return dict(ok=False, alasan=f"kuartal laba tersedia {len(s)} (< {n_min})")
    med = float(s.median())
    mad = float((s - med).abs().median())
    batas = 3 * max(mad, 0.1 * abs(med), 1.0)
    outl = [k for k, v in s.items() if abs(v - med) > batas]
    last = s.index[-1]
    keys = [_prev_q(last, i) for i in range(4)]
    if not all(k in npf.index and pd.notna(npf[k]) for k in keys):
        return dict(ok=False, alasan="4 kuartal berurutan terakhir tidak lengkap", outliers=[label(k) for k in outl])
    waras = [v for k, v in s.items() if k not in outl]
    med2 = float(np.median(waras)) if waras else med
    asli = sum(float(npf[k]) for k in keys)
    nilai = sum((med2 if k in outl else float(npf[k])) for k in keys)
    return dict(ok=True, nilai=nilai, asli=asli, median=med2, outliers=[label(k) for k in keys if k in outl])


def _kesimpulan_tesis(d: dict) -> dict:
    """Tesis investasi dari data keuangan saja (tanpa proyeksi). Tiga keluaran: mendukung, melemahkan, belum diketahui."""
    pend, lem, hilang = [], [], d["hilang"]
    belum = ["proyeksi laba & arus kas", "katalis dan prospek bisnis", "kualitas manajemen / tata kelola",
             "nilai wajar DCF/NAV (butuh proyeksi & discount rate; tidak tersedia di data Invezgo)"]
    if hilang:
        return dict(kesimpulan="TIDAK DAPAT DINILAI", alasan=f"data inti hilang: {', '.join(hilang)}", mendukung=[], melemahkan=[],
                    belum_diketahui=belum, pantau=[])
    ni, nn, cfo, fcf, nd, eq, cc = d["ni"], d["ni_norm"], d["cfo"], d["fcf"], d["net_debt"], d["ekuitas"], d["cash_conv"]
    bank = d["bank"]
    if ni <= 0:
        lem.append(f"laba bersih LTM negatif ({_rp(ni)})")
    elif pd.notna(nn) and nn <= 0:
        lem.append("laba normalisasi LTM tidak positif (laba reguler lemah)")
    else:
        pend.append(f"laba bersih LTM positif ({_rp(ni)})" + (f"; normalisasi {_rp(nn)}" if pd.notna(nn) and abs(nn - ni) > 0.1 * abs(ni) else ""))
    if not bank:
        if pd.notna(cfo) and cfo < 0:
            lem.append(f"arus kas operasi LTM negatif ({_rp(cfo)})")
        elif pd.notna(cc) and cc >= 0.8:
            pend.append(f"konversi kas baik (CFO/laba {cc:.1f}x)")
        elif pd.notna(cc) and cc < 0.5:
            lem.append(f"konversi kas rendah (CFO/laba {cc:.1f}x)")
        if pd.notna(fcf) and fcf < 0:
            lem.append(f"free cash flow LTM negatif ({_rp(fcf)}; capex/ekspansi melebihi kas operasi)")
        if pd.notna(nd) and pd.notna(eq) and eq > 0:
            if nd < 0:
                pend.append(f"kas bersih ({_rp(-nd)} lebih besar dari utang berbunga)")
            elif nd / eq > 1:
                lem.append(f"utang bersih {nd / eq:.1f}x ekuitas")
    if d.get("g_rev") is not None and _pos(d["g_rev"]) and d["g_rev"] >= 0.10:
        pend.append(f"pendapatan tumbuh {d['g_rev']:.0%} YoY")
    if d.get("outliers"):
        lem.append(f"laba kuartal ekstrem ({', '.join(d['outliers'])}): PER/ROE reported tidak andal")
    if len(lem) >= 2 or ni <= 0:
        kes = "TESIS LEMAH"
    elif not lem and len(pend) >= 2:
        kes = "LAYAK DITELITI"
    else:
        kes = "CAMPURAN: BUTUH RISET TAMBAHAN"
    return dict(kesimpulan=kes, alasan="", mendukung=pend, melemahkan=lem, belum_diketahui=belum,
                pantau=[f"membaik/memburuk: {x}" for x in lem])


# ------------------------------------------------------------------ analisa
def analisa(info: dict | None, is_raw, bs_raw, cf_raw, keystat_raw, harga: float, fy: dict | None = None) -> dict:
    IS, BS, CF = parse_panel(is_raw), parse_panel(bs_raw), parse_panel(cf_raw)
    fy_pakai: list[str] = []
    for nm in ("IS", "BS", "CF"):
        raw = (fy or {}).get(nm)
        pf = parse_panel(raw) if raw else pd.DataFrame()
        if not pf.empty:
            fy_pakai.append(nm)
            if nm == "IS":
                IS = gabung(IS, pf)
            elif nm == "BS":
                BS = gabung(BS, pf)
            else:
                CF = gabung(CF, pf)
    if IS.empty and BS.empty:
        raise ad.AdimologyError("Laporan keuangan tidak tersedia untuk emiten ini.")
    info = info if isinstance(info, dict) else {}
    teks_sektor = " ".join(str(info.get(k) or "") for k in ("sector", "subsector", "industry")).lower()
    bank = "bank" in teks_sektor or (_pick(BS, ["Jumlah aset lancar"]) is None and _pick(IS, ["Pendapatan bunga"]) is not None
                                      and _pick(IS, ["Penjualan dan pendapatan usaha", "Jumlah laba bruto"]) is None)

    # ---- PnL (YTD -> kuartal)
    rev_ytd = _first(_pick(IS, ["Pendapatan bunga", "Jumlah pendapatan"]) if bank else None,
                     _pick(IS, ["Penjualan dan pendapatan usaha", "Pendapatan usaha", "Penjualan", "Pendapatan neto",
                   "Penjualan neto", "Jumlah pendapatan", "Pendapatan"], r"^(penjualan|pendapatan (usaha|neto|bersih))"))
    gp_ytd = _pick(IS, ["Jumlah laba bruto", "Laba bruto"])
    pbt_ytd = _pick(IS, ["Jumlah laba (rugi) sebelum pajak penghasilan"], r"sebelum pajak penghasilan")
    np_ytd = _pick(IS, ["Laba (rugi) yang dapat diatribusikan ke entitas induk", "Jumlah laba (rugi)"])
    rev, gp, pbt, npf = (per_kuartal(x) for x in (rev_ytd, gp_ytd, pbt_ytd, np_ytd))

    # ---- arus kas (YTD -> kuartal)
    cfo_ytd = _pick(CF, [], r"^jumlah arus kas bersih .*aktivitas operasi")
    cfi_ytd = _pick(CF, [], r"^jumlah arus kas bersih .*aktivitas investasi")
    cff_ytd = _pick(CF, [], r"^jumlah arus kas bersih .*aktivitas pendanaan")
    capex_ytd = _pick(CF, ["Pembayaran untuk perolehan aset tetap"], r"perolehan aset tetap")
    div_ytd = _pick(CF, [], r"^pembayaran dividen")
    cfo, cfi, cff, capex, div = (per_kuartal(x) for x in (cfo_ytd, cfi_ytd, cff_ytd, capex_ytd, div_ytd))
    fcf = cfo - capex.abs().reindex(cfo.index).fillna(0) if cfo is not None and capex is not None else cfo
    fcf_ytd = cfo_ytd - capex_ytd.abs().reindex(cfo_ytd.index).fillna(0) if cfo_ytd is not None and capex_ytd is not None else cfo_ytd

    # ---- neraca (posisi akhir kuartal, tidak kumulatif)
    aset = _pick(BS, ["Jumlah aset"])
    liab = _pick(BS, ["Jumlah liabilitas"])
    ekuitas = _pick(BS, ["Jumlah ekuitas"])
    ek_induk = _first(_pick(BS, ["Jumlah ekuitas yang diatribusikan kepada pemilik entitas induk"]), ekuitas)
    al = _pick(BS, ["Jumlah aset lancar"])
    ll = _pick(BS, ["Jumlah liabilitas jangka pendek"])
    kas = _pick(BS, ["Kas dan setara kas"])
    utang = _sum_pola(BS, r"(utang bank|pinjaman|obligasi|sewa pembiayaan|liabilitas sewa)",
                      kecuali=r"(piutang|aset|penerimaan|pembayaran)")

    # ---- kolom yang ditampilkan
    semua = sorted(set().union(*[s.dropna().index for s in (rev, npf, aset, cfo, rev_ytd, np_ytd, cfo_ytd) if s is not None]))
    if not semua:
        raise ad.AdimologyError("Laporan keuangan kosong.")
    kol = semua[-N_TAMPIL:]
    last = kol[-1]

    flags: dict = {}                 # nama baris -> daftar bool per kolom (True = nilai kumulatif YTD, bukan kuartal)

    def row(s, ytd=None, nama=None, neg=False):
        vals, fl = gabung_tampil(s, ytd, kol)
        if neg:
            vals = [-abs(v) if pd.notna(v) else np.nan for v in vals]
        if nama and any(fl):
            flags[nama] = fl
        return vals

    peringatan: list[str] = []
    sorotan: list[str] = []
    mark: dict = {}          # (tabel, baris, kolom) -> css

    # ================= PnL
    rev_nm = "Pendapatan" + (" bunga" if bank else "")
    pnl_rows = {
        rev_nm: row(rev, rev_ytd, rev_nm),
        "Laba kotor": row(gp, gp_ytd, "Laba kotor") if not bank else [np.nan] * len(kol),
        "Laba sebelum pajak": row(pbt, pbt_ytd, "Laba sebelum pajak"),
        "Laba bersih (induk)": row(npf, np_ytd, "Laba bersih (induk)"),
    }
    m_rev, m_np, m_cfo = peta(rev, rev_ytd), peta(npf, np_ytd), peta(cfo, cfo_ytd)
    gpm = [_ratio(g, r) for g, r in zip(row(gp, gp_ytd), pnl_rows[rev_nm])]
    npm = [_ratio(n, r) for n, r in zip(pnl_rows["Laba bersih (induk)"], pnl_rows[rev_nm])]
    g_rev_yoy = [tumbuh(m_rev, k, 4) for k in kol]
    g_rev_qoq = [tumbuh(m_rev, k, 1) for k in kol]
    g_np_yoy = [tumbuh(m_np, k, 4) for k in kol]
    g_np_qoq = [tumbuh(m_np, k, 1) for k in kol]
    pnl_pct = {"GPM": gpm, "NPM": npm, "Pendapatan YoY": g_rev_yoy, "Pendapatan QoQ": g_rev_qoq,
               "Laba bersih YoY": g_np_yoy, "Laba bersih QoQ": g_np_qoq}
    rev_name = list(pnl_rows)[0]
    for j, k in enumerate(kol):
        for nama, gs in ((rev_name, g_rev_yoy), ("Laba bersih (induk)", g_np_yoy)):
            if _pos(gs[j]):
                mark[("pnl", nama, label(k))] = HIJAU
        for nama in ("Pendapatan YoY", "Pendapatan QoQ", "Laba bersih YoY", "Laba bersih QoQ"):
            v = pnl_pct[nama][j]
            if _pos(v):
                mark[("pnl", nama, label(k))] = HIJAU
        if pd.notna(pnl_rows["Laba bersih (induk)"][j]) and pnl_rows["Laba bersih (induk)"][j] < 0:
            mark[("pnl", "Laba bersih (induk)", label(k))] = KUNING
        if pd.notna(pnl_rows[rev_name][j]) and pnl_rows[rev_name][j] < 0:
            mark[("pnl", rev_name, label(k))] = KUNING
            peringatan.append(f"Pendapatan {label(k)} negatif setelah diubah dari YTD: kemungkinan revisi/data sumber tidak konsisten.")
    pnl = _tabel(pnl_rows, pnl_pct, kol, flags=flags)

    # ================= neraca
    der = [_ratio(l_, e) for l_, e in zip(row(liab), row(ekuitas))]
    cr = [_ratio(a, b) for a, b in zip(row(al), row(ll))]
    bs_rows = {"Total aset": row(aset), "Total liabilitas": row(liab), "Total ekuitas": row(ekuitas),
               "Ekuitas induk": row(ek_induk), "Kas & setara kas": row(kas), "Utang berbunga (estimasi)": row(utang)}
    if not bank:
        bs_rows.update({"Aset lancar": row(al), "Liabilitas jangka pendek": row(ll)})
    bs_x = {"DER (liabilitas / ekuitas)": der}
    if not bank:
        bs_x["Current ratio"] = cr
    for j, k in enumerate(kol):
        L = label(k)
        a_, l_, e_ = bs_rows["Total aset"][j], bs_rows["Total liabilitas"][j], bs_rows["Total ekuitas"][j]
        if pd.notna(a_) and pd.notna(l_) and l_ > a_:
            mark[("bs", "Total liabilitas", L)] = KUNING
        if pd.notna(e_) and e_ < 0:
            mark[("bs", "Total ekuitas", L)] = KUNING
        if not bank and pd.notna(der[j]) and der[j] > 2:
            mark[("bs", "DER (liabilitas / ekuitas)", L)] = KUNING
        if not bank and pd.notna(cr[j]) and cr[j] < 1:
            mark[("bs", "Current ratio", L)] = KUNING
        if pd.notna(a_) and pd.notna(l_) and pd.notna(e_) and a_ > 0 and abs(a_ - l_ - e_) / a_ > 0.01:
            mark[("bs", "Total aset", L)] = KUNING
        if j > 0:
            for nama in ("Total aset", "Total ekuitas", "Kas & setara kas"):
                cur, prv = bs_rows[nama][j], bs_rows[nama][j - 1]
                if pd.notna(cur) and pd.notna(prv) and prv > 0 and cur > prv and (("bs", nama, L) not in mark):
                    mark[("bs", nama, L)] = HIJAU
    bsdf = _tabel(bs_rows, {}, kol, rasio=bs_x, flags={})

    # ================= arus kas
    cf_rows = {"Arus kas operasi (CFO)": row(cfo, cfo_ytd, "Arus kas operasi (CFO)"),
               "Arus kas investasi": row(cfi, cfi_ytd, "Arus kas investasi"),
               "Arus kas pendanaan": row(cff, cff_ytd, "Arus kas pendanaan"),
               "Belanja modal (capex)": row(capex, capex_ytd, "Belanja modal (capex)", neg=True),
               "Free cash flow (CFO − capex)": row(fcf, fcf_ytd, "Free cash flow (CFO − capex)"),
               "Dividen dibayar": row(div, div_ytd, "Dividen dibayar", neg=True)}
    for j, k in enumerate(kol):
        L = label(k)
        for nama in ("Arus kas operasi (CFO)", "Free cash flow (CFO − capex)"):
            v = cf_rows[nama][j]
            if pd.notna(v) and v < 0:
                mark[("cf", nama, L)] = KUNING
        if _pos(tumbuh(m_cfo, k, 4)):
            mark[("cf", "Arus kas operasi (CFO)", L)] = HIJAU
    cfdf = _tabel(cf_rows, {}, kol, flags=flags)

    # ================= TTM & valuasi
    rev_ttm, rev_basis = ltm_value(rev_ytd, rev)
    np_ttm, np_basis = ltm_value(np_ytd, npf)
    cfo_ttm, cfo_basis = ltm_value(cfo_ytd, cfo)
    fcf_ttm, _ = ltm_value(fcf_ytd, fcf)
    capex_ttm, _ = ltm_value(capex_ytd, capex)
    nrm = normalisasi(npf)
    np_norm = nrm["nilai"] if nrm.get("ok") else np.nan
    saham = _shares(keystat_raw)
    mcap = harga * saham if pd.notna(saham) and harga else np.nan
    e_last = ek_induk.get(last, np.nan) if ek_induk is not None else np.nan
    a_last = aset.get(last, np.nan) if aset is not None else np.nan
    l_last = liab.get(last, np.nan) if liab is not None else np.nan
    per_ = _ratio(mcap, np_ttm)
    pbv = _ratio(mcap, e_last)
    roe = _ratio(np_ttm, e_last)
    roa = _ratio(np_ttm, a_last)
    kas_last = kas.get(last, np.nan) if kas is not None else np.nan
    utang_last = utang.get(last, np.nan) if utang is not None else np.nan
    net_debt = utang_last - kas_last if pd.notna(utang_last) and pd.notna(kas_last) else np.nan
    cash_conv = _ratio(cfo_ttm, np_ttm) if pd.notna(np_ttm) and np_ttm > 0 else np.nan
    per_norm = _ratio(mcap, np_norm) if pd.notna(np_norm) and np_norm > 0 else np.nan
    val = [
        ("Harga terakhir", harga, "rp"), ("Saham beredar", saham, "lembar"), ("Kapitalisasi pasar", mcap, "rpm"),
        (f"Pendapatan ({rev_basis})", rev_ttm, "rpm"), (f"Laba bersih ({np_basis})", np_ttm, "rpm"),
        ("EPS", _ratio(np_ttm, saham), "rp"), ("BVPS", _ratio(e_last, saham), "rp"),
        ("PER", per_, "x"), ("PBV", pbv, "x"), ("PS", _ratio(mcap, rev_ttm), "x"),
        ("Earnings yield", _ratio(np_ttm, mcap), "pct"), ("ROE", roe, "pct"), ("ROA", roa, "pct"),
        ("FCF yield", _ratio(fcf_ttm, mcap), "pct"),
        ("Laba bersih normalisasi (LTM)", np_norm, "rpm"), ("PER (laba normalisasi)", per_norm, "x"),
        ("ROE (laba normalisasi)", _ratio(np_norm, e_last), "pct"), ("Konversi kas (CFO/laba)", cash_conv, "x"),
        ("Utang bersih (utang berbunga - kas)", net_debt, "rpm"),
    ]
    vmark = {}
    if pd.notna(per_) and per_ < 0:
        vmark["PER"] = KUNING; peringatan.append("PER negatif: emiten rugi dalam 12 bulan terakhir.")
    elif pd.notna(per_) and per_ > 50:
        vmark["PER"] = KUNING; peringatan.append(f"PER {per_:.1f}x sangat tinggi: harga mahal atau laba sangat tipis.")
    if pd.notna(pbv) and pbv < 0:
        vmark["PBV"] = KUNING
    if pd.notna(roe) and roe < 0:
        vmark["ROE"] = KUNING
    if pd.notna(roe) and roe >= 0.15:
        vmark["ROE"] = HIJAU; sorotan.append(f"ROE {roe:.1%} (≥ 15%): profitabilitas modal tinggi.")
    if pd.isna(saham):
        peringatan.append("Jumlah saham tidak tersedia, valuasi per saham (PER/PBV) tidak bisa dihitung.")
    if nrm.get("ok") and nrm["outliers"]:
        vmark["PER"] = KUNING
        peringatan.append(f"PER/ROE reported tidak andal: laba kuartal ekstrem ({', '.join(nrm['outliers'])}) ikut dalam LTM "
                          f"(reported {_rp(nrm['asli'])} vs normalisasi {_rp(nrm['nilai'])}).")
    elif not nrm.get("ok"):
        peringatan.append(f"Laba tidak dapat dinormalisasi: {nrm.get('alasan')}.")
    if np_basis == "YTD disetahunkan":
        peringatan.append("Laba/pendapatan LTM dari YTD disetahunkan (FY tahun lalu tidak tersedia): musiman dan laba satu kali bisa menyesatkan.")
    valdf = pd.DataFrame([{"Metrik": n, "Nilai": _fmt(v, t)} for n, v, t in val])

    # ================= ringkasan peringatan & sorotan (periode terakhir)
    L = label(last)
    if pd.notna(a_last) and pd.notna(l_last) and l_last > a_last:
        peringatan.append(f"Total liabilitas ({_rp(l_last)}) LEBIH BESAR dari total aset ({_rp(a_last)}) di {L}: ekuitas negatif, risiko solvabilitas.")
    elif pd.notna(e_last) and e_last < 0:
        peringatan.append(f"Ekuitas negatif di {L}.")
    if not bank and pd.notna(der[-1]) and der[-1] > 2:
        peringatan.append(f"DER {der[-1]:.2f}x (> 2x): utang tinggi dibanding modal.")
    if not bank and pd.notna(cr[-1]) and cr[-1] < 1:
        peringatan.append(f"Current ratio {cr[-1]:.2f} (< 1): aset lancar tidak menutup utang jangka pendek.")
    if pd.notna(np_ttm) and np_ttm < 0:
        peringatan.append(f"Rugi bersih {np_basis}: {_rp(np_ttm)}.")
    if pd.notna(cfo_ttm) and pd.notna(np_ttm) and np_ttm > 0 and cfo_ttm < 0:
        peringatan.append("Laba positif tetapi arus kas operasi 12 bulan negatif: kualitas laba perlu dicek (piutang/persediaan).")
    if pd.notna(fcf_ttm) and fcf_ttm < 0:
        peringatan.append(f"Free cash flow 12 bulan negatif ({_rp(fcf_ttm)}).")
    if any(m == KUNING for (t, n, c), m in mark.items() if t == "bs" and n == "Total aset"):
        peringatan.append("Neraca tidak seimbang (aset ≠ liabilitas + ekuitas) di sebagian periode: cek data sumber.")

    for g_, teks in ((g_rev_yoy[-1], "Pendapatan {L} naik {g:.1%} YoY."), (g_rev_qoq[-1], "Pendapatan {L} naik {g:.1%} dibanding kuartal sebelumnya."),
                     (g_np_yoy[-1], "Laba bersih {L} naik {g:.1%} YoY."), (g_np_qoq[-1], "Laba bersih {L} naik {g:.1%} dibanding kuartal sebelumnya.")):
        if _pos(g_):
            sorotan.append(teks.format(L=L, g=g_))
    if pd.notna(cfo_ttm) and pd.notna(np_ttm) and np_ttm > 0 and cfo_ttm > np_ttm:
        sorotan.append("Arus kas operasi 12 bulan melebihi laba bersih: kualitas laba baik.")

    # ================= kelengkapan data, sensitivitas, tesis investasi
    hilang = [nm for nm, v in (("pendapatan", rev_ttm), ("laba bersih", np_ttm), ("arus kas operasi", cfo_ttm if not bank else 0.0),
                               ("total aset", a_last)) if v is None or pd.isna(v)]
    if hilang:
        peringatan.append(f"Data hilang di sumber/parsing: {', '.join(hilang)}. Metrik turunannya tidak dihitung (tidak diisi asumsi).")
    ytd_kol = sorted({label(k) for nm, fl in flags.items() for k, f_ in zip(kol, fl) if f_})
    if ytd_kol:
        peringatan.append(f"Kolom {', '.join(ytd_kol)} berisi nilai KUMULATIF sejak awal tahun (*): kuartal sebelumnya tidak dilaporkan; "
                          "pertumbuhan hanya dibandingkan sesama kumulatif.")
    eps_n = _ratio(np_norm, saham)
    bvps = _ratio(e_last, saham)
    sens = dict(eps_norm=eps_n, bvps=bvps,
                per=[(x, eps_n * x) for x in (6, 10, 14)] if pd.notna(eps_n) and eps_n > 0 else [],
                pbv=[(x, bvps * x) for x in (0.8, 1.0, 1.5)] if pd.notna(bvps) and bvps > 0 else [])
    tesis = _kesimpulan_tesis(dict(
        hilang=hilang, ni=np_ttm, ni_norm=np_norm, cfo=cfo_ttm, fcf=fcf_ttm, net_debt=net_debt, ekuitas=e_last, cash_conv=cash_conv,
        bank=bank, g_rev=g_rev_yoy[-1], outliers=nrm.get("outliers") if nrm.get("ok") else []))
    profil = {k: info.get(k) for k in ("board", "listing_date", "category", "administration", "notation")} if info else {}

    def _lv(v, n=1):                                   # nilai valid ke-n dari belakang
        vv = [x for x in (v or []) if x is not None and pd.notna(x)]
        return float(vv[-n]) if len(vv) >= n else float("nan")
    npl = pnl_rows["Laba bersih (induk)"]
    metrik = dict(der=_lv(der), der_prev=_lv(der, 3), cr=_lv(cr) if not bank else float("nan"), cr_prev=_lv(cr, 3) if not bank else float("nan"),
                  ll=_lv(bs_rows.get("Liabilitas jangka pendek")), ll_prev=_lv(bs_rows.get("Liabilitas jangka pendek"), 3),
                  utang=_lv(bs_rows["Utang berbunga (estimasi)"]), utang_prev=_lv(bs_rows["Utang berbunga (estimasi)"], 3),
                  g_rev_yoy=_lv(g_rev_yoy), g_rev_qoq=_lv(g_rev_qoq), g_np_yoy=_lv(g_np_yoy), g_np_qoq=_lv(g_np_qoq), gpm=_lv(gpm), npm=_lv(npm),
                  roe=roe, roa=roa, pbv=pbv, per=per_, per_norm=per_norm, rev_ttm=rev_ttm, np_ttm=np_ttm, cfo_ttm=cfo_ttm, fcf_ttm=fcf_ttm, capex_ttm=capex_ttm,
                  ekuitas=e_last, aset=a_last, liab=l_last, rugi_kuartal=int(sum(1 for x in npl[-4:] if pd.notna(x) and x < 0)),
                  n_np_kuartal=int(sum(1 for x in npl[-4:] if pd.notna(x))), rev_kuartal=[None if pd.isna(x) else float(x) for x in pnl_rows[rev_nm][-4:]],
                  np_kuartal=[None if pd.isna(x) else float(x) for x in npl[-4:]],
                  cfo_kuartal=[None if pd.isna(x) else float(x) for x in cf_rows["Arus kas operasi (CFO)"][-4:]],
                  periode=[label(k) for k in kol][-4:], gpm_seri=[None if pd.isna(x) else float(x) for x in gpm[-4:]],
                  npm_seri=[None if pd.isna(x) else float(x) for x in npm[-4:]], rev_yoy_seri=[None if pd.isna(x) else float(x) for x in g_rev_yoy[-4:]],
                  np_yoy_seri=[None if pd.isna(x) else float(x) for x in g_np_yoy[-4:]], div_ttm=ltm_value(div_ytd, div)[0] if div_ytd is not None else float("nan"),
                  saham=saham, mcap=mcap, eps=_ratio(np_ttm, saham), eps_norm=_ratio(np_norm, saham), bvps=_ratio(e_last, saham), kas=kas_last,
                  gearing=_ratio(utang_last, e_last) if pd.notna(utang_last) else float("nan"), np_norm=np_norm, rev_basis=rev_basis, np_basis=np_basis)

    return dict(nama=info.get("name") or "", sektor=info.get("sector") or "", subsektor=info.get("subsector") or "",
                bank=bank, periode=L, kolom=[label(k) for k in kol],
                pnl=pnl, bs=bsdf, cf=cfdf, val=valdf, mark=mark, vmark=vmark,
                peringatan=list(dict.fromkeys(peringatan)), sorotan=sorotan,
                chart=pd.DataFrame({"periode": [label(k) for k in kol], "pendapatan": row(rev, rev_ytd), "laba_bersih": row(npf, np_ytd)}),
                ltm=dict(rev=(rev_ttm, rev_basis), ni=(np_ttm, np_basis), cfo=(cfo_ttm, cfo_basis), fcf=fcf_ttm, capex=capex_ttm,
                         ni_norm=np_norm, norm=nrm, net_debt=net_debt, cash_conv=cash_conv, per=per_, per_norm=per_norm, saham=saham, mcap=mcap),
                kualitas=dict(hilang=hilang, fy=fy_pakai, ytd_kolom=ytd_kol, n_kolom=len(kol)), sens=sens, tesis=tesis, profil=profil, metrik=metrik)


# ------------------------------------------------------------------ format
def _rp(v) -> str:
    if pd.isna(v):
        return "-"
    a = abs(v)
    if a >= 1e12:
        return f"Rp{v / 1e12:,.2f} T"
    if a >= 1e9:
        return f"Rp{v / 1e9:,.1f} M"
    return f"Rp{v:,.0f}"


def _fmt(v, t: str) -> str:
    if v is None or pd.isna(v):
        return "-"
    return {"rp": f"Rp{v:,.0f}", "rpm": _rp(v), "lembar": f"{v:,.0f}", "x": f"{v:,.2f}x", "pct": f"{v:.1%}"}[t]


def _tabel(nominal: dict, persen: dict, kol: list, rasio: dict | None = None, flags: dict | None = None) -> pd.DataFrame:
    """Tabel tampilan (string): nominal dalam Rp miliar, persen, dan rasio x."""
    cols = [label(k) for k in kol]
    rows = {}
    for n, vals in nominal.items():
        if all(pd.isna(v) for v in vals):
            continue
        fl = (flags or {}).get(n) or [False] * len(vals)
        rows[n] = ["-" if pd.isna(v) else f"{v / 1e9:,.0f}" + ("*" if f_ else "") for v, f_ in zip(vals, fl)]
    for n, vals in persen.items():
        if all(pd.isna(v) for v in vals):
            continue
        rows[n] = ["-" if pd.isna(v) else "n/m" if v == NM else f"{v:+.1%}" if "YoY" in n or "QoQ" in n else f"{v:.1%}" for v in vals]
    for n, vals in (rasio or {}).items():
        if all(pd.isna(v) for v in vals):
            continue
        rows[n] = ["-" if pd.isna(v) else f"{v:,.2f}" for v in vals]
    return pd.DataFrame.from_dict(rows, orient="index", columns=cols)


def style(df: pd.DataFrame, mark: dict, nama_tabel: str):
    """Styler: hijau = tumbuh, kuning = perlu dicek."""
    def css(_):
        out = pd.DataFrame("", index=df.index, columns=df.columns)
        for (t, r, c), m in mark.items():
            if t == nama_tabel and r in out.index and c in out.columns:
                out.loc[r, c] = m
        return out
    return df.style.apply(css, axis=None)
