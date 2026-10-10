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
            if not re.fullmatch(r"Q[1-4]", per):
                continue
            try:
                vals[(int(v.get("year")), int(per[1]))] = float(v.get("amount"))
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


# ------------------------------------------------------------------ analisa
def analisa(info: dict | None, is_raw, bs_raw, cf_raw, keystat_raw, harga: float) -> dict:
    IS, BS, CF = parse_panel(is_raw), parse_panel(bs_raw), parse_panel(cf_raw)
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
    semua = sorted(set().union(*[s.dropna().index for s in (rev, npf, aset, cfo) if s is not None]))
    if not semua:
        raise ad.AdimologyError("Laporan keuangan kosong.")
    kol = semua[-N_TAMPIL:]
    last = kol[-1]

    def row(s):
        return [s.get(k, np.nan) if s is not None else np.nan for k in kol]

    peringatan: list[str] = []
    sorotan: list[str] = []
    mark: dict = {}          # (tabel, baris, kolom) -> css

    # ================= PnL
    pnl_rows = {
        "Pendapatan" + (" bunga" if bank else ""): row(rev),
        "Laba kotor": row(gp) if not bank else [np.nan] * len(kol),
        "Laba sebelum pajak": row(pbt),
        "Laba bersih (induk)": row(npf),
    }
    gpm = [_ratio(g, r) for g, r in zip(row(gp), row(rev))]
    npm = [_ratio(n, r) for n, r in zip(row(npf), row(rev))]
    g_rev_yoy = [growth(rev, k, 4) for k in kol]
    g_rev_qoq = [growth(rev, k, 1) for k in kol]
    g_np_yoy = [growth(npf, k, 4) for k in kol]
    g_np_qoq = [growth(npf, k, 1) for k in kol]
    pnl_pct = {"GPM": gpm, "NPM": npm, "Pendapatan YoY": g_rev_yoy, "Pendapatan QoQ": g_rev_qoq,
               "Laba bersih YoY": g_np_yoy, "Laba bersih QoQ": g_np_qoq}
    rev_name = list(pnl_rows)[0]
    for j, k in enumerate(kol):
        for nama, gs in ((rev_name, g_rev_yoy), ("Laba bersih (induk)", g_np_yoy)):
            if pd.notna(gs[j]) and gs[j] > 0:
                mark[("pnl", nama, label(k))] = HIJAU
        for nama in ("Pendapatan YoY", "Pendapatan QoQ", "Laba bersih YoY", "Laba bersih QoQ"):
            v = pnl_pct[nama][j]
            if pd.notna(v) and v > 0:
                mark[("pnl", nama, label(k))] = HIJAU
        if pd.notna(pnl_rows["Laba bersih (induk)"][j]) and pnl_rows["Laba bersih (induk)"][j] < 0:
            mark[("pnl", "Laba bersih (induk)", label(k))] = KUNING
        if pd.notna(pnl_rows[rev_name][j]) and pnl_rows[rev_name][j] < 0:
            mark[("pnl", rev_name, label(k))] = KUNING
            peringatan.append(f"Pendapatan {label(k)} negatif setelah diubah dari YTD: kemungkinan revisi/data sumber tidak konsisten.")
    pnl = _tabel(pnl_rows, pnl_pct, kol)

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
    bsdf = _tabel(bs_rows, {}, kol, rasio=bs_x)

    # ================= arus kas
    cf_rows = {"Arus kas operasi (CFO)": row(cfo), "Arus kas investasi": row(cfi), "Arus kas pendanaan": row(cff),
               "Belanja modal (capex)": [-abs(v) if pd.notna(v) else np.nan for v in row(capex)],
               "Free cash flow (CFO − capex)": row(fcf), "Dividen dibayar": [-abs(v) if pd.notna(v) else np.nan for v in row(div)]}
    for j, k in enumerate(kol):
        L = label(k)
        for nama in ("Arus kas operasi (CFO)", "Free cash flow (CFO − capex)"):
            v = cf_rows[nama][j]
            if pd.notna(v) and v < 0:
                mark[("cf", nama, L)] = KUNING
        g = growth(cfo, k, 4)
        if pd.notna(g) and g > 0:
            mark[("cf", "Arus kas operasi (CFO)", L)] = HIJAU
    cfdf = _tabel(cf_rows, {}, kol)

    # ================= TTM & valuasi
    rev_ttm, rev_basis = ttm(rev, rev_ytd)
    np_ttm, np_basis = ttm(npf, np_ytd)
    cfo_ttm, _ = ttm(cfo, cfo_ytd)
    fcf_ttm, _ = ttm(fcf, None) if fcf is not None else (np.nan, "-")
    saham = _shares(keystat_raw)
    mcap = harga * saham if pd.notna(saham) and harga else np.nan
    e_last = ek_induk.get(last, np.nan) if ek_induk is not None else np.nan
    a_last = aset.get(last, np.nan) if aset is not None else np.nan
    l_last = liab.get(last, np.nan) if liab is not None else np.nan
    per_ = _ratio(mcap, np_ttm)
    pbv = _ratio(mcap, e_last)
    roe = _ratio(np_ttm, e_last)
    roa = _ratio(np_ttm, a_last)
    val = [
        ("Harga terakhir", harga, "rp"), ("Saham beredar", saham, "lembar"), ("Kapitalisasi pasar", mcap, "rpm"),
        (f"Pendapatan ({rev_basis})", rev_ttm, "rpm"), (f"Laba bersih ({np_basis})", np_ttm, "rpm"),
        ("EPS", _ratio(np_ttm, saham), "rp"), ("BVPS", _ratio(e_last, saham), "rp"),
        ("PER", per_, "x"), ("PBV", pbv, "x"), ("PS", _ratio(mcap, rev_ttm), "x"),
        ("Earnings yield", _ratio(np_ttm, mcap), "pct"), ("ROE", roe, "pct"), ("ROA", roa, "pct"),
        ("FCF yield", _ratio(fcf_ttm, mcap), "pct"),
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

    if pd.notna(g_rev_yoy[-1]) and g_rev_yoy[-1] > 0:
        sorotan.append(f"Pendapatan {L} naik {g_rev_yoy[-1]:.1%} YoY.")
    if pd.notna(g_rev_qoq[-1]) and g_rev_qoq[-1] > 0:
        sorotan.append(f"Pendapatan {L} naik {g_rev_qoq[-1]:.1%} dibanding kuartal sebelumnya.")
    if pd.notna(g_np_yoy[-1]) and g_np_yoy[-1] > 0:
        sorotan.append(f"Laba bersih {L} naik {g_np_yoy[-1]:.1%} YoY.")
    if pd.notna(g_np_qoq[-1]) and g_np_qoq[-1] > 0:
        sorotan.append(f"Laba bersih {L} naik {g_np_qoq[-1]:.1%} dibanding kuartal sebelumnya.")
    if pd.notna(cfo_ttm) and pd.notna(np_ttm) and np_ttm > 0 and cfo_ttm > np_ttm:
        sorotan.append("Arus kas operasi 12 bulan melebihi laba bersih: kualitas laba baik.")

    return dict(nama=info.get("name") or "", sektor=info.get("sector") or "", subsektor=info.get("subsector") or "",
                bank=bank, periode=L, kolom=[label(k) for k in kol],
                pnl=pnl, bs=bsdf, cf=cfdf, val=valdf, mark=mark, vmark=vmark,
                peringatan=list(dict.fromkeys(peringatan)), sorotan=sorotan,
                chart=pd.DataFrame({"periode": [label(k) for k in kol], "pendapatan": row(rev), "laba_bersih": row(npf)}))


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


def _tabel(nominal: dict, persen: dict, kol: list, rasio: dict | None = None) -> pd.DataFrame:
    """Tabel tampilan (string): nominal dalam Rp miliar, persen, dan rasio x."""
    cols = [label(k) for k in kol]
    rows = {}
    for n, vals in nominal.items():
        if all(pd.isna(v) for v in vals):
            continue
        rows[n] = ["-" if pd.isna(v) else f"{v / 1e9:,.0f}" for v in vals]
    for n, vals in persen.items():
        if all(pd.isna(v) for v in vals):
            continue
        rows[n] = ["-" if pd.isna(v) else f"{v:+.1%}" if "YoY" in n or "QoQ" in n else f"{v:.1%}" for v in vals]
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
