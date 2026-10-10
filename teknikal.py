"""Analisa teknikal lanjutan untuk tab Rekomendasi.

Support/resistance, MA 5/20/50/100/200, RSI 14, MACD (12, 26, 9), ATR 14, dan nilai bandar
dari broker summary. Tanpa Streamlit, sehingga bisa diuji. Edukasi, bukan ajakan transaksi.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

import adimology as ad

MA_PERIODS = (5, 20, 50, 100, 200)


# ------------------------------------------------------------------- data harga
def get_chart_panjang(client: ad.InvezgoClient, kode: str, tanggal_akhir: dt.date | None,
                      hari_kalender: int = 450) -> pd.DataFrame:
    """OHLCV harian ~1,5 tahun (cukup untuk MA200 + pemanasan MACD)."""
    end = tanggal_akhir or dt.date.today()
    raw = client.get(f"/analysis/chart/stock/{kode}",
                     {"from": (end - dt.timedelta(days=hari_kalender)).isoformat(), "to": end.isoformat()})
    df = pd.DataFrame(raw)
    if df.empty:
        raise ad.AdimologyError(f"Tidak ada data harga untuk {kode}.")
    df["tgl"] = (pd.to_datetime(df["date"], utc=True) + pd.Timedelta(hours=7)).dt.date
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values("tgl").drop_duplicates("tgl", keep="last")
    if tanggal_akhir:
        df = df[df["tgl"] <= tanggal_akhir]
    if df.empty:
        raise ad.AdimologyError("Tidak ada data harga sampai tanggal yang dipilih.")
    df, events = sesuaikan_split(df.reset_index(drop=True))
    df.attrs["split"] = events
    return df


RASIO_SPLIT = (2, 3, 4, 5, 8, 10, 15, 20, 25, 30, 40, 50, 100)


def sesuaikan_split(df: pd.DataFrame, ambang: float = 1.6) -> tuple[pd.DataFrame, list[dict]]:
    """Sesuaikan harga lama untuk stock split / reverse split.

    Data Invezgo tidak disesuaikan. Batas auto-reject BEI membuat lompatan > ~35% sehari mustahil
    tanpa aksi korporasi, jadi rasio close kemarin / open hari ini >= ambang (atau <= 1/ambang)
    dianggap split. Rasio dicocokkan ke rasio split umum bila dekat (DSSA 9-4-2026: 67.000 -> open 2.600
    = 25,8x -> 1:25).
    """
    df = df.copy()
    events = []
    for i in range(1, len(df)):
        prev, cur = df.at[i - 1, "close"], df.at[i, "open"] or df.at[i, "close"]
        if not prev or not cur or pd.isna(prev) or pd.isna(cur):
            continue
        r = prev / cur
        if r >= ambang or r <= 1 / ambang:
            f = r if r >= 1 else 1 / r
            umum = min(RASIO_SPLIT, key=lambda c: abs(np.log(f / c)))
            bulat = abs(np.log(f / umum)) < np.log(1.15)      # dalam 15%: harga hari itu ikut bergerak
            f = umum if bulat else f
            faktor = f if r >= 1 else 1 / f
            df.loc[: i - 1, ["open", "high", "low", "close"]] /= faktor
            df.loc[: i - 1, "volume"] *= faktor
            if bulat:
                jenis = f"stock split 1:{f:g}" if r >= 1 else f"reverse split {f:g}:1"
            else:
                jenis = f"penyesuaian aksi korporasi (rasio {faktor:.2f})"
            events.append({"tgl": df.at[i, "tgl"], "faktor": faktor, "jenis": jenis})
    return df, events


def _floor_tick(x: float) -> float:
    f = ad.fraksi(x)
    return np.floor(x / f + 1e-9) * f


# --------------------------------------------------------------------- indikator
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """RSI Wilder."""
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn
    out = 100 - 100 / (1 + rs)
    out[(dn == 0) & (up > 0)] = 100.0
    out[(dn == 0) & (up == 0)] = 50.0
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    m = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    s = m.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": m, "signal": s, "hist": m - s})


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def indikator(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for n in MA_PERIODS:
        df[f"ma{n}"] = df["close"].rolling(n).mean()
    df["rsi"] = rsi(df["close"])
    df = pd.concat([df, macd(df["close"])], axis=1)
    df["atr"] = atr(df)
    return df


# ------------------------------------------------------------ support/resistance
def _pivots(df: pd.DataFrame, k: int) -> tuple[list[float], list[float]]:
    """Swing high/low: titik tertinggi/terendah di jendela +-k bar."""
    hi, lo = df["high"].to_numpy(), df["low"].to_numpy()
    highs, lows = [], []
    for i in range(k, len(df) - k):
        if hi[i] == hi[i - k:i + k + 1].max():
            highs.append(float(hi[i]))
        if lo[i] == lo[i - k:i + k + 1].min():
            lows.append(float(lo[i]))
    return highs, lows


def _cluster(levels: list[float], tol: float) -> list[dict]:
    """Gabungkan level yang berjarak < tol (relatif). Kekuatan = jumlah sentuhan."""
    out: list[dict] = []
    for lv in sorted(levels):
        if out and lv <= out[-1]["harga"] * (1 + tol):
            c = out[-1]
            c["n"] += 1
            c["harga"] += (lv - c["harga"]) / c["n"]
        else:
            out.append({"harga": lv, "n": 1})
    return out


def support_resistance(df: pd.DataFrame, harga: float, lookback: int = 250, k: int = 5,
                       tol: float = 0.02) -> tuple[list[dict], list[dict]]:
    """Support (di bawah harga, terdekat dulu) dan resistance (di atas harga, terdekat dulu)."""
    d = df.tail(lookback).reset_index(drop=True)
    highs, lows = _pivots(d, k)
    lv = _cluster(highs + lows, tol)
    for c in lv:
        c["harga"] = ad.round_tick(c["harga"])
    sup = sorted([c for c in lv if c["harga"] < harga], key=lambda c: -c["harga"])
    res = sorted([c for c in lv if c["harga"] > harga], key=lambda c: c["harga"])
    return sup, res


# ------------------------------------------------------------------ nilai bandar
def bandar_value(bs: pd.DataFrame, kode_bandar: str | list | None, excl: tuple = ()) -> dict:
    """Net value (Rp) bandar terpilih dan top-5 akumulator vs top-5 distributor di window."""
    if bs is None or bs.empty or "net_value" not in bs:
        return dict(bandar=np.nan, top5_beli=np.nan, top5_jual=np.nan, net_top5=np.nan, status="TIDAK TERUKUR")
    b = bs[~bs["code"].isin(set(excl))]
    beli = float(b.nlargest(5, "net_value")["net_value"].clip(lower=0).sum())
    jual = float(b.nsmallest(5, "net_value")["net_value"].clip(upper=0).sum())
    codes = list(kode_bandar) if isinstance(kode_bandar, (list, tuple)) else [kode_bandar]
    row = bs[bs["code"].isin(codes)]
    nb = float(row["net_value"].sum()) if not row.empty else np.nan
    net = beli + jual
    status = "AKUMULASI" if net > 0 else "DISTRIBUSI" if net < 0 else "NETRAL"
    return dict(bandar=nb, top5_beli=beli, top5_jual=jual, net_top5=net, status=status)


# ------------------------------------------------------------------ rekomendasi
def _val(x):
    return None if x is None or pd.isna(x) else float(x)


def rekomendasi(df: pd.DataFrame, bv: dict | None = None, min_rr: float = 1.5, min_bar: int = 60) -> dict:
    """Rangkum indikator jadi area entry, SL, target jual, dan skor sinyal."""
    ind = indikator(df)
    last = ind.iloc[-1]
    harga = float(last["close"])
    sup, res = support_resistance(ind, harga)
    atr_ = _val(last["atr"]) or harga * 0.02

    # area entry: support terdekat (atau MA20/MA50 di bawah harga, atau low 20 hari)
    if sup:
        s1, basis = sup[0]["harga"], "support"
    else:
        ma_bawah = [v for v in (_val(last["ma20"]), _val(last["ma50"])) if v and v < harga]
        s1, basis = (max(ma_bawah), "MA") if ma_bawah else (float(ind["low"].tail(20).min()), "low 20 hari")
    entry_lo = ad.round_tick(s1)
    entry_hi_sup = ad.round_tick(max(s1 + 0.5 * atr_, s1 * 1.01))   # batas atas zona support

    # SL: di bawah support terdekat sejauh 1 ATR
    sl = ad.round_tick(entry_lo - atr_)

    # target: resistance di atas zona support; cadangan = proyeksi 2R / 3R
    res_atas = [c["harga"] for c in res if c["harga"] > entry_hi_sup]
    risk0 = max((entry_lo + entry_hi_sup) / 2 - sl, 1e-9)
    tp1 = res_atas[0] if res_atas else ad.round_tick((entry_lo + entry_hi_sup) / 2 + 2 * risk0)
    tp2 = res_atas[1] if len(res_atas) > 1 else ad.round_tick(max(tp1, entry_lo) + risk0)
    tp_basis = "resistance" if res_atas else "proyeksi R"

    # harga entry maksimal agar R/R >= min_rr:  (tp - e) / (e - sl) >= min_rr  ->  e <= (tp + min_rr*sl) / (1 + min_rr)
    def entry_maks_untuk(tp):
        return _floor_tick((tp + min_rr * sl) / (1 + min_rr))
    target, target_nama = tp1, "TP1"
    entry_maks = entry_maks_untuk(tp1)
    if entry_maks < entry_lo and tp2 > tp1:
        target, target_nama, entry_maks = tp2, "TP2", entry_maks_untuk(tp2)
    entry_hi = min(entry_hi_sup, entry_maks)
    zona_valid = entry_hi >= entry_lo
    entry_mid = (entry_lo + entry_hi) / 2 if zona_valid else entry_lo
    rr = (target - entry_mid) / max(entry_mid - sl, 1e-9)
    rr_harga = (target - harga) / (harga - sl) if harga > sl else float("nan")

    # skor: tren, momentum, bandar
    poin = []
    def add(ok, teks_ya, teks_tidak, w=1):
        if ok is None:
            return
        poin.append((w if ok else -w, teks_ya if ok else teks_tidak))

    ma = {n: _val(last[f"ma{n}"]) for n in MA_PERIODS}
    add(None if ma[200] is None else harga > ma[200], "harga di atas MA200 (tren besar naik)", "harga di bawah MA200 (tren besar turun)")
    add(None if ma[50] is None or ma[200] is None else ma[50] > ma[200], "MA50 di atas MA200 (golden zone)", "MA50 di bawah MA200 (death zone)")
    add(None if ma[20] is None else harga > ma[20], "harga di atas MA20", "harga di bawah MA20")
    add(None if ma[5] is None or ma[20] is None else ma[5] > ma[20], "MA5 di atas MA20 (momentum pendek naik)", "MA5 di bawah MA20 (momentum pendek turun)")
    r = _val(last["rsi"])
    if r is not None:
        if r >= 70:
            poin.append((-1, f"RSI {r:.0f} jenuh beli (overbought)"))
        elif r <= 30:
            poin.append((1, f"RSI {r:.0f} jenuh jual (oversold), potensi pantul"))
        else:
            poin.append((0, f"RSI {r:.0f} di zona normal"))
    m, s, h = _val(last["macd"]), _val(last["signal"]), _val(last["hist"])
    h_prev = _val(ind["hist"].iloc[-2]) if len(ind) > 1 else None
    add(None if m is None else m > s, "MACD di atas signal (bullish)", "MACD di bawah signal (bearish)")
    add(None if h is None or h_prev is None else h > h_prev, "histogram MACD membesar", "histogram MACD mengecil")
    if bv and bv.get("status") in ("AKUMULASI", "DISTRIBUSI"):
        add(bv["status"] == "AKUMULASI", "top-5 broker net akumulasi", "top-5 broker net distribusi", w=2)

    skor = sum(p for p, _ in poin)
    skor_max = sum(abs(p) for p, _ in poin) or 1
    di_area = zona_valid and entry_lo <= harga <= entry_hi
    jarak_entry = harga / entry_hi - 1 if zona_valid else float("nan")

    if len(ind) < min_bar:
        verdict, alasan = "TIDAK TERUKUR", f"data harga hanya {len(ind)} hari bursa (minimal {min_bar})"
    elif harga <= sl:
        verdict, alasan = "HINDARI", "harga sudah menembus SL di bawah support"
    elif skor <= 0:
        verdict, alasan = "HINDARI", "mayoritas indikator dan aliran bandar melawan"
    elif not zona_valid:
        verdict, alasan = "TUNGGU", (f"tidak ada harga di zona support ({entry_lo:,.0f}-{entry_hi_sup:,.0f}) yang memberi "
                                     f"R/R {min_rr:.1f} ke target {target:,.0f}; tunggu breakout di atas {tp1:,.0f} atau koreksi lebih dalam")
    elif di_area:
        verdict, alasan = "BELI DI AREA ENTRY", f"harga di area beli, R/R {rr:.2f} ke {target_nama}, indikator mendukung"
    else:
        verdict, alasan = "TUNGGU KE AREA ENTRY", (f"harga {jarak_entry:+.1%} di atas entry maksimal {entry_hi:,.0f} "
                                                   f"(R/R {min_rr:.1f}); jangan kejar")

    return dict(harga=harga, tgl=ind["tgl"].iloc[-1], ind=ind, sup=sup, res=res,
                entry_lo=entry_lo, entry_hi=entry_hi, entry_hi_sup=entry_hi_sup, entry_maks=entry_maks,
                zona_valid=zona_valid, target=target, target_nama=target_nama, min_rr=min_rr, rr_harga=rr_harga,
                entry_basis=basis, sl=sl, split=df.attrs.get("split", []),
                tp1=tp1, tp2=tp2, tp_basis=tp_basis, rr=rr, atr=atr_,
                ma=ma, rsi=r, macd=m, signal=s, hist=h, bandar=bv or {},
                poin=poin, skor=skor, skor_max=skor_max, verdict=verdict, alasan=alasan,
                n_bar=len(ind))
