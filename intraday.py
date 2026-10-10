"""Statistik heuristik BPJP / BSJP dari candle 30 menit Invezgo (satu panggilan API).

BSJP (beli sore jual pagi): beli di close hari D, jual di sesi 1 hari D+1.
BPJP (beli pagi jual pagi): baca volume 30 menit pertama, beli di ~09:30 (close candle 09:00),
jual paling lambat akhir sesi 1 (11:30-12:00) hari yang sama.

Volume "tinggi" = volume jendela itu >= AMBANG_RVOL x median 20 hari sebelumnya pada jendela yang sama.
Jam candle Invezgo sudah WIB (meski berakhiran Z): 08:30 = pra-pembukaan, 16:00 = lelang penutupan.
Edukasi, bukan ajakan transaksi.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

import adimology as ad

AMBANG_RVOL = 1.5
TARGET = (0.01, 0.02, 0.03)
PAGI_AKHIR = "10:00"      # jam pertama: pra-pembukaan + 09:00 + 09:30
SORE_AWAL = "15:00"       # jam terakhir: 15:00, 15:30, lelang penutupan 16:00
S1_AKHIR = "11:30"        # candle terakhir sesi 1
BUKA_AKHIR = "09:30"      # 30 menit pertama: pra-pembukaan + candle 09:00


def get_bars(client: ad.InvezgoClient, kode: str, tanggal_akhir: dt.date | None, hari_kalender: int = 180) -> pd.DataFrame:
    end = tanggal_akhir or dt.date.today()
    raw = client.get(f"/analysis/chart/multi-time/{kode}",
                     {"from": (end - dt.timedelta(days=hari_kalender)).isoformat(),
                      "to": (end + dt.timedelta(days=1)).isoformat(), "timeframe": "30"})
    df = pd.DataFrame(raw)
    if df.empty:
        raise ad.AdimologyError("Data candle 30 menit tidak tersedia.")
    ts = pd.to_datetime(df["date"].astype(str).str.replace("Z", "", regex=False).str[:19])
    df = df.assign(tgl=ts.dt.date, jam=ts.dt.strftime("%H:%M"))
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values(["tgl", "jam"])
    if tanggal_akhir:
        df = df[df["tgl"] <= tanggal_akhir]
    return df.reset_index(drop=True)


def per_hari(bars: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tgl, g in bars.groupby("tgl", sort=True):
        pagi, sore = g[g["jam"] < PAGI_AKHIR], g[g["jam"] >= SORE_AWAL]
        buka = g[g["jam"] < BUKA_AKHIR]
        s1 = g[g["jam"] <= S1_AKHIR]
        s1_lanjut = s1[s1["jam"] >= BUKA_AKHIR]            # setelah entri BPJP 09:30
        if s1.empty:
            continue
        rows.append(dict(
            tgl=tgl, open=float(g["open"].iloc[0]), close=float(g["close"].iloc[-1]), high=float(g["high"].max()),
            vol=float(g["volume"].sum()), vol_pagi=float(pagi["volume"].sum()), vol_sore=float(sore["volume"].sum()),
            vol_buka=float(buka["volume"].sum()),
            high_pagi=float(pagi["high"].max()) if not pagi.empty else np.nan,
            high_s1=float(s1["high"].max()), close_s1=float(s1["close"].iloc[-1]),
            jam_puncak_s1=str(s1.loc[s1["high"].idxmax(), "jam"]),
            entri_bpjp=float(buka["close"].iloc[-1]) if not buka.empty else np.nan,
            high_bpjp=float(s1_lanjut["high"].max()) if not s1_lanjut.empty else np.nan,
            jam_puncak_bpjp=str(s1_lanjut.loc[s1_lanjut["high"].idxmax(), "jam"]) if not s1_lanjut.empty else None,
            harga_sore_awal=float(sore["open"].iloc[0]) if not sore.empty else np.nan,
            lengkap=g["jam"].iloc[-1] >= "15:30"))
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    for c in ("vol_pagi", "vol_sore", "vol_buka"):
        base = d[c].shift(1).rolling(20, min_periods=10).median()
        d[f"rvol_{c[4:]}"] = d[c] / base.replace(0, np.nan)
    return d


def _ringkas(x: pd.DataFrame, naik: str, puncak: str, gap: str | None, akhir: str, fee: float) -> dict:
    n = len(x)
    if n == 0:
        return dict(n=0)
    out = dict(n=n, median_naik=float(x[naik].median()), p_rugi_akhir=float((x[akhir] < 0).mean()),
               median_akhir=float(x[akhir].median()), net_median=float(x[naik].median() - fee))
    for t in TARGET:
        out[f"p_{int(t * 100)}"] = float((x[naik] >= t + fee).mean())
    if gap:
        out["p_gap_naik"] = float((x[gap] > 0).mean())
        out["median_gap"] = float(x[gap].median())
    vc = x[puncak].dropna().value_counts(normalize=True)
    out["jam_puncak"] = vc.index[0] if len(vc) else None
    out["p_jam_puncak"] = float(vc.iloc[0]) if len(vc) else np.nan
    out["sebaran_puncak"] = vc.sort_index()
    return out


def statistik(bars: pd.DataFrame, fee: float = 0.004) -> dict:
    d = per_hari(bars)
    if len(d) < 25:
        raise ad.AdimologyError(f"Data intraday terlalu sedikit ({len(d)} hari) untuk statistik BPJP/BSJP.")
    hist = d[d["lengkap"]].reset_index(drop=True)
    hari_ini = d.iloc[-1]

    # ---- BSJP: pasangan (D, D+1); lompatan > 35% = aksi korporasi -> dibuang
    nx = hist.shift(-1)
    bsjp = pd.DataFrame({
        "tgl": hist["tgl"], "rvol_sore": hist["rvol_sore"],
        "gap": nx["open"] / hist["close"] - 1,
        "naik_pagi": nx["high_pagi"] / hist["close"] - 1,
        "naik_s1": nx["high_s1"] / hist["close"] - 1,
        "akhir_s1": nx["close_s1"] / hist["close"] - 1,
        "jam_puncak": nx["jam_puncak_s1"]}).iloc[:-1]
    bsjp = bsjp[bsjp["gap"].abs() < 0.35].dropna(subset=["gap", "naik_s1"])

    # ---- BPJP: hari yang sama, entri ~09:30
    bp = d.copy()
    bpjp = pd.DataFrame({
        "tgl": bp["tgl"], "rvol_buka": bp["rvol_buka"],
        "naik": bp["high_bpjp"] / bp["entri_bpjp"] - 1,
        "akhir_s1": bp["close_s1"] / bp["entri_bpjp"] - 1,
        "jam_puncak": bp["jam_puncak_bpjp"]}).iloc[:-1]           # hari ini belum tentu selesai sesi 1
    bpjp = bpjp.dropna(subset=["naik", "akhir_s1"])

    def kelompok(df, kol):
        return {"Semua hari": df, f"Volume tinggi (≥{AMBANG_RVOL:g}x)": df[df[kol] >= AMBANG_RVOL],
                "Volume normal": df[df[kol] < AMBANG_RVOL]}

    res = dict(n_hari=len(d), dari=d["tgl"].iloc[0], sampai=d["tgl"].iloc[-1], fee=fee,
               bsjp={k: _ringkas(v, "naik_s1", "jam_puncak", "gap", "akhir_s1", fee) for k, v in kelompok(bsjp, "rvol_sore").items()},
               bpjp={k: _ringkas(v, "naik", "jam_puncak", None, "akhir_s1", fee) for k, v in kelompok(bpjp, "rvol_buka").items()},
               data_bsjp=bsjp, data_bpjp=bpjp,
               hari_ini=dict(tgl=hari_ini["tgl"], lengkap=bool(hari_ini["lengkap"]), close=hari_ini["close"],
                             rvol_sore=_f(hari_ini.get("rvol_sore")), rvol_buka=_f(hari_ini.get("rvol_buka")),
                             rvol_pagi=_f(hari_ini.get("rvol_pagi")), entri_bpjp=_f(hari_ini.get("entri_bpjp"))))
    med = d.tail(20)
    res["ambang"] = dict(buka=float(med["vol_buka"].median() * AMBANG_RVOL), sore=float(med["vol_sore"].median() * AMBANG_RVOL),
                         pagi=float(med["vol_pagi"].median() * AMBANG_RVOL))
    res["sinyal"] = _sinyal(res)
    return res


def _f(x):
    return None if x is None or pd.isna(x) else float(x)


def _sinyal(res: dict) -> dict:
    """Baca volume hari terakhir -> kelompok historis yang berlaku -> target & waktu jual."""
    h, fee = res["hari_ini"], res["fee"]
    out = {}
    for jenis, rv, ref in (("bsjp", h["rvol_sore"], h["close"]), ("bpjp", h["rvol_buka"], h["entri_bpjp"])):
        if rv is None or ref is None:
            out[jenis] = None
            continue
        kel = f"Volume tinggi (≥{AMBANG_RVOL:g}x)" if rv >= AMBANG_RVOL else "Volume normal"
        s = res[jenis][kel] if res[jenis][kel].get("n", 0) >= 8 else res[jenis]["Semua hari"]
        if s.get("n", 0) == 0:
            out[jenis] = None
            continue
        naik = max(s["median_naik"], fee + 0.005)
        target = _ceil_tick(ref * (1 + naik))
        layak = s["p_1"] >= 0.5 and s["net_median"] > 0
        out[jenis] = dict(rvol=rv, kelompok=kel if s is res[jenis][kel] else "Semua hari (sampel kelompok < 8)",
                          ref=ref, target=target, naik=naik, p_target=s["p_1"], jam_jual=s["jam_puncak"],
                          p_jam=s["p_jam_puncak"], p_gap=s.get("p_gap_naik"), p_rugi=s["p_rugi_akhir"], n=s["n"],
                          layak=layak)
    return out


def label_jam(j) -> str:
    """Candle 30 menit: 08:30 = pra-pembukaan (harga pembukaan), 16:00 = lelang penutupan."""
    if not j:
        return "-"
    return {"08:30": "08:30 (pra-buka)", "16:00": "16:00 (lelang tutup)"}.get(j, f"{j}–{_tambah30(j)}")


def _tambah30(j: str) -> str:
    h, m = map(int, j.split(":"))
    m += 30
    return f"{h + m // 60:02d}:{m % 60:02d}"


def _ceil_tick(x: float) -> float:
    f = ad.fraksi(x)
    return float(np.ceil(x / f - 1e-9) * f)


def tabel(blok: dict, bsjp: bool) -> pd.DataFrame:
    """Tabel tampilan: baris metrik x kolom kelompok."""
    def pct(v):
        return "-" if v is None or pd.isna(v) else f"{v:+.1%}"

    def p(v):
        return "-" if v is None or pd.isna(v) else f"{v:.0%}"
    rows = {}
    for kel, s in blok.items():
        if not s.get("n"):
            rows[kel] = {"Jumlah hari": "0"}
            continue
        r = {"Jumlah hari": f"{s['n']}"}
        if bsjp:
            r["Peluang gap naik besok"] = p(s["p_gap_naik"])
            r["Median gap pembukaan"] = pct(s["median_gap"])
        r["Median kenaikan tertinggi sesi 1"] = pct(s["median_naik"])
        for t in TARGET:
            r[f"Peluang sentuh +{int(t * 100)}% bersih fee"] = p(s[f"p_{int(t * 100)}"])
        r["Peluang rugi bila tahan s/d akhir sesi 1"] = p(s["p_rugi_akhir"])
        r["Jam puncak paling sering"] = f"{label_jam(s['jam_puncak'])} ({p(s['p_jam_puncak'])})" if s["jam_puncak"] else "-"
        rows[kel] = r
    return pd.DataFrame(rows)
