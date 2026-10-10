"""Backtest offline metode target Adimology (bandarmology) dengan data Invezgo.

Pertanyaan yang diuji: apakah sinyal + target Adimology (avg bandar + 5% + P papan) memberi hasil lebih
baik daripada (a) sekadar "top net buyer sedang akumulasi", (b) saham/tanggal acak, dan (c) sinyal yang
sama tetapi target tetap (median upside), dengan aturan holding yang sama.

Tidak mengubah rumus di adimology.py; hanya memakai building block-nya (get_orderbook, broker_summary,
step_up, round_tick, count_papan). Pada tanggal evaluasi t hanya data sampai t yang dipakai;
entry = open hari bursa berikutnya (t+1).

Pemakaian (lihat backtest/README.md):
    export INVEZGO_API_KEY=...          # jangan ditulis ke file yang di-commit
    python backtest/run_backtest.py                 # cetak estimasi panggilan API lalu berhenti
    python backtest/run_backtest.py --yes           # jalankan (semua respons di-cache ke backtest/cache/)
    python backtest/run_backtest.py --offline       # hanya dari cache, tanpa panggilan API

Edukasi, bukan ajakan transaksi.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import adimology as ad  # noqa: E402
import teknikal as tk  # noqa: E402

HERE = Path(__file__).resolve().parent
OB_START = dt.date(2026, 3, 25)          # orderbook historis Invezgo tersedia mulai tanggal ini
RETAIL_DEFAULT = ("XL", "YP", "XC", "PD")
UNIVERSE_DEFAULT = (
    # LQ45 / big caps
    "BBCA BBRI BMRI BBNI TLKM ASII UNVR ICBP INDF ADRO PTBA ANTM INCO MDKA CPIN KLBF UNTR AMRT MEDC PGAS "
    # second liners
    "BRIS ARTO ESSA AKRA MAPI ERAA SIDO HRUM ELSA TINS DOID BUMI ENRG RAJA SCMA MNCN BUKA AUTO SMRA CTRA"
).split()
CHART_LOOKBACK_DAYS = 60                 # kalender sebelum start, untuk window broker 20 hari


# ----------------------------------------------------------------------- config
@dataclass
class Config:
    tickers: list = field(default_factory=lambda: list(UNIVERSE_DEFAULT))
    start: dt.date = dt.date(2024, 6, 1)
    end: dt.date = field(default_factory=dt.date.today)
    windows: tuple = (5,)
    modes: tuple = ("top1",)             # top1 = broker net beli terbesar; top3 = gabungan 3 akumulator (ad.Params.mode_bandar)
    study: str = "all"                   # all | target | imbalance
    horizons: tuple = (5, 10, 20)
    retail: tuple = RETAIL_DEFAULT
    investor: str = "all"
    market: str = "RG"
    margin_a: float = 0.05
    faktor_realistis: float = 0.5
    min_upside: float = 0.03
    maks_kejar: float = 0.05
    distribusi_pct: float = 0.25
    fee_beli: float = 0.0015
    fee_jual: float = 0.0025
    ob_time: str = "14:30"               # sama dengan snapshot studi imbalance, jadi cache-nya terpakai bersama
    imb_time: str = "14:30"
    imb_horizons: tuple = (1, 5)
    proxy_scale: float = 1.0
    chart_chunk_days: int = 400
    random_mult: int = 5
    n_boot: int = 2000
    seed: int = 7
    max_calls: int | None = None
    cache_dir: Path = HERE / "cache"
    results_dir: Path = HERE / "results"


class BudgetHabis(ad.AdimologyError):
    """Batas --max-calls tercapai. Cache tetap tersimpan; jalankan ulang untuk melanjutkan."""


# ------------------------------------------------------------------------ cache
class CachingClient(ad.InvezgoClient):
    """Membungkus client lain; setiap respons (termasuk 404/422) disimpan ke disk sebagai JSON.

    Kunci cache = hash(path + params). API key tidak pernah ikut ditulis.
    """

    def __init__(self, inner: ad.InvezgoClient | None, cache_dir: Path, max_calls: int | None = None):
        super().__init__("")
        self.inner, self.cache_dir, self.max_calls = inner, Path(cache_dir), max_calls
        self.live_calls = self.cache_hits = self.misses_offline = 0

    def _file(self, path: str, items: tuple) -> Path:
        h = hashlib.sha256(json.dumps([path, list(items)]).encode()).hexdigest()[:24]
        folder = re.sub(r"[^A-Za-z0-9]+", "_", path.split("/")[2] if path.count("/") >= 2 else path).strip("_")
        return self.cache_dir / (folder or "misc") / f"{h}.json"

    def get(self, path: str, params: dict | None = None) -> Any:
        items = tuple(sorted((k, str(v)) for k, v in (params or {}).items() if v is not None))
        f = self._file(path, items)
        if f.exists():
            self.cache_hits += 1
            obj = json.loads(f.read_text())
            return _replay(obj)
        if self.inner is None:
            self.misses_offline += 1
            raise ad.AdimologyError(f"offline: tidak ada di cache ({path})")
        if self.max_calls is not None and self.live_calls >= self.max_calls:
            raise BudgetHabis(f"batas --max-calls {self.max_calls} tercapai")
        self.live_calls += 1
        try:
            data = self.inner.get(path, params)
            obj = {"ok": data}
        except ad.ParamDitolak as e:
            obj = {"err": "422", "msg": str(e)}
        except ad.AdimologyError as e:
            if not str(e).startswith("Data tidak ditemukan"):   # jaringan/401: jangan di-cache
                raise
            obj = {"err": "404", "msg": str(e)}
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps({"path": path, "params": list(items), **obj}))
        tmp.replace(f)
        return _replay(obj)


def _replay(obj: dict) -> Any:
    if "ok" in obj:
        return obj["ok"]
    if obj.get("err") == "422":
        raise ad.ParamDitolak(obj.get("msg", "422"))
    raise ad.AdimologyError(obj.get("msg", "Data tidak ditemukan"))


# ------------------------------------------------------------------- estimasi
def _fridays(start: dt.date, end: dt.date) -> list[dt.date]:
    d = start + dt.timedelta(days=(4 - start.weekday()) % 7)
    out = []
    while d <= end:
        out.append(d)
        d += dt.timedelta(days=7)
    return out


def estimate_calls(cfg: Config) -> dict:
    """Batas atas panggilan API (sebelum cache). Dihitung tanpa memanggil API."""
    fr = _fridays(cfg.start, cfg.end)
    span = (cfg.end - cfg.start).days + CHART_LOOKBACK_DAYS
    chart = math.ceil(span / cfg.chart_chunk_days)
    target = cfg.study in ("all", "target")
    imb = cfg.study in ("all", "imbalance")
    window = len(fr) * len(cfg.windows) if target else 0       # mode top1/top3 memakai respons yang sama
    lastday = len(fr) if target else 0
    ob_sig = sum(1 for d in fr if d >= OB_START) if target else 0
    hari_ob = _weekdays(max(cfg.start, OB_START), cfg.end) if imb else 0
    if imb and cfg.ob_time == cfg.imb_time:                     # snapshot Jumat sudah termasuk di studi imbalance
        ob_sig = 0
    per = chart + window + lastday + ob_sig + hari_ob
    n = len(cfg.tickers)
    return dict(tickers=n, eval_dates=len(fr), chart=chart * n, window_summary=window * n,
                lastday_summary=lastday * n, orderbook=ob_sig * n, orderbook_imbalance=hari_ob * n, total=per * n)


def _weekdays(a: dt.date, b: dt.date) -> int:
    return sum(1 for k in range((b - a).days + 1) if (a + dt.timedelta(days=k)).weekday() < 5) if b >= a else 0


def count_cached(cfg: Config) -> int:
    return sum(1 for _ in cfg.cache_dir.rglob("*.json")) if cfg.cache_dir.exists() else 0


# ------------------------------------------------------------------ data harga
def load_chart(client: ad.InvezgoClient, kode: str, cfg: Config) -> pd.DataFrame:
    """OHLCV harian tidak disesuaikan (seperti broker avg pada saat itu), diambil per potongan."""
    lo, end = cfg.start - dt.timedelta(days=CHART_LOOKBACK_DAYS), cfg.end
    raws = []
    while lo <= end:
        hi = min(lo + dt.timedelta(days=cfg.chart_chunk_days - 1), end)
        try:
            raws.extend(client.get(f"/analysis/chart/stock/{kode}", {"from": lo.isoformat(), "to": hi.isoformat()}) or [])
        except ad.ParamDitolak:
            pass
        lo = hi + dt.timedelta(days=1)
    df = pd.DataFrame(raws)
    if df.empty:
        raise ad.AdimologyError(f"Tidak ada data harga untuk {kode}.")
    df["tgl"] = (pd.to_datetime(df["date"], utc=True) + pd.Timedelta(hours=7)).dt.date
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[df["close"] > 0].sort_values("tgl").drop_duplicates("tgl", keep="last").reset_index(drop=True)
    return df


def split_dates(chart: pd.DataFrame) -> list[dt.date]:
    """Tanggal aksi korporasi (split/reverse split) terdeteksi; sinyal yang melintasinya dibuang."""
    _, events = tk.sesuaikan_split(chart[["tgl", "open", "high", "low", "close", "volume"]].reset_index(drop=True))
    return [e["tgl"] for e in events]


def eval_indices(chart: pd.DataFrame, start: dt.date, end: dt.date) -> list[int]:
    """Indeks hari bursa terakhir tiap minggu (penutupan Jumat, atau hari bursa sebelumnya bila libur)."""
    out = []
    for fri in _fridays(start, end):
        mon = fri - dt.timedelta(days=4)
        m = chart.index[(chart["tgl"] >= mon) & (chart["tgl"] <= fri)]
        if len(m):
            out.append(int(m[-1]))
    return out


# ------------------------------------------------------------------- sinyal
def lpp_proxy(hist: pd.DataFrame, total_lot: float, scale: float = 1.0) -> float:
    """Proksi lot per papan sebelum ada orderbook historis.

    = rata-rata lot transaksi per hari di window (dari broker summary, satuan sama dengan net bandar)
      / median jumlah papan rentang harian (count_papan(low, high), minimal 1).
    """
    papan = [max(ad.count_papan(lo, hi), 1.0) for lo, hi in zip(hist["low"], hist["high"])]
    return scale * (total_lot / max(len(hist), 1)) / float(np.median(papan))


def signal_at(client: ad.InvezgoClient, kode: str, chart: pd.DataFrame, i: int, window: int,
              cfg: Config, mode: str = "top1") -> dict | None:
    """Hitung bandar, target dan verdict Adimology pada penutupan chart.iloc[i] (hanya data <= t)."""
    if i + 1 < window:
        return None
    hist = chart.iloc[i + 1 - window: i + 1]
    t, harga = hist["tgl"].iloc[-1], float(hist["close"].iloc[-1])
    d0 = hist["tgl"].iloc[0]
    try:
        bs = ad.broker_summary(client, kode, d0, t, cfg.investor, cfg.market)
    except ad.AdimologyError:
        return None
    if bs.empty or "net_volume" not in bs:
        return None
    total_lot = float(bs["buy_volume"].sum()) / 100
    excl = {c.strip().upper() for c in cfg.retail}
    kand = bs[~bs["code"].isin(excl)].sort_values("net_volume", ascending=False)
    if kand.empty:
        return None
    if mode == "top3":                     # sama dengan ad.analyze(mode_bandar="top3")
        top = kand[kand["net_volume"] > 0].head(3)
        w = top["buy_volume"].where(top["buy_avg"].notna(), 0)
        if top.empty or w.sum() <= 0:
            return None
        avg = float((top["buy_avg"].fillna(0) * w).sum() / w.sum())
        codes, net_vol = list(top["code"]), float(top["net_volume"].sum())
    else:
        row = kand.iloc[0]
        avg = float(row["buy_avg"]) if pd.notna(row["buy_avg"]) else float("nan")
        codes, net_vol = [row["code"]], float(row["net_volume"])
    barang = net_vol / 100
    porsi = net_vol / (total_lot * 100) * 100 if total_lot else 0.0
    sl = float(hist["low"].min())
    out = dict(kode=kode, t=t, window=window, mode_bandar=mode, w_from=d0, harga=harga, bandar="+".join(codes), avg=avg,
               barang_lot=barang, porsi_net_pct=porsi, sl=sl, total_lot=total_lot)
    if not np.isfinite(avg) or avg <= 0:
        return None

    # distribusi: net bandar di hari terakhir window (aturan sama dengan ad.analyze)
    net_terakhir = 0.0
    if barang > 0:
        try:
            x = ad.broker_summary(client, kode, t, t, cfg.investor, cfg.market)
        except ad.AdimologyError:
            x = pd.DataFrame()
        r = x[x["code"].isin(codes)] if not x.empty else x
        net_terakhir = float(r["net_volume"].sum()) / 100 if not r.empty else 0.0
    distribusi = net_terakhir < 0 and abs(net_terakhir) >= cfg.distribusi_pct * max(barang, 1e-9)
    akumulasi = barang > 0 and not distribusi
    kejar = harga > avg * (1 + cfg.maks_kejar)
    out.update(net_terakhir_lot=net_terakhir, distribusi=distribusi, akumulasi=akumulasi, kejar=kejar)

    # lot per papan: orderbook historis (>= OB_START) atau proksi. Orderbook hanya diminta bila
    # filter yang tidak bergantung pada lpp lolos (hemat kuota).
    proxy = lpp_proxy(hist, total_lot, cfg.proxy_scale)
    lpp, src, ob_lpp = proxy, "proxy", float("nan")
    if t >= OB_START and akumulasi and not kejar:
        p = ad.Params(kode=kode, market=cfg.market, ob_date=t.isoformat(), ob_time=cfg.ob_time)
        try:
            ob = ad.get_orderbook(client, kode, p)
        except ad.AdimologyError:
            ob = None
        if ob and ob["lot_per_papan"] > 0:
            lpp, src, ob_lpp = ob["lot_per_papan"], "orderbook", ob["lot_per_papan"]
        else:
            src = "proxy_fallback"

    dasar = avg * (1 + cfg.margin_a)
    pp = barang / lpp if lpp > 0 else float("nan")
    t_real = ad.step_up(dasar, max(pp, 0) * cfg.faktor_realistis)
    t_max = ad.step_up(dasar, max(pp, 0))
    upside = t_real / harga - 1
    setup_ok = akumulasi and harga < t_real and upside >= cfg.min_upside and not kejar
    out.update(lpp=lpp, lpp_sumber=src, lpp_proxy=proxy, lpp_orderbook=ob_lpp, p=pp,
               t_real=t_real, t_max=t_max, t_real_r=ad.round_tick(t_real), t_max_r=ad.round_tick(t_max),
               upside_real=upside, upside_max=t_max / harga - 1, setup_ok=setup_ok)
    return out


# ---------------------------------------------------------------- simulasi trade
def simulate(fut: pd.DataFrame, tp: float, sl: float, fee_beli: float, fee_jual: float) -> dict:
    """Entry di open hari pertama `fut`, keluar di TP / SL / close hari terakhir.

    Konservatif: bila TP dan SL tersentuh di hari yang sama, dianggap SL dulu.
    Gap: open di bawah SL -> keluar di open; open di atas TP -> keluar di open.
    """
    o, h, l, c = (fut[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    entry = o[0]
    exit_px, hasil, hari = c[-1], "waktu", len(fut)
    for j in range(len(fut)):
        if o[j] <= sl:
            exit_px, hasil, hari = o[j], "sl", j + 1
            break
        if o[j] >= tp:
            exit_px, hasil, hari = o[j], "tp", j + 1
            break
        if l[j] <= sl:
            exit_px, hasil, hari = sl, "sl", j + 1
            break
        if h[j] >= tp:
            exit_px, hasil, hari = tp, "tp", j + 1
            break
    net = exit_px * (1 - fee_jual) / (entry * (1 + fee_beli)) - 1
    return dict(entry=entry, exit=exit_px, hasil=hasil, hari_pegang=hari, net_ret=net,
                mfe=h.max() / entry - 1, mae=l.min() / entry - 1, max_high=h.max())


def outcomes(chart: pd.DataFrame, i: int, tp: float, sl: float, cfg: Config, tp_max: float | None = None) -> dict:
    """Hasil untuk tiap horizon H (hari bursa setelah t). NaN bila data masa depan belum cukup."""
    out = {}
    for H in cfg.horizons:
        fut = chart.iloc[i + 1: i + 1 + H]
        if len(fut) < H:
            out.update({f"hasil_{H}": None, f"net_{H}": np.nan, f"hit_{H}": np.nan, f"sl_{H}": np.nan,
                        f"mfe_{H}": np.nan, f"mae_{H}": np.nan, f"kalibrasi_{H}": np.nan})
            if tp_max is not None:
                out[f"hit_max_{H}"] = np.nan
            continue
        s = simulate(fut, tp, sl, cfg.fee_beli, cfg.fee_jual)
        out.update({"entry": s["entry"], f"hasil_{H}": s["hasil"], f"net_{H}": s["net_ret"],
                    f"hit_{H}": float(s["hasil"] == "tp"), f"sl_{H}": float(s["hasil"] == "sl"),
                    f"mfe_{H}": s["mfe"], f"mae_{H}": s["mae"], f"kalibrasi_{H}": s["max_high"] / tp})
        if tp_max is not None:
            out[f"hit_max_{H}"] = float(simulate(fut, tp_max, sl, cfg.fee_beli, cfg.fee_jual)["hasil"] == "tp")
    return out


# ------------------------------------------------------------------ main loop
def run(client: ad.InvezgoClient, cfg: Config, log=print) -> pd.DataFrame:
    """Kembalikan DataFrame semua baris (strategi adimology, akumulasi, target_tetap, acak)."""
    return run_all(client, cfg, log)[0]


def run_all(client: ad.InvezgoClient, cfg: Config, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(baris backtest target, baris studi imbalance orderbook) sesuai cfg.study."""
    rows, charts, splits, imb = [], {}, {}, []
    do_target = cfg.study in ("all", "target")
    hmax = max(cfg.horizons)
    for kode in cfg.tickers:
        try:
            chart = load_chart(client, kode, cfg)
        except BudgetHabis:
            raise
        except ad.AdimologyError as e:
            log(f"  {kode}: dilewati ({e})")
            continue
        charts[kode], splits[kode] = chart, split_dates(chart)
        if cfg.study in ("all", "imbalance"):
            imb += imbalance_rows(client, kode, chart, cfg)
        if not do_target:
            continue
        idx = eval_indices(chart, cfg.start, cfg.end)
        n_sig = 0
        for i in idx:
            for w, m in _settings(cfg):
                s = signal_at(client, kode, chart, i, w, cfg, m)
                if s is None:
                    continue
                s["split_dalam_rentang"] = _crosses_split(chart, i, w, hmax, splits[kode])
                base = {k: s[k] for k in s}
                if s["setup_ok"]:
                    n_sig += 1
                    rows.append({**base, "strategi": "adimology",
                                 **outcomes(chart, i, s["t_real"], s["sl"], cfg, tp_max=s["t_max"])})
                if s["akumulasi"]:
                    rows.append({**base, "strategi": "akumulasi", "_i": i})
        log(f"  {kode}: {len(idx)} tanggal evaluasi, {n_sig} sinyal Adimology (semua window/mode)")

    imb_df = pd.DataFrame(imb)
    df = pd.DataFrame(rows)
    if df.empty:
        return df, imb_df
    # Target tetap untuk baseline = median upside Adimology per window (tanpa melihat harga masa depan).
    adim = df[df["strategi"] == "adimology"]
    med_up = {}
    for w, m in _settings(cfg):
        sel = adim[(adim["window"] == w) & (adim["mode_bandar"] == m)]
        med_up[(w, m)] = float(sel["upside_real"].median()) if len(sel) else 0.10
    extra = []
    for _, r in adim.iterrows():                         # (c) sinyal sama, target = median upside
        i = int(charts[r["kode"]].index[charts[r["kode"]]["tgl"] == r["t"]][0])
        tp = r["harga"] * (1 + med_up[(r["window"], r["mode_bandar"])])
        extra.append({**_base(r), "strategi": "target_tetap", "tp": tp, **outcomes(charts[r["kode"]], i, tp, r["sl"], cfg)})
    akum = df[df["strategi"] == "akumulasi"]
    for _, r in akum.iterrows():                         # (a) top net buyer akumulasi, target = median upside
        tp = r["harga"] * (1 + med_up[(r["window"], r["mode_bandar"])])
        extra.append({**_base(r), "strategi": "akumulasi", "tp": tp,
                      **outcomes(charts[r["kode"]], int(r["_i"]), tp, r["sl"], cfg)})
    extra += _random_rows(charts, splits, cfg, med_up, n_target=len(adim))   # (b) acak
    df = pd.concat([adim.assign(tp=adim["t_real"]), pd.DataFrame(extra)], ignore_index=True)
    df = df.drop(columns=[c for c in ("_i",) if c in df])
    # Regime menurut tanggal: sejak OB_START lot per papan dari orderbook historis, sebelumnya proksi.
    df["regime"] = np.where(df["t"].map(lambda d: d >= OB_START), "orderbook", "proxy")
    return df, imb_df


def _settings(cfg: Config) -> list[tuple]:
    return [(w, m) for w in cfg.windows for m in cfg.modes]


def _base(r: pd.Series) -> dict:
    keep = [k for k in r.index if not re.match(r"(hasil|net|hit|hit_max|sl|mfe|mae|kalibrasi)_\d+$", k)
            and k not in ("strategi", "_i", "entry", "tp")]
    return {k: r[k] for k in keep}


def _crosses_split(chart: pd.DataFrame, i: int, window: int, hmax: int, events: list) -> bool:
    if not events:
        return False
    lo = chart["tgl"].iloc[max(i + 1 - window, 0)]
    hi = chart["tgl"].iloc[min(i + hmax, len(chart) - 1)]
    return any(lo < d <= hi for d in events)


def _random_rows(charts: dict, splits: dict, cfg: Config, med_up: dict, n_target: int) -> list:
    """(b) Saham + tanggal evaluasi acak; SL = low window, TP = median upside. Hanya butuh data harga."""
    pool = []
    for kode, ch in charts.items():
        for i in eval_indices(ch, cfg.start, cfg.end):
            for w, m in _settings(cfg):
                if i + 1 >= w:
                    pool.append((kode, i, w, m))
    if not pool:
        return []
    rng = np.random.default_rng(cfg.seed)
    n = min(len(pool), max(cfg.random_mult * n_target, 200 * len(_settings(cfg))))
    out = []
    for k in rng.choice(len(pool), size=n, replace=False):
        kode, i, w, m = pool[k]
        ch = charts[kode]
        hist = ch.iloc[i + 1 - w: i + 1]
        harga, sl = float(hist["close"].iloc[-1]), float(hist["low"].min())
        tp = harga * (1 + med_up[(w, m)])
        out.append(dict(kode=kode, t=ch["tgl"].iloc[i], window=w, mode_bandar=m, harga=harga, sl=sl, tp=tp, strategi="acak",
                        lpp_sumber="-", split_dalam_rentang=_crosses_split(ch, i, w, max(cfg.horizons), splits[kode]),
                        **outcomes(ch, i, tp, sl, cfg)))
    return out


# ------------------------------------------------------- studi imbalance orderbook
# Klaim slide 12: total offer lebih tebal dari total bid -> harga cenderung naik; bid lebih tebal -> sideways/koreksi.
IMB_BUCKETS = ((0.0, 0.67, "bid jauh lebih tebal (<0,67)"), (0.67, 1.0, "bid lebih tebal (0,67-1)"),
               (1.0, 1.5, "offer lebih tebal (1-1,5)"), (1.5, float("inf"), "offer jauh lebih tebal (>=1,5)"))


def imbalance_rows(client: ad.InvezgoClient, kode: str, chart: pd.DataFrame, cfg: Config) -> list[dict]:
    """Per hari bursa >= OB_START: rasio total offer / total bid snapshot `imb_time`, lalu return close-to-close
    dari close hari itu (snapshot diambil sebelum close, jadi tidak ada look-ahead)."""
    p = ad.Params(kode=kode, market=cfg.market, ob_time=cfg.imb_time)
    c = chart["close"].to_numpy(float)
    out = []
    for i, d in enumerate(chart["tgl"]):
        if d < max(OB_START, cfg.start) or d > cfg.end:
            continue
        try:
            ob = ad.get_orderbook(client, kode, replace(p, ob_date=d.isoformat()))
        except BudgetHabis:
            raise
        except ad.AdimologyError:
            ob = None
        if not ob or ob["total_bid"] <= 0 or ob["total_offer"] <= 0:
            continue
        row = dict(kode=kode, t=d, close=c[i], total_bid=ob["total_bid"], total_offer=ob["total_offer"],
                   rasio_offer_bid=ob["total_offer"] / ob["total_bid"], n_levels=ob["n_levels"])
        for H in cfg.imb_horizons:
            row[f"ret_{H}"] = c[i + H] / c[i] - 1 if i + H < len(c) else np.nan
        out.append(row)
    return out


def _bucket(r: float) -> str:
    return next(lbl for lo, hi, lbl in IMB_BUCKETS if lo <= r < hi)


def summarize_imbalance(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rata-rata return per bucket vs baseline semua hari, CI bootstrap klaster per MINGGU
    (return 5 hari saling tumpang tindih antar hari berurutan)."""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    d = df.copy()
    d["bucket"] = d["rasio_offer_bid"].map(_bucket)
    d["t_hari"] = d["t"]
    d["t"] = d["t"].map(lambda x: x - dt.timedelta(days=x.weekday()))    # klaster = minggu
    summ, comp = [], []
    for H in cfg.imb_horizons:
        col = f"ret_{H}"
        dh = d[d[col].notna()].assign(naik=lambda x: (x[col] > 0).astype(float))
        groups = [("semua hari (baseline)", dh)] + [(lbl, dh[dh["bucket"] == lbl]) for *_, lbl in IMB_BUCKETS]
        for lbl, g in groups:
            if g.empty:
                continue
            b = _boot_mean([g], col, cfg.n_boot, cfg.seed)[:, 0]
            summ.append(dict(H=H, bucket=lbl, n=len(g), n_minggu=g["t"].nunique(), rata_return=g[col].mean(),
                             rata_return_ci=_fmt_ci(_ci(b)), median_return=g[col].median(), pct_naik=g["naik"].mean(),
                             rasio_median=g["rasio_offer_bid"].median()))
        pairs = [(lbl, g, "semua hari (baseline)", dh) for lbl, g in groups[1:]]
        offer, bid = dh[dh["rasio_offer_bid"] >= 1], dh[dh["rasio_offer_bid"] < 1]
        pairs.append(("offer >= bid", offer, "bid > offer", bid))
        for la, a, lb, b_ in pairs:
            if a.empty or b_.empty:
                continue
            for metrik, c in (("rata_return", col), ("pct_naik", "naik")):
                bb = _boot_mean([a, b_], c, cfg.n_boot, cfg.seed)
                lo, hi = _ci(bb[:, 0] - bb[:, 1])
                comp.append(dict(H=H, kelompok=la, pembanding=lb, metrik=metrik, n_kelompok=len(a),
                                 n_pembanding=len(b_), selisih=a[c].mean() - b_[c].mean(), ci_lo=lo, ci_hi=hi,
                                 kesimpulan=_verdict(lo, hi, "kelompok LEBIH TINGGI", "kelompok LEBIH RENDAH")))
    return pd.DataFrame(summ), pd.DataFrame(comp)


# ------------------------------------------------------------------- ringkasan
def _boot_mean(groups: list[pd.DataFrame], col: str, n_boot: int, seed: int) -> np.ndarray:
    """Bootstrap klaster per tanggal evaluasi (sinyal di minggu yang sama berkorelasi).

    groups: daftar DataFrame yang di-resample bersama (tanggal yang sama ditarik untuk semua grup).
    Hasil: array (n_boot, len(groups)) berisi rata-rata `col`.
    """
    dates = sorted(set().union(*[set(g["t"]) for g in groups]))
    if not dates:
        return np.full((n_boot, len(groups)), np.nan)
    pos = {d: k for k, d in enumerate(dates)}
    sums, cnts = [], []
    for g in groups:
        s, c = np.zeros(len(dates)), np.zeros(len(dates))
        v = g[["t", col]].dropna()
        np.add.at(s, v["t"].map(pos).to_numpy(int), v[col].to_numpy(float))
        np.add.at(c, v["t"].map(pos).to_numpy(int), 1)
        sums.append(s); cnts.append(c)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(dates), size=(n_boot, len(dates)))
    out = np.empty((n_boot, len(groups)))
    for j in range(len(groups)):
        num, den = sums[j][draw].sum(1), cnts[j][draw].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, j] = num / den
    return out


def _ci(a: np.ndarray) -> tuple[float, float]:
    a = a[np.isfinite(a)]
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if len(a) else (np.nan, np.nan)


STRATEGI = ("adimology", "akumulasi", "target_tetap", "acak")


def summarize(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tabel ringkasan per (window, regime, strategi, H) dan perbandingan Adimology vs baseline."""
    d = df[~df["split_dalam_rentang"].astype(bool)] if "split_dalam_rentang" in df else df
    summ, comp = [], []
    for w, m in _settings(cfg):
        dw = d[(d["window"] == w) & (d["mode_bandar"] == m)]
        for regime in ("semua", "proxy", "orderbook"):
            for H in cfg.horizons:
                grp = {}
                for s in STRATEGI:
                    g = dw[dw["strategi"] == s]
                    if regime != "semua":
                        g = g[g["regime"] == regime]
                    g = g[g[f"net_{H}"].notna()]
                    grp[s] = g
                    if g.empty:
                        continue
                    b = _boot_mean([g], f"net_{H}", cfg.n_boot, cfg.seed)[:, 0]
                    bh = _boot_mean([g], f"hit_{H}", cfg.n_boot, cfg.seed)[:, 0]
                    row = dict(window=w, mode_bandar=m, regime=regime, H=H, strategi=s, n=len(g), n_tanggal=g["t"].nunique(),
                               hit_tp=g[f"hit_{H}"].mean(), hit_tp_ci=_fmt_ci(_ci(bh)),
                               kena_sl=g[f"sl_{H}"].mean(), win_rate=(g[f"net_{H}"] > 0).mean(),
                               expectancy=g[f"net_{H}"].mean(), expectancy_ci=_fmt_ci(_ci(b)),
                               median_net=g[f"net_{H}"].median(), mfe_median=g[f"mfe_{H}"].median(),
                               mae_median=g[f"mae_{H}"].median(), kalibrasi_median=g[f"kalibrasi_{H}"].median())
                    if s == "adimology":
                        row["hit_tmax"] = g[f"hit_max_{H}"].mean()
                        row["upside_median"] = g["upside_real"].median()
                    summ.append(row)
                a = grp["adimology"]
                if a.empty:
                    continue
                for s in STRATEGI[1:]:
                    g = grp[s]
                    if g.empty:
                        continue
                    for metric in ("net", "hit"):
                        bb = _boot_mean([a, g], f"{metric}_{H}", cfg.n_boot, cfg.seed)
                        diff = bb[:, 0] - bb[:, 1]
                        lo, hi = _ci(diff)
                        point = a[f"{metric}_{H}"].mean() - g[f"{metric}_{H}"].mean()
                        comp.append(dict(window=w, mode_bandar=m, regime=regime, H=H, baseline=s,
                                         metrik="expectancy" if metric == "net" else "hit_tp",
                                         n_adimology=len(a), n_baseline=len(g), selisih=point,
                                         ci_lo=lo, ci_hi=hi, kesimpulan=_verdict(lo, hi)))
    return pd.DataFrame(summ), pd.DataFrame(comp)


def _fmt_ci(ci: tuple) -> str:
    return "" if not all(np.isfinite(ci)) else f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"


def _verdict(lo: float, hi: float, naik: str = "Adimology LEBIH BAIK", turun: str = "Adimology LEBIH BURUK") -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "data kurang"
    if lo > 0:
        return naik
    if hi < 0:
        return turun
    return "tidak berbeda nyata"


def diagnostics(df: pd.DataFrame) -> dict:
    a = df[df["strategi"] == "adimology"]
    both = a[a["lpp_orderbook"].notna() & (a["lpp_proxy"] > 0)] if "lpp_orderbook" in a else a.iloc[0:0]
    ratio = (both["lpp_orderbook"] / both["lpp_proxy"]).median() if len(both) else np.nan
    return dict(sinyal_adimology=len(a),
                orderbook_gagal_pakai_proxy=int((a.get("lpp_sumber") == "proxy_fallback").sum()),
                dibuang_split=int(df.get("split_dalam_rentang", pd.Series(dtype=bool))
                                                             .astype(bool).sum()),
                rasio_lpp_orderbook_per_proxy_median=ratio, n_pembanding_lpp=len(both))


# ------------------------------------------------------------------------- CLI
def parse_args(argv=None) -> tuple[Config, argparse.Namespace]:
    ap = argparse.ArgumentParser(description="Backtest target Adimology (Invezgo).")
    ap.add_argument("--tickers", help="daftar kode dipisah koma (default ~40 LQ45 + second liner)")
    ap.add_argument("--start", default="2024-06-01")
    ap.add_argument("--end", default=dt.date.today().isoformat())
    ap.add_argument("--windows", default="5", help="window broker summary, mis. 5,10,20")
    ap.add_argument("--modes", default="top1", help="mode bandar: top1, top3, atau top1,top3 (tanpa panggilan tambahan)")
    ap.add_argument("--study", default="all", choices=("all", "target", "imbalance"),
                    help="target = backtest target Adimology; imbalance = studi offer/bid (slide 12)")
    ap.add_argument("--imb-time", default="14:30", help="jam snapshot orderbook studi imbalance (WIB)")
    ap.add_argument("--horizons", default="5,10,20", help="horizon holding (hari bursa)")
    ap.add_argument("--exclude", default="", help="broker ritel tambahan yang dikecualikan (selain XL,YP,XC,PD)")
    ap.add_argument("--ob-time", default="14:30", help="jam snapshot orderbook historis (WIB, HH:mm)")
    ap.add_argument("--proxy-scale", type=float, default=1.0, help="pengali proksi lot per papan")
    ap.add_argument("--chart-chunk-days", type=int, default=400)
    ap.add_argument("--max-calls", type=int, help="batas panggilan API baru dalam satu run")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cache-dir", default=str(HERE / "cache"))
    ap.add_argument("--results-dir", default=str(HERE / "results"))
    ap.add_argument("--offline", action="store_true", help="hanya pakai cache, tanpa API key / panggilan")
    ap.add_argument("--yes", action="store_true", help="setuju menjalankan setelah melihat estimasi")
    a = ap.parse_args(argv)
    ints = lambda s: tuple(int(x) for x in s.split(",") if x.strip())
    tickers = [x.strip().upper() for x in a.tickers.split(",") if x.strip()] if a.tickers else list(UNIVERSE_DEFAULT)
    extra = tuple(x.strip().upper() for x in a.exclude.split(",") if x.strip())
    cfg = Config(tickers=tickers, start=dt.date.fromisoformat(a.start), end=dt.date.fromisoformat(a.end),
                 windows=ints(a.windows), modes=tuple(x.strip().lower() for x in a.modes.split(",") if x.strip()),
                 study=a.study, imb_time=a.imb_time, horizons=ints(a.horizons), retail=RETAIL_DEFAULT + extra,
                 ob_time=a.ob_time, proxy_scale=a.proxy_scale, chart_chunk_days=a.chart_chunk_days,
                 max_calls=a.max_calls, n_boot=a.n_boot, seed=a.seed,
                 cache_dir=Path(a.cache_dir), results_dir=Path(a.results_dir))
    return cfg, a


def main(argv=None, client: ad.InvezgoClient | None = None, log=print) -> int:
    """client: untuk tes (FakeClient). Tanpa client, dipakai InvezgoClient dengan env INVEZGO_API_KEY."""
    cfg, a = parse_args(argv)
    if set(cfg.modes) - {"top1", "top3"}:
        log("--modes hanya boleh top1 dan/atau top3.")
        return 2
    est = estimate_calls(cfg)
    log(f"Universe {est['tickers']} saham, {est['eval_dates']} tanggal evaluasi (Jumat), "
        f"{cfg.start}..{cfg.end}, window {list(cfg.windows)}, horizon {list(cfg.horizons)}")
    log(f"Estimasi panggilan API (batas atas, sebelum cache): {est['total']:,} "
        f"(chart {est['chart']:,}, broker window {est['window_summary']:,}, broker hari-terakhir "
        f"{est['lastday_summary']:,}, orderbook sinyal {est['orderbook']:,}, "
        f"orderbook studi imbalance {est['orderbook_imbalance']:,})")
    log(f"File cache yang sudah ada: {count_cached(cfg):,} (respons yang sudah di-cache tidak memakai kuota)")
    if a.offline:
        inner = None
    else:
        if not a.yes:
            log("Belum dijalankan. Tambahkan --yes untuk melanjutkan (atau --offline untuk cache saja).")
            return 2
        if client is None:
            key = os.environ.get("INVEZGO_API_KEY", "").strip()
            if not key:
                log("INVEZGO_API_KEY belum di-set di environment. Lihat backtest/README.md.")
                return 3
            client = ad.InvezgoClient(key)
        inner = client
    cc = CachingClient(inner, cfg.cache_dir, cfg.max_calls)
    try:
        df, imb = run_all(cc, cfg, log=log)
    except BudgetHabis as e:
        log(f"Berhenti: {e}. {cc.live_calls} panggilan baru tersimpan di cache; jalankan ulang untuk melanjutkan.")
        return 4
    log(f"Panggilan API baru: {cc.live_calls:,}, dari cache: {cc.cache_hits:,}"
        + (f", tidak ada di cache: {cc.misses_offline:,}" if inner is None else ""))
    if df.empty and imb.empty:
        log("Tidak ada baris hasil (tidak ada data atau tidak ada sinyal).")
        return 1
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return _report(df, imb, cfg, stamp, log)


def _report(df: pd.DataFrame, imb: pd.DataFrame, cfg: Config, stamp: str, log) -> int:
    opts = ("display.width", 220, "display.max_columns", 30, "display.max_rows", 1000,
            "display.float_format", "{:.4f}".format)
    files = []
    if not df.empty:
        summ, comp = summarize(df, cfg)
        for name, x in (("sinyal", df), ("ringkasan", summ), ("perbandingan", comp)):
            x.to_csv(cfg.results_dir / f"{name}_{stamp}.csv", index=False)
            files.append(f"{name}_{stamp}.csv")
        with pd.option_context(*opts):
            cols = ["window", "mode_bandar", "regime", "H", "strategi", "n", "n_tanggal", "hit_tp", "hit_tp_ci",
                    "kena_sl", "expectancy", "expectancy_ci", "median_net", "kalibrasi_median"]
            log("\n=== A. Backtest target Adimology: ringkasan (net return setelah fee beli 0,15% / jual 0,25%) ===")
            log(summ[[c for c in cols if c in summ]].to_string(index=False))
            if not comp.empty:
                log("\n=== A. Adimology vs baseline (selisih rata-rata, CI 95% bootstrap klaster per tanggal) ===")
                log(comp.to_string(index=False))
        log(f"\nDiagnostik: {diagnostics(df)}")
    elif cfg.study in ("all", "target"):
        log("Backtest target: tidak ada baris (tidak ada data atau sinyal).")
    if not imb.empty:
        isumm, icomp = summarize_imbalance(imb, cfg)
        for name, x in (("imbalance", imb), ("imbalance_ringkasan", isumm), ("imbalance_perbandingan", icomp)):
            x.to_csv(cfg.results_dir / f"{name}_{stamp}.csv", index=False)
            files.append(f"{name}_{stamp}.csv")
        with pd.option_context(*opts):
            log(f"\n=== B. Studi imbalance orderbook (slide 12), snapshot {cfg.imb_time}, return close-to-close "
                "tanpa fee ===")
            log(isumm.to_string(index=False))
            if not icomp.empty:
                log("\n=== B. Selisih kelompok vs pembanding (CI 95% bootstrap klaster per minggu) ===")
                log(icomp.to_string(index=False))
    elif cfg.study in ("all", "imbalance"):
        log("Studi imbalance: tidak ada snapshot orderbook (data historis baru ada sejak 2026-03-25).")
    log(f"\nHasil disimpan di {cfg.results_dir}/: {', '.join(files)}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
