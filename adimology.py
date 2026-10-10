"""Mesin hitung Adimology (bandarmology) berbasis REST API Invezgo.

Tidak bergantung pada Streamlit, sehingga bisa diuji dan dipakai ulang (mis. backtest).
Edukasi, bukan ajakan transaksi.
"""
from __future__ import annotations

import datetime as dt
import re
import time
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://api.invezgo.com"


class AdimologyError(Exception):
    """Error yang pesannya aman ditampilkan ke pengguna."""


class ParamDitolak(AdimologyError):
    """API membalas 422: parameter tidak bisa dipenuhi (mis. limit melebihi data yang tersedia)."""


# --------------------------------------------------------------------------- util
def _num(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


# ------------------------------------------------------------------ fraksi harga
# Standar BEI: <200:1 | 200-<500:2 | 500-<2000:5 | 2000-<5000:10 | >=5000:25.
TIERS = [(0, 1), (200, 2), (500, 5), (2000, 10), (5000, 25)]


def fraksi(price: float) -> int:
    f = TIERS[0][1]
    for low, fr in TIERS:
        if price >= low:
            f = fr
    return f


def _next_boundary(price: float) -> float:
    for low, _ in TIERS:
        if low > price:
            return float(low)
    return float("inf")


def count_papan(low: float, high: float) -> float:
    """Jumlah tick dari low ke high dengan fraksi yang berubah per rentang."""
    if high <= low:
        return 0.0
    p, n = float(low), 0.0
    while p < high - 1e-9:
        nxt = min(high, _next_boundary(p))
        n += (nxt - p) / fraksi(p)
        p = nxt
    return n


def step_up(price: float, n_ticks: float) -> float:
    """Naikkan harga n_ticks (boleh pecahan) dengan fraksi yang berubah per rentang."""
    p, n = float(price), float(n_ticks)
    while n > 1e-12:
        f = fraksi(p)
        room = (_next_boundary(p) - p) / f
        if n <= room:
            p += n * f
            n = 0
        else:
            p = _next_boundary(p)
            n -= room
    return p


def round_tick(price: float) -> float:
    f = fraksi(price)
    return round(price / f) * f


# ------------------------------------------------------------------- API client
def fetch_json(api_key: str, path: str, params_items: tuple, retries: int = 3) -> Any:
    """GET ke Invezgo dengan retry. params_items = tuple((k, v), ...) agar mudah di-cache."""
    headers = {"Authorization": f"Bearer {api_key}"}
    last = None
    for i in range(retries):
        try:
            r = requests.get(BASE_URL + path, headers=headers, params=dict(params_items), timeout=30)
        except requests.RequestException as e:  # jaringan
            last = e
            time.sleep(1.5 * (i + 1))
            continue
        if r.status_code == 429:
            last = RuntimeError("rate limit")
            time.sleep(2 * (i + 1))
            continue
        if r.status_code in (401, 403):
            raise AdimologyError("API key Invezgo ditolak (401/403). Periksa key dan paket langganan Anda.")
        if r.status_code == 404:
            raise AdimologyError(f"Data tidak ditemukan ({path}). Periksa kode saham.")
        if r.status_code == 422:  # tidak perlu diulang; pemanggil boleh mencoba parameter lain
            raise ParamDitolak(f"Parameter ditolak API (422) untuk {path}.")
        try:
            r.raise_for_status()
        except requests.RequestException as e:
            last = e
            time.sleep(1.5 * (i + 1))
            continue
        return r.json()
    raise AdimologyError(f"Gagal mengambil {path}: {last}")


class InvezgoClient:
    """Subclass dan override _fetch bila ingin menambah cache (lihat app.py)."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def _fetch(self, path: str, params_items: tuple) -> Any:
        return fetch_json(self.api_key, path, params_items)

    def get(self, path: str, params: dict | None = None) -> Any:
        items = tuple(sorted((k, str(v)) for k, v in (params or {}).items() if v is not None))
        return self._fetch(path, items)


# ----------------------------------------------------------------------- params
@dataclass(frozen=True)
class Params:
    kode: str = "MCAS"
    tanggal_akhir: dt.date | None = None      # None = hari bursa terakhir
    window_hari: int = 5
    investor: str = "all"                      # all / f / d
    market: str = "RG"                         # RG / NG / TN
    broker_manual: str | None = None
    broker_dikecualikan: tuple = ("XL",)
    margin_a: float = 0.05
    faktor_realistis: float = 0.5
    ob_date: str | None = None                 # snapshot historis (WIB), keduanya wajib
    ob_time: str | None = None
    lot_per_papan_override: float = 0.0        # >0 = abaikan hitungan orderbook
    min_upside: float = 0.03
    maks_kejar: float = 0.05
    distribusi_pct: float = 0.25
    modal: float = 10_000_000
    risiko_pct: float = 0.01
    fee_beli: float = 0.0015
    fee_jual: float = 0.0025


@dataclass
class Result:
    params: Params
    tgl_awal: dt.date
    tgl_akhir: dt.date
    harga: float
    chart: pd.DataFrame
    bs: pd.DataFrame
    total_vol_lot: float
    bandar: dict
    harian: pd.DataFrame
    net_terakhir: float
    distribusi: bool
    floating: float
    ob: dict | None
    lot_per_papan: float
    lpp_sumber: str
    calc: dict
    verdict: str
    setup_ok: bool
    alasan: list = field(default_factory=list)
    plan: dict = field(default_factory=dict)
    sens: pd.DataFrame | None = None


# ------------------------------------------------------------------- pengambilan
def get_chart(client: InvezgoClient, kode: str, tanggal_akhir: dt.date | None) -> pd.DataFrame:
    end = tanggal_akhir or dt.date.today()
    raw = client.get(f"/analysis/chart/stock/{kode}",
                     {"from": (end - dt.timedelta(days=90)).isoformat(), "to": end.isoformat()})
    df = pd.DataFrame(raw)
    if df.empty:
        raise AdimologyError(f"Tidak ada data harga untuk {kode}. Periksa kode saham.")
    df["tgl"] = (pd.to_datetime(df["date"], utc=True) + pd.Timedelta(hours=7)).dt.date
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("tgl").reset_index(drop=True)
    if tanggal_akhir:
        df = df[df["tgl"] <= tanggal_akhir].reset_index(drop=True)
    if df.empty:
        raise AdimologyError("Tidak ada data harga sampai tanggal yang dipilih.")
    return df


def broker_summary(client: InvezgoClient, kode: str, d_from, d_to, investor: str, market: str) -> pd.DataFrame:
    raw = client.get(f"/analysis/summary/stock/{kode}",
                     {"from": str(d_from), "to": str(d_to), "investor": investor, "market": market})
    df = pd.DataFrame(raw)
    if df.empty:
        return df
    for c in ["buy_freq", "buy_volume", "buy_value", "sell_freq", "sell_volume", "sell_value",
              "buy_avg", "sell_avg", "net_value", "net_volume", "net_freq"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _parse_levels(items, side: str) -> dict:
    levels: dict = {}
    for it in (items or []):
        if not isinstance(it, dict):
            continue
        for k, v in it.items():
            m = re.fullmatch(rf"{side}(\d+)price", k)
            if m:
                price, lot = _num(v), _num(it.get(f"{side}{m.group(1)}lot"))
                if price > 0:
                    levels[price] = levels.get(price, 0) + lot
    return levels


def get_orderbook(client: InvezgoClient, kode: str, p: Params) -> dict | None:
    params = {"market": p.market}
    if p.ob_date and p.ob_time:
        params.update({"date": p.ob_date, "time": p.ob_time})
    ob = client.get(f"/analysis/order-book/{kode}", params)
    if isinstance(ob, dict) and isinstance(ob.get("data"), dict):
        ob = ob["data"]
    if not isinstance(ob, dict):
        return None
    bids, offers = _parse_levels(ob.get("bid"), "bid"), _parse_levels(ob.get("offer"), "offer")
    if not bids or not offers:
        return None
    papan = max(count_papan(min(bids), max(offers)), 1.0)
    total_bid, total_offer = sum(bids.values()), sum(offers.values())
    return dict(bids=bids, offers=offers, total_bid=total_bid, total_offer=total_offer,
                bid_terbawah=min(bids), offer_teratas=max(offers),
                best_bid=max(bids), best_offer=min(offers),
                papan=papan, lot_per_papan=(total_bid + total_offer) / papan,
                n_levels=len(bids) + len(offers))


# --------------------------------------------------------------------- analisis
def analyze(client: InvezgoClient, p: Params) -> Result:
    kode = p.kode.strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{4}", kode):
        raise AdimologyError("Kode saham harus 4 karakter, mis. MCAS.")
    p = replace(p, kode=kode)

    chart = get_chart(client, kode, p.tanggal_akhir)
    hari = chart["tgl"].tolist()[-p.window_hari:]
    tgl_awal, tgl_akhir = hari[0], hari[-1]
    harga = float(chart["close"].iloc[-1])

    bs = broker_summary(client, kode, tgl_awal, tgl_akhir, p.investor, p.market)
    if bs.empty:
        raise AdimologyError("Broker summary kosong untuk window ini.")
    total_vol = float(bs["buy_volume"].sum())
    bs = bs.copy()
    bs["net_lot"] = bs["net_volume"] / 100
    bs["porsi_net_%"] = bs["net_volume"] / total_vol * 100 if total_vol else 0.0

    excl = {c.strip().upper() for c in p.broker_dikecualikan}
    if p.broker_manual:
        sel = bs[bs["code"] == p.broker_manual.strip().upper()]
        if sel.empty:
            raise AdimologyError(f"Broker {p.broker_manual} tidak ada di window ini.")
        row = sel.iloc[0]
    else:
        kand = bs[~bs["code"].isin(excl)].sort_values("net_volume", ascending=False)
        if kand.empty:
            raise AdimologyError("Tidak ada broker kandidat bandar.")
        row = kand.iloc[0]
    bandar = dict(code=row["code"], name=row["name"], avg=float(row["buy_avg"]) if pd.notna(row["buy_avg"]) else float("nan"),
                  net_lot=float(row["net_volume"]) / 100, porsi=float(row["porsi_net_%"]))
    if not np.isfinite(bandar["avg"]):
        raise AdimologyError(f"Broker {bandar['code']} tidak punya transaksi beli di window ini (avg beli kosong).")

    rows = []
    for d in hari:
        x = broker_summary(client, kode, d, d, p.investor, p.market)
        r = x[x["code"] == bandar["code"]] if not x.empty else x
        if r.empty:
            rows.append(dict(tgl=d, net_lot=0.0, avg_beli=np.nan, avg_jual=np.nan))
        else:
            r = r.iloc[0]
            rows.append(dict(tgl=d, net_lot=r["net_volume"] / 100, avg_beli=r["buy_avg"], avg_jual=r["sell_avg"]))
    harian = pd.DataFrame(rows)
    harian["net_kumulatif_lot"] = harian["net_lot"].cumsum()
    net_terakhir = float(harian["net_lot"].iloc[-1])
    barang = bandar["net_lot"]
    distribusi = net_terakhir < 0 and abs(net_terakhir) >= p.distribusi_pct * max(barang, 1e-9)
    floating = harga / bandar["avg"] - 1

    ob = get_orderbook(client, kode, p)
    if p.lot_per_papan_override > 0:
        lpp, lpp_sumber = float(p.lot_per_papan_override), "manual"
    elif ob:
        lpp, lpp_sumber = ob["lot_per_papan"], "orderbook"
    else:
        raise AdimologyError("Orderbook kosong (pasar tutup / tanpa antrean). Isi 'lot per papan manual', "
                             "pakai snapshot historis, atau jalankan saat jam bursa.")

    a = bandar["avg"] * p.margin_a
    dasar = bandar["avg"] + a
    pp = barang / lpp if lpp > 0 else float("nan")
    t_real = step_up(dasar, max(pp, 0) * p.faktor_realistis)
    t_max = step_up(dasar, max(pp, 0))
    calc = dict(a=a, dasar=dasar, barang=barang, p=pp, t_real=t_real, t_max=t_max,
                t_real_r=round_tick(t_real), t_max_r=round_tick(t_max),
                upside_real=t_real / harga - 1, upside_max=t_max / harga - 1)

    # ---- verdict (aturan eksplisit)
    sl = float(chart[chart["tgl"].isin(hari)]["low"].min())
    upside = calc["upside_real"]
    rr = (t_real - harga) / (harga - sl) if harga > sl else float("nan")
    alasan, setup_ok = [], False
    if barang <= 0:
        verdict = "TIDAK ADA SINYAL"; alasan.append("tidak ada akumulator bersih (net beli <= 0)")
    elif distribusi:
        verdict = "TIDAK ADA SINYAL"; alasan.append(f"bandar mulai distribusi (net hari terakhir {net_terakhir:,.0f} lot)")
    elif harga >= t_real:
        verdict = "TIDAK ADA SINYAL"; alasan.append("harga sudah di atas target realistis")
    elif upside < p.min_upside:
        verdict = "TIDAK ADA SINYAL"; alasan.append(f"upside {upside:.1%} < minimal {p.min_upside:.0%}")
    elif harga > bandar["avg"] * (1 + p.maks_kejar):
        verdict = "TIDAK ADA SINYAL"; alasan.append(f"harga {harga / bandar['avg'] - 1:+.1%} di atas avg bandar (mengejar)")
    else:
        verdict = "SETUP TERPENUHI"; setup_ok = True
        alasan.append("akumulasi bersih, tidak distribusi, harga dekat modal bandar")
    if bandar["porsi"] < 3:
        alasan.append(f"catatan: porsi net {bandar['porsi']:.1f}% dari volume, bandar tidak dominan")
    if floating < 0:
        alasan.append(f"catatan: bandar rugi mengambang {floating:.1%}")
    if lpp_sumber == "orderbook" and ob and ob["n_levels"] <= 4:
        alasan.append("catatan: orderbook hanya 1-2 level, lot per papan tidak representatif")

    plan = dict(sl=sl, rr=rr, upside=upside, lot=0, modal_pakai=0.0, laba_target=0.0, rugi_sl=0.0)
    if harga > sl:
        lot_risiko = int((p.modal * p.risiko_pct) / ((harga - sl) * 100))
        lot_modal = int(p.modal / (harga * 100 * (1 + p.fee_beli)))
        lot = max(min(lot_risiko, lot_modal), 0)
        modal_pakai = harga * lot * 100 * (1 + p.fee_beli)
        plan.update(lot=lot, modal_pakai=modal_pakai,
                    laba_target=calc["t_real_r"] * lot * 100 * (1 - p.fee_jual) - modal_pakai,
                    rugi_sl=sl * lot * 100 * (1 - p.fee_jual) - modal_pakai)

    # ---- sensitivitas
    lpps = sorted({round(lpp), round(lpp * 0.5), round(lpp * 2), 1000, 3000, 5000, 10000})
    bbs = sorted({round(barang), round(barang * 0.5), round(max(-net_terakhir, 0)) or round(barang * 0.3)})
    rows = []
    for bb in bbs:
        for l in lpps:
            q = bb / l if l else float("nan")
            rows.append(dict(barang_bandar_lot=bb, lot_per_papan=l, p=q,
                             realistis=round_tick(step_up(dasar, q * p.faktor_realistis)),
                             max=round_tick(step_up(dasar, q))))
    sens = pd.DataFrame(rows)

    return Result(params=p, tgl_awal=tgl_awal, tgl_akhir=tgl_akhir, harga=harga, chart=chart, bs=bs,
                  total_vol_lot=total_vol / 100, bandar=bandar, harian=harian, net_terakhir=net_terakhir,
                  distribusi=distribusi, floating=floating, ob=ob, lot_per_papan=lpp, lpp_sumber=lpp_sumber,
                  calc=calc, verdict=verdict, setup_ok=setup_ok, alasan=alasan, plan=plan, sens=sens)
