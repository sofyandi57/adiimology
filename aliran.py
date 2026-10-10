"""Aliran dana lanjutan untuk melengkapi Adimology (materi 'The Hidden Flow of Money').

- Asing: net beli/jual investor asing di window (investor=f).
- Negosiasi: transaksi pasar NG (crossing) di window; bandar bisa keluar-masuk lewat nego tanpa menggerakkan harga.
- Konsistensi: akumulator teratas di window 1, 3, 7, 20 hari (cek bandar yang sama muncul terus).
- Afiliasi: apakah broker bandar dipakai pemegang saham >= 5% (laporan KSEI/IDX).
Edukasi, bukan ajakan transaksi.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

import adimology as ad

RITEL_DEFAULT = ("XL", "YP", "XC", "PD")      # Stockbit, Mirae, Ajaib, Indo Premier: dominan ritel
WINDOWS = (1, 3, 7, 20)


def _sum(df: pd.DataFrame, col: str) -> float:
    return float(pd.to_numeric(df[col], errors="coerce").sum()) if not df.empty and col in df else 0.0


def asing(client: ad.InvezgoClient, kode: str, awal, akhir, market: str = "RG") -> dict:
    bs = ad.broker_summary(client, kode, awal, akhir, "f", market)
    if bs.empty:
        return dict(net_value=0.0, net_lot=0.0, status="TIDAK ADA TRANSAKSI ASING", top_beli=[], top_jual=[])
    nv, nl = _sum(bs, "net_value"), _sum(bs, "net_volume") / 100
    srt = bs.sort_values("net_value")
    return dict(net_value=nv, net_lot=nl, status="NET BUY ASING" if nv > 0 else "NET SELL ASING" if nv < 0 else "NETRAL",
                top_beli=list(srt[srt["net_value"] > 0].tail(3)["code"])[::-1],
                top_jual=list(srt[srt["net_value"] < 0].head(3)["code"]))


def negosiasi(client: ad.InvezgoClient, kode: str, awal, akhir, vol_rg_lot: float, kode_bandar: list[str]) -> dict:
    ng = ad.broker_summary(client, kode, awal, akhir, "all", "NG")
    if ng.empty:
        return dict(vol_lot=0.0, porsi=0.0, bandar_net_lot=0.0, status="TIDAK ADA NEGO", top=[])
    vol = _sum(ng, "buy_volume") / 100
    bnd = ng[ng["code"].isin(kode_bandar)]
    bnet = _sum(bnd, "net_volume") / 100
    porsi = vol / vol_rg_lot if vol_rg_lot else 0.0
    if bnet < 0 and abs(bnet) > 0:
        status = "BANDAR JUAL DI NEGO"
    elif porsi >= 0.2:
        status = "NEGO BESAR"
    else:
        status = "NEGO KECIL"
    top = ng.reindex(ng["net_volume"].abs().sort_values(ascending=False).index).head(5)
    return dict(vol_lot=vol, porsi=porsi, bandar_net_lot=bnet, status=status,
                top=[dict(code=r["code"], net_lot=float(r["net_volume"]) / 100) for _, r in top.iterrows()])


def konsistensi(client: ad.InvezgoClient, kode: str, hari_bursa: list, excl, investor: str = "all",
                market: str = "RG", kode_bandar: list[str] | None = None) -> pd.DataFrame:
    """Akumulator teratas per window; hari_bursa = daftar tanggal bursa urut (dari chart)."""
    rows = []
    for w in WINDOWS:
        if len(hari_bursa) < w:
            continue
        bs = ad.broker_summary(client, kode, hari_bursa[-w], hari_bursa[-1], investor, market)
        if bs.empty:
            continue
        b = bs[~bs["code"].isin(set(excl))].sort_values("net_volume", ascending=False)
        if b.empty:
            continue
        t = b.iloc[0]
        rows.append(dict(window=f"{w} hari", dari=hari_bursa[-w], kode=t["code"], broker=t["name"],
                         net_lot=float(t["net_volume"]) / 100, avg_beli=float(t["buy_avg"]) if pd.notna(t["buy_avg"]) else None,
                         sama_dengan_bandar=bool(kode_bandar and t["code"] in kode_bandar)))
    return pd.DataFrame(rows)


def afiliasi(client: ad.InvezgoClient, kode: str, kode_bandar: list[str], hari: int = 365) -> dict:
    """Pemegang saham >= 5% yang menyimpan sahamnya di broker yang sama dengan bandar."""
    akhir = dt.date.today()
    raw = client.get("/analysis/shareholder-above", {"code": kode, "from": (akhir - dt.timedelta(days=hari)).isoformat(),
                                                    "to": akhir.isoformat(), "page": 1, "limit": 100})
    data = raw.get("data", []) if isinstance(raw, dict) else (raw or [])
    df = pd.DataFrame(data)
    if df.empty or "format_securities" not in df:
        return dict(cocok=[], semua_broker=[], n=0)
    df["format_securities"] = df["format_securities"].astype(str).str.upper().str.strip()
    cocok = df[df["format_securities"].isin(kode_bandar)]
    hasil = (cocok.sort_values("date").groupby("name").tail(1)
             [["name", "format_securities", "next_pct", "date"]].to_dict("records") if not cocok.empty else [])
    return dict(cocok=hasil, semua_broker=sorted(df["format_securities"].unique()), n=int(len(df)))


def ringkas(r: ad.Result, a: dict | None) -> list[dict]:
    """Tabel konfirmasi aliran (dipakai app, PDF, Excel)."""
    k = r.konsentrasi
    rows = [{"Konfirmasi": "Konsentrasi (top 3 beli vs top 15 jual)", "Status": k.get("status", "-"),
             "Detail": f"{k.get('top3_beli_lot', 0):,.0f} vs {k.get('top15_jual_lot', 0):,.0f} lot "
                       f"(rasio {k.get('rasio', float('nan')):.2f}); {k.get('n_beli', 0)} broker beli vs {k.get('n_jual', 0)} jual"}]
    if a:
        s = a["asing"]
        rows.append({"Konfirmasi": "Asing (window)", "Status": s["status"],
                     "Detail": f"net {s['net_value'] / 1e9:+,.2f} M · beli {', '.join(s['top_beli']) or '-'} · jual {', '.join(s['top_jual']) or '-'}"})
        n = a["nego"]
        rows.append({"Konfirmasi": "Pasar negosiasi (window)", "Status": n["status"],
                     "Detail": f"{n['vol_lot']:,.0f} lot ({n['porsi']:.0%} dari volume RG) · net bandar di nego {n['bandar_net_lot']:+,.0f} lot"})
        kc = a["konsistensi"]
        if not kc.empty:
            rows.append({"Konfirmasi": "Konsistensi bandar 1/3/7/20 hari",
                         "Status": f"{int(kc['sama_dengan_bandar'].sum())} dari {len(kc)} window",
                         "Detail": " · ".join(f"{w}: {c}" for w, c in zip(kc["window"], kc["kode"]))})
        f = a.get("afiliasi")
        if f is not None:
            rows.append({"Konfirmasi": "Afiliasi pemegang saham ≥5%",
                         "Status": "TERAFILIASI" if f["cocok"] else "TIDAK DITEMUKAN",
                         "Detail": "; ".join(f"{x['name']} ({x['format_securities']}, {x['next_pct']:.1f}%)" for x in f["cocok"][:3])
                                   or f"broker bandar tidak dipakai pemegang ≥5% (12 bln, {f['n']} laporan)"})
    return rows


# ----------------------------------------------------------------- minat ritel
RITEL_ANALISA = ("XL", "XC", "PD", "CC", "YP")      # definisi ritel pengguna untuk membaca minat ritel (bukan daftar eksklusi bandar)


def ritel_flow(bs: pd.DataFrame, kode_ritel, harga: float | None = None, ambang: float = 0.02) -> dict:
    """Net beli/jual broker ritel di broker summary: net jual = ritel 'dikeringkan', net beli = ritel akumulasi."""
    codes = [c.strip().upper() for c in kode_ritel if c and c.strip()]
    if bs is None or bs.empty or not codes:
        return dict(status="TIDAK DIUKUR", codes=codes)
    d = bs.copy()
    for c in ("net_volume", "buy_volume", "sell_volume", "buy_avg", "sell_avg"):
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    total = float(d["buy_volume"].sum()) / 100
    b = d[d["code"].isin(codes)]
    if b.empty or total <= 0:
        return dict(status="TIDAK ADA TRANSAKSI RITEL", codes=codes, per_broker=[], net_lot=0.0, net_pct=0.0)
    net = float(b["net_volume"].sum()) / 100
    wb = b["buy_volume"].where(b["buy_avg"].notna(), 0)
    avg_beli = float((b["buy_avg"].fillna(0) * wb).sum() / wb.sum()) if wb.sum() > 0 else float("nan")
    pct = net / total
    status = "RITEL NET JUAL (dikeringkan)" if pct <= -ambang else "RITEL NET BELI (akumulasi ritel)" if pct >= ambang else "RITEL NETRAL"
    per = [dict(code=r["code"], net_lot=float(r["net_volume"]) / 100, buy_avg=None if pd.isna(r["buy_avg"]) else float(r["buy_avg"]),
                sell_avg=None if pd.isna(r["sell_avg"]) else float(r["sell_avg"])) for _, r in b.sort_values("net_volume").iterrows()]
    return dict(status=status, codes=codes, net_lot=net, net_pct=pct, porsi_beli=float(b["buy_volume"].sum()) / 100 / total,
                porsi_jual=float(b["sell_volume"].sum()) / 100 / total, avg_beli=avg_beli,
                floating=(harga / avg_beli - 1) if harga and avg_beli and avg_beli > 0 else float("nan"), per_broker=per)


def minat_ritel(sid: dict | None, fw: dict | None, fl: dict | None = None, ret20: float | None = None) -> dict:
    """Gabungkan SID (jumlah investor) dan aliran broker ritel jadi satu bacaan minat ritel.

    SID turun = minat ritel menurun (saham sedang tidak menarik). Bila ritel juga net jual dan harga sideways/turun,
    itu pola 'ritel dikeringkan' yang dalam materi dianggap tahap akumulasi pihak besar, tetapi BUKAN sinyal naik segera.
    """
    ada_sid = bool(sid and sid.get("status") not in (None, "TIDAK DIUKUR"))
    arah = sid["arah"] if ada_sid else None
    f = fw if fw and fw.get("status") not in ("TIDAK DIUKUR",) else None
    jual = f is not None and f["status"].startswith("RITEL NET JUAL")
    beli = f is not None and f["status"].startswith("RITEL NET BELI")
    poin = []
    if ada_sid:
        poin.append(f"SID {sid['sid_awal']:,.0f} -> {sid['sid_akhir']:,.0f} ({sid['chg']:+.1%}, {sid['dari']}..{sid['sampai']}); "
                    f"bulan terakhir {sid['chg_1b']:+.1%}. Data bulanan (KSEI), bisa terlambat.")
    if f:
        poin.append(f"Broker ritel {'/'.join(f['codes'])} window: net {f['net_lot']:+,.0f} lot ({f['net_pct']:+.1%} dari volume) -> {f['status']}")
    if fl and fl.get("status") not in (None, "TIDAK DIUKUR", "TIDAK ADA TRANSAKSI RITEL"):
        poin.append(f"Periode panjang: net {fl['net_lot']:+,.0f} lot ({fl['net_pct']:+.1%}) -> {fl['status']}")
    sideways = ret20 is not None and -0.10 <= ret20 <= 0.10
    if arah == "TURUN" and jual:
        status = "RITEL KELUAR / DIKERINGKAN"
        ring = ("SID turun dan broker ritel net jual: minat ritel menurun (saham sedang tidak menarik)."
                + (" Harga sideways/turun: pola yang dalam materi dianggap akumulasi pihak besar, tetapi belum sinyal naik; "
                   "konfirmasi dengan konsentrasi beli bandar." if sideways or (ret20 is not None and ret20 < 0) else
                   " Harga sudah naik: ritel menjual ke kenaikan, waspada distribusi."))
    elif arah == "TURUN" and beli:
        status = "CAMPURAN: SID TURUN, RITEL NET BELI"
        ring = "Jumlah investor turun tetapi broker ritel membeli: sisa ritel menambah posisi; minat umum menurun."
    elif arah == "TURUN":
        status = "MINAT RITEL MENURUN"
        ring = "SID turun: saham sedang tidak menarik bagi ritel."
    elif arah == "NAIK" and beli:
        status = "RITEL MASUK / AKUMULASI RITEL"
        ring = ("SID naik dan broker ritel net beli: minat ritel meningkat. Bila harga sudah naik atau bandar net jual, "
                "waspada bandar mendistribusikan barang ke ritel.")
    elif arah == "NAIK" and jual:
        status = "CAMPURAN: SID NAIK, RITEL NET JUAL"
        ring = "Investor baru bertambah tetapi broker ritel net jual: sinyal tidak searah."
    elif arah == "NAIK":
        status = "MINAT RITEL MENINGKAT"
        ring = "SID naik: saham makin diminati ritel."
    elif f is not None:
        status = "RITEL DIKERINGKAN (tanpa data SID)" if jual else "RITEL AKUMULASI (tanpa data SID)" if beli else "RITEL NETRAL"
        ring = "Hanya aliran broker ritel yang terukur; SID tidak tersedia."
    else:
        status, ring = "TIDAK DIUKUR", "SID dan aliran broker ritel tidak tersedia."
    minat = {"TURUN": "MENURUN (tidak menarik saat ini)", "NAIK": "MENINGKAT", "STABIL": "STABIL"}.get(arah, "TIDAK DIUKUR")
    return dict(status=status, minat=minat, ringkas=ring, poin=poin, sid=sid if ada_sid else None, window=f, panjang=fl)
