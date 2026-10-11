"""Buku besar broker harian yang persisten: penyimpanan, impor (backfill + inkremental), rekonsiliasi, dan validasi integritas.

Sumber: Invezgo /analysis/summary/stock/{kode}?from=d&to=d (satu panggilan per hari bursa; seluruh broker; pasar RG).
Satuan SUMBER (terverifikasi dari data): volume = LEMBAR (bukan lot), nilai = Rupiah, harga rata-rata = nilai / lembar.
Respons mentah disimpan terpisah dari rekaman ternormalisasi; kunci unik: ticker + tanggal + kode broker + sumber data.
Edukasi, bukan ajakan transaksi. Kode broker = perantara banyak nasabah, bukan satu pemilik.
"""
from __future__ import annotations

import datetime as dt
import json
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

import adimology as ad
import teknikal as tk

SHARES_PER_LOT = 100
LOT_TOL = 100                         # toleransi selisih beli-jual vs net sumber (1 lot = 100 lembar)
SUMBER = "invezgo_summary_daily"
SESI_DEFAULT = 200                    # default riwayat = 200 SESI BURSA (bukan hari kalender)
SOURCE_VERSION = "invezgo_summary_daily:v1"
EP_SUMMARY = "/analysis/summary/stock"
EP_CHART = "/analysis/chart/stock"
WIB = ZoneInfo("Asia/Jakarta")
JAM_FINAL = dt.time(18, 0)             # broker summary EOD diperbarui ~17:30-18:00 WIB

META = sa.MetaData()
T_RAW = sa.Table(
    "ledger_raw", META, sa.Column("id", sa.Integer, primary_key=True, autoincrement=True), sa.Column("ticker", sa.String(10), nullable=False),
    sa.Column("endpoint", sa.String(80), nullable=False), sa.Column("params_key", sa.String(120), nullable=False), sa.Column("trade_date", sa.Date),
    sa.Column("fetched_at", sa.DateTime), sa.Column("n_rows", sa.Integer), sa.Column("is_final", sa.Boolean), sa.Column("payload", sa.Text),
    sa.UniqueConstraint("ticker", "endpoint", "params_key", name="uq_ledger_raw"))
T_BROKER = sa.Table(
    "ledger_broker_daily", META, sa.Column("ticker", sa.String(10), primary_key=True), sa.Column("trade_date", sa.Date, primary_key=True),
    sa.Column("broker_code", sa.String(4), primary_key=True), sa.Column("data_source", sa.String(40), primary_key=True),
    sa.Column("broker_name", sa.String(80)), sa.Column("buy_shares", sa.Float), sa.Column("sell_shares", sa.Float), sa.Column("buy_value", sa.Float),
    sa.Column("sell_value", sa.Float), sa.Column("buy_avg_raw", sa.Float), sa.Column("sell_avg_raw", sa.Float), sa.Column("net_shares_raw", sa.Float),
    sa.Column("net_value_raw", sa.Float), sa.Column("retrieved_at", sa.DateTime), sa.Column("is_final", sa.Boolean))
T_MARKET = sa.Table(
    "ledger_market_daily", META, sa.Column("ticker", sa.String(10), primary_key=True), sa.Column("trade_date", sa.Date, primary_key=True),
    sa.Column("data_source", sa.String(40), primary_key=True), sa.Column("open", sa.Float), sa.Column("high", sa.Float), sa.Column("low", sa.Float),
    sa.Column("close", sa.Float), sa.Column("volume_shares", sa.Float), sa.Column("retrieved_at", sa.DateTime))
T_META = sa.Table(
    "ledger_meta", META, sa.Column("ticker", sa.String(10), primary_key=True), sa.Column("first_available_trade_date", sa.Date),
    sa.Column("last_available_trade_date", sa.Date), sa.Column("sessions_loaded", sa.Integer), sa.Column("target_sessions", sa.Integer),
    sa.Column("market_sessions", sa.Integer), sa.Column("history_complete", sa.Boolean), sa.Column("last_successful_sync", sa.DateTime),
    sa.Column("source_version", sa.String(60)))
T_DERIVED = sa.Table(
    "ledger_derived", META, sa.Column("ticker", sa.String(10), primary_key=True), sa.Column("as_of_date", sa.Date, primary_key=True),
    sa.Column("broker_code", sa.String(4), primary_key=True), sa.Column("calc_version", sa.String(30), primary_key=True),
    sa.Column("computed_at", sa.DateTime), sa.Column("metrics", sa.Text))
T_SIGNAL = sa.Table(
    "ledger_signal", META, sa.Column("ticker", sa.String(10), primary_key=True), sa.Column("signal_date", sa.Date, primary_key=True),
    sa.Column("calc_version", sa.String(30), primary_key=True), sa.Column("computed_at", sa.DateTime), sa.Column("fingerprint", sa.String(80)),
    sa.Column("bandar_score", sa.Float), sa.Column("episode_status", sa.String(24)), sa.Column("confidence_score", sa.Float), sa.Column("payload", sa.Text))
T_LOG = sa.Table(
    "ledger_api_log", META, sa.Column("id", sa.Integer, primary_key=True, autoincrement=True), sa.Column("ts", sa.DateTime), sa.Column("ticker", sa.String(10)),
    sa.Column("endpoint", sa.String(80)), sa.Column("params", sa.String(160)), sa.Column("status", sa.String(10)), sa.Column("n_rows", sa.Integer),
    sa.Column("ms", sa.Integer), sa.Column("error", sa.String(240)))


def _utc_naive(now: dt.datetime | None = None) -> dt.datetime:
    n = now or dt.datetime.now(dt.timezone.utc)
    return n.astimezone(dt.timezone.utc).replace(tzinfo=None) if n.tzinfo else n


def engine(url: str) -> sa.Engine:
    if url.startswith(("postgres://", "postgresql://")):
        url = "postgresql+psycopg://" + url.split("://", 1)[1]
    eng = sa.create_engine(url, pool_pre_ping=True)
    META.create_all(eng)
    return eng


def upsert(conn, table: sa.Table, rows: list[dict], keys: list[str]) -> None:
    """Upsert idempoten (kunci unik); kunci non-PK dikelola lewat delete+insert pada dialek lain."""
    if not rows:
        return
    d = conn.dialect.name
    if d in ("postgresql", "sqlite"):
        ins = (pg_insert if d == "postgresql" else sqlite_insert)(table).values(rows)
        upd = {c.name: ins.excluded[c.name] for c in table.columns if c.name not in keys and not c.primary_key}
        conn.execute(ins.on_conflict_do_update(index_elements=keys, set_=upd) if upd else ins.on_conflict_do_nothing(index_elements=keys))
    else:
        for r in rows:
            conn.execute(table.delete().where(sa.and_(*[table.c[k] == r[k] for k in keys])))
        conn.execute(table.insert(), rows)


# --------------------------------------------------------------------- final?
def hari_final(tgl: dt.date, now: dt.datetime | None = None) -> bool:
    """Data broker sebuah hari dianggap final bila hari itu sudah lewat, atau hari ini setelah 18:00 WIB (EOD)."""
    n = (now or dt.datetime.now(dt.timezone.utc)).astimezone(WIB)
    return tgl < n.date() or (tgl == n.date() and n.time() >= JAM_FINAL)


# ------------------------------------------------------------------ normalisasi
def _f(x):
    try:
        v = float(x)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


def normalisasi_summary(kode: str, tgl: dt.date, payload, retrieved: dt.datetime, final: bool, raw_id: int | None = None) -> list[dict]:
    """Respons summary harian -> rekaman per broker. SATUAN ASLI dipertahankan (lembar, Rupiah); lot dihitung saat dibaca."""
    data = payload.get("data", []) if isinstance(payload, dict) else (payload or [])
    out = []
    for r in data:
        if not isinstance(r, dict) or not r.get("code"):
            continue
        out.append(dict(ticker=kode, trade_date=tgl, broker_code=str(r["code"]).strip().upper(), data_source=SUMBER, broker_name=str(r.get("name") or "")[:80],
                        buy_shares=_f(r.get("buy_volume")) or 0.0, sell_shares=_f(r.get("sell_volume")) or 0.0,
                        buy_value=_f(r.get("buy_value")) or 0.0, sell_value=_f(r.get("sell_value")) or 0.0,
                        buy_avg_raw=_f(r.get("buy_avg")), sell_avg_raw=_f(r.get("sell_avg")), net_shares_raw=_f(r.get("net_volume")),
                        net_value_raw=_f(r.get("net_value")), retrieved_at=retrieved, is_final=final))
    return out


# ------------------------------------------------------------------- impor
def _log(eng, kode, endpoint, params, status, n_rows, ms, error=None, now=None):
    with eng.begin() as c:
        c.execute(T_LOG.insert().values(ts=_utc_naive(now), ticker=kode, endpoint=endpoint, params=str(params)[:160], status=status,
                                        n_rows=n_rows, ms=ms, error=(str(error)[:240] if error else None)))


def _raw_ada(eng, kode, endpoint, key):
    with eng.connect() as c:
        return c.execute(sa.select(T_RAW.c.id, T_RAW.c.is_final, T_RAW.c.payload, T_RAW.c.n_rows).where(
            sa.and_(T_RAW.c.ticker == kode, T_RAW.c.endpoint == endpoint, T_RAW.c.params_key == key))).first()


def ambil_hari(client, eng, kode: str, tgl: dt.date, now: dt.datetime | None = None, paksa: bool = False) -> dict:
    """Satu hari bursa: pakai respons mentah final bila ada (tanpa panggilan API); jika tidak, ambil, simpan mentah, normalisasi, upsert."""
    key, final = f"{tgl}|all|RG", hari_final(tgl, now)
    ada = _raw_ada(eng, kode, EP_SUMMARY, key)
    dari_api = False
    if ada and ada.is_final and not paksa:
        payload, raw_id = json.loads(ada.payload), ada.id
    else:
        t0 = time.time()
        try:
            payload = client.get(f"{EP_SUMMARY}/{kode}", {"from": str(tgl), "to": str(tgl), "investor": "all", "market": "RG"})
        except Exception as e:
            _log(eng, kode, EP_SUMMARY, key, "ERR", 0, int((time.time() - t0) * 1000), e, now)
            raise
        n = len(payload) if isinstance(payload, list) else len((payload or {}).get("data", []))
        _log(eng, kode, EP_SUMMARY, key, "OK", n, int((time.time() - t0) * 1000), None, now)
        retrieved = _utc_naive(now)
        with eng.begin() as c:
            upsert(c, T_RAW, [dict(ticker=kode, endpoint=EP_SUMMARY, params_key=key, trade_date=tgl, fetched_at=retrieved, n_rows=n,
                                   is_final=final, payload=json.dumps(payload))], ["ticker", "endpoint", "params_key"])
            raw_id = c.execute(sa.select(T_RAW.c.id).where(sa.and_(T_RAW.c.ticker == kode, T_RAW.c.endpoint == EP_SUMMARY,
                                                                  T_RAW.c.params_key == key))).scalar()
        dari_api = True
    rows = normalisasi_summary(kode, tgl, payload, _utc_naive(now), final, raw_id)
    with eng.begin() as c:
        upsert(c, T_BROKER, rows, ["ticker", "trade_date", "broker_code", "data_source"])
    return dict(tgl=tgl, n_broker=len(rows), api=dari_api, final=final)


def ambil_pasar(client, eng, kode: str, hari_kalender: int = 450, now: dt.datetime | None = None) -> pd.DataFrame:
    """OHLCV harian (kalender bursa + harga penutupan); disesuaikan untuk split agar sebanding. Satu panggilan API."""
    t0 = time.time()
    try:
        df = tk.get_chart_panjang(client, kode, None, hari_kalender)
    except Exception as e:
        _log(eng, kode, EP_CHART, f"{hari_kalender}d", "ERR", 0, int((time.time() - t0) * 1000), e, now)
        raise
    _log(eng, kode, EP_CHART, f"{hari_kalender}d", "OK", len(df), int((time.time() - t0) * 1000), None, now)
    ret = _utc_naive(now)
    rows = [dict(ticker=kode, trade_date=r.tgl, data_source="invezgo_chart", open=float(r.open), high=float(r.high), low=float(r.low), close=float(r.close),
                 volume_shares=float(r.volume), retrieved_at=ret) for r in df.itertuples()]
    with eng.begin() as c:
        upsert(c, T_MARKET, rows, ["ticker", "trade_date", "data_source"])
    return df


def perbarui(client, eng, kode: str, hari_bursa: int = SESI_DEFAULT, max_calls: int = 60, now: dt.datetime | None = None, jeda: float = 0.1,
             progress=None, hanya_baru: bool = False) -> dict:
    """Backfill sekali + inkremental: hanya tanggal yang belum final yang diambil. Terbaru dulu; dapat dilanjutkan bila max_calls tercapai."""
    kode = kode.strip().upper()
    pasar = ambil_pasar(client, eng, kode, now=now)
    tgl = [d for d in pasar["tgl"].tolist()][-hari_bursa:]
    with eng.connect() as c:
        final_ada = {r[0] for r in c.execute(sa.select(T_RAW.c.trade_date).where(sa.and_(
            T_RAW.c.ticker == kode, T_RAW.c.endpoint == EP_SUMMARY, T_RAW.c.is_final == True)))}  # noqa: E712
    if hanya_baru and final_ada:              # sinkron inkremental: hanya sesi SESUDAH data final terakhir
        terakhir = max(final_ada)
        perlu = [d for d in reversed(tgl) if d > terakhir]
    else:
        perlu = [d for d in reversed(tgl) if d not in final_ada]
    hasil = dict(kode=kode, total=len(tgl), sudah=len(tgl) - len(perlu), diambil=0, gagal=[], tersisa=0, kosong=[])
    for i, d in enumerate(perlu):
        if hasil["diambil"] >= max_calls:
            hasil["tersisa"] = len(perlu) - i
            break
        try:
            r = ambil_hari(client, eng, kode, d, now)
            hasil["diambil"] += int(r["api"])
            if r["n_broker"] == 0:
                hasil["kosong"].append(d)
        except Exception as e:
            hasil["gagal"].append((d, str(e)[:120]))
        if progress:
            progress(i + 1, len(perlu), d)
        time.sleep(jeda)
    try:
        hasil["meta"] = catat_meta(eng, kode, hari_bursa, sukses=not hasil["gagal"], now=now)
    except Exception:
        pass
    return hasil


def catat_meta(eng, kode: str, target: int = SESI_DEFAULT, sukses: bool = True, now: dt.datetime | None = None) -> dict:
    """Cakupan riwayat per ticker: sesi termuat (final), tanggal pertama/terakhir, history_complete. Tidak mengisi tanggal yang hilang."""
    kode = kode.strip().upper()
    with eng.connect() as c:
        pasar = sorted(r[0] for r in c.execute(sa.select(T_MARKET.c.trade_date).where(T_MARKET.c.ticker == kode)))
        final = {r[0] for r in c.execute(sa.select(T_RAW.c.trade_date).where(sa.and_(
            T_RAW.c.ticker == kode, T_RAW.c.endpoint == EP_SUMMARY, T_RAW.c.is_final == True)))}  # noqa: E712
        lama = c.execute(sa.select(T_META.c.last_successful_sync).where(T_META.c.ticker == kode)).scalar()
    jendela = pasar[-target:]
    muat_ = sorted(d for d in jendela if d in final)
    row = dict(ticker=kode, first_available_trade_date=muat_[0] if muat_ else None, last_available_trade_date=muat_[-1] if muat_ else None,
               sessions_loaded=len(muat_), target_sessions=target, market_sessions=len(pasar), history_complete=bool(len(muat_) >= target),
               last_successful_sync=(_utc_naive(now) if sukses else lama), source_version=SOURCE_VERSION)
    with eng.begin() as c:
        upsert(c, T_META, [row], ["ticker"])
    return row


def muat_meta(eng, kode: str) -> dict | None:
    with eng.connect() as c:
        r = c.execute(sa.select(T_META).where(T_META.c.ticker == kode.strip().upper())).mappings().first()
    return dict(r) if r else None


def sinkron_inkremental(client, eng, kode: str, max_calls: int = 5, now: dt.datetime | None = None) -> dict | None:
    """Ticker yang sudah punya ledger: ambil HANYA sesi baru (maks max_calls). Ledger kosong -> tidak melakukan apa pun (backfill harus disengaja)."""
    m = muat_meta(eng, kode)
    if not m or not m.get("sessions_loaded"):
        return None
    return perbarui(client, eng, kode, m.get("target_sessions") or SESI_DEFAULT, max_calls=max_calls, now=now, jeda=0.05, hanya_baru=True)


def rekonsiliasi(client, eng, kode: str, n_hari: int = 3, now: dt.datetime | None = None) -> list[dict]:
    """Ambil ulang N hari bursa terakhir dan bandingkan dengan yang tersimpan; koreksi dicatat."""
    kode = kode.strip().upper()
    with eng.connect() as c:
        tgl = [r[0] for r in c.execute(sa.select(T_MARKET.c.trade_date).where(T_MARKET.c.ticker == kode).order_by(T_MARKET.c.trade_date.desc()).limit(n_hari))]
    out = []
    for d in tgl:
        with eng.connect() as c:
            lama = {(r.broker_code): (r.buy_shares, r.sell_shares, r.buy_value, r.sell_value) for r in c.execute(
                sa.select(T_BROKER).where(sa.and_(T_BROKER.c.ticker == kode, T_BROKER.c.trade_date == d)))}
        ambil_hari(client, eng, kode, d, now, paksa=True)
        with eng.connect() as c:
            baru = {(r.broker_code): (r.buy_shares, r.sell_shares, r.buy_value, r.sell_value) for r in c.execute(
                sa.select(T_BROKER).where(sa.and_(T_BROKER.c.ticker == kode, T_BROKER.c.trade_date == d)))}
        beda = [b for b in set(lama) | set(baru) if lama.get(b) != baru.get(b)]
        out.append(dict(tgl=d, n_lama=len(lama), n_baru=len(baru), n_berubah=len(beda), dikoreksi=bool(lama) and bool(beda)))
    return out


# ------------------------------------------------------------------- muat
def muat(eng, kode: str, hari: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(ledger harian long, pasar harian). Hanya data tersimpan; tidak memanggil API."""
    kode = kode.strip().upper()
    with eng.connect() as c:
        b = pd.read_sql(sa.select(T_BROKER).where(T_BROKER.c.ticker == kode).order_by(T_BROKER.c.trade_date), c)
        m = pd.read_sql(sa.select(T_MARKET).where(T_MARKET.c.ticker == kode).order_by(T_MARKET.c.trade_date), c)
    for df in (b, m):
        if not df.empty:
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    if hari and not m.empty:
        awal = sorted(m["trade_date"])[-hari:][0]
        b, m = b[b["trade_date"] >= awal], m[m["trade_date"] >= awal]
    return b.reset_index(drop=True), m.reset_index(drop=True)


def log_api(eng, kode: str | None = None, hari: int = 7) -> pd.DataFrame:
    q = sa.select(T_LOG).where(T_LOG.c.ts >= _utc_naive() - dt.timedelta(days=hari)).order_by(T_LOG.c.ts.desc())
    if kode:
        q = q.where(T_LOG.c.ticker == kode)
    with eng.connect() as c:
        return pd.read_sql(q, c)


# ---------------------------------------------------------------- validasi
def tandai_crossing(b: pd.DataFrame, m: pd.DataFrame, rasio: float = 0.02, min_porsi: float = 0.03) -> pd.DataFrame:
    """Broker-hari dengan beli ~ jual dan besar (relatif volume hari itu) = crossing/repo; gross-nya tidak mencerminkan niat beli/jual."""
    b = b.copy()
    vol = m.set_index("trade_date")["volume_shares"] if not m.empty else pd.Series(dtype=float)
    v = b["trade_date"].map(vol)
    tot_beli = b.groupby("trade_date")["buy_shares"].transform("sum")
    pembagi = v.where(v.notna() & (v > 0), tot_beli)
    mx = b[["buy_shares", "sell_shares"]].max(axis=1)
    mn = b[["buy_shares", "sell_shares"]].min(axis=1)
    b["crossing"] = (mx > 0) & ((mx - mn) <= rasio * mx) & (mn >= min_porsi * pembagi)
    return b


def validasi(b: pd.DataFrame, m: pd.DataFrame, log: pd.DataFrame | None = None, now: dt.datetime | None = None) -> dict:
    """Pemeriksaan integritas. Status: OK / PERINGATAN / GAGAL. Tidak ada pengisian data dengan tebakan."""
    cek = []
    add = lambda i, s, d: cek.append(dict(id=i, status=s, detail=d))  # noqa: E731
    if b.empty:
        add("data_ada", "GAGAL", "ledger kosong: jalankan 'Perbarui ledger'")
        return dict(status="GAGAL", cek=cek, n_gagal=1, n_peringatan=0, tanggal_hilang=[], cakupan=None)
    dup = int(b.duplicated(["ticker", "trade_date", "broker_code", "data_source"]).sum())
    add("duplikat", "OK" if dup == 0 else "GAGAL", f"{dup} rekaman duplikat (kunci ticker+tanggal+broker+sumber)")
    tgl_pasar = sorted(m["trade_date"]) if not m.empty else []
    tgl_ledger = set(b["trade_date"])
    awal = min(tgl_ledger)
    hilang = [d for d in tgl_pasar if d >= awal and d not in tgl_ledger]
    add("tanggal_hilang", "OK" if not hilang else "PERINGATAN", "tidak ada tanggal bursa yang hilang di dalam rentang" if not hilang else
        f"{len(hilang)} tanggal bursa tanpa data broker: {', '.join(str(d) for d in hilang[:8])}{' ...' if len(hilang) > 8 else ''}")
    nonfinal = sorted(b.loc[~b["is_final"].astype(bool), "trade_date"].unique()) if "is_final" in b else []
    add("data_final", "OK" if not nonfinal else "PERINGATAN", "semua hari final (EOD)" if not nonfinal else f"belum final (sebelum EOD 18:00 WIB): {', '.join(str(d) for d in nonfinal)}")
    n = now or dt.datetime.now(dt.timezone.utc)
    if tgl_pasar:
        lag = (n.astimezone(WIB).date() - max(tgl_ledger)).days
        add("kesegaran", "OK" if lag <= 4 else "PERINGATAN", f"data broker terakhir {max(tgl_ledger)} ({lag} hari sebelum sekarang)")
    if "net_shares_raw" in b:
        ada = b["net_shares_raw"].notna()
        selisih = int((((b.loc[ada, "buy_shares"] - b.loc[ada, "sell_shares"]) - b.loc[ada, "net_shares_raw"]).abs() > LOT_TOL).sum())
        add("net_konsisten", "OK" if selisih == 0 else "PERINGATAN", "beli - jual = net sumber di semua rekaman (toleransi 1 lot)" if selisih == 0 else
            f"{selisih} rekaman dengan beli - jual != net sumber (toleransi 1 lot): net dihitung dari beli - jual, bukan dari field net sumber")
    inv = int(((b["buy_shares"] > 0) & (b["buy_value"] <= 0)).sum() + ((b["sell_shares"] > 0) & (b["sell_value"] <= 0)).sum() + ((b["buy_shares"] < 0) | (b["sell_shares"] < 0)).sum())
    add("nilai_tidak_valid", "OK" if inv == 0 else "GAGAL", f"{inv} rekaman volume>0 dengan nilai<=0 atau volume negatif")
    avg = (b["buy_value"] / b["buy_shares"].replace(0, np.nan))
    sel = b["buy_avg_raw"].notna() & avg.notna()
    sat = int((((avg[sel] / b.loc[sel, "buy_avg_raw"]) - 1).abs() > 0.01).sum())
    add("satuan_konsisten", "OK" if sat == 0 else "GAGAL", "harga rata-rata beli = nilai / lembar di semua rekaman (satuan: lembar)" if sat == 0 else
        f"{sat} rekaman dengan nilai/lembar tidak sama dengan harga rata-rata sumber: periksa satuan (lot vs lembar)")
    if not m.empty:
        mm = m.set_index("trade_date")
        j = b[b["buy_avg_raw"].notna() & (b["buy_shares"] > 0)].copy()
        j["lo"], j["hi"] = j["trade_date"].map(mm["low"]), j["trade_date"].map(mm["high"])
        luar = int(((j["buy_avg_raw"] < j["lo"] * 0.97) | (j["buy_avg_raw"] > j["hi"] * 1.03)).sum())
        add("harga_dalam_rentang", "OK" if luar == 0 else "PERINGATAN", "harga rata-rata beli berada dalam rentang low-high harian (±3%)" if luar == 0 else
            f"{luar} rekaman harga rata-rata di luar rentang harian (±3%): kemungkinan crossing di harga nego atau aksi korporasi")
        tot = b.groupby("trade_date")["buy_shares"].sum()
        rasio = (tot / mm["volume_shares"].reindex(tot.index)).dropna()
        if len(rasio):
            off = int(((rasio - 1).abs() > 0.02).sum())
            add("total_vs_pasar", "OK" if off == 0 else "PERINGATAN",
                f"jumlah beli seluruh broker vs volume chart: median {rasio.median():.2f}x; {off} hari selisih >2% (pasar negosiasi/tunai bisa termasuk di chart)")
        adj, ev = tk.sesuaikan_split(m.rename(columns={"trade_date": "tgl", "volume_shares": "volume"}).sort_values("tgl").reset_index(drop=True))
        add("aksi_korporasi", "OK" if not ev else "PERINGATAN", "tidak ada split terdeteksi" if not ev else
            "split terdeteksi: " + "; ".join(f"{e['jenis']} ({e['tgl']})" for e in ev) + ". Lot/harga broker sebelum tanggal itu TIDAK disesuaikan: akumulasi lintas tanggal tidak sebanding.")
    bx = tandai_crossing(b, m)
    nx = int(bx["crossing"].sum())
    pangsa = float(bx.loc[bx["crossing"], "buy_shares"].sum() / max(b["buy_shares"].sum(), 1))
    add("crossing", "OK" if nx == 0 else "PERINGATAN", "tidak ada broker-hari crossing" if nx == 0 else
        f"{nx} broker-hari crossing/repo ({pangsa:.0%} dari gross beli): gross beli/jual menggelembung; ditandai dan dipisahkan, net tidak terpengaruh")
    if log is not None and not log.empty:
        err = log[log["status"] == "ERR"]
        add("api_error", "OK" if err.empty else "PERINGATAN", "tidak ada error API tercatat" if err.empty else f"{len(err)} error API (terakhir: {str(err.iloc[0]['error'])[:80]})")
    g = [c for c in cek if c["status"] == "GAGAL"]
    p = [c for c in cek if c["status"] == "PERINGATAN"]
    return dict(status="GAGAL" if g else "PERINGATAN" if p else "OK", cek=cek, n_gagal=len(g), n_peringatan=len(p), tanggal_hilang=hilang,
                cakupan=dict(hari=len(tgl_ledger), awal=awal, akhir=max(tgl_ledger), n_broker=int(b["broker_code"].nunique())))


# ------------------------------------------------------- ledger sebagai sumber (hemat API)
def agregat_summary(eng, kode: str, dari, sampai) -> list[dict] | None:
    """Respons ala /summary/stock untuk rentang [dari, sampai] dihitung dari ledger tersimpan; None bila ledger belum menutup SELURUH hari bursa
    dalam rentang itu dengan data final (maka pemanggil jatuh ke API). Frekuensi transaksi tidak disimpan ledger -> 0 (tidak dipakai app)."""
    kode = kode.strip().upper()
    dari, sampai = dt.date.fromisoformat(str(dari)[:10]), dt.date.fromisoformat(str(sampai)[:10])
    try:
        with eng.connect() as c:
            hari = {r[0] for r in c.execute(sa.select(T_MARKET.c.trade_date).where(sa.and_(
                T_MARKET.c.ticker == kode, T_MARKET.c.trade_date >= dari, T_MARKET.c.trade_date <= sampai)))}
            if not hari or min(hari) > dari + dt.timedelta(days=5):
                return None
            final = {r[0] for r in c.execute(sa.select(T_RAW.c.trade_date).where(sa.and_(
                T_RAW.c.ticker == kode, T_RAW.c.endpoint == EP_SUMMARY, T_RAW.c.is_final == True,  # noqa: E712
                T_RAW.c.trade_date >= dari, T_RAW.c.trade_date <= sampai)))}
            if not hari <= final:
                return None
            rows = c.execute(sa.select(T_BROKER).where(sa.and_(
                T_BROKER.c.ticker == kode, T_BROKER.c.trade_date >= dari, T_BROKER.c.trade_date <= sampai))).mappings().all()
    except Exception:
        return None
    ag: dict[str, dict] = {}
    for r in rows:
        a = ag.setdefault(r["broker_code"], dict(code=r["broker_code"], name=r["broker_name"], bs=0.0, ss=0.0, bv=0.0, sv=0.0))
        a["bs"] += r["buy_shares"] or 0.0
        a["ss"] += r["sell_shares"] or 0.0
        a["bv"] += r["buy_value"] or 0.0
        a["sv"] += r["sell_value"] or 0.0
    out = []
    for a in ag.values():
        out.append(dict(code=a["code"], name=a["name"], buy_freq=0, buy_volume=a["bs"], buy_value=a["bv"], sell_freq=0, sell_volume=a["ss"], sell_value=a["sv"],
                        buy_avg=(a["bv"] / a["bs"] if a["bs"] else 0), sell_avg=(a["sv"] / a["ss"] if a["ss"] else 0),
                        net_value=a["bv"] - a["sv"], net_volume=a["bs"] - a["ss"], net_freq=0))
    return out


def summary_hari_payload(client, eng, kode: str, tgl: dt.date, now: dt.datetime | None = None):
    """Respons ringkasan broker satu hari: dari ledger bila sudah final (0 panggilan API), jika tidak diambil & disimpan (ledger tumbuh otomatis)."""
    kode = kode.strip().upper()
    key = f"{tgl}|all|RG"
    ada = _raw_ada(eng, kode, EP_SUMMARY, key)
    if not (ada and ada.is_final):
        ambil_hari(client, eng, kode, tgl, now)
        ada = _raw_ada(eng, kode, EP_SUMMARY, key)
    return json.loads(ada.payload)


def simpan_chart(eng, kode: str, raw, now: dt.datetime | None = None) -> int:
    """Simpan OHLCV (disesuaikan split) dari respons chart ke tabel pasar ledger agar cakupan hari bursa diketahui."""
    df = tk.parse_chart(raw)
    ret = _utc_naive(now)
    rows = [dict(ticker=kode.strip().upper(), trade_date=r.tgl, data_source="invezgo_chart", open=float(r.open), high=float(r.high), low=float(r.low),
                 close=float(r.close), volume_shares=float(r.volume), retrieved_at=ret) for r in df.itertuples()]
    with eng.begin() as c:
        upsert(c, T_MARKET, rows, ["ticker", "trade_date", "data_source"])
    return len(rows)
