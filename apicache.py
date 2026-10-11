"""Cache respons Invezgo di database (Neon/SQLite) + penghitung pemakaian kuota bulanan.

Tujuan: setiap respons API disimpan terkompresi dan dipakai ulang selama masih berlaku, sehingga kuota (30.000 req/bulan)
tidak terbuang untuk data yang tidak berubah. Masa berlaku mengikuti sifat data:
  realtime (orderbook, IEP)         -> tidak disimpan di DB
  laporan keuangan, info, keystat   -> 7 hari
  kalender, SID, kepemilikan        -> 1 hari
  data harga/broker dengan tanggal akhir sudah lewat -> 30 hari (EOD final, tidak berubah)
  data yang mencakup hari ini       -> 10 menit sebelum 18:00 WIB (EOD belum final), 12 jam sesudahnya
Respons 422 (parameter ditolak) disimpan juga agar tidak ditanyakan ulang ke API.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import threading
import zlib
from zoneinfo import ZoneInfo

import sqlalchemy as sa

import adimology as ad

WIB = ZoneInfo("Asia/Jakarta")
KUOTA_BULAN = 30_000
BATAS_BYTE = 150 * 1024 * 1024          # batas total cache (Neon gratis 512 MB dibagi dengan ledger & histori)

META = sa.MetaData()
T_CACHE = sa.Table(
    "api_cache", META,
    sa.Column("k", sa.String(40), primary_key=True),
    sa.Column("path", sa.String(120), nullable=False),
    sa.Column("params", sa.String(300)),
    sa.Column("fetched_at", sa.DateTime, nullable=False),
    sa.Column("expires_at", sa.DateTime, nullable=False, index=True),
    sa.Column("ukuran", sa.Integer),
    sa.Column("payload", sa.LargeBinary, nullable=False))
T_USAGE = sa.Table(
    "api_usage", META,
    sa.Column("bulan", sa.String(7), primary_key=True),
    sa.Column("req", sa.Integer, nullable=False, default=0),
    sa.Column("hit", sa.Integer, nullable=False, default=0))

_lock = threading.Lock()
_puts = [0]


def siapkan(eng: sa.Engine) -> sa.Engine:
    META.create_all(eng)
    return eng


def _naive_utc(now: dt.datetime | None = None) -> dt.datetime:
    return (now or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc).replace(tzinfo=None)


def kunci(path: str, params_items: tuple) -> str:
    return hashlib.sha1((path + "?" + json.dumps(sorted(params_items), default=str)).encode()).hexdigest()


def ttl_detik(path: str, params_items: tuple, now: dt.datetime | None = None) -> int | None:
    """None = jangan simpan di DB (realtime / dikelola modul lain)."""
    p = dict(params_items)
    if "order-book" in path or "intraday-data" in path or "running-trade" in path:
        return None
    if "/summary/stock/" in path and p.get("from") and p.get("from") == p.get("to"):
        return None                                   # ringkasan harian dikelola ledger (respons mentah sendiri)
    if any(s in path for s in ("financial-statement", "/information/", "/keystat/")):
        return 7 * 86400
    if any(s in path for s in ("calendar", "shareholder")):
        return 86400
    n = (now or dt.datetime.now(dt.timezone.utc)).astimezone(WIB)
    akhir = p.get("to") or p.get("date")
    if akhir:
        try:
            tgl = dt.date.fromisoformat(str(akhir)[:10])
        except ValueError:
            return 300
        if tgl < n.date():
            return 30 * 86400
        return 12 * 3600 if n.time() >= dt.time(18, 0) else 600
    return 300


def _simpan(eng, k, path, params_items, payload_obj, ttl, now):
    raw = zlib.compress(json.dumps(payload_obj, default=str).encode(), 6)
    t = _naive_utc(now)
    row = dict(k=k, path=path[:120], params=json.dumps(sorted(params_items), default=str)[:300], fetched_at=t,
               expires_at=t + dt.timedelta(seconds=ttl), ukuran=len(raw), payload=raw)
    with eng.begin() as c:
        if eng.dialect.name == "postgresql":
            from sqlalchemy.dialects.postgresql import insert
        else:
            from sqlalchemy.dialects.sqlite import insert
        st = insert(T_CACHE).values(**row)
        c.execute(st.on_conflict_do_update(index_elements=["k"], set_={x: st.excluded[x] for x in row if x != "k"}))
    with _lock:
        _puts[0] += 1
        n = _puts[0]
    if n % 25 == 0:
        bersihkan(eng, now)


def bersihkan(eng, now=None) -> None:
    """Hapus entri kedaluwarsa >2 hari; bila total > BATAS_BYTE hapus yang terlama."""
    t = _naive_utc(now)
    with eng.begin() as c:
        c.execute(sa.delete(T_CACHE).where(T_CACHE.c.expires_at < t - dt.timedelta(days=2)))
        total = c.execute(sa.select(sa.func.coalesce(sa.func.sum(T_CACHE.c.ukuran), 0))).scalar() or 0
        while total > BATAS_BYTE:
            ids = [r[0] for r in c.execute(sa.select(T_CACHE.c.k).order_by(T_CACHE.c.fetched_at).limit(50))]
            if not ids:
                break
            c.execute(sa.delete(T_CACHE).where(T_CACHE.c.k.in_(ids)))
            total = c.execute(sa.select(sa.func.coalesce(sa.func.sum(T_CACHE.c.ukuran), 0))).scalar() or 0


def _hitung(eng, req: int, hit: int, now=None) -> None:
    bulan = (now or dt.datetime.now(dt.timezone.utc)).astimezone(WIB).strftime("%Y-%m")
    if eng.dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        from sqlalchemy.dialects.sqlite import insert
    st = insert(T_USAGE).values(bulan=bulan, req=req, hit=hit)
    with eng.begin() as c:
        c.execute(st.on_conflict_do_update(index_elements=["bulan"], set_={"req": T_USAGE.c.req + req, "hit": T_USAGE.c.hit + hit}))


def ambil(eng, path: str, params_items: tuple, api_key: str, now: dt.datetime | None = None):
    """Cache-DB lalu API. Error DB tidak boleh mematikan aplikasi (jatuh ke API langsung)."""
    ttl = ttl_detik(path, params_items, now) if eng is not None else None
    k = kunci(path, params_items)
    if ttl is not None:
        try:
            with eng.connect() as c:
                row = c.execute(sa.select(T_CACHE.c.payload).where(sa.and_(T_CACHE.c.k == k, T_CACHE.c.expires_at > _naive_utc(now)))).first()
            if row:
                obj = json.loads(zlib.decompress(row[0]))
                _hitung(eng, 0, 1, now)
                if isinstance(obj, dict) and obj.get("__ditolak"):
                    raise ad.ParamDitolak(obj["__ditolak"])
                return obj
        except ad.ParamDitolak:
            raise
        except Exception:
            pass
    n0 = ad.REQ_COUNT[0]
    try:
        hasil = ad.fetch_json(api_key, path, params_items)
    except ad.ParamDitolak as e:
        _catat(eng, ad.REQ_COUNT[0] - n0, now)
        if ttl is not None:
            try:
                _simpan(eng, k, path, params_items, {"__ditolak": str(e)}, 30 * 86400, now)
            except Exception:
                pass
        raise
    _catat(eng, ad.REQ_COUNT[0] - n0, now)
    if ttl is not None:
        try:
            _simpan(eng, k, path, params_items, hasil, ttl, now)
        except Exception:
            pass
    return hasil


def _catat(eng, n: int, now=None) -> None:
    if eng is not None and n > 0:
        try:
            _hitung(eng, n, 0, now)
        except Exception:
            pass


class DbClient(ad.InvezgoClient):
    """Klien dengan cache DB + penghitung kuota (dipakai ledger; app memakai lapisan memori Streamlit di atasnya)."""

    def __init__(self, api_key: str, eng=None):
        super().__init__(api_key)
        self.eng = eng

    def _fetch(self, path, params_items):
        return ambil(self.eng, path, params_items, self.api_key)


def pemakaian(eng, now=None) -> dict:
    bulan = (now or dt.datetime.now(dt.timezone.utc)).astimezone(WIB).strftime("%Y-%m")
    try:
        with eng.connect() as c:
            r = c.execute(sa.select(T_USAGE.c.req, T_USAGE.c.hit).where(T_USAGE.c.bulan == bulan)).first()
            n, ukuran = c.execute(sa.select(sa.func.count(), sa.func.coalesce(sa.func.sum(T_CACHE.c.ukuran), 0))).first()
    except Exception:
        return dict(bulan=bulan, req=None, hit=None, kuota=KUOTA_BULAN, entri=None, ukuran_mb=None)
    req, hit = (r[0], r[1]) if r else (0, 0)
    return dict(bulan=bulan, req=int(req), hit=int(hit), kuota=KUOTA_BULAN, entri=int(n), ukuran_mb=round((ukuran or 0) / 1e6, 2))
