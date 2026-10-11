"""Lapisan turunan & sinyal ledger broker: rolling flow per sesi bursa, Bandar Score 0-10, status episode, sinyal teraudit.

Lapisan: raw (ledger_raw/ledger_broker_daily; tidak pernah ditimpa) -> derived (rolling per broker, snapshot disimpan) -> signal (skor, status,
bukti pro/kontra, kelengkapan data, versi formula). Semua jendela = SESI BURSA yang termuat; tanggal awal-akhir selalu dicatat.
Bandarmologi = lapisan positioning/timing, BUKAN identitas atau niat pihak tertentu, dan bukan dasar rekomendasi tunggal.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import sqlalchemy as sa

import ledger as lg
import ledger_calc as lc

CALC_VERSION = "bandar-ledger-2026.10.1"
LOT = lg.SHARES_PER_LOT
JENDELA = (1, 5, 20, 60, 120)

ST_ACC, ST_MARKUP, ST_DIST, ST_MARKDOWN, ST_MIXED, ST_INSUF = ("ACCUMULATING", "MARKUP", "DISTRIBUTING", "MARKDOWN", "MIXED_OR_UNCLEAR", "INSUFFICIENT_DATA")


@dataclass
class Cfg2:
    kelompok_excl: tuple = ("XL", "YP", "XC", "PD")     # platform ritel besar: bukan 'kelompok kunci'
    top_n: int = 5
    min_sesi: int = 60                                  # di bawah ini -> INSUFFICIENT_DATA
    min_nilai_miliar: float = 5.0                       # ambang likuiditas (parameter, bukan hukum)
    markup_ret20: float = 0.12
    markdown_ret20: float = -0.08
    partisipasi_min: float = 0.02
    konsentrasi_tinggi: float = 0.60
    konsentrasi_sedang: float = 0.40
    sell_streak: int = 3


# --------------------------------------------------------------- rolling
def _wide(L: pd.DataFrame, kolom: str) -> pd.DataFrame:
    return L.pivot_table(index="trade_date", columns="broker_code", values=kolom, aggfunc="sum").sort_index().fillna(0.0)


def pasar_harian(L: pd.DataFrame) -> pd.DataFrame:
    """Harga & volume pasar SEKALI per tanggal (bukan dijumlah per baris broker)."""
    m = L.drop_duplicates("trade_date").set_index("trade_date")[["close_price", "volume_lots"]].sort_index()
    m["return_1d"] = m["close_price"].pct_change()
    m["return_5d"] = m["close_price"].pct_change(5)
    m["return_20d"] = m["close_price"].pct_change(20)
    m["volume_ratio_20d"] = m["volume_lots"] / m["volume_lots"].shift(1).rolling(20, min_periods=10).mean()
    m["value_traded"] = m["close_price"] * m["volume_lots"] * LOT            # proksi: close x volume (nilai transaksi sebenarnya tidak ada di chart)
    return m


def rolling_broker(L: pd.DataFrame) -> pd.DataFrame:
    """Fitur rolling per broker per sesi (long). 1D = satu sesi bursa. YTD = sejak sesi pertama tahun kalender tanggal itu."""
    net, nv = _wide(L, "net_lots"), _wide(L, "net_value")
    bs, bv = _wide(L, "buy_shares"), _wide(L, "buy_value")
    cross = L.pivot_table(index="trade_date", columns="broker_code", values="crossing", aggfunc="max").sort_index().fillna(False).astype(bool)
    bs_x, bv_x = bs.where(~cross, 0.0), bv.where(~cross, 0.0)
    M = pasar_harian(L)
    vol = M["volume_lots"]
    out = {}
    for n in JENDELA:
        out[f"net_lots_{n}d"] = net.rolling(n, min_periods=n).sum()
    tahun = pd.Series([d.year for d in net.index], index=net.index)
    out["net_lots_ytd"] = net.groupby(tahun.values).cumsum()
    for n in (5, 20):
        out[f"net_value_{n}d"] = nv.rolling(n, min_periods=n).sum()
        out[f"avg_buy_{n}d_ex_crossing"] = bv_x.rolling(n, min_periods=n).sum() / bs_x.rolling(n, min_periods=n).sum().replace(0, np.nan)
    out["net_flow_pct_volume_20d"] = net.rolling(20, min_periods=20).sum().div(vol.rolling(20, min_periods=20).sum(), axis=0)
    out["net_buy_days_20d"] = (net > 0).astype(int).rolling(20, min_periods=20).sum()
    out["net_sell_days_20d"] = (net < 0).astype(int).rolling(20, min_periods=20).sum()
    streak = net.copy() * 0
    for c in net.columns:
        run, vals = 0, []
        for v in net[c].to_numpy():
            run = run + 1 if v < 0 else 0
            vals.append(run)
        streak[c] = vals
    out["net_sell_streak"] = streak
    # respons harga pada sesi net jual (sesaat, bukan prediktif): rata-rata return_1d pada sesi net jual non-crossing, 20 sesi
    r1 = M["return_1d"]
    jual = (net < 0) & (~cross)
    out["price_response_to_net_sell_20d"] = (jual.mul(r1, axis=0)).rolling(20, min_periods=20).sum() / jual.astype(int).rolling(20, min_periods=20).sum().replace(0, np.nan)
    frames = []
    for k, w in out.items():
        f = w.stack(future_stack=True).rename(k) if hasattr(w, "stack") else w
        frames.append(f)
    R = pd.concat(frames, axis=1).reset_index()
    R.columns = ["trade_date", "broker_code"] + list(R.columns[2:])
    R = R.merge(M.reset_index()[["trade_date", "close_price", "return_1d", "return_5d", "return_20d", "volume_lots", "volume_ratio_20d", "value_traded"]], on="trade_date", how="left")
    R["net_lots_1d"] = R["net_lots_1d"]
    return R


def snapshot_derived(R: pd.DataFrame, tgl) -> pd.DataFrame:
    return R[R["trade_date"] == tgl].reset_index(drop=True)


# ----------------------------------------------------------------- kelompok
def kelompok_kunci(L: pd.DataFrame, cfg: Cfg2) -> list[str]:
    net60 = L[L["trade_date"].isin(sorted(L["trade_date"].unique())[-60:])].groupby("broker_code")["net_lots"].sum()
    net60 = net60[~net60.index.isin(cfg.kelompok_excl)]
    return [k for k in net60.sort_values(ascending=False).index[: cfg.top_n] if net60[k] > 0]


def _na(x):
    return x is None or (isinstance(x, float) and np.isnan(x))


# ------------------------------------------------------------------- skor
def hitung(res: dict, meta: dict | None, cfg: Cfg2 | None = None, now: dt.datetime | None = None) -> dict:
    """Sinyal teraudit dari hasil lc.jalankan(). Mengembalikan dict sesuai spesifikasi (lihat SINYAL_FIELDS)."""
    cfg = cfg or Cfg2()
    now = now or dt.datetime.now(dt.timezone.utc)
    ts = now.astimezone(lg.WIB).strftime("%Y-%m-%d %H:%M WIB")
    if not res.get("ok"):
        return _kosong(meta, "ledger kosong", ts)
    L, val = res["ledger"], res["validasi"]
    tgl = sorted(L["trade_date"].unique())
    n_sesi = len(tgl)
    kode = L["ticker"].iloc[0]
    lengkap = dict(sessions_loaded=n_sesi, target_sessions=(meta or {}).get("target_sessions", lg.SESI_DEFAULT), history_complete=bool((meta or {}).get("history_complete", False)),
                   first=str(tgl[0]), last=str(tgl[-1]), validasi=val["status"], tanggal_hilang=len(val.get("tanggal_hilang") or []))
    base = dict(ticker=kode, signal_date=str(tgl[-1]), observation_start_date=str(tgl[0]), observation_end_date=str(tgl[-1]), data_completeness=lengkap,
                formula_version=CALC_VERSION, computed_at=ts)
    if val["status"] == "GAGAL" or n_sesi < cfg.min_sesi:
        alasan = "validasi GAGAL" if val["status"] == "GAGAL" else f"hanya {n_sesi} sesi termuat (minimal {cfg.min_sesi})"
        return dict(base, episode_id="NONE", bandar_score=None, label="N/A", score_components=[], episode_status=ST_INSUF, confidence_score=0.0,
                    confidence_reason=[alasan], supporting_evidence=[], contradicting_evidence=[f"[OBS] {alasan}"], key_brokers=[])
    M = pasar_harian(L)
    grp = kelompok_kunci(L, cfg)
    Lg = L[L["broker_code"].isin(grp)]
    gnet = Lg.groupby("trade_date")["net_lots"].sum().reindex(tgl).fillna(0.0)
    vol = M["volume_lots"].reindex(tgl)
    close = M["close_price"].reindex(tgl)
    sup, lawan, komp = [], [], []

    # 1 persistensi
    pos, tot = [], []
    for n in (5, 20, 60, 120):
        if n_sesi >= n:
            v = float(gnet.tail(n).sum())
            tot.append(n)
            if v > 0:
                pos.append(n)
            (sup if v > 0 else lawan).append(f"[OBS] net kelompok {n}D {v:+,.0f} lot ({tgl[-n]}..{tgl[-1]})")
    p1 = 2.0 * len(pos) / len(tot) if tot else 0.0
    komp.append(dict(dimensi="Persistensi net buy", poin=round(p1, 2), maks=2, detail=f"{len(pos)}/{len(tot)} jendela positif"))

    # 2 konsentrasi (atas semua broker net beli selain platform ritel; bukan asumsi kerja sama)
    net60 = L[L["trade_date"].isin(tgl[-60:])].groupby("broker_code")["net_lots"].sum()
    net60 = net60[~net60.index.isin(cfg.kelompok_excl)]
    beli = net60[net60 > 0].sort_values(ascending=False)
    share3 = float(beli.head(3).sum() / beli.sum()) if len(beli) and beli.sum() > 0 else 0.0
    p2 = 2.0 if share3 >= cfg.konsentrasi_tinggi else 1.0 if share3 >= cfg.konsentrasi_sedang else 0.0
    komp.append(dict(dimensi="Konsentrasi broker", poin=p2, maks=2, detail=f"3 broker net beli teratas = {share3:.0%} dari seluruh net beli 60D"))
    (sup if p2 >= 1 else lawan).append(f"[OBS] konsentrasi net beli 60D: top-3 {', '.join(beli.head(3).index)} = {share3:.0%} (bukan bukti kerja sama)")

    # 3 pola akumulasi (episode pada net harian kelompok)
    cfg_lc = lc.Cfg()
    aktif = np.ones(len(gnet), dtype=bool)
    ep = lc._episode(gnet.to_numpy(), tgl, float(vol.sum()), aktif, cfg_lc)
    if ep and ep["hari"] >= 10:
        p3 = 2.0
    elif ep:
        p3 = 1.0
    else:
        p3 = 0.0
    episode_id = f"{kode}-GRP-{ep['mulai']}" if ep else "NONE"
    komp.append(dict(dimensi="Pola akumulasi", poin=p3, maks=2, detail=(f"episode {ep['mulai']}..{ep['puncak']} ({ep['hari']} sesi, {ep['frac_net_beli']:.0%} sesi net beli)" if ep else "tidak ada episode berkelanjutan")))
    (sup if ep else lawan).append("[EST] episode " + (f"{ep['mulai']}..{ep['puncak']}, {ep['hari']} sesi, +{ep['gain_lots']:,.0f} lot (lembah->puncak, bukan hari net-buy pertama)" if ep else "akumulasi berkelanjutan tidak ditemukan"))

    # 4 konfirmasi harga-volume
    ret20 = float(close.iloc[-1] / close.iloc[-21] - 1) if n_sesi > 20 else float("nan")
    net20 = float(gnet.tail(20).sum())
    part = net20 / max(float(vol.tail(20).sum()), 1.0)
    vr = float(vol.tail(5).mean() / max(vol.iloc[-25:-5].mean(), 1.0)) if n_sesi > 25 else float("nan")
    bx = Lg[(Lg["trade_date"].isin(tgl[-60:])) & (~Lg["crossing"])]
    biaya = float(bx["buy_value"].sum() / (bx["buy_shares"].sum())) if bx["buy_shares"].sum() > 0 else float("nan")
    pts = 0.0
    for ok, txt in ((not _na(ret20) and ret20 >= -0.02, f"harga 20D {ret20:+.1%} (tidak melemah saat net beli)"),
                    (not _na(vr) and vr >= 1.0, f"volume 5D / 20D = {vr:.2f}x"),
                    (part >= cfg.partisipasi_min, f"partisipasi net 20D = {part:.1%} dari volume (ambang {cfg.partisipasi_min:.0%})"),
                    (not _na(biaya) and close.iloc[-1] >= biaya, f"harga {close.iloc[-1]:,.0f} vs proksi biaya kelompok 60D {biaya:,.0f}" if not _na(biaya) else "proksi biaya N/A")):
        pts += 0.5 if ok else 0.0
        (sup if ok else lawan).append("[OBS] " + txt + (" : mendukung" if ok else " : tidak mendukung"))
    komp.append(dict(dimensi="Konfirmasi harga-volume", poin=pts, maks=2, detail=f"return20D {ret20:+.1%}; vol5/20 {vr:.2f}x; partisipasi {part:.1%}"))

    # 5 risiko distribusi (2 = tidak ada tanda)
    run = 0
    for v in gnet.to_numpy()[::-1]:
        if v < 0:
            run += 1
        else:
            break
    net5, net60g = float(gnet.tail(5).sum()), float(gnet.tail(60).sum())
    sv = Lg[~Lg["crossing"]].groupby("trade_date")["sell_value"].sum().reindex(tgl).fillna(0.0)
    sv5, svr = float(sv.tail(5).mean()), float(sv.mean())
    hi60 = float(close.tail(60).max())
    risiko = []
    if run >= cfg.sell_streak:
        risiko.append((1.0, f"net jual kelompok {run} sesi beruntun"))
    if net5 < 0 < net60g:
        risiko.append((0.5, f"net 5D {net5:+,.0f} lot berlawanan dengan net 60D {net60g:+,.0f}"))
    if svr > 0 and sv5 >= 1.5 * svr:
        risiko.append((0.5, f"nilai jual 5D = {sv5 / svr:.1f}x rata-rata"))
    if close.iloc[-1] >= 0.97 * hi60 and net5 < 0:
        risiko.append((0.5, "harga dekat high 60D sementara kelompok net jual"))
    p5 = max(0.0, 2.0 - sum(x for x, _ in risiko))
    komp.append(dict(dimensi="Risiko distribusi", poin=p5, maks=2, detail="; ".join(t for _, t in risiko) or "tidak ada tanda distribusi"))
    for _, t in risiko:
        lawan.append("[OBS] " + t)
    if not risiko:
        sup.append("[OBS] tidak ada tanda distribusi pada kelompok kunci (sell streak/rasio jual/divergensi)")

    skor = round(sum(c["poin"] for c in komp), 1)
    # label: skala 0-10; di bawah 4 didefinisikan eksplisit
    if skor >= 8:
        label = "INDIKASI AKUMULASI KUAT"
    elif skor >= 6:
        label = "INDIKASI AKUMULASI MODERAT"
    elif skor >= 4:
        label = "NETRAL / CAMPURAN"
    elif net20 < 0 or p5 <= 0.5:
        label = "DISTRIBUSI DOMINAN (skor <4: net 20D negatif atau risiko distribusi tinggi)"
    else:
        label = "LEMAH / TIDAK ADA POLA (skor <4)"

    # status episode (aturan terdokumentasi)
    pernah_acc = bool(ep) or float(gnet.tail(120).sum()) > 0
    if net20 > 0 and net60g > 0 and not _na(ret20) and ret20 >= cfg.markup_ret20:
        status = ST_MARKUP
    elif net20 > 0 and net60g > 0:
        status = ST_ACC
    elif net20 < 0 and not _na(ret20) and ret20 <= cfg.markdown_ret20:
        status = ST_MARKDOWN
    elif (net20 < 0 or run >= cfg.sell_streak) and pernah_acc:
        status = ST_DIST
    else:
        status = ST_MIXED

    # confidence (0-1) + alasan
    conf, why = 1.0, []
    fk = min(n_sesi / lengkap["target_sessions"], 1.0)
    if fk < 1:
        conf *= max(0.5, fk)
        why.append(f"riwayat {n_sesi}/{lengkap['target_sessions']} sesi")
    if not lengkap["history_complete"]:
        conf *= 0.9
        why.append("history_complete=false")
    if val["status"] == "PERINGATAN":
        conf *= 0.85
        why.append("validasi data ada peringatan: " + ", ".join(c["id"] for c in val["cek"] if c["status"] != "OK")[:80])
    if lengkap["tanggal_hilang"]:
        conf *= 0.85
        why.append(f"{lengkap['tanggal_hilang']} sesi tanpa data broker")
    nilai20 = float(M["value_traded"].tail(20).mean())
    likuid = nilai20 >= cfg.min_nilai_miliar * 1e9
    if not likuid:
        conf *= 0.8
        why.append(f"likuiditas rendah: nilai transaksi rata-rata 20D Rp{nilai20 / 1e9:,.1f} M < ambang Rp{cfg.min_nilai_miliar:.0f} M (risiko slippage/kapasitas)")
        lawan.append(f"[OBS] likuiditas rendah (Rp{nilai20 / 1e9:,.1f} M/hari)")
    if not grp:
        conf *= 0.5
        why.append("tidak ada kelompok net beli")
    return dict(base, episode_id=episode_id, bandar_score=skor, label=label, score_components=komp, episode_status=status, confidence_score=round(conf, 2),
                confidence_reason=why or ["data lengkap dan tervalidasi"], supporting_evidence=sup, contradicting_evidence=lawan, key_brokers=grp,
                likuid=likuid, nilai_transaksi_20d=nilai20, cfg=asdict(cfg))


def _kosong(meta, alasan, ts):
    return dict(ticker=None, signal_date=None, episode_id="NONE", bandar_score=None, label="N/A", score_components=[], episode_status=ST_INSUF, confidence_score=0.0,
                confidence_reason=[alasan], supporting_evidence=[], contradicting_evidence=[f"[OBS] {alasan}"], key_brokers=[], observation_start_date=None,
                observation_end_date=None, data_completeness=dict(meta or {}), formula_version=CALC_VERSION, computed_at=ts)


# ------------------------------------------------------------ persistensi
def fingerprint(res: dict, cfg: Cfg2) -> str:
    L = res["ledger"]
    isi = f"{L['trade_date'].max()}|{L['trade_date'].nunique()}|{round(float(L['buy_shares'].sum()))}|{round(float(L['sell_shares'].sum()))}|{CALC_VERSION}|{json.dumps(asdict(cfg), sort_keys=True)}"
    return hashlib.sha1(isi.encode()).hexdigest()[:40]


def simpan(eng, sinyal: dict, R: pd.DataFrame | None, fp: str) -> None:
    """Simpan sinyal (satu baris per ticker+tanggal+versi) dan snapshot derived (tanggal terakhir, semua broker aktif). Idempoten."""
    if not sinyal.get("ticker"):
        return
    t = lg._utc_naive()
    row = dict(ticker=sinyal["ticker"], signal_date=dt.date.fromisoformat(sinyal["signal_date"]), calc_version=CALC_VERSION, computed_at=t, fingerprint=fp,
               bandar_score=sinyal.get("bandar_score"), episode_status=sinyal["episode_status"], confidence_score=sinyal["confidence_score"],
               payload=json.dumps(sinyal, default=str, ensure_ascii=False))
    with eng.begin() as c:
        lg.upsert(c, lg.T_SIGNAL, [row], ["ticker", "signal_date", "calc_version"])
        if R is not None and not R.empty:
            tgl = R["trade_date"].max()
            snap = R[R["trade_date"] == tgl]
            rows = [dict(ticker=sinyal["ticker"], as_of_date=tgl, broker_code=r["broker_code"], calc_version=CALC_VERSION, computed_at=t,
                         metrics=json.dumps({k: (None if _na(v) else (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v))) for k, v in r.items()
                                             if k not in ("trade_date", "broker_code")}, ensure_ascii=False))
                    for _, r in snap.iterrows() if abs(r.get("net_lots_60d", 0) or 0) > 0 or abs(r.get("net_lots_5d", 0) or 0) > 0]
            if rows:
                lg.upsert(c, lg.T_DERIVED, rows, ["ticker", "as_of_date", "broker_code", "calc_version"])


def sinyal_tersimpan(eng, kode: str, fp: str) -> dict | None:
    """Sinyal terakhir bila fingerprint input sama (tidak perlu dihitung ulang)."""
    with eng.connect() as c:
        r = c.execute(sa.select(lg.T_SIGNAL.c.payload, lg.T_SIGNAL.c.fingerprint).where(sa.and_(
            lg.T_SIGNAL.c.ticker == kode.upper(), lg.T_SIGNAL.c.calc_version == CALC_VERSION)).order_by(lg.T_SIGNAL.c.signal_date.desc()).limit(1)).first()
    return json.loads(r[0]) if r and r[1] == fp else None


def hitung_dan_simpan(eng, res: dict, meta: dict | None, cfg: Cfg2 | None = None) -> tuple[dict, bool]:
    """(sinyal, dihitung_ulang). Bila input (tanggal terakhir, jumlah sesi, total beli/jual, versi, konfigurasi) tidak berubah dipakai sinyal tersimpan."""
    cfg = cfg or Cfg2()
    if not res.get("ok"):
        return hitung(res, meta, cfg), True
    fp = fingerprint(res, cfg)
    ada = sinyal_tersimpan(eng, res["ledger"]["ticker"].iloc[0], fp)
    if ada:
        return ada, False
    s = hitung(res, meta, cfg)
    R = rolling_broker(res["ledger"])
    simpan(eng, s, R, fp)
    return s, True


# ------------------------------------------------------------- laporan
def baris(s: dict | None) -> list[str]:
    L = ["### Bandar Score ledger (0-10; lapisan positioning/timing, BUKAN dasar rekomendasi tunggal; bobot maks 10% menurut spesifikasi)"]
    if not s or s.get("bandar_score") is None:
        L.append(f"N/A: {'; '.join((s or {}).get('confidence_reason', ['ledger belum ada']))}. status={ (s or {}).get('episode_status', ST_INSUF)}")
        return L
    L.append(f"[EST] skor={s['bandar_score']}/10 · {s['label']} · status={s['episode_status']} · confidence={s['confidence_score']} ({'; '.join(s['confidence_reason'])})")
    L.append(f"observasi {s['observation_start_date']}..{s['observation_end_date']} ({s['data_completeness']['sessions_loaded']}/{s['data_completeness']['target_sessions']} sesi; history_complete={s['data_completeness']['history_complete']}); "
             f"kelompok kunci (ex platform ritel): {', '.join(s['key_brokers']) or '-'}; formula {s['formula_version']}")
    L.append("Dimensi | Poin | Detail")
    for c in s["score_components"]:
        L.append(f"{c['dimensi']} | {c['poin']}/{c['maks']} | {c['detail']}")
    L.append("Bukti mendukung:")
    L += [f"- {x}" for x in s["supporting_evidence"][:8]]
    L.append("Bukti berlawanan:")
    L += [f"- {x}" for x in s["contradicting_evidence"][:8]]
    return L
