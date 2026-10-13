"""Kualitas data & fitur ledger broker yang aman dibaca LLM: memisahkan FAKTA (data sumber), HASIL HITUNG (fitur), dan HIPOTESIS (status kandidat).

Aturan utama (spesifikasi improvement ledger bandarmologi v1.0, P0+P1):
- Tidak ada kebocoran masa depan: status/fitur pada tanggal T hanya memakai data s.d. T (daily_flow_status, rolling_flow_status).
  Ringkasan akhir periode (period_end_summary) disimpan di tabel terpisah dan tidak pernah disalin ke baris historis.
- Volume pasar disimpan SEKALI per tanggal (market_daily), tidak diulang di baris broker; market_volume_shares = lot x 100 tervalidasi.
- NULL (tidak diketahui / tidak ada baris sumber) dibedakan dari nol. Posisi/modal yang tidak dapat diestimasi = NOT_AVAILABLE + alasan;
  estimasi modal broker beranomali = UNRELIABLE. Net kumulatif = cumulative_net_flow_proxy (bukan posisi).
- Status hanya kandidat berbasis proksi: ACCUMULATION_CANDIDATE / DISTRIBUTION_CANDIDATE / NEUTRAL / INSUFFICIENT_DATA.
  Kode broker = perantara banyak nasabah; bukan identitas atau niat pihak mana pun.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

import ledger as lg

FORMULA_VERSION = "ledger-quality-2026.10.1"
ACC_C, DIST_C, NEUTRAL, INSUF = "ACCUMULATION_CANDIDATE", "DISTRIBUTION_CANDIDATE", "NEUTRAL", "INSUFFICIENT_DATA"
NA, UNRELIABLE, PROXY = "NOT_AVAILABLE", "UNRELIABLE", "PROXY"
JENDELA_FITUR = (1, 3, 5, 10, 20)
LOT = lg.SHARES_PER_LOT
ADJ_BASIS = "harga chart disesuaikan split oleh aplikasi bila split terdeteksi; lot/harga broker summary TIDAK disesuaikan"
BELUM_VERIF = "belum diverifikasi sumber eksternal"

# definisi fitur rolling (denominator eksplisit); jendela = sesi bursa yang termuat s.d. as_of_date
DEFINISI = {
    "net_lots": ("sum(buy_lots - sell_lots) dalam jendela; sesi tanpa baris sumber dihitung 0 untuk penjumlahan dan dicatat di source_coverage", "-"),
    "market_volume_lots": ("sum(volume pasar harian, lot) dalam jendela", "-"),
    "net_flow_pct_market_volume": ("net_lots / market_volume_lots", "market_volume_lots jendela (chart, semua segmen; broker = pasar reguler)"),
    "buy_sell_imbalance": ("(buy_lots - sell_lots) / (buy_lots + sell_lots)", "buy_lots + sell_lots jendela (> 0)"),
    "positive_flow_days_pct": ("jumlah hari net > 0 / jumlah hari teramati", "hari dengan baris sumber dalam jendela"),
    "longest_positive_run": ("run terpanjang hari net > 0 berturut-turut", "-"),
    "direction_changes": ("pergantian tanda net antar hari bertransaksi berurutan (hari net 0/NULL dilewati)", "-"),
    "share_of_total_net_buy": ("max(net_lots broker, 0) / sum max(net_lots, 0) seluruh broker", "total net beli broker yang net beli dalam jendela"),
    "share_of_total_gross_buy": ("buy_lots broker / sum buy_lots seluruh broker", "total gross beli seluruh broker dalam jendela"),
}


@dataclass
class CfgQ:
    """Ambang kualitas & status (semua relatif terhadap volume pasar; parameter, bukan hukum)."""
    harian_min_pct: float = 0.01          # |net hari| / volume pasar hari itu untuk kandidat harian
    harian_min_imbalance: float = 0.2     # |beli-jual| / (beli+jual) hari itu
    rolling_hari: int = 20                # lookback rolling_flow_status
    rolling_min_pct: float = 0.02         # |net jendela| / volume jendela
    rolling_persist: float = 0.6          # porsi hari teramati searah
    rolling_min_obs_frac: float = 0.25    # hari teramati / lookback minimal
    satu_sisi_rasio: float = 0.02         # sisi lawan < 2% sisi dominan (kasus beli 754.563 vs jual 1.429 lot)
    satu_sisi_min_hari: int = 5           # hari satu sisi (lawan 0/NULL)
    satu_sisi_porsi_hari: float = 0.5     # porsi hari aktif yang satu sisi
    material_pct: float = 0.005           # gross sisi dominan / volume pasar periode agar anomali diperiksa
    ekstrem_net_pct: float = 0.25         # |net hari| / volume pasar hari = ekstrem
    harga_tol: float = 0.03               # rata-rata harga di luar [low, high] ± 3%
    muncul_hilang_hari: int = 5
    muncul_hilang_pct: float = 0.01       # (beli+jual) / (2 x volume periode) agar muncul/hilang diperiksa
    dominasi_hari: float = 0.5            # |net hari terbesar| / |net kumulatif|
    dominasi_min_pct: float = 0.005       # |net kumulatif| / volume periode agar dominasi diperiksa


def _iso(x) -> pd.Series:
    """Waktu UTC naive dari database -> ISO 8601 dengan penanda Z (UTC)."""
    return pd.to_datetime(pd.Series(x), errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy()


def _cfg(cfg) -> CfgQ:
    return getattr(cfg, "kualitas", None) or (cfg if isinstance(cfg, CfgQ) else CfgQ())


def label_kandidat(rule_status: str) -> str:
    """Status aturan ledger_calc -> kategori kandidat berbasis proksi."""
    return {"ACCUMULATION": ACC_C, "DISTRIBUTION": DIST_C, "MIXED": NEUTRAL, "NEUTRAL": NEUTRAL}.get(rule_status, INSUF)


# ----------------------------------------------------------------- matriks
def _wide(L: pd.DataFrame, kolom: str) -> pd.DataFrame:
    """tanggal x broker; NaN dipertahankan (tidak diketahui), BUKAN 0."""
    return L.pivot(index="trade_date", columns="broker_code", values=kolom).sort_index()


def _vol(L: pd.DataFrame) -> pd.Series:
    return L.drop_duplicates("trade_date").set_index("trade_date")["volume_lots"].sort_index()


def _run_terpanjang(pos: np.ndarray) -> np.ndarray:
    """pos: (..., n) bool -> run True terpanjang sepanjang sumbu terakhir."""
    run = np.zeros(pos.shape[:-1])
    best = np.zeros(pos.shape[:-1])
    for k in range(pos.shape[-1]):
        run = (run + 1) * pos[..., k]
        best = np.maximum(best, run)
    return best


def _ganti_arah(s: np.ndarray) -> np.ndarray:
    """s: (..., n) tanda {-1,0,1} -> jumlah pergantian tanda antar hari bertanda (0 dilewati), hanya di dalam jendela."""
    last = np.zeros(s.shape[:-1])
    ch = np.zeros(s.shape[:-1])
    for k in range(s.shape[-1]):
        sk = s[..., k]
        ch += (sk != 0) & (last != 0) & (sk != last)
        last = np.where(sk != 0, sk, last)
    return ch


def _geser(df: pd.DataFrame, n: int, fn) -> pd.DataFrame:
    a = df.to_numpy(dtype=float)
    out = np.full(a.shape, np.nan)
    if len(a) >= n:
        w = sliding_window_view(a, n, axis=0)                 # (T-n+1, B, n)
        out[n - 1:] = fn(w)
    return pd.DataFrame(out, index=df.index, columns=df.columns)


# ------------------------------------------------------------- fitur rolling
def fitur_rolling(L: pd.DataFrame, tanggal_pasar: list | None = None, jendela=JENDELA_FITUR) -> dict:
    """{n: {fitur: DataFrame tanggal x broker}} + 'source_coverage' & 'quality_flag'. Nilai pada T hanya memakai sesi <= T."""
    net, buy, sell = _wide(L, "net_lots"), _wide(L, "buy_lots"), _wide(L, "sell_lots")
    vol = _vol(L).reindex(net.index)
    obs = net.notna().astype(float)
    net0, buy0, sell0 = net.fillna(0.0), buy.fillna(0.0), sell.fillna(0.0)
    pos = (net0 > 0).astype(float)
    sgn = np.sign(net0)
    tgl = list(net.index)
    idx_pasar = {d: i for i, d in enumerate(sorted(tanggal_pasar))} if tanggal_pasar else None
    out = {}
    for n in jendela:
        r = lambda x: x.rolling(n, min_periods=n).sum()  # noqa: E731
        nw, bw, sw, ow = r(net0), r(buy0), r(sell0), r(obs)
        vw = vol.rolling(n, min_periods=n).sum()
        ada = ow > 0
        f = dict(net_lots=nw.where(ada), buy_lots=bw.where(ada), sell_lots=sw.where(ada),
                 market_volume_lots=pd.DataFrame({c: vw for c in net.columns}, index=net.index))
        f["net_flow_pct_market_volume"] = nw.div(vw.where(vw > 0), axis=0).where(ada)
        f["buy_sell_imbalance"] = ((bw - sw) / (bw + sw).where((bw + sw) > 0)).where(ada)
        f["positive_flow_days_pct"] = r(pos) / ow.where(ada)
        f["longest_positive_run"] = _geser(pos, n, _run_terpanjang).where(ada)
        f["direction_changes"] = _geser(sgn, n, _ganti_arah).where(ada)
        npos = nw.clip(lower=0)
        f["share_of_total_net_buy"] = npos.div(npos.sum(axis=1).where(npos.sum(axis=1) > 0), axis=0).where(ada)
        f["share_of_total_gross_buy"] = bw.div(bw.sum(axis=1).where(bw.sum(axis=1) > 0), axis=0).where(ada)
        cov = ow / n
        q = pd.DataFrame("OK", index=net.index, columns=net.columns)
        q = q.mask(cov < 1, "PARTIAL_COVERAGE").mask(~ada, "NO_OBSERVATION")
        if idx_pasar:
            span = pd.Series([idx_pasar.get(tgl[i], np.nan) - idx_pasar.get(tgl[i - n + 1], np.nan) + 1 if i >= n - 1 else np.nan for i in range(len(tgl))], index=net.index)
            gap = span > n
            q = q.mask(pd.DataFrame({c: gap for c in net.columns}) & q.eq("OK"), "SESSION_GAP")
        q = q.mask(nw.isna(), "INSUFFICIENT_WINDOW")
        f["source_coverage"], f["quality_flag"] = cov, q
        out[n] = f
    return out


def fitur_long(F: dict, broker: list | None = None, tanggal: list | None = None, calculated_at: str | None = None) -> pd.DataFrame:
    """Bentuk panjang broker_flow_features: as_of_date, broker_code, lookback_days, feature_name, feature_value, denominator, ... ."""
    rows = []
    for n, f in F.items():
        cov, q = f["source_coverage"], f["quality_flag"]
        idx = [d for d in cov.index if tanggal is None or d in set(tanggal)]
        cols = [c for c in cov.columns if broker is None or c in set(broker)]
        for nama, (_, den) in DEFINISI.items():
            w = f[nama].loc[idx, cols]
            s = w.stack(future_stack=True).rename("feature_value").reset_index()
            s.columns = ["as_of_date", "broker_code", "feature_value"]
            s["lookback_days"], s["feature_name"], s["denominator"] = n, nama, den
            s["source_coverage"] = cov.loc[idx, cols].stack(future_stack=True).to_numpy()
            s["quality_flag"] = q.loc[idx, cols].stack(future_stack=True).to_numpy()
            rows.append(s)
    if not rows:
        return pd.DataFrame()
    R = pd.concat(rows, ignore_index=True)
    R["formula_version"], R["calculated_at"] = FORMULA_VERSION, calculated_at or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return R.sort_values(["as_of_date", "broker_code", "lookback_days", "feature_name"]).reset_index(drop=True)[
        ["as_of_date", "broker_code", "lookback_days", "feature_name", "feature_value", "denominator", "source_coverage", "quality_flag", "formula_version", "calculated_at"]]


def konsentrasi(F: dict) -> pd.DataFrame:
    """flow_concentration per tanggal x jendela: kontribusi top-1/top-3 broker terhadap total net beli dan total gross beli (denominator eksplisit)."""
    rows = []
    for n, f in F.items():
        nw, bw = f["net_lots"], f["buy_lots"]
        for d in nw.index:
            npos = nw.loc[d].clip(lower=0).dropna().sort_values(ascending=False)
            g = bw.loc[d].dropna().sort_values(ascending=False)
            if nw.loc[d].isna().all():
                continue
            tn, tg = float(npos.sum()), float(g.sum())
            rows.append(dict(as_of_date=d, lookback_days=n, top1_broker=(npos.index[0] if tn > 0 else None),
                             top1_share_net_buy=(float(npos.iloc[0]) / tn if tn > 0 else np.nan), top3_share_net_buy=(float(npos.head(3).sum()) / tn if tn > 0 else np.nan),
                             denominator_net_buy_lots=tn, top1_share_gross_buy=(float(g.iloc[0]) / tg if tg > 0 else np.nan),
                             top3_share_gross_buy=(float(g.head(3).sum()) / tg if tg > 0 else np.nan), denominator_gross_buy_lots=tg,
                             denominator_definition="net beli = sum max(net,0) seluruh broker; gross beli = sum buy_lots seluruh broker (jendela sama)",
                             formula_version=FORMULA_VERSION))
    return pd.DataFrame(rows)


# --------------------------------------------------------------- status
def status_harian(L: pd.DataFrame, cfg=None) -> pd.DataFrame:
    """daily_flow_status: klasifikasi flow HARI ITU SAJA (tidak memakai data sebelum/sesudahnya selain harga/volume hari itu)."""
    q = _cfg(cfg)
    d = L[["trade_date", "broker_code", "buy_lots", "sell_lots", "net_lots", "volume_lots", "crossing", "data_completeness", "row_flags"]].copy()
    pct = d["net_lots"] / d["volume_lots"].where(d["volume_lots"] > 0)
    imb = d["net_lots"] / (d["buy_lots"] + d["sell_lots"]).where((d["buy_lots"] + d["sell_lots"]) > 0)
    st = np.where(d["data_completeness"] != "COMPLETE", INSUF, np.where(d["crossing"], NEUTRAL, np.where(
        (pct >= q.harian_min_pct) & (imb >= q.harian_min_imbalance), ACC_C, np.where((pct <= -q.harian_min_pct) & (imb <= -q.harian_min_imbalance), DIST_C, NEUTRAL))))
    st = np.where(pct.isna() & (d["data_completeness"] == "COMPLETE"), INSUF, st)
    ev_for, ev_ag, inval, conf = [], [], [], []
    for s, p, im, c, comp, fl in zip(st, pct, imb, d["crossing"], d["data_completeness"], d["row_flags"]):
        if s == INSUF:
            ev_for.append("")
            ev_ag.append("tidak ada baris sumber pada sesi ini (NULL, bukan 0)" if comp == NA else "lot beli/jual tidak lengkap (NULL)" if comp == "PARTIAL" else "volume pasar tidak tersedia")
            inval.append("tidak berlaku (data tidak cukup)")
            conf.append("Insufficient Data")
            continue
        o = f"net {p:+.2%} volume pasar hari itu; imbalance {im:+.2f}"
        lawan = [x for x in (("indikasi crossing (beli~jual besar): gross tidak mencerminkan arah") if c else "",
                             ("anomali baris: " + fl) if fl else "", "satu hari saja; kode broker = banyak nasabah") if x]
        if s == ACC_C:
            ev_for.append(o)
            inval.append("net jual pada sesi berikutnya >= net beli hari ini")
        elif s == DIST_C:
            ev_for.append(o)
            inval.append("net beli pada sesi berikutnya >= net jual hari ini")
        else:
            ev_for.append("")
            lawan.insert(0, o + f" (di bawah ambang ±{q.harian_min_pct:.0%} / ±{q.harian_min_imbalance:.1f})")
            inval.append("tidak berlaku (netral)")
        ev_ag.append("; ".join(lawan))
        conf.append("Low" if (s == NEUTRAL or fl or c or abs(p) < 3 * q.harian_min_pct) else "Medium")
    dq = np.where(d["data_completeness"] != "COMPLETE", d["data_completeness"], np.where(d["row_flags"] != "", "FLAGGED: " + d["row_flags"], "COMPLETE"))
    return pd.DataFrame(dict(trade_date=d["trade_date"], broker_code=d["broker_code"], status_type="daily", status=st, evidence_for=ev_for, evidence_against=ev_ag,
                             data_quality=dq, confidence=conf, as_of_date=d["trade_date"], lookback_days=1, invalidation_condition=inval,
                             formula_version=FORMULA_VERSION)).reset_index(drop=True)


def status_rolling(L: pd.DataFrame, F: dict | None = None, cfg=None) -> pd.DataFrame:
    """rolling_flow_status: jendela berjalan `rolling_hari` sesi, hanya data s.d. tanggal itu."""
    q = _cfg(cfg)
    n = q.rolling_hari
    F = F if F and n in F else fitur_rolling(L, None, (n,))
    f = F[n]
    net = _wide(L, "net_lots").fillna(0.0)
    S = pd.DataFrame({k: f[k].stack(future_stack=True) for k in ("net_flow_pct_market_volume", "positive_flow_days_pct", "source_coverage", "quality_flag", "net_lots",
                                                                 "buy_lots", "sell_lots")})
    S["mx"] = net.abs().rolling(n, min_periods=n).max().stack(future_stack=True)
    rows = []
    obs_min = max(3, int(np.ceil(q.rolling_min_obs_frac * n)))
    for (d, br), p, pdp, c, flag, nl, b_, s_, mxv in S.itertuples(name=None):
        n_obs = 0 if pd.isna(c) else int(round(c * n))
        dq = []
        if flag not in ("OK",):
            dq.append(flag)
        satu = (not pd.isna(b_) and not pd.isna(s_) and max(b_, s_) > 0 and min(b_, s_) < q.satu_sisi_rasio * max(b_, s_))
        if satu:
            dq.append("ONE_SIDED_IN_WINDOW")
        dom = (not pd.isna(nl) and abs(nl) > 0 and mxv / abs(nl) >= q.dominasi_hari and n > 1)
        if dom:
            dq.append("SINGLE_DAY_DOMINANCE")
        if pd.isna(nl) or n_obs < obs_min or pd.isna(p):
            rows.append((d, br, INSUF, "", f"hari teramati {n_obs}/{n} (minimal {obs_min}) atau jendela belum lengkap", ";".join(dq) or "INSUFFICIENT_WINDOW",
                         "Insufficient Data", f"dinilai setelah >= {obs_min} hari teramati dalam {n} sesi"))
            continue
        negd = 1 - pdp if not pd.isna(pdp) else np.nan
        o = f"net {n}D {nl:+,.0f} lot = {p:+.2%} volume pasar {n}D; hari net beli {pdp:.0%} dari {n_obs} hari teramati"
        lawan = []
        if p >= q.rolling_min_pct and pdp >= q.rolling_persist:
            s = ACC_C
            inv = f"net {n}D berbalik <= 0 atau porsi hari net beli < {q.rolling_persist:.0%}"
        elif p <= -q.rolling_min_pct and negd >= q.rolling_persist:
            s = DIST_C
            inv = f"net {n}D berbalik >= 0 atau porsi hari net jual < {q.rolling_persist:.0%}"
        else:
            s = NEUTRAL
            inv = "tidak berlaku (netral)"
            lawan.append(o + f" (ambang ±{q.rolling_min_pct:.0%} dan porsi {q.rolling_persist:.0%} tidak terpenuhi)")
        if satu:
            lawan.append(f"satu sisi dalam jendela (beli {b_:,.0f} vs jual {s_:,.0f} lot): kemungkinan cakupan/pemetaan data, perlu rekonsiliasi")
        if dom:
            lawan.append(f"satu hari = {mxv / abs(nl):.0%} dari |net {n}D|")
        if c < 1:
            lawan.append(f"cakupan {c:.0%} sesi (sesi tanpa baris sumber = NULL)")
        lawan.append("net beli broker tidak membuktikan posisi masih dipegang")
        conf = "Low" if (s == NEUTRAL or satu or dom or c < 0.8) else "Medium"
        rows.append((d, br, s, o if s != NEUTRAL else "", "; ".join(lawan), ";".join(dq) or "OK", conf, inv))
    R = pd.DataFrame(rows, columns=["trade_date", "broker_code", "status", "evidence_for", "evidence_against", "data_quality", "confidence", "invalidation_condition"])
    R.insert(2, "status_type", "rolling")
    R["as_of_date"], R["lookback_days"], R["formula_version"] = R["trade_date"], n, FORMULA_VERSION
    return R[["trade_date", "broker_code", "status_type", "status", "evidence_for", "evidence_against", "data_quality", "confidence", "as_of_date",
              "lookback_days", "invalidation_condition", "formula_version"]]


# --------------------------------------------------------------- anomali
def anomali(L: pd.DataFrame, cfg=None) -> dict:
    """{broker: [dict(flag, detail)]} untuk SELURUH periode (hanya untuk ringkasan akhir periode & estimasi; tidak disalin ke tanggal historis)."""
    q = _cfg(cfg)
    vol = _vol(L)
    vt = float(vol.sum())
    tgl = list(vol.index)
    n = len(tgl)
    out = {}
    for br, g in L.groupby("broker_code"):
        g = g.sort_values("trade_date")
        b, s, net = g["buy_lots"], g["sell_lots"], g["net_lots"]
        tb, ts = float(b.sum()), float(s.sum())
        aktif = (b.fillna(0) > 0) | (s.fillna(0) > 0)
        n_akt = int(aktif.sum())
        fl = []
        for nama, dom, lawan, td, tl in (("ONE_SIDED_BUY", b, s, tb, ts), ("ONE_SIDED_SELL", s, b, ts, tb)):
            if td < q.material_pct * vt or td <= 0:
                continue
            hari = int(((dom.fillna(0) > 0) & (lawan.isna() | (lawan == 0))).sum())
            if tl < q.satu_sisi_rasio * td or (hari >= q.satu_sisi_min_hari and n_akt and hari / n_akt >= q.satu_sisi_porsi_hari):
                fl.append(dict(flag=nama, detail=f"{'beli' if nama.endswith('BUY') else 'jual'} {td:,.0f} lot vs {'jual' if nama.endswith('BUY') else 'beli'} {tl:,.0f} lot; "
                                                    f"{hari} hari tanpa sisi lawan (0/NULL): bisa karakter transaksi, keterbatasan cakupan, atau masalah pemetaan; rekonsiliasi dulu"))
        v = g["trade_date"].map(vol)
        ek = g.loc[(net.abs() / v.where(v > 0)) >= q.ekstrem_net_pct, "trade_date"].tolist()
        if ek:
            fl.append(dict(flag="EXTREME_NET_VS_VOLUME", detail=f"{len(ek)} hari |net| >= {q.ekstrem_net_pct:.0%} volume pasar ({', '.join(map(str, ek[:4]))})"))
        if "high_price" in g and g["high_price"].notna().any():
            lo, hi = g["low_price"] * (1 - q.harga_tol), g["high_price"] * (1 + q.harga_tol)
            luar = int(((g["avg_buy_price"] < lo) | (g["avg_buy_price"] > hi)).sum() + ((g["avg_sell_price"] < lo) | (g["avg_sell_price"] > hi)).sum())
            if luar:
                fl.append(dict(flag="AVG_PRICE_OUT_OF_RANGE", detail=f"{luar} harga rata-rata di luar rentang harian ±{q.harga_tol:.0%} (crossing/nego/aksi korporasi?)"))
        if n >= 2 * q.muncul_hilang_hari and n_akt and (tb + ts) >= q.muncul_hilang_pct * 2 * vt:
            ia = [tgl.index(d) for d in g.loc[aktif, "trade_date"]]
            if ia[0] >= n // 2:
                fl.append(dict(flag="APPEARED", detail=f"baru aktif sejak {tgl[ia[0]]} (sesi ke-{ia[0] + 1} dari {n}) dengan gross material"))
            if ia[-1] <= n - 1 - q.muncul_hilang_hari:
                fl.append(dict(flag="DISAPPEARED", detail=f"tidak aktif sejak {tgl[ia[-1]]} ({n - 1 - ia[-1]} sesi terakhir tanpa transaksi teramati)"))
        cum = float(net.sum())
        if abs(cum) >= q.dominasi_min_pct * vt and n_akt >= 2:
            same = net.where(np.sign(net) == np.sign(cum))
            if same.notna().any():
                i = same.abs().idxmax()
                share = abs(float(net[i])) / abs(cum)
                if share >= q.dominasi_hari:
                    fl.append(dict(flag="SINGLE_DAY_DOMINANCE", detail=f"{g.loc[i, 'trade_date']}: {share:.0%} dari net kumulatif {cum:+,.0f} lot berasal dari satu hari"))
        if fl:
            out[br] = fl
    return out


# ------------------------------------------------------------ crossing & ekstrem
def kontribusi_crossing(L: pd.DataFrame) -> pd.DataFrame:
    """Per broker + TOTAL: kontribusi baris yang ditandai crossing vs pasar reguler non-crossing (segmen tidak dikenal = UNKNOWN)."""
    d = L.assign(_x=L["crossing"].astype(bool))
    rows = []
    for (br, seg), g in d.groupby(["broker_code", "market_segment"]):
        x, r = g[g["_x"]], g[~g["_x"]]
        rows.append(dict(broker_code=br, market_segment=seg, net_lots_total=float(g["net_lots"].sum()), net_lots_non_crossing=float(r["net_lots"].sum()),
                         net_lots_crossing_flagged=float(x["net_lots"].sum()), gross_buy_non_crossing=float(r["buy_lots"].sum()),
                         gross_buy_crossing_flagged=float(x["buy_lots"].sum()), crossing_days=int(len(x))))
    R = pd.DataFrame(rows)
    if R.empty:
        return R
    tot = R.drop(columns=["broker_code", "market_segment"]).sum()
    R = pd.concat([R, pd.DataFrame([dict(broker_code="TOTAL", market_segment="/".join(sorted(R["market_segment"].unique())), **tot.to_dict())])], ignore_index=True)
    gb = R["gross_buy_non_crossing"] + R["gross_buy_crossing_flagged"]
    R["crossing_pct_gross_buy"] = R["gross_buy_crossing_flagged"] / gb.where(gb > 0)
    R["comparable_note"] = np.where(R["market_segment"].str.contains("UNKNOWN"), "segmen tidak diketahui: jangan digabung dengan reguler",
                                    "crossing = heuristik (beli~jual besar) di pasar reguler; gross-nya menggelembung, net hampir tidak terpengaruh")
    return R


def dampak_ekstrem(L: pd.DataFrame, cfg=None) -> pd.DataFrame:
    """Net kumulatif dengan dan tanpa hari ekstrem (data asli tidak dihapus; ini hanya simulasi pembanding)."""
    q = _cfg(cfg)
    vol = _vol(L)
    rows = []
    for br, g in L.groupby("broker_code"):
        net = g["net_lots"]
        if net.notna().sum() == 0:
            continue
        tot = float(net.sum())
        i = net.abs().idxmax()
        v = g["trade_date"].map(vol)
        ek = (net.abs() / v.where(v > 0)) >= q.ekstrem_net_pct
        tanpa1, tanpa_ek = tot - float(net[i]), float(net[~ek].sum())
        rows.append(dict(broker_code=br, net_lots_total=tot, top_abs_day=g.loc[i, "trade_date"], top_abs_day_net=float(net[i]),
                         top_day_share=(abs(float(net[i])) / abs(tot) if tot else np.nan), net_lots_ex_top_day=tanpa1, n_extreme_days=int(ek.sum()),
                         net_lots_ex_extreme_days=tanpa_ek, sign_flip_ex_top_day=bool(tot != 0 and np.sign(tanpa1) != np.sign(tot)),
                         sign_flip_ex_extreme=bool(tot != 0 and ek.any() and np.sign(tanpa_ek) != np.sign(tot))))
    return pd.DataFrame(rows)


# --------------------------------------------------------------- tabel ekspor
def market_daily(L: pd.DataFrame, m: pd.DataFrame, verif: dict | None = None) -> pd.DataFrame:
    """SATU baris per tanggal (sumber utama). Volume pasar tidak pernah diulang per broker; join ke broker_flow_daily via trade_date."""
    m = m.sort_values("trade_date")
    m = m[(m["trade_date"] >= L["trade_date"].min()) & (m["trade_date"] <= L["trade_date"].max())].reset_index(drop=True)
    g = L.groupby("trade_date")
    shares = m["volume_shares"].astype(float)
    kol = lambda c: m[c].to_numpy(dtype=float) if c in m else np.full(len(m), np.nan)  # noqa: E731
    D = pd.DataFrame(dict(trade_date=m["trade_date"].to_numpy(), open=kol("open"), high=kol("high"), low=kol("low"), close=kol("close"),
                          market_volume_lots=(shares / LOT).to_numpy(), market_volume_shares=shares.to_numpy()))
    D["market_value_idr"], D["frequency"] = np.nan, np.nan               # tidak tersedia di chart tersimpan (NULL, bukan 0)
    tg = D["trade_date"]
    D["broker_data_available"] = tg.isin(set(L["trade_date"])).to_numpy()
    D["n_brokers_observed"] = tg.map(g["data_completeness"].apply(lambda s: int((s != NA).sum()))).to_numpy()
    D["regular_buy_lots_all_brokers"] = tg.map(g["buy_lots"].sum()).to_numpy()
    D["crossing_flagged_buy_lots"] = tg.map(L[L["crossing"].astype(bool)].groupby("trade_date")["buy_lots"].sum()).fillna(0.0).where(D["broker_data_available"]).to_numpy()
    rasio = D["regular_buy_lots_all_brokers"] / D["market_volume_lots"].where(D["market_volume_lots"] > 0)
    D["broker_to_market_volume_ratio"] = rasio
    D["non_regular_or_unknown_lots"] = (D["market_volume_lots"] - D["regular_buy_lots_all_brokers"]).clip(lower=0).where(D["broker_data_available"])
    kelipatan = (np.abs(D["market_volume_shares"] - D["market_volume_lots"] * LOT) < 1e-6) & (np.mod(D["market_volume_shares"], LOT) == 0)
    D["volume_unit_check"] = np.where(~kelipatan, "NOT_LOT_MULTIPLE", np.where(rasio.notna() & ((rasio < 0.05) | (rasio > 1.5)), "UNIT_MISMATCH_SUSPECTED", "OK"))
    D["source_id"] = (m["data_source"].to_numpy() if "data_source" in m else lg.SUMBER_PASAR)
    D["ingested_at"] = _iso(m["retrieved_at"]) if "retrieved_at" in m else None
    D["unit_volume"], D["adjustment_basis"] = lg.UNIT_VOLUME, ADJ_BASIS
    ext = pd.Series(BELUM_VERIF, index=D.index)
    if verif and verif.get("n_dibanding"):
        lain = ", ".join(verif["sumber"][1:])
        ext[D["trade_date"].isin(set(verif.get("tanggal_dibanding") or []))] = f"COCOK dengan {lain} (dalam toleransi)"
        K = verif.get("konflik")
        if K is not None and not K.empty:
            for d, k in K.groupby("trade_date"):
                ext[D["trade_date"] == d] = "KONFLIK: " + "; ".join(f"{r.field} {r.source_a}={r.value_a:,.2f} vs {r.source_b}={r.value_b:,.2f}" for r in k.itertuples())
    D["external_verification"] = ext
    return D.reset_index(drop=True)


def cek_ledger(L: pd.DataFrame) -> list[dict]:
    """Pemeriksaan tingkat baris pada ledger turunan: net = beli - jual, duplikasi kunci (tanggal, broker, segmen), non-negatif, NULL vs 0."""
    out = []
    both = L["buy_lots"].notna() & L["sell_lots"].notna()
    salah = int((both & ((L["net_lots"] - (L["buy_lots"] - L["sell_lots"])).abs() > 1e-6)).sum() + (~both & L["net_lots"].notna()).sum())
    out.append(dict(id="net_lots_valid", status="OK" if salah == 0 else "GAGAL", detail="net_lots = buy_lots - sell_lots di semua baris; NULL bila salah satu sisi NULL" if salah == 0 else
                    f"{salah} baris net_lots tidak sama dengan buy_lots - sell_lots"))
    dup = int(L.duplicated(["trade_date", "broker_code", "market_segment"]).sum())
    out.append(dict(id="duplikat_kunci_ledger", status="OK" if dup == 0 else "GAGAL", detail=f"{dup} duplikat kunci (trade_date, broker_code, market_segment)"))
    neg = int(((L[["buy_lots", "sell_lots", "buy_value", "sell_value"]] < 0).any(axis=1)).sum())
    out.append(dict(id="nilai_negatif", status="OK" if neg == 0 else "GAGAL", detail=f"{neg} baris dengan lot/nilai negatif"))
    na, part = int((L["data_completeness"] == NA).sum()), int((L["data_completeness"] == "PARTIAL").sum())
    out.append(dict(id="null_vs_nol", status="OK" if part == 0 else "PERINGATAN",
                    detail=f"{na} broker-sesi tanpa baris sumber = NULL/NOT_AVAILABLE (bukan 0); {part} baris dengan salah satu sisi NULL"))
    unk = int((L["market_segment"] == "UNKNOWN").sum())
    out.append(dict(id="segmen", status="OK" if unk == 0 else "PERINGATAN", detail="semua baris pasar REGULAR (market=RG)" if unk == 0 else
                    f"{unk} baris segmen UNKNOWN: jangan digabung dengan pasar reguler"))
    return out


def _estimasi(v: dict, anom: list, opening: float | None) -> dict:
    ep = v.get("episode")
    if opening is not None:
        pos = dict(estimated_position_status=PROXY, estimated_position_lots=v["posisi"].get("dasar"),
                   estimated_position_reason="persediaan awal dinyatakan pengguna + cumulative_net_flow_proxy; tetap bukan inventory aktual")
    else:
        pos = dict(estimated_position_status=NA, estimated_position_lots=np.nan,
                   estimated_position_reason="persediaan awal & inventory broker tidak tersedia; hanya cumulative_net_flow_proxy (bukan posisi)")
    biaya = ep.get("avg_buy_ex_crossing") if ep else np.nan
    if biaya is None or pd.isna(biaya):
        cst, crs = NA, ("tidak ada episode akumulasi berkelanjutan" if not ep else "tidak ada beli non-crossing dalam episode")
    elif anom:
        cst, crs = UNRELIABLE, "anomali broker: " + ", ".join(a["flag"] for a in anom) + " (transaksi beli/jual mungkin tidak lengkap)"
    else:
        cst, crs = PROXY, "rata-rata beli ex-crossing dalam episode (weighted: nilai/lembar); bukan harga pokok pemegang"
    if anom and pos["estimated_position_status"] == PROXY:
        pos.update(estimated_position_status=UNRELIABLE, estimated_position_reason=pos["estimated_position_reason"] + "; anomali broker")
    return dict(pos, avg_cost_proxy=biaya if cst != NA else np.nan, avg_cost_status=cst, avg_cost_reason=crs)


INVALIDASI_AKHIR = {ACC_C: "net kumulatif turun >= 30% dari puncak episode, atau >= 3 sesi net jual beruntun disertai harga melemah",
                    DIST_C: "net beli kembali persisten (>= 60% sesi aktif net beli) dan harga bertahan di atas zona distribusi",
                    NEUTRAL: "tidak ada hipotesis aktif; status berubah bila aturan kandidat salah satu sisi terpenuhi",
                    INSUF: "dinilai setelah jumlah sesi aktif mencukupi"}


def period_end_summary(L: pd.DataFrame, hasil: dict, anom: dict, ekstrem: pd.DataFrame, cross: pd.DataFrame, opening: float | None = None) -> pd.DataFrame:
    """Ringkasan FINAL akhir periode per broker (tabel terpisah; tidak disalin ke baris historis)."""
    as_of = L["trade_date"].max()
    n_sesi = int(L["trade_date"].nunique())
    E = ekstrem.set_index("broker_code") if not ekstrem.empty else pd.DataFrame()
    X = cross[cross["broker_code"] != "TOTAL"].groupby("broker_code")[["net_lots_crossing_flagged", "net_lots_non_crossing"]].sum() if not cross.empty else pd.DataFrame()
    rows = []
    for k, v in hasil.items():
        st = label_kandidat(v["status"])
        an = anom.get(k, [])
        sa_, sd_ = v.get("sinyal_acc") or {}, v.get("sinyal_dist") or {}
        if st == DIST_C:
            pro, kon = [s for s, ok in sd_.items() if ok], [s for s, ok in sa_.items() if ok]
        else:
            pro, kon = [s for s, ok in sa_.items() if ok], [s for s, ok in sd_.items() if ok]
            if st == NEUTRAL and v["status"] == "MIXED":
                kon.insert(0, "pola campuran: sinyal akumulasi dan distribusi sama-sama terpenuhi")
        kon += [f"{a['flag']}: {a['detail']}" for a in an]
        if v.get("catatan"):
            kon.append(v["catatan"])
        kon.append("net beli broker tidak membuktikan posisi masih dipegang; kode broker = banyak nasabah")
        conf = v.get("confidence", "Low")
        if an and conf != "Insufficient Data":
            conf = "Low"
        dq = "OK" if not an else ("UNRELIABLE (" if any(a["flag"].startswith("ONE_SIDED") for a in an) else "LIMITED (") + ", ".join(a["flag"] for a in an) + ")"
        e = E.loc[k] if k in E.index else None
        rows.append(dict(broker_code=k, broker_name=v.get("nama"), as_of_date=as_of, lookback_days=n_sesi, status=st, rule_status=v["status"],
                         evidence_for="; ".join(pro), evidence_against="; ".join(kon), data_quality=dq, confidence=conf,
                         invalidation_condition=INVALIDASI_AKHIR[st], gross_buy_lots=v["gross_buy_lots"], gross_sell_lots=v["gross_sell_lots"],
                         cumulative_net_flow_proxy=v["net_flow_lots"],
                         net_lots_non_crossing=(float(X.loc[k, "net_lots_non_crossing"]) if k in X.index else np.nan),
                         net_lots_crossing_flagged=(float(X.loc[k, "net_lots_crossing_flagged"]) if k in X.index else np.nan),
                         net_lots_ex_top_day=(e["net_lots_ex_top_day"] if e is not None else np.nan),
                         net_lots_ex_extreme_days=(e["net_lots_ex_extreme_days"] if e is not None else np.nan),
                         anomaly_flags=",".join(a["flag"] for a in an), **_estimasi(v, an, opening), label="NET_FLOW_PROXY", formula_version=FORMULA_VERSION))
    return pd.DataFrame(rows).sort_values("cumulative_net_flow_proxy", ascending=False).reset_index(drop=True)


def broker_flow_daily(L: pd.DataFrame) -> pd.DataFrame:
    """Satu baris per tanggal x broker x segmen. TANPA volume/harga pasar (lihat market_daily). Status hanya memakai data s.d. tanggal itu."""
    cols = ["trade_date", "ticker", "broker_code", "broker_name", "market_segment", "data_completeness", "buy_lots", "sell_lots", "net_lots", "buy_value", "sell_value",
            "net_value", "avg_buy_price", "avg_sell_price", "crossing", "cumulative_net_flow_proxy", "daily_flow_status", "rolling_flow_status", "rolling_lookback_days",
            "row_flags", "is_final", "source_id", "ingested_at"]
    D = L[[c for c in cols if c in L]].copy().rename(columns={"crossing": "crossing_flag", "row_flags": "quality_flags", "buy_value": "buy_value_idr",
                                                               "sell_value": "sell_value_idr", "net_value": "net_value_idr"})
    D["data_status"] = np.where(D.pop("is_final").astype(bool), "FINAL", "PARTIAL")
    if "ingested_at" in D:
        D["ingested_at"] = _iso(D["ingested_at"])
    D["unit_volume"], D["unit_price"], D["source_version"], D["formula_version"] = lg.UNIT_VOLUME, lg.UNIT_HARGA, lg.SOURCE_VERSION, FORMULA_VERSION
    D["label"] = "NET_FLOW_PROXY"
    return D.sort_values(["trade_date", "broker_code"]).reset_index(drop=True)
