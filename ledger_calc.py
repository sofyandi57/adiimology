"""Analisis buku besar broker: episode akumulasi, biaya rata-rata (proksi), posisi tersisa (skenario), status aliran, zona distribusi, peringatan.

Penanda: [OBS] teramati dari data tersimpan, [EST] estimasi model, [INF] inferensi, [DEC] keputusan.
PRINSIP: broker = perantara banyak nasabah; net kumulatif dan biaya rata-rata adalah PROKSI ALIRAN (NET_FLOW_PROXY), bukan inventaris
atau harga pokok sebenarnya. Net positif saja tidak pernah cukup untuk label ACCUMULATION. Data tidak lengkap -> N/A, bukan tebakan.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import ledger as lg
import ledger_kualitas as lq

LOT = lg.SHARES_PER_LOT
ACC, DIST, MIXED, NEUTRAL, INSUF = "ACCUMULATION", "DISTRIBUTION", "MIXED", "NEUTRAL", "INSUFFICIENT_DATA"


@dataclass
class Cfg:
    min_hari_aktif: int = 20          # di bawah ini: INSUFFICIENT_DATA
    jendela_persist: int = 20         # sesi aktif terakhir untuk konsistensi
    persist_frac: float = 0.6
    min_partisipasi: float = 0.02     # net 20h / volume 20h
    min_gross_pct: float = 0.005      # di bawah ini (gross/volume) dianggap tidak aktif -> NEUTRAL
    episode_min_hari: int = 5
    episode_min_net_pct: float = 0.005
    dist_beruntun: int = 3
    dist_nilai_mult: float = 1.5
    turun_dari_puncak: float = 0.30
    dekat_zona: float = 0.02
    zona_lebar: float = 0.02
    reaksi_hari: int = 5
    breakout_naik: float = 0.03
    opening_inventory: float | None = None      # lot; hanya bila pengguna menyatakannya
    kualitas: lq.CfgQ = field(default_factory=lq.CfgQ)   # ambang anomali/status harian-rolling (relatif volume pasar)


# --------------------------------------------------------------- ledger harian
def bangun(b: pd.DataFrame, m: pd.DataFrame, cfg: Cfg | None = None) -> pd.DataFrame:
    """Ledger harian per broker (satuan lot, Rupiah) + cumulative_net_flow_proxy per broker pada jendela data yang tersedia.
    Broker tanpa baris sumber pada suatu sesi -> lot/nilai NULL dan data_completeness=NOT_AVAILABLE (BUKAN 0).
    close_price/volume_lots/high/low di sini hanya join internal; ekspor memakai market_daily terpisah (tidak diulang per broker)."""
    cfg = cfg or Cfg()
    m, _ = lg.pisah_sumber(m)
    if b.empty or m.empty:
        return pd.DataFrame()
    bx = lg.tandai_crossing(b, m)
    mm = m.sort_values("trade_date").reset_index(drop=True)
    mm = mm[mm["trade_date"].isin(set(b["trade_date"]))].reset_index(drop=True)      # hanya sesi yang BENAR-BENAR termuat: tidak ada nol buatan untuk sesi hilang
    if mm.empty:
        return pd.DataFrame()
    tgl = mm["trade_date"].tolist()
    base = pd.DataFrame({"trade_date": tgl, "close_price": mm["close"].to_numpy(), "volume_lots": mm["volume_shares"].to_numpy() / LOT,
                         "high_price": mm["high"].to_numpy() if "high" in mm else np.nan, "low_price": mm["low"].to_numpy() if "low" in mm else np.nan})
    kode = bx["ticker"].iloc[0]
    rows = []
    for br, g in bx.groupby("broker_code"):
        g = g.set_index("trade_date")
        d = base.copy()
        for c in ("buy_shares", "sell_shares", "buy_value", "sell_value"):
            d[c] = d["trade_date"].map(g[c])                               # NaN = tidak diketahui / tidak ada baris sumber
        d["crossing"] = d["trade_date"].map(g["crossing"]).fillna(False).astype(bool)
        d["buy_avg_raw"] = d["trade_date"].map(g["buy_avg_raw"])
        d["sell_avg_raw"] = d["trade_date"].map(g["sell_avg_raw"])
        d["broker_name"] = g["broker_name"].iloc[0]
        d["broker_code"] = br
        d["ticker"] = kode
        d["is_final"] = d["trade_date"].map(g["is_final"]).fillna(True).astype(bool)
        d["missing_data_flag"] = ~d["trade_date"].isin(g.index)          # True = broker tidak punya baris sumber di sesi itu (NULL; BUKAN tanggal hilang)
        src = g["data_source"] if "data_source" in g else pd.Series(lg.SUMBER, index=g.index)
        d["market_segment"] = d["trade_date"].map(src.map(lg.segmen)).fillna("UNKNOWN")
        d["source_id"] = [lg.source_id_broker(kode, t, src.get(t, lg.SUMBER)) if t in g.index else None for t in d["trade_date"]]
        d["ingested_at"] = d["trade_date"].map(g["retrieved_at"]) if "retrieved_at" in g else pd.NaT
        rows.append(d)
    L = pd.concat(rows, ignore_index=True)
    L["source_timestamp"] = L["ingested_at"]
    L["buy_lots"], L["sell_lots"] = L["buy_shares"] / LOT, L["sell_shares"] / LOT
    L["net_lots"] = L["buy_lots"] - L["sell_lots"]                         # NULL bila salah satu sisi NULL
    L["net_value"] = L["buy_value"] - L["sell_value"]
    L["avg_buy_price"] = np.where(L["buy_shares"] > 0, L["buy_value"] / L["buy_shares"].where(L["buy_shares"] > 0), np.nan)
    L["avg_sell_price"] = np.where(L["sell_shares"] > 0, L["sell_value"] / L["sell_shares"].where(L["sell_shares"] > 0), np.nan)
    L["data_completeness"] = np.where(L["missing_data_flag"], lq.NA, np.where(L["buy_lots"].isna() | L["sell_lots"].isna(), "PARTIAL", "COMPLETE"))
    L["row_flags"] = _flag_baris(L, cfg.kualitas)
    L = L.sort_values(["broker_code", "trade_date"]).reset_index(drop=True)
    L["cumulative_net_flow_proxy"] = L["net_lots"].fillna(0.0).groupby(L["broker_code"]).cumsum()   # proksi aliran, BUKAN posisi
    return L


def _flag_baris(L: pd.DataFrame, q: lq.CfgQ) -> pd.Series:
    """Flag kualitas tingkat baris yang hanya memakai data HARI ITU (aman untuk tanggal historis)."""
    f = []
    lo, hi = L["low_price"] * (1 - q.harga_tol), L["high_price"] * (1 + q.harga_tol)
    for nama, cond in (("SIDE_NULL", (L["data_completeness"] == "PARTIAL")),
                       ("CROSSING_SUSPECTED", L["crossing"]),
                       ("AVG_BUY_OUT_OF_RANGE", (L["avg_buy_price"] < lo) | (L["avg_buy_price"] > hi)),
                       ("AVG_SELL_OUT_OF_RANGE", (L["avg_sell_price"] < lo) | (L["avg_sell_price"] > hi)),
                       ("NET_EXTREME_VS_VOLUME", (L["net_lots"].abs() / L["volume_lots"].where(L["volume_lots"] > 0)) >= q.ekstrem_net_pct)):
        f.append(np.where(cond.fillna(False).astype(bool), nama, ""))
    return pd.Series([";".join(x for x in t if x) for t in zip(*f)], index=L.index)


# ------------------------------------------------------------------ per broker
def _episode(net: np.ndarray, tgl: list, vol_total: float, aktif: np.ndarray, cfg: Cfg) -> dict | None:
    """Episode akumulasi = dari lembah net kumulatif terakhir sebelum puncak sampai puncak; bukan hari net-buy pertama."""
    ext = np.r_[0.0, np.cumsum(net)]
    p = int(np.argmax(ext))
    s = int(np.where(ext[:p + 1] == ext[:p + 1].min())[0][-1])
    gain = ext[p] - ext[s]
    if p <= s or gain <= 0:
        return None
    sl = slice(s, p)                                      # hari bursa index s .. p-1
    hari = p - s
    pos = (net[sl] > 0) & aktif[sl]
    n_aktif = int(aktif[sl].sum())
    frac = float(pos.sum() / n_aktif) if n_aktif else 0.0
    if hari < cfg.episode_min_hari or gain < cfg.episode_min_net_pct * vol_total or frac < cfg.persist_frac:
        return None
    return dict(i0=s, i1=p - 1, mulai=tgl[s], puncak=tgl[p - 1], hari=hari, gain_lots=float(gain), frac_net_beli=frac,
                masih_naik=(p == len(ext) - 1), turun_dari_puncak=float(ext[p] - ext[-1]), turun_pct=float((ext[p] - ext[-1]) / gain), ext=ext)


def _skenario_posisi(net: np.ndarray, opening: float | None) -> dict:
    ext = np.r_[0.0, np.cumsum(net)]
    cum_end = float(ext[-1])
    i_min = float(max(0.0, -ext.min()))
    return dict(label="NET_FLOW_PROXY", net_flow_lots=cum_end, rendah=max(0.0, cum_end), tinggi=cum_end + i_min,
                dasar=(max(0.0, opening + cum_end) if opening is not None else None), i_min=i_min,
                asumsi=("rendah = persediaan awal 0 (net negatif dianggap 0); tinggi = persediaan awal minimal agar posisi tidak pernah negatif; "
                        "dasar = persediaan awal yang Anda nyatakan" if opening is not None else
                        "rendah = persediaan awal 0; tinggi = persediaan awal minimal agar posisi tidak pernah negatif; dasar = N/A (persediaan awal tidak diketahui)"))


def analisa_broker(g: pd.DataFrame, cfg: Cfg, vol_total_lots: float, validasi_gagal: bool = False) -> dict:
    """g = baris ledger satu broker (urut tanggal, seluruh hari bursa)."""
    g = g.sort_values("trade_date").reset_index(drop=True)
    for c in ("buy_lots", "sell_lots", "net_lots", "buy_shares", "sell_shares", "buy_value", "sell_value", "net_value"):
        if c in g:                    # agregasi: sesi tanpa baris sumber (NULL) tidak menyumbang transaksi teramati; nilai NULL tetap NULL di ledger
            g[c] = g[c].fillna(0.0)
    tgl, close = g["trade_date"].tolist(), g["close_price"].to_numpy()
    net, buy, sell = g["net_lots"].to_numpy(), g["buy_lots"].to_numpy(), g["sell_lots"].to_numpy()
    aktif = ((g["buy_shares"] > 0) | (g["sell_shares"] > 0)).to_numpy()
    cx = g["crossing"].to_numpy()
    n_hari, n_aktif = len(g), int(aktif.sum())
    gb, gs = float(buy.sum()), float(sell.sum())
    gb_x, gs_x = float(buy[~cx].sum()), float(sell[~cx].sum())
    bv, sv = float(g["buy_value"].sum()), float(g["sell_value"].sum())
    bv_x, sv_x = float(g.loc[~g["crossing"], "buy_value"].sum()), float(g.loc[~g["crossing"], "sell_value"].sum())
    avg_all = bv / (gb * LOT) if gb > 0 else np.nan
    avg_x = bv_x / (gb_x * LOT) if gb_x > 0 else np.nan
    net_sh = (gb - gs) * LOT
    proxy_net = (bv - sv) / net_sh if net_sh > 0 and (bv - sv) > 0 else np.nan
    ep = _episode(net, tgl, vol_total_lots, aktif, cfg)
    res = dict(broker=g["broker_code"].iloc[0], nama=g["broker_name"].iloc[0], n_hari=n_hari, n_aktif=n_aktif, gross_buy_lots=gb, gross_sell_lots=gs,
               gross_buy_ex_crossing=gb_x, gross_sell_ex_crossing=gs_x, net_flow_lots=gb - gs, buy_value=bv, sell_value=sv,
               avg_buy_gross=avg_all, avg_buy_ex_crossing=avg_x, net_flow_cost_proxy=proxy_net, crossing_pct_gross_buy=(gb - gb_x) / gb if gb > 0 else 0.0,
               episode=ep, posisi=_skenario_posisi(net, cfg.opening_inventory))
    if ep:
        sl = slice(ep["i0"], ep["i1"] + 1)
        eb, ebx = float(buy[sl].sum()), float(buy[sl][~cx[sl]].sum())
        ev = float(g["buy_value"].to_numpy()[sl].sum())
        evx = float(g["buy_value"].to_numpy()[sl][~cx[sl]].sum())
        ep.update(gross_buy_lots=eb, gross_sell_lots=float(sell[sl].sum()), net_lots=float(net[sl].sum()),
                  avg_buy_gross=ev / (eb * LOT) if eb > 0 else np.nan, avg_buy_ex_crossing=evx / (ebx * LOT) if ebx > 0 else np.nan)
    k20j = slice(max(0, n_hari - 20), n_hari)
    gb20, gs20 = float(buy[k20j].sum()), float(sell[k20j].sum())
    jual_sistematis = bool(int((net[k20j] < 0).sum()) >= 15 and gs20 > 0 and gb20 < 0.05 * gs20)   # penjual sistematis: net jual >=15 dari 20 sesi, gross beli < 5% gross jual
    res["jual_sistematis"] = jual_sistematis
    if n_aktif < cfg.min_hari_aktif:
        res.update(status=INSUF, confidence="Insufficient Data", sinyal_acc={}, sinyal_dist={}, catatan=f"hari aktif {n_aktif} < {cfg.min_hari_aktif}")
        return res
    if (gb + gs) < cfg.min_gross_pct * vol_total_lots * 2:
        res.update(status=NEUTRAL, confidence="Low", sinyal_acc={}, sinyal_dist={}, catatan="aktivitas gross terlalu kecil terhadap volume pasar")
        return res
    w = cfg.jendela_persist
    idx_aktif = np.where(aktif)[0][-w:]
    nl = net[idx_aktif]
    s1 = bool(len(nl) and (nl > 0).mean() >= cfg.persist_frac)
    s2 = bool(res["net_flow_lots"] > 0 and ep is not None)
    k20 = slice(max(0, n_hari - 20), n_hari)
    beli_hari = [i for i in range(k20.start, n_hari) if net[i] > 0]
    pull = [i for i in beli_hari if i >= 5 and close[i] <= close[i - 5] * 1.01]
    s3 = bool(len(beli_hari) >= 3 and len(pull) / len(beli_hari) >= 0.4)
    est_biaya = (ep["avg_buy_ex_crossing"] if ep and pd.notna(ep.get("avg_buy_ex_crossing")) else avg_x)
    s4 = bool(pd.notna(est_biaya) and close[-1] > est_biaya)
    ret20 = close[-1] / close[max(0, n_hari - 21)] - 1
    s5 = bool(ret20 > 0.02 and net[k20].sum() > 0)
    part = float(net[k20].sum() / max(float(g["volume_lots"].iloc[k20].sum()), 1.0))
    s6 = bool(part >= cfg.min_partisipasi)
    sig_a = {"S1 net beli persisten (20 sesi aktif)": s1, "S2 net kumulatif positif & ada episode": s2, "S3 beli saat konsolidasi/koreksi": s3,
             "S4 estimasi biaya di bawah harga": s4, "S5 harga naik saat net beli": s5, "S6 partisipasi >= ambang terhadap volume": s6}
    run = 0
    for v in net[::-1]:
        if v < 0:
            run += 1
        else:
            break
    d2 = run >= cfg.dist_beruntun
    jual_nilai = g["sell_value"].where(~g["crossing"], 0.0).to_numpy()
    rata = float(jual_nilai[aktif].mean()) if aktif.any() else 0.0
    d3 = bool(rata > 0 and float(jual_nilai[-5:].mean()) >= cfg.dist_nilai_mult * rata)
    d4 = bool(sum(1 for v in net[-10:] if v > 0) >= 6 and close[-1] / close[-11] - 1 <= 0) if n_hari > 11 else False
    d5 = bool(n_hari > 10 and close[-1] / close[-6] - 1 < 0 and net[-5:].sum() < net[-10:-5].sum() and net[-5:].sum() < 0)
    hi60 = np.maximum.accumulate(close)
    jual_idx = [i for i in range(max(0, n_hari - 20), n_hari) if net[i] < 0]
    dekat = [i for i in jual_idx if close[i] >= 0.98 * float(close[max(0, i - 59):i + 1].max())]
    d6 = bool(len(dekat) >= 2 and len(dekat) / max(len(jual_idx), 1) >= 0.3)
    sig_d = {"D1 net jual beruntun >= %d sesi" % cfg.dist_beruntun: d2, "D2 setelah episode akumulasi": bool(ep and d2), "D3 nilai jual 5h >= %.1fx rata-rata" % cfg.dist_nilai_mult: d3,
             "D4 harga gagal naik meski sering net beli": d4, "D5 harga melemah & net jual meningkat": d5, "D6 menjual di dekat resistance (high 60h)": d6}
    n_a = sum([s3, s4, s5, s6])
    dist_ok = d2 and (d3 or d5 or d6) and (ep is not None or res["net_flow_lots"] > 0)
    acc_ok = s1 and s2 and n_a >= 2
    if acc_ok and dist_ok:
        st = MIXED
    elif acc_ok:
        st = ACC
    elif dist_ok:
        st = DIST
    elif s1 and d2:
        st = MIXED
    else:
        st = NEUTRAL
    if jual_sistematis and st != DIST:
        st = DIST
        sig_d["D7 penjual sistematis (net jual >=15 dari 20 sesi, gross beli <5% gross jual)"] = True
    n_sig = (sum(sig_a.values()) if st in (ACC, MIXED) else sum(sig_d.values()) if st == DIST else 0)
    if validasi_gagal or st == NEUTRAL:
        conf = "Low"
    elif st in (ACC, DIST) and n_sig >= 4 and n_aktif >= 40:
        conf = "High"
    elif n_sig >= 3 and n_aktif >= 30:
        conf = "Medium"
    else:
        conf = "Low"
    catatan = ("net positif tanpa sinyal pendukung yang cukup (net positif saja TIDAK cukup untuk ACCUMULATION)" if st == NEUTRAL and res["net_flow_lots"] > 0 else "")
    res.update(status=st, confidence=conf, sinyal_acc=sig_a, sinyal_dist=sig_d, partisipasi_20h=part, ret20=ret20, run_net_jual=run, catatan=catatan)
    return res


# ------------------------------------------------------------------ zona distribusi
def _reaksi(close: np.ndarray, i: int, h: int) -> float:
    return float(close[min(i + h, len(close) - 1)] / close[i] - 1) if i < len(close) - 1 else np.nan


def zona_distribusi(L: pd.DataFrame, hasil: dict, cfg: Cfg) -> dict:
    """Zona dari penjualan teramati broker kunci (bukan target harga broker: niat jual tidak teramati)."""
    kunci = [k for k, v in hasil.items() if v["status"] in (ACC, MIXED, DIST) or (v.get("episode") and v["status"] != INSUF)]
    ev = []
    for k in kunci:
        g = L[L["broker_code"] == k].sort_values("trade_date").reset_index(drop=True)
        close = g["close_price"].to_numpy()
        ep = hasil[k].get("episode")
        mulai = ep["i0"] if ep else 0
        for i in range(mulai, len(g)):
            if g["net_lots"].iloc[i] < 0 and not g["crossing"].iloc[i] and g["sell_lots"].iloc[i] > 0:
                ev.append(dict(tgl=g["trade_date"].iloc[i], broker=k, close=float(close[i]), net_lots=float(g["net_lots"].iloc[i]), sell_lots=float(g["sell_lots"].iloc[i]),
                               sell_value=float(g["sell_value"].iloc[i]), reaksi=_reaksi(close, i, cfg.reaksi_hari)))
    out = dict(kunci=kunci, n_event=len(ev), zona={})
    if not ev:
        return out
    E = pd.DataFrame(ev).sort_values("tgl").reset_index(drop=True)

    def paket(sub, nama, inval):
        n_tgl, n_br = sub["tgl"].nunique(), sub["broker"].nunique()
        return dict(nama=nama, harga_rendah=float(sub["close"].min()), harga_tinggi=float(sub["close"].max()), tanggal=[str(d) for d in sorted(sub["tgl"].unique())[:6]],
                    broker=sorted(sub["broker"].unique()), gross_sell_lots=float(sub["sell_lots"].sum()), net_lots=float(sub["net_lots"].sum()),
                    sell_value=float(sub["sell_value"].sum()), reaksi_rata=float(sub["reaksi"].mean()) if sub["reaksi"].notna().any() else np.nan,
                    confidence=("High" if n_tgl >= 5 and n_br >= 2 else "Medium" if n_tgl >= 3 else "Low"), invalidasi=inval, n_tanggal=n_tgl)
    awal = E.groupby("broker").head(3)
    out["zona"]["early"] = paket(awal, "Early distribution watch",
                                 "gugur bila penjualan awal itu terbukti crossing/transfer atau harga menembus batas atas zona dengan net beli kelompok positif 3 sesi")
    lebar = max(float(E["close"].median()) * cfg.zona_lebar, 1e-9)
    E["bin"] = np.floor(E["close"] / lebar).astype(int)
    top = E.groupby("bin")["sell_value"].sum().idxmax()
    out["zona"]["major"] = paket(E[E["bin"] == top], "Major distribution watch",
                                 "gugur bila harga ditutup 3 sesi di atas batas atas zona dan net kelompok kembali positif (resistance diserap)")
    br = E[E["reaksi"] >= cfg.breakout_naik]
    out["zona"]["breakout"] = (paket(br, "Breakout continuation (harga naik meski ada penjualan)",
                                     "gugur bila harga kembali ditutup di bawah batas bawah zona dalam 3 sesi (breakout gagal)") if not br.empty else None)
    return out


# --------------------------------------------------------------------- peringkat
def peringkat(hasil: dict) -> dict:
    df = pd.DataFrame([{k: v[k] for k in ("broker", "nama", "gross_buy_lots", "gross_sell_lots", "gross_buy_ex_crossing", "gross_sell_ex_crossing", "net_flow_lots",
                                          "status", "confidence")} for v in hasil.values()])
    if df.empty:
        return {}
    ep_ok = [k for k, v in hasil.items() if v.get("episode") and v["status"] in (ACC, MIXED)]
    geser = [k for k, v in hasil.items() if v.get("episode") and v["status"] in (DIST, MIXED) and v["episode"].get("turun_pct", 0) > 0.1]
    return dict(top_net_buyer=df.nlargest(8, "net_flow_lots"), top_net_seller=df.nsmallest(8, "net_flow_lots"),
                gross_buyer=df.nlargest(8, "gross_buy_ex_crossing"), gross_seller=df.nlargest(8, "gross_sell_ex_crossing"),
                persisten=df[df["broker"].isin(ep_ok)].sort_values("net_flow_lots", ascending=False), bergeser=df[df["broker"].isin(geser)])


# ------------------------------------------------------------------------ alert
def alert(L: pd.DataFrame, hasil: dict, zona: dict, val: dict, cfg: Cfg, now: dt.datetime | None = None) -> list[dict]:
    ts = (now or dt.datetime.now(dt.timezone.utc)).astimezone(lg.WIB).strftime("%Y-%m-%d %H:%M WIB")
    out = []
    mk = L.drop_duplicates("trade_date").sort_values("trade_date")
    close = mk["close_price"].to_numpy()
    tgl = mk["trade_date"].tolist()
    last = close[-1]
    window = f"{tgl[0]}..{tgl[-1]}"
    for k, v in hasil.items():
        ep = v.get("episode")
        if ep and v["status"] in (DIST, MIXED) and v.get("run_net_jual", 0) >= cfg.dist_beruntun:
            out.append(dict(tipe="Broker akumulator net jual beruntun", level="WASPADA", broker=k, ts=ts, jendela=f"{v['run_net_jual']} sesi terakhir",
                            bukti=f"{k}: {v['run_net_jual']} sesi net jual berturut-turut setelah episode {ep['mulai']}..{ep['puncak']}", confidence=v["confidence"]))
        if ep and ep["turun_pct"] >= cfg.turun_dari_puncak and ep["hari"] >= cfg.episode_min_hari:
            out.append(dict(tipe="Net kumulatif turun material dari puncak", level="WASPADA", broker=k, ts=ts, jendela=f"puncak {ep['puncak']}",
                            bukti=f"{k}: net kumulatif turun {ep['turun_pct']:.0%} dari kenaikan episode ({ep['turun_dari_puncak']:,.0f} lot)", confidence=v["confidence"]))
    for nm, z in (zona.get("zona") or {}).items():
        if z and z["harga_rendah"] * (1 - cfg.dekat_zona) <= last <= z["harga_tinggi"] * (1 + cfg.dekat_zona):
            out.append(dict(tipe=f"Harga mendekati zona {z['nama']}", level="INFO", broker=",".join(z["broker"]), ts=ts, jendela=window,
                            bukti=f"harga {last:,.0f} dalam ±{cfg.dekat_zona:.0%} dari {z['harga_rendah']:,.0f}-{z['harga_tinggi']:,.0f} ({z['n_tanggal']} tanggal penjualan teramati)",
                            confidence=z["confidence"]))
    akum = [k for k, v in hasil.items() if v["status"] == ACC]
    if akum and len(mk) > 21:
        net5 = float(L[L["broker_code"].isin(akum) & (L["trade_date"].isin(tgl[-5:]))]["net_lots"].sum())
        if last > float(np.max(close[-21:-1])) and net5 > 0:
            out.append(dict(tipe="Harga tembus high 20h saat akumulasi berlanjut", level="INFO", broker=",".join(akum), ts=ts, jendela="20 sesi / net 5 sesi",
                            bukti=f"close {last:,.0f} > high 20h {float(np.max(close[-21:-1])):,.0f}; net {','.join(akum)} 5 sesi {net5:+,.0f} lot", confidence="Medium"))
    key = [k for k, v in hasil.items() if v["status"] in (ACC, MIXED, DIST)]
    if key and len(mk) > 11:
        netg = float(L[L["broker_code"].isin(key) & (L["trade_date"].isin(tgl[-10:]))]["net_lots"].sum())
        r10 = last / close[-11] - 1
        if (r10 >= 0.03 and netg < 0) or (r10 <= -0.03 and netg > 0):
            out.append(dict(tipe="Divergensi flow vs harga (10 sesi)", level="WASPADA", broker=",".join(key), ts=ts, jendela="10 sesi",
                            bukti=f"harga {r10:+.1%} sedangkan net broker kunci {netg:+,.0f} lot", confidence="Low"))
    vol = mk["volume_lots"].to_numpy()
    if len(vol) > 25 and float(np.mean(vol[-5:])) < 0.5 * float(np.median(vol[-25:-5])):
        out.append(dict(tipe="Likuiditas memburuk", level="WASPADA", broker="-", ts=ts, jendela="5 vs 20 sesi",
                        bukti=f"rata-rata volume 5 sesi {float(np.mean(vol[-5:])):,.0f} lot < 50% median 20 sesi sebelumnya {float(np.median(vol[-25:-5])):,.0f} lot: risiko eksekusi naik", confidence="Medium"))
    for c in val["cek"]:
        if c["status"] != "OK" and c["id"] in ("tanggal_hilang", "data_final", "api_error", "aksi_korporasi", "satuan_konsisten", "duplikat", "kesegaran"):
            out.append(dict(tipe="Integritas data", level="WASPADA" if c["status"] == "PERINGATAN" else "KRITIS", broker="-", ts=ts, jendela=window,
                            bukti=f"{c['id']}: {c['detail']}", confidence="High"))
    return out


# --------------------------------------------------------------------- orkestrasi
def jalankan(b: pd.DataFrame, m: pd.DataFrame, cfg: Cfg | None = None, log: pd.DataFrame | None = None, now: dt.datetime | None = None) -> dict:
    cfg = cfg or Cfg()
    m, m_lain = lg.pisah_sumber(m)                      # sumber harga kedua hanya untuk verifikasi silang (source_conflict)
    val = lg.validasi(b, m, log, now)
    L = bangun(b, m, cfg)
    if L.empty:
        return dict(ok=False, validasi=val, alasan="ledger kosong", cfg=cfg)
    _gabung_cek(val, lq.cek_ledger(L))
    q = cfg.kualitas
    vol_total = float(L.drop_duplicates("trade_date")["volume_lots"].sum())
    hasil = {k: analisa_broker(g, cfg, vol_total, val["status"] == "GAGAL") for k, g in L.groupby("broker_code")}
    zona = zona_distribusi(L, hasil, cfg)
    pr = peringkat(hasil)
    al = alert(L, hasil, zona, val, cfg, now)
    # status per tanggal: HANYA data s.d. tanggal itu (tidak ada status akhir periode yang disalin ke tanggal historis)
    F = lq.fitur_rolling(L, sorted(set(m["trade_date"])), tuple(sorted(set(lq.JENDELA_FITUR) | {q.rolling_hari})))
    SH, SR = lq.status_harian(L, q), lq.status_rolling(L, F, q)
    L = L.merge(SH[["trade_date", "broker_code", "status"]].rename(columns={"status": "daily_flow_status"}), on=["trade_date", "broker_code"], how="left")
    L = L.merge(SR[["trade_date", "broker_code", "status"]].rename(columns={"status": "rolling_flow_status"}), on=["trade_date", "broker_code"], how="left")
    L["rolling_lookback_days"] = q.rolling_hari
    anom = lq.anomali(L, q)
    cross = lq.kontribusi_crossing(L)
    ekstrem = lq.dampak_ekstrem(L, q)
    verif = lg.periksa_sumber(m, m_lain)
    pe = lq.period_end_summary(L, hasil, anom, ekstrem, cross, cfg.opening_inventory)
    return dict(ok=True, ledger=L, hasil=hasil, zona=zona, peringkat=pr, alert=al, validasi=val, cfg=cfg, vol_total_lots=vol_total,
                tgl_awal=L["trade_date"].min(), tgl_akhir=L["trade_date"].max(), fitur=F, status_detail=pd.concat([SH, SR], ignore_index=True),
                anomali=anom, crossing=cross, ekstrem=ekstrem, sumber=verif, period_end=pe, pasar=lq.market_daily(L, m, verif),
                konsentrasi=lq.konsentrasi({n: F[n] for n in lq.JENDELA_FITUR if n in F}))


def _gabung_cek(val: dict, cek: list[dict]) -> None:
    val["cek"] += cek
    g = [c for c in val["cek"] if c["status"] == "GAGAL"]
    p = [c for c in val["cek"] if c["status"] == "PERINGATAN"]
    val.update(status="GAGAL" if g else "PERINGATAN" if p else "OK", n_gagal=len(g), n_peringatan=len(p))


def tabel_posisi(res: dict) -> pd.DataFrame:
    pe = res["period_end"].set_index("broker_code") if res.get("period_end") is not None and not res["period_end"].empty else None
    rows = []
    for k, v in res["hasil"].items():
        ep, ps = v.get("episode"), v["posisi"]
        e = pe.loc[k] if pe is not None and k in pe.index else {}
        rows.append({"Broker": k, "Nama": v["nama"], "Start Date (episode)": ep["mulai"] if ep else None, "Gross Buy Lots": v["gross_buy_lots"],
                     "Gross Sell Lots": v["gross_sell_lots"], "Cumulative net flow proxy": v["net_flow_lots"], "Avg Buy (gross)": v["avg_buy_gross"],
                     "Avg Buy (ex-crossing)": v["avg_buy_ex_crossing"], "Net-flow cost proxy": v["net_flow_cost_proxy"],
                     "Status estimasi modal": e.get("avg_cost_status", lq.NA), "Alasan modal": e.get("avg_cost_reason", ""),
                     "Status posisi": e.get("estimated_position_status", lq.NA), "Alasan posisi": e.get("estimated_position_reason", ""),
                     "Est. Remaining Low": ps["rendah"], "Est. Remaining High": ps["tinggi"], "Est. Remaining Base": ps["dasar"],
                     "Crossing % gross buy": v["crossing_pct_gross_buy"], "Anomali": e.get("anomaly_flags", ""),
                     "Status (kandidat)": lq.label_kandidat(v["status"]), "Confidence": e.get("confidence", v["confidence"]), "Label": ps["label"]})
    return pd.DataFrame(rows).sort_values("Cumulative net flow proxy", ascending=False).reset_index(drop=True)


def csv_ledger(res: dict) -> bytes:
    """broker_flow_daily seluruh broker (tanpa volume/harga pasar; join ke market_daily per trade_date)."""
    return lq.broker_flow_daily(res["ledger"]).to_csv(index=False).encode("utf-8")


def _relevan(res: dict) -> set:
    """Broker relevan untuk ekspor AI: 5 net buyer + 5 net seller akhir periode + broker berstatus/beranomali (seleksi saja; nilai per tanggal tidak berubah)."""
    H = res["hasil"]
    urut = sorted(H, key=lambda k: H[k]["net_flow_lots"])
    return set(urut[:5]) | set(urut[-5:]) | {k for k, v in H.items() if v["status"] in (ACC, DIST, MIXED) or v.get("episode")} | set(res.get("anomali") or {})


def csv_market(res: dict | None) -> bytes:
    if not res or not res.get("ok"):
        return b"info\nledger kosong: perbarui di tab Ledger Broker\n"
    return res["pasar"].to_csv(index=False).encode("utf-8")


def csv_period_end(res: dict | None) -> bytes:
    if not res or not res.get("ok"):
        return b"info\nledger kosong: perbarui di tab Ledger Broker\n"
    return res["period_end"].to_csv(index=False).encode("utf-8")


def csv_fitur(res: dict | None, n_asof: int = 5, broker: set | None = None) -> bytes:
    """broker_flow_features (bentuk panjang) untuk n_asof sesi terakhir dan broker relevan."""
    if not res or not res.get("ok"):
        return b"info\nledger kosong: perbarui di tab Ledger Broker\n"
    tgl = sorted(res["ledger"]["trade_date"].unique())[-n_asof:]
    F = {n: f for n, f in res["fitur"].items() if n in lq.JENDELA_FITUR}
    return lq.fitur_long(F, sorted(broker or _relevan(res)), tgl).to_csv(index=False).encode("utf-8")


def ekspor_ai(res: dict | None) -> dict[str, bytes]:
    """Paket CSV untuk AI: tabel terpisah (tidak ada volume pasar yang diulang per broker). Kunci = akhiran nama file."""
    return {"broker_flow_daily.csv": csv_ai(res), "market_daily.csv": csv_market(res), "period_end_summary.csv": csv_period_end(res),
            "broker_flow_features.csv": csv_fitur(res, n_asof=1)}


# ----------------------------------------------------------------------- jawaban
def _fmt_px(x):
    return "N/A" if x is None or pd.isna(x) else f"{x:,.1f}"


def _fmt_lot(x):
    return "N/A" if x is None or pd.isna(x) else f"{x:,.0f}"


def jawaban(res: dict, kode: str) -> dict:
    """Sepuluh jawaban dalam Bahasa Indonesia + ringkasan. Setiap angka dapat ditelusuri ke ledger tersimpan."""
    if not res.get("ok"):
        return dict(jawab=[], ringkasan=[("Status saat ini", "INSUFFICIENT_DATA: ledger kosong")])
    H, Z, val = res["hasil"], res["zona"], res["validasi"]
    ranking = sorted(((k, v) for k, v in H.items() if v["status"] in (ACC, MIXED, DIST) and v.get("episode")), key=lambda kv: kv[1]["episode"]["frac_net_beli"] * max(kv[1]["episode"]["gain_lots"], 0), reverse=True)
    kuat = ranking[0] if ranking else None
    qs = []
    if kuat:
        k, v = kuat
        ep = v["episode"]
        qs.append(("Broker mana yang menunjukkan akumulasi paling konsisten?",
                   f"[OBS] {k} ({v['nama']}): {ep['frac_net_beli']:.0%} sesi aktif dalam episode net beli, cumulative_net_flow_proxy {_fmt_lot(v['net_flow_lots'])} lot, status {lq.label_kandidat(v['status'])} (confidence {v['confidence']}). "
                   f"[INF] Ini broker/perantara, bukan identitas pemilik akhir."))
        qs.append(("Sejak kapan episode akumulasi diperkirakan dimulai?",
                   f"[EST] {ep['mulai']} (lembah net kumulatif terakhir sebelum puncak {ep['puncak']}); {ep['hari']} sesi bursa; "
                   f"{'masih naik' if ep['masih_naik'] else f'net kumulatif turun {ep['turun_pct']:.0%} dari puncak'}. Bukan hari net-buy pertama."))
        qs.append(("Berapa gross buy, gross sell, dan cumulative net flow?",
                   f"[OBS] jendela {res['tgl_awal']}..{res['tgl_akhir']}: gross buy {_fmt_lot(v['gross_buy_lots'])} lot (ex-crossing {_fmt_lot(v['gross_buy_ex_crossing'])}), "
                   f"gross sell {_fmt_lot(v['gross_sell_lots'])} lot (ex-crossing {_fmt_lot(v['gross_sell_ex_crossing'])}), net {_fmt_lot(v['net_flow_lots'])} lot. Crossing {v['crossing_pct_gross_buy']:.0%} dari gross beli."))
        qs.append(("Berapa estimasi average buy price dan bagaimana cara menghitungnya?",
                   f"[EST] gross {v['avg_buy_gross']:,.1f}; ex-crossing {v['avg_buy_ex_crossing']:,.1f}; dalam episode {ep.get('avg_buy_ex_crossing', float('nan')):,.1f}; "
                   f"net-flow cost proxy {_fmt_px(v['net_flow_cost_proxy'])}. "
                   "Cara: total nilai beli / total lembar beli (lembar = lot x 100); nilai dan lembar dari broker summary harian tersimpan. Proksi, bukan harga pokok."))
        ps = v["posisi"]
        qs.append(("Apakah posisi tersisa dapat diestimasi, atau hanya net-flow proxy?",
                   f"[EST] HANYA NET_FLOW_PROXY: persediaan awal tidak diketahui. Skenario rendah {_fmt_lot(ps['rendah'])} lot (persediaan awal 0), tinggi {_fmt_lot(ps['tinggi'])} lot "
                   f"(persediaan awal minimal agar posisi tak negatif), dasar {_fmt_lot(ps['dasar'])}. Jangan dibaca sebagai kepemilikan aktual."))
        st = {ACC: "kandidat akumulasi (proksi aliran)", DIST: "kandidat distribusi (proksi aliran)", MIXED: "pola campuran"}[v["status"]]
        qs.append(("Apakah broker masih mengakumulasi, mulai distribusi, atau menunjukkan pola campuran?",
                   f"[INF] {k}: {st} ({lq.label_kandidat(v['status'])}). Sinyal akumulasi terpenuhi: {sum(v['sinyal_acc'].values())}/6; sinyal distribusi: {sum(v['sinyal_dist'].values())}/6."))
        sup = [s for s, ok in v["sinyal_acc"].items() if ok] + []
        bantah = [s for s, ok in v["sinyal_acc"].items() if not ok] + [s for s, ok in v["sinyal_dist"].items() if ok]
        qs.append(("Apa bukti yang mendukung dan membantah hipotesis akumulasi?",
                   "[OBS] Mendukung: " + ("; ".join(sup) or "tidak ada") + ". Membantah/belum terpenuhi: " + ("; ".join(bantah) or "tidak ada") + "."))
    else:
        for q in ("Broker mana yang menunjukkan akumulasi paling konsisten?", "Sejak kapan episode akumulasi diperkirakan dimulai?", "Berapa gross buy, gross sell, dan cumulative net flow?",
                  "Berapa estimasi average buy price dan bagaimana cara menghitungnya?", "Apakah posisi tersisa dapat diestimasi, atau hanya net-flow proxy?",
                  "Apakah broker masih mengakumulasi, mulai distribusi, atau menunjukkan pola campuran?", "Apa bukti yang mendukung dan membantah hipotesis akumulasi?"):
            qs.append((q, "N/A: tidak ada broker dengan episode akumulasi berkelanjutan pada jendela data ini (net positif saja tidak cukup). Lihat peringkat net buyer dan catatan di bawah."))
    zr = []
    for key in ("early", "major", "breakout"):
        z = Z.get("zona", {}).get(key)
        if z:
            zr.append(f"{z['nama']}: {z['harga_rendah']:,.0f}-{z['harga_tinggi']:,.0f} ({z['n_tanggal']} tanggal, broker {','.join(z['broker'])}, confidence {z['confidence']})")
    qs.insert(7, ("Pada rentang harga berapa distribusi historis pernah terjadi?",
                  ("[OBS/EST] " + "; ".join(zr) + ". Ini area penjualan teramati, BUKAN target harga broker.") if zr else "N/A: belum ada penjualan teramati oleh broker kunci setelah episode akumulasi."))
    ubah = ["net kelompok 5 sesi berbalik arah", "harga menembus zona distribusi dengan net beli positif (resistance diserap)", "data baru memperlihatkan crossing/transfer menjelaskan lot terakumulasi"]
    qs.append(("Apa kondisi yang mengubah kesimpulan?", "[DEC] " + "; ".join(ubah) + "."))
    kurang = []
    if val["status"] != "OK":
        kurang += [f"{c['id']} ({c['status']})" for c in val["cek"] if c["status"] != "OK"][:4]
    kurang += ["persediaan awal broker (untuk posisi tersisa aktual)", "identitas pemilik akhir (tidak tersedia di broker summary)", "histori lebih panjang untuk episode sebelumnya"]
    qs.append(("Data apa yang masih dibutuhkan untuk meningkatkan confidence?", "; ".join(kurang) + "."))
    # ringkasan
    if kuat:
        k, v = kuat
        ep = v["episode"]
        rg = [("Observed accumulation", f"{k}: cumulative_net_flow_proxy {_fmt_lot(v['net_flow_lots'])} lot sejak {ep['mulai']}; {sum(v['sinyal_acc'].values())}/6 sinyal terpenuhi"),
              ("Estimated cost basis", f"{ep.get('avg_buy_ex_crossing', float('nan')):,.1f} (rata-rata beli ex-crossing dalam episode; proksi, bukan harga pokok)"),
              ("Estimated remaining position", f"N/A sebagai posisi aktual; NET_FLOW_PROXY {_fmt_lot(v['posisi']['rendah'])}-{_fmt_lot(v['posisi']['tinggi'])} lot")]
    else:
        rg = [("Observed accumulation", "tidak ada broker yang memenuhi syarat ACCUMULATION_CANDIDATE"), ("Estimated cost basis", "N/A"), ("Estimated remaining position", "N/A")]
    rg.append(("Distribution watch zones", "; ".join(zr) if zr else "N/A (belum ada penjualan teramati)"))
    stat_all = "INSUFFICIENT_DATA" if val["status"] == "GAGAL" else (lq.label_kandidat(kuat[1]["status"]) if kuat else NEUTRAL)
    conf = kuat[1]["confidence"] if kuat else "Low"
    rg.append(("Current status", stat_all))
    rg.append(("Confidence", f"{conf} (hari data {val['cakupan']['hari'] if val.get('cakupan') else 0}; validasi {val['status']})"))
    impl = "avoid" if stat_all == lq.DIST_C else ("wait for confirmation" if stat_all in (lq.NEUTRAL, lq.INSUF) or conf != "High" else "monitor")
    rg.append(("Trading implication", f"{impl}. Invalidasi: " + "; ".join(ubah[:2]) + "."))
    return dict(jawab=qs, ringkasan=rg)


# ------------------------------------------------------------- ekspor untuk AI
def _pct(x):
    return "N/A" if x is None or pd.isna(x) else f"{x:+.1%}"


def baris_ai(res: dict | None, kode: str) -> list[str]:
    """Bagian 'LEDGER BROKER' untuk TXT/MD dengan format OBSERVASI / INFERENSI / BUKTI YANG MELAWAN / KUALITAS DATA / KESIMPULAN / INVALIDASI / DATA TAMBAHAN.
    Tabel memakai '|' agar otomatis menjadi tabel di Markdown. Tidak ada klaim posisi aktual atau identitas bandar."""
    L = ["## LEDGER BROKER (data tersimpan di database; tanpa panggilan API saat ekspor)"]
    if not res or not res.get("ok"):
        L.append("N/A: belum ada data ledger tersimpan untuk saham ini (buka tab Ledger Broker lalu Perbarui/backfill). Akumulasi/distribusi broker harian TIDAK dapat dinilai.")
        return L
    val, H, Z = res["validasi"], res["hasil"], res["zona"]
    pe, Mk, cross, ver = res["period_end"], res["pasar"], res["crossing"], res["sumber"]
    q = res["cfg"].kualitas
    L.append("PERINGATAN METODE: broker=perantara banyak nasabah; net kumulatif = cumulative_net_flow_proxy (NET_FLOW_PROXY), bukan posisi/harga pokok aktual; "
             "bukan identitas 'bandar'; status = kandidat (ACCUMULATION_CANDIDATE/DISTRIBUTION_CANDIDATE/NEUTRAL/INSUFFICIENT_DATA), bukan fakta niat pelaku.")
    top = pe.reindex(pe["cumulative_net_flow_proxy"].abs().sort_values(ascending=False).index).head(8)
    tgl = res["tgl_akhir"]
    # ------------------------------------------------------------------ OBSERVASI
    L.append("### OBSERVASI")
    L.append(f"[OBS] jendela {res['tgl_awal']}..{tgl}; {val['cakupan']['hari']} sesi; {val['cakupan']['n_broker']} broker; satuan {lg.UNIT_VOLUME}, harga {lg.UNIT_HARGA}; "
             f"sumber {lg.SOURCE_VERSION} (pasar REGULAR) + {', '.join(ver['sumber'])} (harga/volume); formula {lq.FORMULA_VERSION}")
    vol20 = Mk["market_volume_lots"].tail(20)
    L.append(f"[OBS] volume pasar dicatat SEKALI per tanggal (market_daily): rata-rata 20 sesi {vol20.mean():,.0f} lot; total broker reguler / volume pasar median "
             f"{Mk['broker_to_market_volume_ratio'].median():.2f}x. Net {val['cakupan']['n_broker']} broker terlihat BUKAN net flow seluruh pasar.")
    L.append("Broker | Net lot periode | Net ex-crossing | Net crossing | Net tanpa hari terbesar | Gross beli | Gross jual")
    for r in top.itertuples():
        L.append(f"{r.broker_code} | {r.cumulative_net_flow_proxy:,.0f} | {r.net_lots_non_crossing:,.0f} | {r.net_lots_crossing_flagged:,.0f} | {r.net_lots_ex_top_day:,.0f} | "
                 f"{r.gross_buy_lots:,.0f} | {r.gross_sell_lots:,.0f}")
    F20 = res["fitur"].get(20)
    if F20 is not None and tgl in F20["net_lots"].index:
        L.append(f"Fitur rolling per {tgl} (hanya data s.d. tanggal itu; persen terhadap volume pasar jendela yang sama):")
        L.append("Broker | Net 1D | Net 3D | Net 5D | Net 10D | Net 20D | Net/volume 20D | Hari net beli 20D | Run beli terpanjang 20D | Ganti arah 20D | Mutu")
        for b in top["broker_code"]:
            nl = [res["fitur"][n]["net_lots"].loc[tgl].get(b, np.nan) for n in lq.JENDELA_FITUR]
            L.append(f"{b} | " + " | ".join(_fmt_lot(x) for x in nl) + f" | {_pct(F20['net_flow_pct_market_volume'].loc[tgl].get(b))} | "
                     f"{('N/A' if pd.isna(F20['positive_flow_days_pct'].loc[tgl].get(b)) else format(F20['positive_flow_days_pct'].loc[tgl].get(b), '.0%'))} | "
                     f"{_fmt_lot(F20['longest_positive_run'].loc[tgl].get(b))} | {_fmt_lot(F20['direction_changes'].loc[tgl].get(b))} | {F20['quality_flag'].loc[tgl].get(b)}")
    K = res.get("konsentrasi")
    if K is not None and not K.empty:
        k20 = K[(K["as_of_date"] == tgl) & (K["lookback_days"] == 20)]
        if not k20.empty and k20.iloc[0]["denominator_net_buy_lots"] <= 0:
            L.append("[OBS] konsentrasi 20D: tidak ada broker dengan net beli 20D (denominator net beli = 0)")
        elif not k20.empty:
            r = k20.iloc[0]
            L.append(f"[OBS] konsentrasi 20D: top-1 {r['top1_broker']} = {r['top1_share_net_buy']:.0%} dan top-3 = {r['top3_share_net_buy']:.0%} dari total net beli broker yang net beli "
                     f"({r['denominator_net_buy_lots']:,.0f} lot); top-3 gross beli = {r['top3_share_gross_buy']:.0%} dari total gross beli ({r['denominator_gross_buy_lots']:,.0f} lot)")
    tot = cross[cross["broker_code"] == "TOTAL"]
    if not tot.empty:
        t = tot.iloc[0]
        L.append(f"[OBS] crossing (heuristik beli~jual besar, pasar reguler): {t['crossing_days']:.0f} broker-hari, {('N/A' if pd.isna(t['crossing_pct_gross_buy']) else format(t['crossing_pct_gross_buy'], '.0%'))} "
                 f"dari gross beli; net crossing {t['net_lots_crossing_flagged']:+,.0f} lot vs net non-crossing {t['net_lots_non_crossing']:+,.0f} lot (dipisah, tidak digabung ke sinyal)")
    zs = [z for z in (Z.get("zona") or {}).values() if z]
    if zs:
        L.append("Zona distribusi teramati (area penjualan teramati, bukan target harga broker):")
        L.append("Zona | Harga | Tanggal | Broker | Gross jual lot | Reaksi | Conf")
        for z in zs:
            L.append(f"{z['nama'].split(' (')[0]} | {z['harga_rendah']:,.0f}-{z['harga_tinggi']:,.0f} | {z['n_tanggal']} tgl ({z['tanggal'][0]}..) | {','.join(z['broker'])} | "
                     f"{z['gross_sell_lots']:,.0f} | {('N/A' if pd.isna(z['reaksi_rata']) else format(z['reaksi_rata'], '+.1%'))} | {z['confidence']}")
    else:
        L.append("Zona distribusi: N/A (belum ada penjualan teramati broker kunci).")
    # ------------------------------------------------------------------ INFERENSI
    L.append("### INFERENSI")
    L.append("[INF] status akhir periode per broker (period_end_summary; TIDAK berlaku surut ke tanggal historis):")
    L.append("Broker | Status akhir periode | Confidence | Mutu data | Bukti mendukung")
    for r in top.itertuples():
        L.append(f"{r.broker_code} | {r.status} | {r.confidence} | {r.data_quality} | {(r.evidence_for or '-')[:160]}")
    Ld = res["ledger"]
    akhir = Ld[Ld["trade_date"] == tgl].set_index("broker_code")
    L.append(f"[INF] rolling_flow_status {q.rolling_hari}D per {tgl}: " + "; ".join(f"{b}={akhir['rolling_flow_status'].get(b, 'N/A')}" for b in top["broker_code"]))
    sg = res.get("sinyal")
    if sg is not None:
        import ledger_score as ls_
        L += ls_.baris(sg)
    # ------------------------------------------------------------------ BUKTI YANG MELAWAN
    L.append("### BUKTI YANG MELAWAN")
    for r in top.itertuples():
        if r.status in (lq.ACC_C, lq.DIST_C):
            L.append(f"- {r.broker_code} ({r.status}): {(r.evidence_against or '-')[:300]}")
    for a in [a for a in res["alert"] if a["level"] != "INFO"][:6]:
        L.append(f"- [{a['level']}] {a['tipe']}: {a['bukti']} (jendela {a['jendela']})")
    if not any(r.status in (lq.ACC_C, lq.DIST_C) for r in top.itertuples()):
        L.append("- tidak ada broker berstatus kandidat; net positif/negatif saja tidak cukup untuk hipotesis")
    # ------------------------------------------------------------------ KUALITAS DATA
    L.append("### KUALITAS DATA")
    L.append(f"validasi {val['status']}" + (": " + "; ".join(f"{c['id']} ({c['status']}): {c['detail'][:140]}" for c in val["cek"] if c["status"] != "OK") if val["status"] != "OK" else ""))
    an = res.get("anomali") or {}
    if an:
        L.append("Anomali broker (perlu pemeriksaan manual; estimasi modal broker ini = UNRELIABLE):")
        for b, fl in list(an.items())[:10]:
            L.append(f"- {b}: " + "; ".join(f"{a['flag']} ({a['detail']})" for a in fl))
    else:
        L.append("Anomali broker: tidak ada yang melewati ambang.")
    L.append("Estimasi posisi: NOT_AVAILABLE untuk semua broker (persediaan awal/inventory tidak tersedia; hanya cumulative_net_flow_proxy)."
             if res["cfg"].opening_inventory is None else "Estimasi posisi: memakai persediaan awal yang dinyatakan pengguna (tetap proksi).")
    L.append("Estimasi modal: " + ", ".join(f"{r.broker_code}={r.avg_cost_status}" for r in top.itertuples()) + " (PROXY = rata-rata beli ex-crossing episode; bukan harga pokok)")
    L.append(f"Verifikasi sumber harga: {ver['status']} ({ver['catatan']})")
    nk = Mk["volume_unit_check"].value_counts().to_dict()
    L.append(f"Satuan volume: market_volume_shares = market_volume_lots x 100; cek per tanggal {nk}. Segmen: " +
             ", ".join(f"{k}={v}" for k, v in Ld["market_segment"].value_counts().items()) + "; data_completeness: " +
             ", ".join(f"{k}={v}" for k, v in Ld["data_completeness"].value_counts().items()) + " (NOT_AVAILABLE = tidak ada baris sumber, bukan 0)")
    # ------------------------------------------------------------------ KESIMPULAN
    j = jawaban(res, kode)
    rg = dict(j["ringkasan"])
    L.append("### KESIMPULAN")
    L.append(f"Status sementara: {rg.get('Current status')} · confidence {rg.get('Confidence')} · implikasi {rg.get('Trading implication', '').split('.')[0]}. "
             "Ini hipotesis berbasis proksi, bukan kepastian; tidak ada klaim posisi aktual atau identitas pemilik dana.")
    # ------------------------------------------------------------------ INVALIDASI
    L.append("### INVALIDASI")
    for r in top.itertuples():
        if r.status in (lq.ACC_C, lq.DIST_C):
            L.append(f"- {r.broker_code} ({r.status}): {r.invalidation_condition}")
    for z in zs:
        L.append(f"- {z['nama'].split(' (')[0]}: {z['invalidasi']}")
    L.append("- umum: net kelompok 5 sesi berbalik arah; data baru memperlihatkan crossing/transfer menjelaskan lot terakumulasi")
    # ------------------------------------------------------------------ DATA TAMBAHAN
    L.append("### DATA TAMBAHAN")
    tambah = ["inventory/kepemilikan aktual (KSEI) untuk memvalidasi cumulative_net_flow_proxy",
              "rekonsiliasi broker beranomali (" + ", ".join(an) + ")" if an else "",
              "sumber harga/volume kedua untuk verifikasi silang" if ver["status"] == "BELUM_DIVERIFIKASI" else "",
              "data pasar negosiasi/tunai per broker (saat ini hanya pasar reguler)", "riwayat lebih panjang untuk episode sebelumnya"]
    L += [f"- {t}" for t in tambah if t]
    return L


def csv_ai(res: dict | None) -> bytes:
    """broker_flow_daily ringkas untuk AI (broker relevan). Tanpa close/volume pasar (lihat market_daily.csv; join via trade_date).
    Status per baris hanya memakai data s.d. tanggal itu; ringkasan akhir periode ada di period_end_summary.csv."""
    if not res or not res.get("ok"):
        return b"info\nledger kosong: perbarui di tab Ledger Broker\n"
    D = lq.broker_flow_daily(res["ledger"])
    D = D[D["broker_code"].isin(_relevan(res))].drop(columns=["ticker", "broker_name", "buy_value_idr", "sell_value_idr", "unit_price"])
    for c in ("buy_lots", "sell_lots", "net_lots", "cumulative_net_flow_proxy"):
        D[c] = D[c].round(0)
    for c in ("avg_buy_price", "avg_sell_price"):
        D[c] = D[c].round(1)
    D["net_value_idr"] = D["net_value_idr"].round(0)
    return D.to_csv(index=False).encode("utf-8")
