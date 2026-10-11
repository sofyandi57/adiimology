"""Analisis buku besar broker: episode akumulasi, biaya rata-rata (proksi), posisi tersisa (skenario), status aliran, zona distribusi, peringatan.

Penanda: [OBS] teramati dari data tersimpan, [EST] estimasi model, [INF] inferensi, [DEC] keputusan.
PRINSIP: broker = perantara banyak nasabah; net kumulatif dan biaya rata-rata adalah PROKSI ALIRAN (NET_FLOW_PROXY), bukan inventaris
atau harga pokok sebenarnya. Net positif saja tidak pernah cukup untuk label ACCUMULATION. Data tidak lengkap -> N/A, bukan tebakan.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

import ledger as lg

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


# --------------------------------------------------------------- ledger harian
def bangun(b: pd.DataFrame, m: pd.DataFrame, cfg: Cfg | None = None) -> pd.DataFrame:
    """Ledger harian per broker (satuan lot, Rupiah) + net kumulatif per broker pada jendela data yang tersedia."""
    cfg = cfg or Cfg()
    if b.empty or m.empty:
        return pd.DataFrame()
    bx = lg.tandai_crossing(b, m)
    mm = m.sort_values("trade_date").reset_index(drop=True)
    mm = mm[mm["trade_date"].isin(set(b["trade_date"]))].reset_index(drop=True)      # hanya sesi yang BENAR-BENAR termuat: tidak ada nol buatan untuk sesi hilang
    if mm.empty:
        return pd.DataFrame()
    tgl = mm["trade_date"].tolist()
    base = pd.DataFrame({"trade_date": tgl, "close_price": mm["close"].to_numpy(), "volume_lots": mm["volume_shares"].to_numpy() / LOT})
    kode = bx["ticker"].iloc[0]
    rows = []
    for br, g in bx.groupby("broker_code"):
        g = g.set_index("trade_date")
        d = base.copy()
        for c in ("buy_shares", "sell_shares", "buy_value", "sell_value"):
            d[c] = d["trade_date"].map(g[c]).fillna(0.0)
        d["crossing"] = d["trade_date"].map(g["crossing"]).fillna(False).astype(bool)
        d["buy_avg_raw"] = d["trade_date"].map(g["buy_avg_raw"])
        d["sell_avg_raw"] = d["trade_date"].map(g["sell_avg_raw"])
        d["broker_name"] = g["broker_name"].iloc[0]
        d["broker_code"] = br
        d["ticker"] = kode
        d["is_final"] = d["trade_date"].map(g["is_final"]).fillna(True).astype(bool)
        d["missing_data_flag"] = ~d["trade_date"].isin(g.index)          # True = broker tidak punya baris sumber di sesi itu (tidak aktif; BUKAN tanggal hilang)
        d["source_timestamp"] = d["trade_date"].map(g["retrieved_at"]) if "retrieved_at" in g else pd.NaT
        rows.append(d)
    L = pd.concat(rows, ignore_index=True)
    L["buy_lots"], L["sell_lots"] = L["buy_shares"] / LOT, L["sell_shares"] / LOT
    L["net_lots"] = L["buy_lots"] - L["sell_lots"]
    L["net_value"] = L["buy_value"] - L["sell_value"]
    L["avg_buy_price"] = np.where(L["buy_shares"] > 0, L["buy_value"] / L["buy_shares"].replace(0, np.nan), np.nan)
    L["avg_sell_price"] = np.where(L["sell_shares"] > 0, L["sell_value"] / L["sell_shares"].replace(0, np.nan), np.nan)
    L = L.sort_values(["broker_code", "trade_date"]).reset_index(drop=True)
    L["cumulative_net_lots"] = L.groupby("broker_code")["net_lots"].cumsum()
    return L


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
    val = lg.validasi(b, m, log, now)
    L = bangun(b, m, cfg)
    if L.empty:
        return dict(ok=False, validasi=val, alasan="ledger kosong", cfg=cfg)
    vol_total = float(L.drop_duplicates("trade_date")["volume_lots"].sum())
    hasil = {k: analisa_broker(g, cfg, vol_total, val["status"] == "GAGAL") for k, g in L.groupby("broker_code")}
    zona = zona_distribusi(L, hasil, cfg)
    pr = peringkat(hasil)
    al = alert(L, hasil, zona, val, cfg, now)
    # kolom status & posisi estimasi pada ledger (nilai akhir broker; bukan status harian)
    L["flow_status_akhir"] = L["broker_code"].map({k: v["status"] for k, v in hasil.items()})
    L["confidence_akhir"] = L["broker_code"].map({k: v["confidence"] for k, v in hasil.items()})
    L["estimated_position_lots_low"] = L["broker_code"].map({k: v["posisi"]["rendah"] for k, v in hasil.items()})
    L["estimated_avg_cost_proxy"] = L["broker_code"].map({k: (v["episode"]["avg_buy_ex_crossing"] if v.get("episode") else np.nan) for k, v in hasil.items()})
    return dict(ok=True, ledger=L, hasil=hasil, zona=zona, peringkat=pr, alert=al, validasi=val, cfg=cfg, vol_total_lots=vol_total,
                tgl_awal=L["trade_date"].min(), tgl_akhir=L["trade_date"].max())


def tabel_posisi(res: dict) -> pd.DataFrame:
    rows = []
    for k, v in res["hasil"].items():
        ep, ps = v.get("episode"), v["posisi"]
        rows.append({"Broker": k, "Nama": v["nama"], "Start Date (episode)": ep["mulai"] if ep else None, "Gross Buy Lots": v["gross_buy_lots"],
                     "Gross Sell Lots": v["gross_sell_lots"], "Net Flow Lots": v["net_flow_lots"], "Avg Buy (gross)": v["avg_buy_gross"],
                     "Avg Buy (ex-crossing)": v["avg_buy_ex_crossing"], "Net-flow cost proxy": v["net_flow_cost_proxy"],
                     "Est. Remaining Low": ps["rendah"], "Est. Remaining High": ps["tinggi"], "Est. Remaining Base": ps["dasar"],
                     "Crossing % gross buy": v["crossing_pct_gross_buy"], "Status": v["status"], "Confidence": v["confidence"], "Label": ps["label"]})
    return pd.DataFrame(rows).sort_values("Net Flow Lots", ascending=False).reset_index(drop=True)


def csv_ledger(res: dict) -> bytes:
    cols = ["trade_date", "ticker", "broker_code", "broker_name", "buy_lots", "sell_lots", "net_lots", "buy_value", "sell_value", "net_value", "avg_buy_price",
            "avg_sell_price", "close_price", "volume_lots", "crossing", "cumulative_net_lots", "estimated_position_lots_low", "estimated_avg_cost_proxy",
            "flow_status_akhir", "confidence_akhir", "is_final"]
    out = res["ledger"][cols + ["missing_data_flag", "source_timestamp"]].copy()
    out["data_status"] = np.where(out.pop("is_final"), "FINAL", "PARTIAL")
    out["duplicate_flag"] = False                                        # kunci unik di database: duplikat tidak mungkin tersimpan
    out["source_version"] = lg.SOURCE_VERSION
    out["calculation_version"] = __import__("ledger_score").CALC_VERSION
    return out.to_csv(index=False).encode("utf-8")


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
                   f"[OBS] {k} ({v['nama']}): {ep['frac_net_beli']:.0%} sesi aktif dalam episode net beli, net {_fmt_lot(v['net_flow_lots'])} lot, status {v['status']} (confidence {v['confidence']}). "
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
        st = {ACC: "masih mengakumulasi", DIST: "mulai/ sedang distribusi", MIXED: "pola campuran"}[v["status"]]
        qs.append(("Apakah broker masih mengakumulasi, mulai distribusi, atau menunjukkan pola campuran?",
                   f"[INF] {k}: {st} ({v['status']}). Sinyal akumulasi terpenuhi: {sum(v['sinyal_acc'].values())}/6; sinyal distribusi: {sum(v['sinyal_dist'].values())}/6."))
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
        rg = [("Observed accumulation", f"{k}: net {_fmt_lot(v['net_flow_lots'])} lot sejak {ep['mulai']}; {sum(v['sinyal_acc'].values())}/6 sinyal terpenuhi"),
              ("Estimated cost basis", f"{ep.get('avg_buy_ex_crossing', float('nan')):,.1f} (rata-rata beli ex-crossing dalam episode; proksi, bukan harga pokok)"),
              ("Estimated remaining position", f"N/A sebagai posisi aktual; NET_FLOW_PROXY {_fmt_lot(v['posisi']['rendah'])}-{_fmt_lot(v['posisi']['tinggi'])} lot")]
    else:
        rg = [("Observed accumulation", "tidak ada broker yang memenuhi syarat ACCUMULATION"), ("Estimated cost basis", "N/A"), ("Estimated remaining position", "N/A")]
    rg.append(("Distribution watch zones", "; ".join(zr) if zr else "N/A (belum ada penjualan teramati)"))
    stat_all = "INSUFFICIENT_DATA" if val["status"] == "GAGAL" else (kuat[1]["status"] if kuat else NEUTRAL)
    conf = kuat[1]["confidence"] if kuat else "Low"
    rg.append(("Current status", stat_all))
    rg.append(("Confidence", f"{conf} (hari data {val['cakupan']['hari'] if val.get('cakupan') else 0}; validasi {val['status']})"))
    impl = "avoid" if stat_all in (DIST,) else ("wait for confirmation" if stat_all in (MIXED, NEUTRAL, INSUF) or conf != "High" else "monitor")
    rg.append(("Trading implication", f"{impl}. Invalidasi: " + "; ".join(ubah[:2]) + "."))
    return dict(jawab=qs, ringkasan=rg)


# ------------------------------------------------------------- ekspor untuk AI
def baris_ai(res: dict | None, kode: str) -> list[str]:
    """Bagian 'LEDGER BROKER' ringkas untuk TXT/MD. Tabel memakai '|' agar otomatis menjadi tabel di Markdown."""
    L = ["## LEDGER BROKER (data tersimpan di database; tanpa panggilan API saat ekspor)"]
    if not res or not res.get("ok"):
        L.append("N/A: belum ada data ledger tersimpan untuk saham ini (buka tab Ledger Broker lalu Perbarui/backfill). Akumulasi/distribusi broker harian TIDAK dapat dinilai.")
        return L
    val, H, Z = res["validasi"], res["hasil"], res["zona"]
    L.append(f"[OBS] jendela {res['tgl_awal']}..{res['tgl_akhir']}; {val['cakupan']['hari']} hari; {val['cakupan']['n_broker']} broker; validasi {val['status']}"
             + (f" (masalah: {', '.join(c['id'] for c in val['cek'] if c['status'] != 'OK')})" if val["status"] != "OK" else ""))
    L.append("PERINGATAN METODE: broker=perantara banyak nasabah; biaya & posisi = NET_FLOW_PROXY (bukan posisi/harga pokok aktual); bukan identitas 'bandar'; net positif saja tidak cukup untuk ACCUMULATION.")
    sg = res.get("sinyal")
    if sg is not None:
        import ledger_score as ls_
        L += ls_.baris(sg)
    j = jawaban(res, kode)
    for k, v in j["ringkasan"]:
        L.append(f"{k}: {v}")
    L.append("### Peringkat broker (lot; ex-crossing)")
    L.append("Broker | Net lot | Gross beli | Gross jual | Status | Conf")
    for k, v in sorted(H.items(), key=lambda kv: -abs(kv[1]["net_flow_lots"]))[:8]:
        L.append(f"{k} | {v['net_flow_lots']:,.0f} | {v['gross_buy_ex_crossing']:,.0f} | {v['gross_sell_ex_crossing']:,.0f} | {v['status']} | {v['confidence']}")
    L.append("### Estimasi posisi (NET_FLOW_PROXY)")
    L.append("Broker | Mulai episode | Avg beli gross | Avg beli ex-cross | Proksi net-flow | Sisa rendah | Sisa tinggi | Crossing")
    for k, v in sorted(H.items(), key=lambda kv: -abs(kv[1]["net_flow_lots"]))[:8]:
        ep, ps = v.get("episode"), v["posisi"]
        L.append(f"{k} | {ep['mulai'] if ep else 'N/A'} | {_fmt_px(v['avg_buy_gross'])} | {_fmt_px(v['avg_buy_ex_crossing'])} | {_fmt_px(v['net_flow_cost_proxy'])} | "
                 f"{_fmt_lot(ps['rendah'])} | {_fmt_lot(ps['tinggi'])} | {v['crossing_pct_gross_buy']:.0%}")
    zs = [z for z in (Z.get("zona") or {}).values() if z]
    if zs:
        L.append("### Zona distribusi teramati (bukan target broker)")
        L.append("Zona | Harga | Tanggal | Broker | Gross jual lot | Reaksi | Conf | Invalidasi")
        for z in zs:
            L.append(f"{z['nama'].split(' (')[0]} | {z['harga_rendah']:,.0f}-{z['harga_tinggi']:,.0f} | {z['n_tanggal']} tgl ({z['tanggal'][0]}..) | {','.join(z['broker'])} | "
                     f"{z['gross_sell_lots']:,.0f} | {('N/A' if pd.isna(z['reaksi_rata']) else format(z['reaksi_rata'], '+.1%'))} | {z['confidence']} | {z['invalidasi']}")
    else:
        L.append("Zona distribusi: N/A (belum ada penjualan teramati broker kunci).")
    if res["alert"]:
        L.append("### Peringatan aktif")
        for a in res["alert"][:8]:
            L.append(f"- [{a['level']}] {a['tipe']}: {a['bukti']} (jendela {a['jendela']}; conf {a['confidence']}; {a['ts']})")
    L.append("### Tanya-jawab ledger")
    for q, a in j["jawab"]:
        L.append(f"- {q} {a}")
    return L


def csv_ai(res: dict | None) -> bytes:
    """CSV ringkas untuk AI: ledger harian hanya untuk broker relevan (episode/status aktif + 5 net buyer + 5 net seller), kolom inti + status."""
    if not res or not res.get("ok"):
        return b"info\nledger kosong: perbarui di tab Ledger Broker\n"
    H = res["hasil"]
    urut = sorted(H, key=lambda k: H[k]["net_flow_lots"])
    rel = set(urut[:5]) | set(urut[-5:]) | {k for k, v in H.items() if v["status"] in (ACC, DIST, MIXED) or v.get("episode")}
    L = res["ledger"]
    L = L[L["broker_code"].isin(rel)]
    cols = ["trade_date", "broker_code", "buy_lots", "sell_lots", "net_lots", "cumulative_net_lots", "avg_buy_price", "avg_sell_price", "close_price", "volume_lots",
            "crossing", "flow_status_akhir", "confidence_akhir", "estimated_position_lots_low", "estimated_avg_cost_proxy"]
    out = L[cols].copy()
    out["label"] = "NET_FLOW_PROXY"
    for c in ("buy_lots", "sell_lots", "net_lots", "cumulative_net_lots", "volume_lots", "estimated_position_lots_low"):
        out[c] = out[c].round(0)
    for c in ("avg_buy_price", "avg_sell_price", "estimated_avg_cost_proxy", "close_price"):
        out[c] = out[c].round(1)
    return out.to_csv(index=False).encode("utf-8")
