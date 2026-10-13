"""Aliran transaksi broker dalam TIGA LAPISAN terpisah: (1) observed flow, (2) price confirmation, (3) interpretation.

Bukan identifikasi 'bandar'. Satu kode broker = banyak nasabah; broker summary tidak menunjukkan pemilik manfaat, posisi yang masih dipegang,
crossing antar-nasabah, atau inventory. Label bersifat operasional (aturan sistem), bukan kepastian tentang pelaku:
  ACCUMULATION_CANDIDATE  aliran beli konsisten, belum dikonfirmasi harga
  ACCUMULATION_CONFIRMED  aliran beli konsisten DAN harga menunjukkan supply terserap / struktur membaik
  DISTRIBUTION_CANDIDATE  aliran jual meningkat atau pembelian melemah
  DISTRIBUTION_CONFIRMED  tekanan jual konsisten DAN harga/volume mengonfirmasi pelemahan
  NEUTRAL                 bukti berimbang
  INSUFFICIENT_DATA       data tidak cukup / kualitas bermasalah (bukan skor netral)

Kelompok ditetapkan SEBELUM membaca hasil dan diberi versi: 'ritel proksi' = daftar broker tetap; 'non-ritel' = semua broker lain.
Label ritel adalah PROKSI (satu broker melayani berbagai tipe nasabah). Evidence level (Low/Medium/High) = kelengkapan & konsistensi bukti,
BUKAN probabilitas profit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

VERSI = "flow-2026.10.1"
RITEL_VERSI = "ritel-proksi-v1 (XL Stockbit, YP Mirae, XC Ajaib, PD Indo Premier)"
RITEL = ("XL", "YP", "XC", "PD")
JENDELA = (1, 3, 5, 10, 20, 60)
ACC_C, ACC_K, DIS_C, DIS_K, NETRAL, INSUF = ("ACCUMULATION_CANDIDATE", "ACCUMULATION_CONFIRMED", "DISTRIBUTION_CANDIDATE", "DISTRIBUTION_CONFIRMED",
                                             "NEUTRAL", "INSUFFICIENT_DATA")
TEKS = {ACC_C: "Kandidat akumulasi (aliran beli konsisten, belum dikonfirmasi harga)", ACC_K: "Akumulasi terkonfirmasi aturan (aliran beli + respons harga)",
        DIS_C: "Kandidat distribusi (aliran jual meningkat / pembelian melemah)", DIS_K: "Distribusi terkonfirmasi aturan (jual konsisten + harga melemah)",
        NETRAL: "Netral (bukti berimbang)", INSUF: "Data tidak cukup"}


@dataclass
class CfgFlow:
    min_sesi: int = 20                 # di bawah ini: INSUFFICIENT_DATA
    persist_frac: float = 0.55         # porsi hari net beli/jual (20 sesi) untuk disebut konsisten
    min_partisipasi: float = 0.01      # |net non-ritel 20D| / volume 20D minimal agar aliran dianggap berarti
    ret_lemah: float = -0.03           # return 20D di bawah ini = harga melemah
    ret_kuat: float = 0.03             # return 20D di atas ini = harga menguat
    rvol_tinggi: float = 1.5           # volume relatif "tinggi"
    absorpsi_min_hari: int = 2
    distribusi_min_hari: int = 2
    min_nilai_miliar: float = 5.0      # ambang likuiditas (parameter, bukan hukum)
    breakout_gagal_hari: int = 3


# ---------------------------------------------------------------- sumber data
def dari_ledger(L: pd.DataFrame) -> dict:
    """Ledger (semua broker, per sesi) -> matriks net lot tanggal x broker."""
    net = L.pivot_table(index="trade_date", columns="broker_code", values="net_lots", aggfunc="sum").sort_index().fillna(0.0)
    m = L.drop_duplicates("trade_date").set_index("trade_date").sort_index()
    cx = L[L["crossing"]].groupby("trade_date")["buy_lots"].sum() if "crossing" in L else pd.Series(dtype=float)
    gb = L.groupby("trade_date")["buy_lots"].sum()
    return dict(net=net, volume=m["volume_lots"], sumber="ledger (semua broker)", lengkap_broker=True,
                crossing_pct=float(cx.sum() / gb.sum()) if gb.sum() > 0 else 0.0)


def dari_inventori(inv: dict) -> dict:
    """Inventory chart Invezgo (net kumulatif lembar, ~15 broker teratas) -> net lot harian."""
    cum = inv["cum"]
    net = cum.diff().fillna(cum.iloc[0] if len(cum) else 0.0) / 100.0
    pr = inv["price"].set_index("date")
    return dict(net=net, volume=pr["volume"] / 100.0, sumber="inventory chart (±15 broker teratas)", lengkap_broker=False, crossing_pct=None)


# ----------------------------------------------------------------- analisa
def _streak(arr) -> tuple[int, int]:
    """(streak terpanjang net beli, streak terakhir bertanda: + beli / - jual)."""
    best = cur = 0
    for v in arr:
        cur = cur + 1 if v > 0 else 0
        best = max(best, cur)
    last = 0
    for v in arr[::-1]:
        if last == 0 and v != 0:
            last = 1 if v > 0 else -1
        elif (v > 0 and last > 0) or (v < 0 and last < 0):
            last += 1 if last > 0 else -1
        else:
            break
    return best, last


def analisa(data: dict | None, ohlc: pd.DataFrame | None, cfg: CfgFlow | None = None, ritel: tuple = RITEL) -> dict:
    """data dari dari_ledger/dari_inventori; ohlc = DataFrame harian (kolom tgl, open, high, low, close, volume[lembar])."""
    cfg = cfg or CfgFlow()
    base = dict(versi=VERSI, ritel_versi=RITEL_VERSI, cfg=asdict(cfg))
    if not data or ohlc is None or ohlc.empty or data["net"].empty:
        return dict(base, label=INSUF, teks=TEKS[INSUF], evidence="Low", alasan_data="tidak ada data broker harian multi-sesi (perlu ledger atau Hitung lengkap)",
                    observed=None, harga=None, invalidasi=[], mendukung=[], melawan=[], kualitas={})
    net = data["net"]
    o = ohlc.copy()
    o["tgl"] = pd.to_datetime(o["tgl"]).dt.date
    o = o.set_index("tgl").sort_index()
    tgl = [d for d in net.index if d in o.index]
    net = net.loc[tgl]
    n = len(tgl)
    kual = dict(sumber=data["sumber"], sesi=n, dari=str(tgl[0]) if tgl else None, sampai=str(tgl[-1]) if tgl else None,
                broker_lengkap=data["lengkap_broker"], crossing_pct=data.get("crossing_pct"))
    if n < cfg.min_sesi:
        return dict(base, label=INSUF, teks=TEKS[INSUF], evidence="Low", alasan_data=f"hanya {n} sesi bersama harga (minimal {cfg.min_sesi})",
                    observed=None, harga=None, invalidasi=[], mendukung=[], melawan=[], kualitas=kual)
    rit = [c for c in net.columns if c in ritel]
    non = [c for c in net.columns if c not in ritel]
    nr = net[non].sum(axis=1)                 # net non-ritel per sesi
    rr = net[rit].sum(axis=1) if rit else pd.Series(0.0, index=net.index)
    vol = (o.loc[tgl, "volume"] / 100.0)      # lot
    close = o.loc[tgl, "close"]
    nilai20 = float((o.loc[tgl, "close"] * o.loc[tgl, "volume"]).tail(20).median())
    kual["nilai_transaksi_median_20d"] = nilai20
    kual["likuid"] = nilai20 >= cfg.min_nilai_miliar * 1e9
    # ---------------- lapisan 1: observed flow
    rows = []
    for w in JENDELA:
        if n < w:
            continue
        s = net.iloc[-w:].sum()
        sn = s[[c for c in s.index if c not in ritel]]
        top_b = sn[sn > 0].sort_values(ascending=False).head(3)
        top_s = sn[sn < 0].sort_values().head(3)
        v = float(vol.iloc[-w:].sum())
        rows.append(dict(jendela=f"{w}D", dari=str(tgl[-w]), sampai=str(tgl[-1]), net_non_ritel=float(nr.iloc[-w:].sum()), net_ritel_proksi=float(rr.iloc[-w:].sum()),
                         pct_volume=float(nr.iloc[-w:].sum() / v) if v > 0 else np.nan, volume_lot=v,
                         top3_beli=", ".join(f"{k} {x:,.0f}" for k, x in top_b.items()), top3_jual=", ".join(f"{k} {x:,.0f}" for k, x in top_s.items()),
                         rasio_konsentrasi=float(top_b.sum() / abs(top_s.sum())) if top_s.sum() < 0 else np.nan))
    obs = pd.DataFrame(rows)
    w20 = min(20, n)
    a20 = nr.iloc[-w20:].to_numpy()
    frac_beli = float((a20 > 0).mean())
    frac_jual = float((a20 < 0).mean())
    best, last = _streak(a20)
    flips = int(sum(1 for x, y in zip(a20[:-1], a20[1:]) if x * y < 0))
    persist = dict(frac_beli=frac_beli, frac_jual=frac_jual, streak_beli_terpanjang=best, streak_terakhir=last, pergantian_arah=flips, sesi=w20)
    # ---------------- lapisan 2: price confirmation
    c_all = o["close"]
    ma20, ma50 = float(c_all.tail(20).mean()), float(c_all.tail(50).mean()) if len(c_all) >= 50 else np.nan
    last_c = float(close.iloc[-1])
    ret = {f"{w}D": float(close.iloc[-1] / close.iloc[-w - 1] - 1) for w in JENDELA if n > w}
    v_all = o["volume"]
    rvol = float(v_all.tail(5).mean() / v_all.iloc[-25:-5].mean()) if len(v_all) > 25 and v_all.iloc[-25:-5].mean() > 0 else np.nan
    hi60 = float(o["high"].tail(60).max())
    dd = last_c / hi60 - 1
    hi20_prev = o["high"].iloc[-23:-3].max() if len(o) > 23 else np.nan
    bo = None
    if len(o) > 23:
        brk = [i for i in range(len(o) - 3, len(o)) if o["close"].iloc[i] > o["high"].iloc[i - 20:i].max()]
        if brk:
            i0 = brk[0]
            gagal = bool(o["close"].iloc[-1] < o["high"].iloc[i0 - 20:i0].max())
            bo = dict(tgl=str(o.index[i0]), level=float(o["high"].iloc[i0 - 20:i0].max()), gagal=gagal)
    rv_d = (o["volume"] / o["volume"].shift(1).rolling(20, min_periods=10).median()).reindex(tgl)
    rng = (o["high"] - o["low"]).replace(0, np.nan).reindex(tgl)
    pos_close = ((o["close"] - o["low"]).reindex(tgl) / rng)
    chg = o["close"].pct_change().reindex(tgl)
    # distribusi: volume tinggi, harga ditutup lemah (setengah bawah & di bawah open), non-ritel net jual
    dist_hari = [str(d) for d in tgl[-20:] if rv_d[d] >= cfg.rvol_tinggi and o.loc[d, "close"] < o.loc[d, "open"] and pos_close[d] <= 0.4 and nr[d] < 0]
    # absorpsi: volume tinggi, ritel proksi menjual (non-ritel menyerap), harga tidak turun; lalu harga merebut kembali level
    abs_hari = [d for d in tgl[-20:] if rv_d[d] >= cfg.rvol_tinggi and nr[d] > 0 and rr[d] < 0 and chg[d] >= -0.005]
    reclaim = False
    if len(abs_hari) >= cfg.absorpsi_min_hari:
        i_first = tgl.index(abs_hari[0])
        ref = float(close.iloc[max(0, i_first - 10):i_first].max()) if i_first > 0 else float(close.iloc[0])
        reclaim = last_c > ref
    absorpsi = dict(hari=[str(d) for d in abs_hari], terkonfirmasi=bool(len(abs_hari) >= cfg.absorpsi_min_hari and reclaim))
    r20 = ret.get("20D", ret.get(f"{n - 1}D", np.nan)) if n > 20 else float(close.iloc[-1] / close.iloc[0] - 1)
    net20 = float(nr.iloc[-w20:].sum())
    v20 = float(vol.iloc[-w20:].sum())
    part = net20 / v20 if v20 > 0 else 0.0
    if net20 > 0 and r20 < cfg.ret_lemah:
        respons = "flow beli belum efektif / supply masih berat (non-ritel net beli tetapi harga turun)"
    elif net20 > 0:
        respons = "flow beli diikuti ketahanan/kenaikan harga"
    elif net20 < 0 and r20 > cfg.ret_kuat:
        respons = "harga naik sementara non-ritel net jual (kenaikan ditopang ritel proksi: rapuh)"
    elif net20 < 0:
        respons = "flow jual dikonfirmasi harga yang melemah"
    else:
        respons = "tidak ada aliran bersih berarti"
    harga = dict(ret=ret, ma20=ma20, ma50=ma50, di_atas_ma20=last_c > ma20, ma20_di_atas_ma50=bool(ma20 > ma50) if not np.isnan(ma50) else None, rvol_5_20=rvol,
                 drawdown_60d=dd, breakout=bo, hari_distribusi=dist_hari, absorpsi=absorpsi, respons=respons, low20=float(o["low"].tail(20).min()), close=last_c)
    # ---------------- lapisan 3: interpretation
    sup, lawan = [], []
    beli_konsisten = net20 > 0 and frac_beli >= cfg.persist_frac and part >= cfg.min_partisipasi
    jual_konsisten = net20 < 0 and frac_jual >= cfg.persist_frac and -part >= cfg.min_partisipasi
    net5 = float(nr.iloc[-min(5, n):].sum())
    melemah = net20 > 0 and net5 < 0
    konf_harga = (r20 >= 0 and last_c > ma20 and not (bo and bo["gagal"])) or absorpsi["terkonfirmasi"]
    lemah_harga = r20 < 0 or last_c < ma20 or bool(bo and bo["gagal"]) or len(dist_hari) >= cfg.distribusi_min_hari
    (sup if beli_konsisten else lawan).append(f"[OBS] non-ritel net {net20:+,.0f} lot dalam {w20} sesi ({part:+.1%} volume); {frac_beli:.0%} sesi net beli, {flips} pergantian arah")
    (sup if net5 >= 0 else lawan).append(f"[OBS] net non-ritel 5 sesi {net5:+,.0f} lot")
    (sup if konf_harga else lawan).append(f"[OBS] harga {r20:+.1%} {w20} sesi; close {'di atas' if last_c > ma20 else 'di bawah'} MA20 {ma20:,.0f}" + (f"; breakout {bo['tgl']} {'GAGAL' if bo['gagal'] else 'bertahan'}" if bo else ""))
    if absorpsi["hari"]:
        (sup if absorpsi["terkonfirmasi"] else lawan).append(f"[OBS] {len(abs_hari)} hari volume tinggi saat ritel proksi menjual dan harga bertahan" + ("; harga merebut kembali level sebelumnya (absorpsi)" if absorpsi["terkonfirmasi"] else "; belum merebut kembali level (bukan absorpsi)"))
    if dist_hari:
        lawan.append(f"[OBS] {len(dist_hari)} hari distribusi (volume tinggi, ditutup lemah, non-ritel net jual): {', '.join(dist_hari[-3:])}")
    if not kual["likuid"]:
        lawan.append(f"[OBS] likuiditas rendah: nilai transaksi median 20D Rp{nilai20 / 1e9:,.1f} M < Rp{cfg.min_nilai_miliar:g} M (risiko slippage)")
    if beli_konsisten and not melemah:
        label = ACC_K if konf_harga else ACC_C
    elif jual_konsisten:
        label = DIS_K if lemah_harga else DIS_C
    elif melemah or (net20 < 0) or len(dist_hari) >= cfg.distribusi_min_hari:
        label = DIS_C
    else:
        label = NETRAL
    # evidence level: kelengkapan & konsistensi bukti (BUKAN probabilitas)
    skor_bukti = (n >= 60) + data["lengkap_broker"] + (abs(part) >= 2 * cfg.min_partisipasi) + (max(frac_beli, frac_jual) >= 0.65) + (konf_harga if label in (ACC_C, ACC_K) else lemah_harga)
    evidence = "High" if skor_bukti >= 4 else "Medium" if skor_bukti >= 2 else "Low"
    if not kual["likuid"] and evidence == "High":
        evidence = "Medium"
    inval = []
    if label in (ACC_C, ACC_K):
        inval = [f"net non-ritel berbalik negatif {3} sesi berturut-turut",
                 f"close di bawah low 20 sesi {harga['low20']:,.0f} dengan volume ≥ {cfg.rvol_tinggi}x median",
                 "pembelian non-ritel tanpa respons harga 10 sesi (return ≤ 0 sementara net > 0)",
                 "muncul ≥ 2 hari distribusi saat gagal menembus resistance",
                 f"nilai transaksi median 20D turun di bawah Rp{cfg.min_nilai_miliar:g} M atau data ledger tidak lengkap"]
    elif label in (DIS_C, DIS_K):
        inval = ["non-ritel kembali net beli konsisten ≥ 55% dari 10 sesi", f"close kembali di atas MA20 {ma20:,.0f} dan bertahan 3 sesi",
                 "absorpsi terkonfirmasi (volume tinggi, harga bertahan lalu merebut kembali level)"]
    return dict(base, label=label, teks=TEKS[label], evidence=evidence, alasan_data="", observed=obs, persistensi=persist, harga=harga,
                invalidasi=inval, mendukung=sup, melawan=lawan, kualitas=kual, net20=net20, partisipasi=part)


def baris(fl: dict | None) -> list[str]:
    L = ["## ALIRAN TRANSAKSI BROKER (3 lapisan; label operasional, bukan identitas 'bandar')"]
    if not fl:
        L.append("N/A")
        return L
    k = fl.get("kualitas") or {}
    L.append(f"[DEC] label={fl['label']} ({fl['teks']}); evidence={fl['evidence']} (tingkat bukti, BUKAN probabilitas); versi {fl['versi']}; kelompok ritel = {fl['ritel_versi']} (proksi)")
    if fl["label"] == INSUF:
        L.append(f"alasan: {fl['alasan_data']}")
        return L
    L.append(f"[OBS] sumber={k.get('sumber')}; {k.get('sesi')} sesi {k.get('dari')}..{k.get('sampai')}; crossing={'-' if k.get('crossing_pct') is None else format(k['crossing_pct'], '.0%')}; "
             f"nilai transaksi median 20D Rp{k.get('nilai_transaksi_median_20d', 0) / 1e9:,.1f} M ({'likuid' if k.get('likuid') else 'TIDAK likuid'})")
    L.append("### Lapisan 1: observed flow (lot; non-ritel = semua broker di luar ritel proksi)")
    L.append("Jendela | Periode | Net non-ritel | % volume | Net ritel proksi | Top-3 beli | Top-3 jual")
    for r in fl["observed"].itertuples():
        L.append(f"{r.jendela} | {r.dari}..{r.sampai} | {r.net_non_ritel:+,.0f} | {r.pct_volume:+.1%} | {r.net_ritel_proksi:+,.0f} | {r.top3_beli or '-'} | {r.top3_jual or '-'}")
    p = fl["persistensi"]
    L.append(f"[OBS] persistensi {p['sesi']} sesi: {p['frac_beli']:.0%} net beli, {p['frac_jual']:.0%} net jual, streak beli terpanjang {p['streak_beli_terpanjang']}, streak terakhir {p['streak_terakhir']:+d}, pergantian arah {p['pergantian_arah']}")
    h = fl["harga"]
    L.append("### Lapisan 2: price confirmation")
    L.append(f"[OBS] return: " + ", ".join(f"{k_} {v:+.1%}" for k_, v in h["ret"].items()) + f"; close {h['close']:,.0f} vs MA20 {h['ma20']:,.0f}; volume 5/20 {h['rvol_5_20']:.2f}x; drawdown dari high 60D {h['drawdown_60d']:+.1%}")
    L.append(f"[OBS] breakout: {('tidak ada' if not h['breakout'] else h['breakout']['tgl'] + (' GAGAL' if h['breakout']['gagal'] else ' bertahan'))}; hari distribusi 20D: {len(h['hari_distribusi'])}; absorpsi: {'terkonfirmasi' if h['absorpsi']['terkonfirmasi'] else ('kandidat ' + str(len(h['absorpsi']['hari'])) + ' hari' if h['absorpsi']['hari'] else 'tidak ada')}")
    L.append(f"[INF] respons harga terhadap flow: {h['respons']}")
    L.append("### Lapisan 3: interpretation")
    L += [f"- (mendukung) {x}" for x in fl["mendukung"]] + [f"- (berlawanan) {x}" for x in fl["melawan"]]
    L += [f"- (invalidasi) {x}" for x in fl["invalidasi"]]
    return L
