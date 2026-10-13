"""Ringkasan TXT padat untuk dianalisa AI (Claude, ChatGPT, dll): data sebanyak mungkin, ukuran sekecil mungkin.

Format: bagian '## NAMA', baris 'kunci=nilai', tabel dipisah '|'. Harga bulat tanpa pemisah ribuan, persen bertanda,
lot = 100 lembar, 'M' = Rp miliar. Edukasi, bukan ajakan transaksi.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

import datetime as dt

import adimology as ad
import audit as au
import bandar as bd
import dinamika as dn
import hengky as hk
import laporan as lp
import konteks as ktx
import skor as sk
import bandar_flow as bf
import situasi_khusus as sxk
import ledger_calc as lgc

HORIZON = (5, 10, 20)


# -------------------------------------------------------------------- format
def _i(x):
    return "-" if x is None or pd.isna(x) else f"{x:.0f}"


def _f(x, d=1):
    return "-" if x is None or pd.isna(x) else f"{x:.{d}f}"


def _p(x, d=1):
    return "-" if x is None or pd.isna(x) else f"{x * 100:+.{d}f}%"


def _pu(x):
    return "-" if x is None or pd.isna(x) else f"{x * 100:.0f}%"


def _m(x):
    return "-" if x is None or pd.isna(x) else f"{x / 1e9:.1f}M"


# ----------------------------------------------------------- statistik historis
def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0 or p is None or np.isnan(p):
        return (np.nan, np.nan)
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    a = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - a) / d, (c + a) / d)


def statistik_harga(df: pd.DataFrame, tp_pct: float | None, sl_pct: float | None, biaya: float = 0.004) -> dict | None:
    """Base rate dari riwayat harga saham ini: peluang naik, sebaran return, sentuh target sebelum stop.

    Bukan prakiraan: jendela saling tumpang tindih (sampel efektif jauh lebih kecil dari n) dan pola masa lalu bisa berubah.
    """
    d = df.dropna(subset=["close", "high", "low"]).reset_index(drop=True)
    if len(d) < 80:
        return None
    c, h, lo = d["close"].to_numpy(), d["high"].to_numpy(), d["low"].to_numpy()
    rsi = d["rsi"].to_numpy() if "rsi" in d else np.full(len(d), np.nan)
    ma50 = d["ma50"].to_numpy() if "ma50" in d else np.full(len(d), np.nan)

    def zona(x):
        return "<30" if x < 30 else ">70" if x > 70 else "30-70"
    kini = (bool(c[-1] > ma50[-1]) if not np.isnan(ma50[-1]) else None, zona(rsi[-1]) if not np.isnan(rsi[-1]) else None)
    out = dict(n_hari=len(d), per_horizon={}, kondisi=dict(di_atas_ma50=kini[0], rsi=kini[1]))
    for H in HORIZON:
        rows = []
        for i in range(len(d) - H):
            ret = c[i + H] / c[i] - 1
            mfe = h[i + 1:i + H + 1].max() / c[i] - 1
            mae = lo[i + 1:i + H + 1].min() / c[i] - 1
            first = None
            if tp_pct and sl_pct:
                first = "none"
                for j in range(i + 1, i + H + 1):
                    hit_sl, hit_tp = lo[j] / c[i] - 1 <= sl_pct, h[j] / c[i] - 1 >= tp_pct
                    if hit_sl:
                        first = "sl"; break             # hari yang sama = SL (konservatif)
                    if hit_tp:
                        first = "tp"; break
            cocok = (not np.isnan(ma50[i]) and kini[0] is not None and bool(c[i] > ma50[i]) == kini[0]
                     and not np.isnan(rsi[i]) and kini[1] is not None and zona(rsi[i]) == kini[1])
            rows.append((ret, mfe, mae, first, cocok))
        a = pd.DataFrame(rows, columns=["ret", "mfe", "mae", "first", "cocok"])

        def ringkas(x):
            if len(x) == 0:
                return None
            n_eff = max(len(x) // H, 1)               # jendela tumpang tindih: sampel efektif ~ n / H
            pn = float((x["ret"] > 0).mean())
            return dict(n=len(x), n_eff=n_eff, ci_naik=wilson(pn, n_eff), exp_bersih=float(x["ret"].mean() - biaya),
                        p_naik=pn, p_5=float((x["ret"] >= 0.05).mean()),
                        p_m5=float((x["ret"] <= -0.05).mean()), med=float(x["ret"].median()),
                        p10=float(x["ret"].quantile(0.1)), p90=float(x["ret"].quantile(0.9)),
                        mfe=float(x["mfe"].median()), mae=float(x["mae"].median()),
                        p_tp=float((x["first"] == "tp").mean()) if tp_pct and sl_pct else None,
                        p_sl=float((x["first"] == "sl").mean()) if tp_pct and sl_pct else None)
        out["per_horizon"][H] = dict(semua=ringkas(a), mirip=ringkas(a[a["cocok"]]))
    r_d = pd.Series(c).pct_change().dropna()
    pk = pd.Series(c).cummax()
    out["vol"] = dict(std_harian=float(r_d.tail(60).std()), vol_tahunan=float(r_d.tail(120).std() * np.sqrt(250)),
                      dd_maks=float((pd.Series(c) / pk - 1).min()),
                      rentang_harian=float(((d["high"] - d["low"]) / d["close"]).tail(20).mean()))
    out["hi_lo"] = dict(hi=float(h[-250:].max()), lo=float(lo[-250:].min()))
    out["biaya"] = biaya
    return out


def kecocokan(r, rk, f, ovn, hk_, vol: float | None, stat: dict | None) -> list[tuple[str, str, str]]:
    """Kecocokan gaya trading (HEURISTIK, belum divalidasi)."""
    atr_pct = rk["atr"] / rk["harga"] if rk else np.nan
    nilai = None
    if rk is not None:
        nilai = float((rk["ind"]["close"] * rk["ind"]["volume"]).tail(20).median())
    up = None
    if hk_:
        for k in hk_["checklist"]["kriteria"]:
            if k["kriteria"].startswith("Uptrend"):
                up = k["status"] == hk.YA
    out = []
    # scalping / BPJP-BSJP
    if nilai is None or pd.isna(atr_pct):
        out.append(("scalping/BPJP-BSJP", "tidak diukur", "data teknikal kurang"))
    else:
        sg = (ovn or {}).get("sinyal", {}).get("bsjp") if ovn else None
        ok = nilai >= 5e9 and atr_pct >= 0.02
        why = f"nilai {nilai / 1e9:.1f}M/hari {'>=' if nilai >= 5e9 else '<'} 5M; ATR {atr_pct:.1%} {'>=' if atr_pct >= 0.02 else '<'} 2%"
        if sg:
            ok = ok and sg["p_target"] >= 0.5
            why += f"; peluang +1% BSJP {sg['p_target']:.0%}"
        out.append(("scalping/BPJP-BSJP", "cocok" if ok else "kurang cocok", why))
    # swing
    if rk is None or pd.isna(atr_pct):
        out.append(("swing", "tidak diukur", "data teknikal kurang"))
    else:
        ok = bool(up) and 0.015 <= atr_pct <= 0.08 and rk["rr"] >= 1.5
        out.append(("swing", "cocok" if ok else "kurang cocok",
                    f"uptrend={'ya' if up else 'tidak' if up is not None else '?'}; ATR {atr_pct:.1%}; R/R {rk['rr']:.1f}"))
    # hold jangka panjang
    if f is None:
        out.append(("hold jangka panjang", "tidak diukur", "fundamental gagal diambil"))
    else:
        fatal = lp._fatal(f)
        val = dict(zip(f["val"]["Metrik"], f["val"]["Nilai"]))
        cfo_neg = any("arus kas operasi" in x.lower() and "negatif" in x.lower() for x in f["peringatan"])
        ok = not fatal and not cfo_neg
        out.append(("hold jangka panjang", "layak dipelajari" if ok else "kurang cocok",
                    f"tanda bahaya besar={'ya' if fatal else 'tidak'}; CFO negatif={'ya' if cfo_neg else 'tidak'}; "
                    f"PER {val.get('PER', '-')}, PBV {val.get('PBV', '-')}, ROE {val.get('ROE', '-')}"))
    return out


# --------------------------------------------------------------------- teks
def _tabel(df: pd.DataFrame, n_kol: int = 6) -> list[str]:
    kol = list(df.columns)[-n_kol:]
    out = ["metrik|" + "|".join(str(k).replace(" 20", "'") for k in kol)]
    for idx, row in df[kol].iterrows():
        out.append(f"{idx}|" + "|".join(str(v) for v in row.values))
    return out


def _data_lines(r, rk, f, ovn, alr, b200, hk_, nilai, vol, ritel, stat) -> list[str]:
    L: list[str] = []
    add = L.append
    add("## HARGA")
    d = rk["ind"] if rk else r.chart
    c = d["close"].to_numpy()
    def ret(n):
        return c[-1] / c[-1 - n] - 1 if len(c) > n else np.nan
    tp_p, sl_p = (rk["target"] / rk["harga"] - 1, rk["sl"] / rk["harga"] - 1) if rk else (None, None)
    add(f"close={_i(r.harga)} tgl={rk['tgl'] if rk else r.chart['tgl'].iloc[-1]} ret1/5/20/60h={_p(ret(1))}/{_p(ret(5))}/{_p(ret(20))}/{_p(ret(60))}")
    if stat:
        v = stat["vol"]
        add(f"hi/lo250h={_i(stat['hi_lo']['hi'])}/{_i(stat['hi_lo']['lo'])} jarak_hi={_p(r.harga / stat['hi_lo']['hi'] - 1)} "
            f"jarak_lo={_p(r.harga / stat['hi_lo']['lo'] - 1)} vol_tahunan={_pu(v['vol_tahunan'])} rentang_harian20={v['rentang_harian'] * 100:.1f}% "
            f"dd_maks={_p(v['dd_maks'])}")
    if rk:
        for ev in rk.get("split", []):
            add(f"split: harga sebelum {ev['tgl']} disesuaikan ({ev['jenis']})")

    if rk:
        add("## TEKNIKAL")
        ma = rk["ma"]
        add("MA5/20/50/100/200=" + "/".join(_i(ma[n]) for n in (5, 20, 50, 100, 200))
            + " harga_vs=" + "/".join(("+" if ma[n] and rk["harga"] > ma[n] else "-") if ma[n] else "?" for n in (5, 20, 50, 100, 200)))
        add(f"RSI14={_f(rk['rsi'])} MACD={_f(rk['macd'], 2)} signal={_f(rk['signal'], 2)} hist={_f(rk['hist'], 2)} ATR={_i(rk['atr'])} ({_pu(rk['atr'] / rk['harga'])})")
        add("support=" + ",".join(f"{_i(x['harga'])}x{x['n']}" for x in rk["sup"][:3]) + " resistance=" + ",".join(f"{_i(x['harga'])}x{x['n']}" for x in rk["res"][:3]))
        add("sinyal=" + "; ".join(f"{'+' if p > 0 else '-' if p < 0 else '0'}{t}" for p, t in rk["poin"]))

        add("## RENCANA APP (harga masuk/SL/TP)")
        rn = hk_["rencana"] if hk_ else None
        add(f"area_beli={_i(rk['entry_lo'])}-{_i(rk['entry_hi'])} valid={'ya' if rk['zona_valid'] else 'tidak'} entry_maks_RR1.5={_i(rk['entry_maks'])} "
            f"SL(support-1ATR)={_i(rk['sl'])} ({_p(rk['sl'] / rk['harga'] - 1)})")
        add(f"area_jual=TP1 {_i(rk['tp1'])} ({_p(rk['tp1'] / rk['harga'] - 1)}) / TP2 {_i(rk['tp2'])} ({_p(rk['tp2'] / rk['harga'] - 1)}) "
            f"target_utama={rk['target_nama']} RR_area={_f(rk['rr'], 2)} RR_harga_kini={_f(rk['rr_harga'], 2)}")
        if rn:
            if rn["support"]:
                s_ = rn["support"]
                add(f"plan_support: level={_i(s_['level'])} stop1tick={_i(s_['stop_tick'])} RR={_f(s_['rr'], 1)}")
            if rn["breakout"]:
                b_ = rn["breakout"]
                add(f"plan_breakout: level={_i(b_['level'])} entry={_i(b_['entry'])} stop1tick={_i(b_['stop_tick'])} target={_i(b_['target'])} "
                    f"vol_konfirmasi={_i(b_['volume_konfirmasi_lot'])}lot")
            add(f"impas_fee={_i(rn['impas'])} ({_p(rn['impas_pct'], 2)})")
    if stat:
        add("## STATISTIK HISTORIS HARGA (base rate saham ini, bukan prakiraan)")
        k = stat["kondisi"]
        add(f"n_hari={stat['n_hari']} kondisi_kini: harga>MA50={k['di_atas_ma50']} RSI={k['rsi']}. TP/SL = target {_p(tp_p)} / stop {_p(sl_p)} dari close.")
        add("H|set|n|n_eff|P(naik)|CI95_naik(n_eff)|P(>=+5%)|P(<=-5%)|med|p10|p90|MFEmed|MAEmed|exp_bersih|P(TP dulu)|P(SL dulu)")
        for H, g in stat["per_horizon"].items():
            for nm in ("semua", "mirip"):
                x = g[nm]
                if x:
                    add(f"{H}h|{nm}|{x['n']}|{x['n_eff']}|{_pu(x['p_naik'])}|{_pu(x['ci_naik'][0])}-{_pu(x['ci_naik'][1])}|{_pu(x['p_5'])}|{_pu(x['p_m5'])}|"
                        f"{_p(x['med'])}|{_p(x['p10'])}|{_p(x['p90'])}|{_p(x['mfe'])}|{_p(x['mae'])}|{_p(x['exp_bersih'], 2)}|{_pu(x['p_tp'])}|{_pu(x['p_sl'])}")
        add(f"catatan: BUKAN model probabilitas tervalidasi. n_eff=n/H (jendela tumpang tindih); CI = Wilson 95% pada n_eff; exp_bersih = rata-rata return "
            f"dikurangi biaya {stat['biaya']:.2%}; 'mirip'=hari dengan posisi vs MA50 & zona RSI sama (definisi sederhana); tanpa pemisahan train/validation/out-of-sample.")

    add("## GAYA TRADING (heuristik)")
    for nm, st_, why in kecocokan(r, rk, f, ovn, hk_, None, stat):
        add(f"{nm}={st_} ({why})")

    add("## BANDAR (window) -- periode & definisi sendiri (top-1 broker, top-3 vs top-15 pada window ini); tidak sepadan dengan bagian 200H")
    add(f"window={r.tgl_awal}..{r.tgl_akhir} ({r.params.window_hari}h) bandar={r.bandar['code']} net={_i(r.bandar['net_lot'])}lot porsi={r.bandar['porsi']:.1f}% "
        f"avg={_f(r.bandar['avg'])} floating={_p(r.floating)} net_hari_terakhir={_i(r.net_terakhir)} distribusi={'ya' if r.distribusi else 'tidak'}")
    k = r.konsentrasi or {}
    add(f"konsentrasi={k.get('status')} top3_beli={_i(k.get('top3_beli_lot'))} top15_jual={_i(k.get('top15_jual_lot'))} broker_beli/jual={k.get('n_beli')}/{k.get('n_jual')}")
    add(f"adimology: target_real={_i(r.calc['t_real_r'])} ({_p(r.calc['upside_real'])}) target_max={_i(r.calc['t_max_r'])} ({_p(r.calc['upside_max'])}) "
        f"lot/papan={_i(r.lot_per_papan)} ({r.lpp_sumber[:24]}) SL_window={_i(r.plan['sl'])}")
    top = r.bs.sort_values("net_volume", ascending=False)
    add("top_beli(kode:net_lot@avg)=" + ",".join(f"{x['code']}:{_i(x['net_lot'])}@{_i(x['buy_avg'])}" for _, x in top.head(5).iterrows()))
    add("top_jual=" + ",".join(f"{x['code']}:{_i(x['net_lot'])}@{_i(x['sell_avg'])}" for _, x in top.tail(5).iloc[::-1].iterrows()))
    add("bandar_harian(net_lot)=" + ",".join(_i(x) for x in r.harian["net_lot"].tail(10)))
    if alr:
        a_, n_ = alr["asing"], alr["nego"]
        add(f"asing={a_['status']} {_m(a_['net_value'])} beli={','.join(a_.get('top_beli', []))} jual={','.join(a_.get('top_jual', []))}")
        add(f"nego={n_['status']} vol={_i(n_['vol_lot'])}lot ({_pu(n_['porsi'])}) bandar_net_nego={_i(n_['bandar_net_lot'])}")
        kc = alr["konsistensi"]
        if not kc.empty:
            add("konsistensi=" + ",".join(f"{w.split()[0]}h:{c_}" for w, c_ in zip(kc["window"], kc["kode"])) + f" sama_dgn_bandar={int(kc['sama_dengan_bandar'].sum())}/{len(kc)}")
        af = alr.get("afiliasi")
        if af is not None:
            add("afiliasi>=5%=" + ("; ".join(f"{x['name']}({x['format_securities']},{x['next_pct']:.1f}%)" for x in af["cocok"][:2]) or "tidak"))

    if b200:
        add(f"## BANDAR {b200['hari']}H ({b200['dari']}..{b200['sampai']}) -- top-5 non-ritel PER PERIODE (definisi berbeda dari kelompok tetap di atas dan dari konsentrasi window): JANGAN dibandingkan langsung")
        add(f"[INF] arah={b200['status'].lower()} {_lbl_nyangkut(b200['nyangkut'])} | [OBS] ref_transaksi_top3={_f(b200['modal_top3'])} net_top3={_i(b200['net_top3_lot'])}lot "
            f"harga_vs_modal={_p(b200['floating_top3'])} acc5/dist5={_m(b200['net_acc5_value'])}/{_m(b200['net_dist5_value'])} konsentrasi={b200['konsentrasi']['status']}")
        add("akumulator(kode:net_lot@avg|harga_vs_avg)=" + ",".join(f"{x['code']}:{_i(x['net_lot'])}@{_i(x['avg'])}|{_p(x['floating'], 0)}" for x in b200["akumulator"]))
        add("distributor=" + ",".join(f"{x['code']}:{_i(x['net_lot'])}@{_i(x['avg'])}" for x in b200["distributor"]))

    if ritel:
        w, pj, s_ = ritel.get("window"), ritel.get("panjang"), ritel.get("sid")
        add(f"## MINAT RITEL (SID & broker ritel {'/'.join((w or {}).get('codes', [])) or '-'})")
        add(f"status={ritel['status']} minat={ritel['minat']}")
        if s_:
            add("SID(bulan:jumlah@harga)=" + ",".join(f"{str(t)[:7]}:{v:.0f}@{_i(pr)}" for t, v, pr in s_["seri"])
                + f" arah={s_['arah']} chg_3b={_p(s_['chg'])} chg_1b={_p(s_['chg_1b'])} (data bulanan KSEI, terlambat)")
        for nm, x in (("window", w), ("periode_panjang", pj)):
            if x and x.get("status") not in (None, "TIDAK DIUKUR"):
                add(f"ritel_{nm}: {x['status']} net={_i(x.get('net_lot'))}lot ({_p(x.get('net_pct'))} vol)"
                    + (f" porsi_beli/jual={_pu(x.get('porsi_beli'))}/{_pu(x.get('porsi_jual'))} avg_beli={_f(x.get('avg_beli'))} harga_vs_avg={_p(x.get('floating'))}"
                       if x.get("porsi_beli") is not None else "")
                    + (" | " + ",".join(f"{b['code']}:{_i(b['net_lot'])}" for b in x["per_broker"]) if x.get("per_broker") else ""))
        add("tafsir=" + ritel["ringkas"] + " SID turun = minat ritel menurun (saham tidak menarik saat ini).")
    if vol:
        add("## VOLUME")
        add(" ".join(f"avg{n}h={_i(vol[n]['lot_rata'])}lot({_f(vol[n]['nilai_rata'] / 1e9)}M)" for n in (20, 60, 200) if n in vol)
            + f" hari_terakhir={_i(vol['hari_terakhir']['lot'])}lot rvol={_f(vol['hari_terakhir']['rvol'])}x")
    if hk_ and hk_.get("pv"):
        pv = hk_["pv"]
        add(f"harga-volume{pv['n']}h={pv['label_n']} ({_p(pv['chg_n'])}, vol {_f(pv['rvol_n'])}x) harian=" + ",".join(f"{x.label[:4]}" for x in pv["harian"].itertuples()))
        if pv["climax"]:
            c_ = pv["climax"]
            add(f"selling_climax={c_['tgl']} vol{_f(c_['rvol'])}x low_bertahan={c_['low_bertahan']}")
        if pv["crossing"]:
            add("candle_kecil_volume_besar=" + ",".join(str(x) for x in pv["crossing"]))
    if hk_ and hk_.get("sid") and hk_["sid"]["status"] != "TIDAK DIUKUR":
        s_ = hk_["sid"]
        add(f"SID={s_['status']} {_i(s_['sid_awal'])}->{_i(s_['sid_akhir'])} ({_p(s_['chg'])}) harga {_p(s_['price_chg'])}")
    if hk_:
        add("hengky_checklist=" + ";".join(f"{k['kriteria'].split(' (')[0]}:{k['status']}" for k in hk_["checklist"]["kriteria"]))

    if r.ob:
        o = r.ob
        bk = sorted(o["bids"], reverse=True)[:5]
        of = sorted(o["offers"])[:5]
        add("## ORDERBOOK")
        add(f"bid={','.join(f'{_i(p_)}x{_i(o['bids'][p_])}' for p_ in bk)} offer={','.join(f'{_i(p_)}x{_i(o['offers'][p_])}' for p_ in of)} "
            f"total_bid={_i(o['total_bid'])} total_offer={_i(o['total_offer'])} level={o['n_levels']}")

    if ovn:
        add(f"## BPJP/BSJP (candle 30m {ovn['dari']}..{ovn['sampai']}, fee {ovn['fee']:.2%})")
        sg, sb = ovn["sinyal"].get("bsjp"), ovn["sinyal"].get("bpjp")
        if sg:
            add(f"BSJP: rvol_sore={_f(sg['rvol'])}x kelompok={sg['kelompok'][:14]} P(gap naik)={_pu(sg['p_gap'])} P(+1% bersih fee)={_pu(sg['p_target'])} "
                f"P(rugi s/d akhir sesi1)={_pu(sg['p_rugi'])} target={_i(sg['target'])} jam_puncak={sg['jam_jual']}({_pu(sg['p_jam'])}) layak={'ya' if sg['layak'] else 'tidak'}")
        for nm, blok in (("BSJP", ovn["bsjp"]), ("BPJP", ovn["bpjp"])):
            for kel, s_ in blok.items():
                if s_.get("n"):
                    add(f"{nm}|{kel[:12]}|n={s_['n']}|med_naik={_p(s_['median_naik'])}|P+1%={_pu(s_['p_1'])}|P+2%={_pu(s_['p_2'])}|P+3%={_pu(s_['p_3'])}|P_rugi={_pu(s_['p_rugi_akhir'])}")
        add(f"ambang_vol_tinggi: sore>={_i(ovn['ambang']['sore'])} buka>={_i(ovn['ambang']['buka'])} lembar")

    if f:
        add(f"## FUNDAMENTAL [OBS] (Rp miliar per kuartal; * = kumulatif YTD; n/m = tidak bermakna; '-' = data hilang, jangan diisi asumsi)")
        for nm, kunci in (("PnL", "pnl"), ("NERACA", "bs"), ("ARUSKAS", "cf")):
            add(f"[{nm}]")
            L.extend(_tabel(f[kunci]))
        lt = f["ltm"]
        add(f"LTM: pendapatan={_m(lt['rev'][0])} [{lt['rev'][1]}] laba={_m(lt['ni'][0])} [{lt['ni'][1]}] laba_normalisasi={_m(lt['ni_norm'])} "
            f"CFO={_m(lt['cfo'][0])} FCF={_m(lt['fcf'])} capex={_m(lt['capex'])} utang_bersih={_m(lt['net_debt'])} konversi_kas={_f(lt['cash_conv'], 2)}x")
        nm_ = lt["norm"]
        add("normalisasi_laba=" + (f"reported {_m(nm_['asli'])} -> normal {_m(nm_['nilai'])}; kuartal_ekstrem={','.join(nm_['outliers']) or 'tidak ada'} (3xMAD; heuristik)"
                                   if nm_.get("ok") else f"tidak dihitung ({nm_.get('alasan')})"))
        add("valuasi=" + "; ".join(f"{a}={b}" for a, b in f["val"].values))
        sn = f["sens"]
        if sn["per"] or sn["pbv"]:
            add("sensitivitas_multiple(BUKAN nilai wajar; harga implisit): " + " ".join(f"PER{x}x={_i(v)}" for x, v in sn["per"])
                + (" " if sn["per"] and sn["pbv"] else "") + " ".join(f"PBV{x}x={_i(v)}" for x, v in sn["pbv"]))
        for w in f["peringatan"]:
            add(f"PERHATIAN: {w}")
        for s_ in f["sorotan"][:4]:
            add(f"TUMBUH: {s_}")
    return L


# ----------------------------------------------------------- blok laporan baru
EDGE = ("strategi_teruji=sinyal Adimology (SETUP TERPENUHI -> target realistis vs SL window; beli di open esok hari), BUKAN penilaian bandar/aturan Hengky/checklist terbaru. "
        "universe=BBRI,ANTM,MDKA,ESSA periode=2025-06..2026-10 n_transaksi=248 (42 tanggal). "
        "hit_target(5/10/20h)=23%/27%/32%; expectancy_bersih_fee(5/10/20h)=-0.13%/-0.28%/+0.13% (CI95 mencakup 0; tidak berbeda nyata dari nol). "
        "benchmark_entry_acak=-0.31%/-0.34%/-0.24%; Adimology tidak terbukti lebih baik (hit_target lebih tinggi hanya di sebagian horizon, expectancy tidak). "
        "dengan_orderbook_asli(sejak 2026-03): target terlalu jauh (median +15%), hit 8-17%, expectancy negatif. "
        "out_of_sample=TIDAK ADA (tanpa walk-forward/train-validation-test). drawdown=belum diukur. slippage=tidak dimodelkan. periode_validitas=hanya periode uji.")


def _lbl_bandar(k: str) -> str:
    return {dn.K_TERKONFIRMASI: "AKUMULASI TERKONFIRMASI oleh perilaku harga (hipotesis dari aliran broker; bukan bukti identitas/niat bandar)",
            dn.K_TERINDIKASI: "AKUMULASI TERINDIKASI (aliran beli ada; respons harga belum mengonfirmasi)",
            dn.K_DISTRIBUSI: "DISTRIBUSI TERINDIKASI", dn.K_ABSORPSI: "ABSORPSI SUPPLY (penjualan besar tidak menekan harga)",
            dn.K_TIDAK: "TIDAK KONKLUSIF", "AKUMULASI KUAT": "INDIKASI AKUMULASI KUAT (checklist lama; kerangka dinamis tidak tersedia)",
            "AKUMULASI LEMAH / BELUM SEARAH": "INDIKASI AKUMULASI LEMAH (checklist lama)",
            "TIDAK ADA BUKTI AKUMULASI": "TIDAK ADA BUKTI AKUMULASI"}.get(k, "INDIKASI DISTRIBUSI / JEBAKAN (waspada)")


def _lbl_nyangkut(k: str) -> str:
    return {"NYANGKUT": "harga DI BAWAH referensi transaksi (estimated_transaction_reference) broker; bukan harga pokok, posisi yang masih dipegang tidak diketahui",
            "IMPAS": "harga dekat referensi transaksi (estimasi)", "TIDAK NYANGKUT": "harga di atas referensi transaksi (estimasi)"}.get(k, k)


def _bandar_lines(din: dict | None) -> list[str]:
    L = ["## BANDARMOLOGI: KERANGKA BUKTI [OBS = angka, INF = hipotesis]"]
    if not din:
        L.append("tidak tersedia (inventory broker gagal/ dinonaktifkan): bandarmologi hanya dari checklist lama; kesimpulan akumulasi/distribusi TIDAK KONKLUSIF.")
        return L
    L.append(f"[INF] kategori={din['kategori']} tingkat_bukti={din['tingkat']} keyakinan={din['keyakinan']} alasan={din['alasan']}")
    if not din["dimensi"]:
        return L
    L.append(f"[OBS] kelompok_tetap(top-3 akumulator non-ritel, 60h)={','.join(f'{k}({din['nama'].get(k, k)[:18]})' for k in din['grup'])} "
             f"distributor={','.join(din['grup_dist']) or '-'}; denominator=volume total tiap jendela; n_hari={din['n_hari']}")
    L.append("bobot dimensi = usulan desain; BUKAN probabilitas keberhasilan; skor 0-100")
    L.append("dimensi|bobot|skor|mendukung|menentang")
    for x in din["dimensi"]:
        L.append(f"{x['dimensi']}|{x['bobot']:.0%}|{x['skor']:.0f}|{'; '.join(x['mendukung'])[:230] or '-'}|{'; '.join(x['menentang'])[:230] or '-'}")
    L.append("jendela_h|net_lot|net_Rp~(lot*close)|%vol|hari_net_beli|ret_harga|ret_per_1%vol")
    for r_ in din["tabel"].itertuples():
        L.append(f"{r_.jendela}|{_i(r_.net_lot)}|{_m(r_.net_rp_perkiraan)}|{_p(r_.pct_volume, 2)}|{_pu(r_.hari_net_positif)}|{_p(r_.ret_harga)}|{_f(r_.respons_per_1pct, 2)}")
    L.append("[OBS] net_kelompok_harian_10h(lot)=" + ",".join(_i(x) for x in din["net_harian_10h"]) + f" (kumulatif {_i(din['kum_10h'])})")
    mo = din["modal"]
    if mo.get("ada"):
        pt = dn.posisi_tertekan(din)
        L.append(f"[OBS] estimasi_modal_transaksi_bersih(proksi, BUKAN harga pokok): rentang={_i(mo['rendah'])}-{_i(mo['tinggi'])} pusat={_i(mo['pusat'])} lebar={_pu(mo['lebar'])} "
                 f"per_jendela(avg_beli|buy/sell/imbalance Rp)=" + ",".join(
                     f"{x['jendela']}h:{_i(x['avg_beli'])}|{_m(x.get('buy_value'))}/{_m(x.get('sell_value'))}/{_f(x.get('imbalance'), 2)}" for x in mo["per_jendela"]) + f" harga_vs_pusat={_p(mo['harga_vs_pusat'])} ({mo['harga_vs_rentang']})")
        L.append(f"[INF] posisi_tertekan={pt[0]}: {pt[1]}")
    else:
        L.append(f"estimasi_modal=tidak tersedia ({mo.get('alasan')})")
    sp, bo, rs = din["supply"], din["breakout"], din["respons"]
    L.append(f"[OBS] supply: jual_besar={sp['n_jual_besar']} diserap={sp['n_absorpsi']} volume_besar_harga_diam={sp['n_volume_besar_harga_diam']} hari_distribusi_20h={sp['n_hari_distribusi']}; "
             f"breakout_60h={bo['n']} gagal={bo['n_gagal']}; korelasi_flow_harga(sama hari/besok)={_f(rs['corr_sama_hari'], 2)}/{_f(rs['corr_hari_berikut'], 2)}; "
             f"konsistensi: hari_net_beli_20h={_pu(din['konsistensi']['frac_pos'])} run_terpanjang={din['konsistensi']['run_terpanjang']} pergantian_arah={din['konsistensi']['pergantian_arah']}")
    L.append("bukti_mendukung=" + ("; ".join(din["mendukung"]) or "tidak ada"))
    L.append("bukti_menentang=" + ("; ".join(din["menentang"]) or "tidak ada"))
    L.append(f"tingkat_keyakinan={din['keyakinan']} (terkonfirmasi_statistik=TIDAK: pola belum diuji out-of-sample)")
    L.append("kondisi_pembatalan=" + "; ".join(din["pembatalan"]))
    for a, b in dn.tujuh_pertanyaan(din, "lihat D. KEPUTUSAN"):
        L.append(f"Q{a}: {b}")
    L.append("catatan: " + dn.CATATAN_BOBOT + " Broker informatif vs tidak, dan nilai tambah bandarmologi dibanding harga-volume saja, BELUM diuji (butuh backtest out-of-sample).")
    return L


def _meta(r, rk, f, ovn, alr, b200, hk_, ritel, v, snapshot, status, nilai_din=None) -> list[str]:
    L = ["## A. METADATA & KUALITAS DATA"]
    tgl = rk["tgl"] if rk else r.chart["tgl"].iloc[-1]
    umur = (snapshot.date() - tgl).days
    L.append(f"snapshot={snapshot:%Y-%m-%d %H:%M} WIB data_harga_s/d={tgl} (umur {umur} hari) broker_window={r.tgl_awal}..{r.tgl_akhir} "
             f"lap_keuangan_terakhir={f['periode'] if f else '-'} basis_laba={f['ltm']['ni'][1] if f else '-'} FY_tersedia={','.join(f['kualitas']['fy']) if f and f['kualitas']['fy'] else 'tidak'}")
    L.append("sumber: harga=/analysis/chart/stock (EOD; volume lembar) | broker=/analysis/summary/stock (EOD 18:00) | orderbook=/analysis/order-book (snapshot) | "
             "intraday=/analysis/chart/multi-time (30m) | keuangan=/analysis/financial-statement (IS/BS/CF kuartalan+FY; Q4 diturunkan) | "
             "inventory_broker=/analysis/inventory-chart/stock (net kumulatif harian top broker) | SID=/analysis/shareholder/number (bulanan) | afiliasi=/analysis/shareholder-above | aksi=/analysis/calendar | profil=/analysis/information")
    kosong = []
    if rk is None:
        kosong.append("teknikal(semua)")
    elif rk["ma"].get(200) is None:
        kosong.append("MA200")
    if r.ob is None or r.ob.get("n_levels", 0) <= 4:
        kosong.append("orderbook(>2 level)")
    if f is None:
        kosong.append("fundamental(semua)")
    else:
        kosong += f["kualitas"]["hilang"]
        if not f.get("profil"):
            kosong.append("profil_emiten")
    if hk_ is not None and not hk_.get("sid"):
        kosong.append("SID")
    if alr is None:
        kosong.append("aliran(asing/nego/konsistensi)")
    elif alr.get("afiliasi") is None:
        kosong.append("afiliasi>=5%")
    if b200 is None:
        kosong.append("bandar_jangka_panjang")
    if nilai_din is None:
        kosong.append("bandar_dinamis(inventory broker)")
    if ovn is None:
        kosong.append("statistik_BPJP_BSJP")
    L.append(f"field_kosong({len(kosong)})=" + (",".join(kosong) or "tidak ada"))
    pr = (f or {}).get("profil") or {}
    if pr:
        notasi = ",".join(f"{x.get('notation')}({str(x.get('description'))[:40]})" for x in (pr.get("notation") or [])) or "tidak ada"
        L.append(f"profil: papan={pr.get('board') or '-'} kategori={','.join(pr.get('category') or []) or '-'} notasi_khusus={notasi} "
                 f"administrasi={pr.get('administration') or '-'} listing={str(pr.get('listing_date') or '-')[:10]}")
    ak = (f or {}).get("aksi")
    L.append("aksi_korporasi=" + ((f or {}).get("aksi_status") or "tidak diukur") + (": " + "; ".join(f"{x['type']}({x['ringkas'][:60]})" for x in ak[:5]) if ak else "")
             + (" | split_terdeteksi=" + ",".join(f"{e['jenis']}@{e['tgl']}" for e in rk.get("split", [])) if rk and rk.get("split") else ""))
    return L


def _validasi_lines(val: dict) -> list[str]:
    L = [f"validasi={val['status']} gagal={val['n_gagal']} peringatan={val['n_peringatan']}"]
    L += [f"  [{c['status']}] {c['id']}: {c['detail']}" for c in val["cek"] if c["status"] != "OK"]
    return L


def _uptrend(rk) -> bool | None:
    return hk.uptrend(rk)[2] if rk else None


def _tesis_trading(rk, hk_, nilai) -> str:
    if not rk:
        return "tidak dapat dinilai (data teknikal tidak tersedia)"
    up = _uptrend(rk)
    if up:
        return "momentum / pullback ke support (tren naik)"
    if rk["rsi"] is not None and rk["rsi"] < 30:
        return "mean reversion / pantulan jenuh jual (MELAWAN tren turun; risiko tinggi)"
    if up is False:
        return "tidak ada setup trend-following (tren tidak naik); hanya pantau pembalikan atau hindari"
    return "konsolidasi: tunggu breakout atau pantulan di support"


def _kesimpulan(r, rk, f, ovn, alr, b200, hk_, nilai, ritel, kep, alasan, stat, pos) -> list[str]:
    L = ["## B. KESIMPULAN (observasi -> inferensi -> keputusan)"]
    c = (rk["ind"] if rk else r.chart)["close"].to_numpy()
    def ret(n):
        return c[-1] / c[-1 - n] - 1 if len(c) > n else np.nan
    ma = rk["ma"] if rk else {}
    L.append(f"[OBS] harga={_i(r.harga)} ret5/20/60h={_p(ret(5))}/{_p(ret(20))}/{_p(ret(60))} vs_MA20/50/200="
             + "/".join(("+" if ma.get(n) and r.harga > ma[n] else "-") if ma.get(n) else "?" for n in (20, 50, 200))
             + (f" RSI={_f(rk['rsi'])}" if rk else ""))
    k = r.konsentrasi or {}
    L.append(f"[OBS] broker_window: kandidat={r.bandar['code']} net={_i(r.bandar['net_lot'])}lot ({r.bandar['porsi']:.1f}% vol) avg={_f(r.bandar['avg'])} harga_vs_avg={_p(r.floating)} "
             f"top3_beli/top15_jual={_i(k.get('top3_beli_lot'))}/{_i(k.get('top15_jual_lot'))}"
             + (f" asing={alr['asing']['status']}" if alr else ""))
    if f:
        lt = f["ltm"]
        L.append(f"[OBS] fundamental LTM: pendapatan={_m(lt['rev'][0])} laba={_m(lt['ni'][0])} CFO={_m(lt['cfo'][0])} FCF={_m(lt['fcf'])} (basis {lt['ni'][1]})")
    din_ = nilai.get("dinamis")
    L.append(f"[INF] bandarmologi={_lbl_bandar(nilai['kategori'])}" + (f"; tingkat_bukti={nilai['tingkat']}; keyakinan={din_['keyakinan']}" if din_ else "")
             + ("; jebakan: " + " | ".join(nilai["jebakan"]) if nilai["jebakan"] else ""))
    if b200:
        L.append(f"[INF] bandar_{b200['hari']}h: arah={b200['status'].lower()} {_lbl_nyangkut(b200['nyangkut'])}")
    if ritel:
        L.append(f"[INF] minat_ritel={ritel['status']} (minat {ritel['minat']})")
    if hk_:
        L.append(f"[INF] disiplin_hengky={hk_['checklist']['hasil']}; profil={hk_['profil']['profil']}")
    if rk:
        L.append(f"[INF] teknikal={rk['verdict']} ({rk['alasan']})")
    L.append(f"[INF] adimology_skenario={r.verdict} (BUKAN dasar keputusan; tidak terbukti di backtest)")
    L.append(f"[KEP] keputusan_app={kep}")

    L.append("### TRADING_SETUP (harga, volume, risiko)")
    L.append("tesis=" + _tesis_trading(rk, hk_, nilai))
    bukti = [t for p_, t in (rk["poin"] if rk else []) if p_ > 0][:4] + [x["kriteria"] for x in nilai["kriteria"] if x["status"] == "ya"][:3]
    lemah = [t for p_, t in (rk["poin"] if rk else []) if p_ < 0][:4] + nilai["jebakan"][:2] + ([x["kriteria"] + ": " + x["bukti"] for x in hk_["checklist"]["kriteria"] if x["status"] == "tidak"][:3] if hk_ else [])
    L.append("bukti=" + ("; ".join(bukti) or "tidak ada yang kuat"))
    L.append("melemahkan=" + ("; ".join(lemah) or "tidak ada yang menonjol"))
    if stat:
        g10, g20 = stat["per_horizon"][10], stat["per_horizon"][20]
        m = g10["mirip"] or g10["semua"]
        pos0 = pos[-1] if pos else None
        L.append(f"peluang_payoff=base_rate saham ini (10h, {'kondisi mirip' if g10['mirip'] else 'semua hari'}, n_eff={m['n_eff']}{' -> TIDAK ANDAL, sampel efektif kecil' if m['n_eff'] < 30 else ''}): P(TP dulu)={_pu(m['p_tp'])} P(SL dulu)={_pu(m['p_sl'])} "
                 f"exp_bersih={_p(m['exp_bersih'], 2)}; payoff TP {_p(rk['target'] / rk['harga'] - 1)} vs SL {_p(rk['sl'] / rk['harga'] - 1)}"
                 + (f" (RR bersih fee {_f(pos0['rr_bersih'], 2)} di harga kini)" if pos0 and pos0.get("lot") else "")
                 + ". Probabilitas strategi TERVALIDASI: BELUM ADA (lihat C).")
    else:
        L.append("peluang_payoff=tidak dapat dihitung (data harga < 80 hari). Probabilitas tervalidasi: BELUM ADA.")
    L.append(f"tindakan={kep}; horizon=" + ("swing 5-20 hari bursa" if rk else "tidak ditentukan")
             + ("; BSJP/BPJP hanya bila peluang +1% >= 50% (lihat bagian BPJP/BSJP)" if ovn else ""))
    tidak = ["probabilitas strategi (tanpa validasi out-of-sample)", "slippage dan gap aktual", "berita/katalis", "niat pihak di balik kode broker"]
    if f:
        tidak += f["kualitas"]["hilang"]
    L.append("belum_diketahui=" + "; ".join(tidak))

    L.append("### INVESTMENT_THESIS (terpisah dari trading; hanya bila data mendukung)")
    if not f:
        L.append("tidak didukung data: laporan keuangan gagal diambil.")
    else:
        t = f["tesis"]
        if t["kesimpulan"] == "TIDAK DAPAT DINILAI":
            L.append(f"kesimpulan=TIDAK DAPAT DINILAI ({t['alasan']}). Tidak ada kesimpulan investasi; jangan menyimpulkan dari valuasi parsial.")
        else:
            L.append(f"kesimpulan={t['kesimpulan']} horizon=>= 6 bulan, ditinjau tiap laporan kuartalan")
            L.append("mendukung=" + ("; ".join(t["mendukung"]) or "tidak ada"))
            L.append("melemahkan=" + ("; ".join(t["melemahkan"]) or "tidak ada"))
        L.append("belum_diketahui=" + "; ".join(t["belum_diketahui"]))
    return L


def _edge_lines(stat, kep) -> list[str]:
    L = ["## C. TRADING EDGE (bukti, bukan klaim)", EDGE]
    L.append("status_edge=BELUM TERBUKTI. Angka pada STATISTIK HISTORIS adalah base rate saham ini, bukan edge strategi; jangan dipakai untuk sizing sebagai probabilitas keberhasilan.")
    return L


def _keputusan_lines(r, rk, nilai, hk_, kep, alasan, pos, ovn) -> list[str]:
    L = ["## D. KEPUTUSAN & INVALIDASI", f"[KEP] rekomendasi={kep}"]
    L.append("alasan=" + " | ".join(alasan))
    if rk:
        rn = hk_["rencana"] if hk_ else None
        if rk["zona_valid"]:
            L.append(f"kondisi_entry=hanya di area beli {_i(rk['entry_lo'])}-{_i(rk['entry_hi'])} (entry maksimal {_i(rk['entry_maks'])} agar RR>={rk['min_rr']:.1f} ke {rk['target_nama']})")
        else:
            L.append(f"kondisi_entry=TIDAK ADA harga entry di zona support {_i(rk['entry_lo'])}-{_i(rk['entry_hi_sup'])} yang memberi RR>={rk['min_rr']:.1f}; tunggu")
        if rn and rn.get("breakout"):
            b_ = rn["breakout"]
            L.append(f"alternatif_breakout=close > {_i(b_['level'])} dengan volume >= {_i(b_['volume_konfirmasi_lot'])} lot, entry {_i(b_['entry'])}, stop 1 tick {_i(b_['stop_tick'])} (stop ketat; false breakout sering)")
        L.append(f"target=TP1 {_i(rk['tp1'])} ({_p(rk['tp1'] / rk['harga'] - 1)}), TP2 {_i(rk['tp2'])} ({_p(rk['tp2'] / rk['harga'] - 1)}); utama {rk['target_nama']}")
        sup = rk["sup"][0]["harga"] if rk["sup"] else None
        L.append(f"invalidasi=close di bawah support {_i(sup)} (SL {_i(rk['sl'])}, {_p(rk['sl'] / rk['harga'] - 1)}); atau bandar net jual 3 hari berturut-turut; atau muncul jebakan baru (jual di nego/distribusi terselubung)")
    ubah = [x["kriteria"] for x in nilai["kriteria"] if x["status"] in ("tidak", "netral")] +            ([x["kriteria"] for x in hk_["checklist"]["kriteria"] if x["status"] in ("tidak", "netral") and x["inti"]] if hk_ else [])
    L.append("rekomendasi_berubah_jika=" + ("; ".join(f"'{u}' menjadi ya" for u in ubah[:7]) or "tidak ada syarat tersisa") + " (dan tidak ada jebakan)")
    if pos:
        L.append("posisi: modal & risiko dari parameter app; STOP = SL teknikal di atas; fee termasuk")
        L.append("skenario|entry|stop|lot|modal_pakai|rugi_di_stop|%modal|laba_di_target|RR_bersih|rugi_jika_gap5%")
        for x in pos:
            if x.get("lot"):
                L.append(f"{x['nama']}|{_i(x['entry'])}|{_i(x['stop'])}|{x['lot']}|Rp{x['modal_pakai']:,.0f}|Rp{x['rugi_stop']:,.0f}|{x['rugi_stop_pct_modal']:.2%}|Rp{x['laba_target']:,.0f}|{_f(x['rr_bersih'], 2)}|Rp{x['rugi_gap']:,.0f} (pengikat lot: {x['pengikat']})")
            else:
                L.append(f"{x['nama']}|lot=0 ({x.get('alasan') or 'risiko/modal tidak cukup'})")
    return L


def _audit_lines(r, rk, f, nilai, hk_, kep, alasan, ritel, b200) -> list[str]:
    p = r.params
    L = ["## E. AUDIT TRAIL"]
    L.append("rumus: area_beli=support..support+max(0.5ATR,1%) dibatasi entry_maks; entry_maks=floor_tick((target+1.5*SL)/2.5) [RR>=1.5]; SL=support-1ATR14; TP=resistance berikutnya (cadangan: proyeksi R); "
             "RR=(target-entry)/(entry-SL); impas=harga*(1+fee_b)/(1-fee_j); LTM=FY(t-1)-YTD(t-1)+YTD(t) (cadangan: 4 kuartal / YTD disetahunkan); Q4=FY-YTD Q3; "
             "laba_normalisasi: kuartal |selisih median|>3xMAD diganti median kuartal lain; base_rate: return/MFE/MAE harian historis, n_eff=n/H; " + au.RUMUS_POSISI)
    L.append(f"asumsi: modal=Rp{p.modal:,.0f} risiko/trade={p.risiko_pct:.1%} fee_beli={p.fee_beli:.2%} fee_jual={p.fee_jual:.2%} (pajak jual dianggap termasuk); lot=100 lembar; window_broker={p.window_hari}h; "
             f"ritel(minat)=XL,XC,PD,CC,YP (default); ambang heuristik belum divalidasi: distribusi {p.distribusi_pct:.0%}, referensi transaksi +-5%, SID +-2% (3 bln), volume tinggi 1.5x, likuid 5M/hari, uptrend=harga>MA50 & MA50 naik")
    sinyal = {"adimology": 1 if r.verdict.startswith("SETUP") else -1,
              "bandarmologi": {"kuat": 1, "waspada": -1, "lain": 0}[bd.gate(nilai)]}
    if rk:
        sinyal["teknikal"] = 1 if rk["verdict"].startswith("BELI") else -1 if rk["verdict"] == "HINDARI" else 0
    if hk_:
        sinyal["hengky_inti"] = 1 if hk_["checklist"]["lolos_inti"] else -1
    if f and f["tesis"]["kesimpulan"] != "TIDAK DAPAT DINILAI":
        sinyal["fundamental"] = 1 if f["tesis"]["kesimpulan"] == "LAYAK DITELITI" else -1 if f["tesis"]["kesimpulan"] == "TESIS LEMAH" else 0
    pos_ = [k for k, v in sinyal.items() if v > 0]
    neg_ = [k for k, v in sinyal.items() if v < 0]
    L.append("sinyal: " + " ".join(f"{k}={'+' if v > 0 else '-' if v < 0 else '0'}" for k, v in sinyal.items()))
    L.append("konflik_sinyal=" + (f"positif {pos_} vs negatif {neg_}" if pos_ and neg_ else "tidak ada konflik tajam"))
    L.append(f"alasan_keputusan_final={kep}: aturan (bandar = sensor risiko + penguat) = HINDARI bila bandar menjual/distribusi (veto, termasuk status ledger DISTRIBUTING/MARKDOWN) / teknikal HINDARI / fundamental fatal / ; BELI bila teknikal di area beli + aturan inti Hengky terpenuhi + ada jejak akumulasi (terkonfirmasi = bukti kuat, terindikasi/absorpsi = bukti sedang); jebakan non-distribusi, tidak konklusif, atau notasi khusus = TUNGGU. "
             "Satu sinyal positif (mis. Adimology) tidak cukup mengalahkan sinyal negatif." + (" Rincian: " + " | ".join(alasan[:4]) if alasan else ""))
    return L


def _baris(r, rk, f, waktu: str, ovn=None, alr=None, b200=None, hk_=None, nilai=None, vol=None, ritel=None,
           status=None, snapshot=None, ledger=None, ctx=None) -> list[str]:
    snapshot = snapshot or dt.datetime.now()
    nilai = nilai or bd.penilaian(r, alr, rk)
    kep, alasan = lp.keputusan(r, rk, f, alr, nilai, hk_, ctx)
    pos = au.skenario_posisi(r, rk)
    val = au.validasi(r, rk, f, pos, snapshot)
    tp_p, sl_p = (rk["target"] / rk["harga"] - 1, rk["sl"] / rk["harga"] - 1) if rk else (None, None)
    stat = statistik_harga(rk["ind"], tp_p, sl_p, r.params.fee_beli + r.params.fee_jual) if rk else None
    sek = "/".join(x for x in ((f or {}).get("sektor"), (f or {}).get("subsektor")) if x)
    L = [f"# {r.params.kode}" + (f" | {f['nama']}" if f and f.get("nama") else "") + (f" | {sek}" if sek else "") + f" | {waktu}"]
    L.append(f"STATUS_VALIDASI={val['status']}" + (" -- JANGAN DIPAKAI: angka tidak konsisten (lihat bagian A)" if val["status"] == "GAGAL" else ""))
    L.append("LEGENDA: [OBS]=data teramati; [INF]=inferensi heuristik app (bukan fakta); [KEP]=keputusan app. n/m=tidak bermakna; *=kumulatif YTD; '-'=data hilang (JANGAN diisi asumsi). "
             "Satuan: harga Rp; lot=100 lembar; M=Rp miliar; % bertanda.")
    L.append("UNTUK AI: jawab terpisah (1) trading setup dan (2) investment thesis; bila data satu sisi tidak mendukung, katakan tidak dapat dinilai. Bedakan observasi, inferensi, keputusan. "
             "Nyatakan 'probabilitas belum diketahui' bila tidak tervalidasi; sebut data hilang/asumsi yang paling memengaruhi keputusan dan syarat yang membatalkan kesimpulan. Bukan nasihat investasi.")
    L += _meta(r, rk, f, ovn, alr, b200, hk_, ritel, val, snapshot, status or {}, nilai.get("dinamis"))
    L += _validasi_lines(val)
    L += _kesimpulan(r, rk, f, ovn, alr, b200, hk_, nilai, ritel, kep, alasan, stat, pos)
    L += _edge_lines(stat, kep)
    L += _keputusan_lines(r, rk, nilai, hk_, kep, alasan, pos, ovn)
    L.append("# ===== DATA RINCI (observasi; baris [INF] = inferensi) =====")
    L += _bandar_lines(nilai.get("dinamis"))
    L += _data_lines(r, rk, f, ovn, alr, b200, hk_, nilai, vol, ritel, stat)
    L += _audit_lines(r, rk, f, nilai, hk_, kep, alasan, ritel, b200)
    L += sk.baris_laporan(sk.hitung(r, rk, f, alr, nilai, hk_, ctx or {}))
    L += bf.baris((ctx or {}).get('flow'))
    L += sxk.baris((ctx or {}).get('situasi'))
    L += ktx.baris_laporan((ctx or {}).get('konteks'))
    L += lgc.baris_ai(ledger, r.params.kode)
    L.append("## BATASAN DATA & ENDPOINT")
    L += au.batasan(status or {}, r, rk, f, ovn, alr, b200, hk_, ritel)
    return L



def txt_bytes(*a, **k) -> bytes:
    return ("\n".join(_baris(*a, **k)) + "\n").encode("utf-8")


# ------------------------------------------------------------------- Markdown
def _md_dari_baris(L: list[str]) -> list[str]:
    """Ubah baris laporan menjadi Markdown terstruktur: heading, tabel (blok baris '|' berkolom sama), daftar, kutipan."""
    out: list[str] = []
    i = 0
    while i < len(L):
        ln = L[i].rstrip()
        if i == 0 and ln.startswith("# "):
            out += [ln.replace(" | ", " — "), ""]
        elif ln.startswith("# ====="):
            out += ["", "---", "", "> " + ln.strip("# =").strip(), ""]
        elif ln.startswith("### "):
            out += ["", ln, ""]
        elif ln.startswith("## "):
            out += ["", ln, ""]
        elif re.fullmatch(r"\[[A-Za-z0-9 ]+\]", ln):
            out += ["", "#### " + ln.strip("[]"), ""]
        elif ln.startswith(("LEGENDA:", "UNTUK AI:", "sumber:")):
            out += ["> " + ln, ""]
        elif ln.startswith("STATUS_VALIDASI="):
            out += ["**" + ln + "**", ""]
        else:
            cells = ln.split("|")
            if len(cells) >= 3 and all("=" not in c for c in cells) and i + 1 < len(L) and L[i + 1].count("|") == ln.count("|"):
                j = i + 1
                rows = []
                while j < len(L) and L[j].count("|") == ln.count("|"):
                    rows.append([c.strip() for c in L[j].split("|")])
                    j += 1
                out += ["", "| " + " | ".join(c.strip() for c in cells) + " |", "|" + "---|" * len(cells)]
                out += ["| " + " | ".join(r_) + " |" for r_ in rows] + [""]
                i = j
                continue
            out.append("- " + ln.strip() if ln.strip() else "")
        i += 1
    return out


def md_bytes(*a, **k) -> bytes:
    md = _md_dari_baris(_baris(*a, **k))
    return ("\n".join(md) + "\n").replace("\n\n\n", "\n\n").encode("utf-8")
