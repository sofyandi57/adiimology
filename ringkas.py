"""Ringkasan TXT padat untuk dianalisa AI (Claude, ChatGPT, dll): data sebanyak mungkin, ukuran sekecil mungkin.

Format: bagian '## NAMA', baris 'kunci=nilai', tabel dipisah '|'. Harga bulat tanpa pemisah ribuan, persen bertanda,
lot = 100 lembar, 'M' = Rp miliar. Edukasi, bukan ajakan transaksi.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import adimology as ad
import bandar as bd
import hengky as hk
import laporan as lp

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
def statistik_harga(df: pd.DataFrame, tp_pct: float | None, sl_pct: float | None) -> dict | None:
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
            return dict(n=len(x), p_naik=float((x["ret"] > 0).mean()), p_5=float((x["ret"] >= 0.05).mean()),
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


def txt_bytes(r, rk, f, waktu: str, ovn=None, alr=None, b200=None, hk_=None, nilai=None, vol=None) -> bytes:
    nilai = nilai or bd.penilaian(r, alr, rk)
    kep, alasan = lp.keputusan(r, rk, f, alr, nilai, hk_)
    L: list[str] = []
    add = L.append
    kode = r.params.kode
    sek = "/".join(x for x in ((f or {}).get("sektor"), (f or {}).get("subsektor")) if x)
    add(f"# {kode}" + (f" | {f['nama']}" if f and f.get("nama") else "") + (f" | {sek}" if sek else "") + f" | {waktu}")
    add("UNTUK AI: ringkasan data saham IDX dari app Adimology Calculator (data Invezgo). Satuan: harga Rp, lot=100 lembar, M=Rp miliar, "
        "% bertanda. Semua aturan/ambang di app = HEURISTIK belum tervalidasi; backtest 4 saham: target Adimology tidak lebih baik dari "
        "entry acak setelah fee. 'Statistik historis' = base rate dari harga saham ini, bukan prakiraan. Jawab dengan peluang & "
        "ketidakpastian, sebut asumsi, bukan nasihat investasi.")
    add("PERTANYAAN PENGGUNA BIASA: peluang naik; cocok hold/swing/scalping; harga masuk; SL/TP; ukuran posisi; bandar nyangkut/akumulasi; risiko.")

    add("## KEPUTUSAN APP")
    add(f"keputusan={kep}")
    add(f"bandarmologi={nilai['kategori']} skor={nilai['skor']:+d}/{nilai['skor_maks']}")
    if hk_:
        add(f"hengky={hk_['checklist']['hasil']} profil={hk_['profil']['profil']}")
    if rk:
        add(f"teknikal={rk['verdict']} ({rk['alasan']}) skor={rk['skor']:+d}/{rk['skor_max']}")
    add(f"adimology(skenario)={r.verdict}: " + "; ".join(r.alasan[:3]))
    for j in nilai["jebakan"]:
        add(f"JEBAKAN: {j}")
    for a in alasan:
        if a.startswith("BELI ditahan"):
            add(a)

    add("## HARGA")
    d = rk["ind"] if rk else r.chart
    c = d["close"].to_numpy()
    def ret(n):
        return c[-1] / c[-1 - n] - 1 if len(c) > n else np.nan
    stat = None
    if rk:
        tp_p, sl_p = rk["target"] / rk["harga"] - 1, rk["sl"] / rk["harga"] - 1
        stat = statistik_harga(rk["ind"], tp_p, sl_p)
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
    add(f"posisi(app): modal={r.params.modal / 1e6:.0f}jt risiko={r.params.risiko_pct:.1%} lot={r.plan['lot']} "
        f"modal_pakai={r.plan['modal_pakai'] / 1e6:.1f}jt rugi_di_SL_window={r.plan['rugi_sl'] / 1e6:+.2f}jt")

    if stat:
        add("## STATISTIK HISTORIS HARGA (base rate saham ini, bukan prakiraan)")
        k = stat["kondisi"]
        add(f"n_hari={stat['n_hari']} kondisi_kini: harga>MA50={k['di_atas_ma50']} RSI={k['rsi']}. TP/SL = target {_p(tp_p)} / stop {_p(sl_p)} dari close.")
        add("H|set|n|P(naik)|P(>=+5%)|P(<=-5%)|med|p10|p90|MFEmed|MAEmed|P(TP dulu)|P(SL dulu)")
        for H, g in stat["per_horizon"].items():
            for nm in ("semua", "mirip"):
                x = g[nm]
                if x:
                    add(f"{H}h|{nm}|{x['n']}|{_pu(x['p_naik'])}|{_pu(x['p_5'])}|{_pu(x['p_m5'])}|{_p(x['med'])}|{_p(x['p10'])}|{_p(x['p90'])}|"
                        f"{_p(x['mfe'])}|{_p(x['mae'])}|{_pu(x['p_tp'])}|{_pu(x['p_sl'])}")
        add("catatan: jendela tumpang tindih; 'mirip'=hari dengan posisi vs MA50 & zona RSI sama; n kecil = tidak andal.")

    add("## GAYA TRADING (heuristik)")
    for nm, st_, why in kecocokan(r, rk, f, ovn, hk_, None, stat):
        add(f"{nm}={st_} ({why})")

    add("## BANDAR (window)")
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
        add(f"## BANDAR {b200['hari']}H ({b200['dari']}..{b200['sampai']})")
        add(f"status={b200['status']} nyangkut={b200['nyangkut']} modal_top3={_f(b200['modal_top3'])} net_top3={_i(b200['net_top3_lot'])}lot "
            f"harga_vs_modal={_p(b200['floating_top3'])} acc5/dist5={_m(b200['net_acc5_value'])}/{_m(b200['net_dist5_value'])} konsentrasi={b200['konsentrasi']['status']}")
        add("akumulator(kode:net_lot@avg|harga_vs_avg)=" + ",".join(f"{x['code']}:{_i(x['net_lot'])}@{_i(x['avg'])}|{_p(x['floating'], 0)}" for x in b200["akumulator"]))
        add("distributor=" + ",".join(f"{x['code']}:{_i(x['net_lot'])}@{_i(x['avg'])}" for x in b200["distributor"]))

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
        add(f"## FUNDAMENTAL (Rp miliar per kuartal, {f['periode']})")
        for nm, kunci in (("PnL", "pnl"), ("NERACA", "bs"), ("ARUSKAS", "cf")):
            add(f"[{nm}]")
            L.extend(_tabel(f[kunci]))
        add("valuasi=" + "; ".join(f"{a}={b}" for a, b in f["val"].values))
        for w in f["peringatan"]:
            add(f"PERHATIAN: {w}")
        for s_ in f["sorotan"][:4]:
            add(f"TUMBUH: {s_}")
    add("## CATATAN SUMBER")
    add("backtest Adimology (4 saham, 248 sinyal): target realistis tercapai 23-32% sebelum SL, return rata-rata -0.3%..+0.1% setelah fee, tidak beda nyata dari acak; "
        "offer>bid tidak terbukti naik. Aturan Hengky = transkrip ASR, klaim akurasi tak teruji.")
    return ("\n".join(L) + "\n").encode("utf-8")
