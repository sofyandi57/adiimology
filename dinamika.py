"""Bandarmologi sebagai kerangka bukti (bukan skor tunggal): aliran dinamis, respons harga, penyerapan supply, estimasi modal.

Prinsip (dari feedback metodologis):
- Pisahkan observasi / inferensi / keputusan; jangan menyebut broker = satu bandar atau mengasumsikan net buy = niat menaikkan harga.
- Metrik sepadan: kelompok broker TETAP (top-3 akumulator non-ritel, 60 hari) diukur di semua jendela dengan denominator volume yang sama.
- Tingkat bukti: Tidak konklusif -> Indikasi awal -> Konfirmasi perilaku harga. 'Terkonfirmasi statistik' TIDAK PERNAH diberikan di sini:
  pola ini belum diuji out-of-sample.
- Bobot dimensi adalah USULAN DESAIN, bukan hasil optimasi, dan bukan probabilitas keberhasilan.
Edukasi, bukan ajakan transaksi.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

import adimology as ad

JENDELA = (1, 3, 7, 20, 60)
BOBOT = {"Konsistensi aliran": 0.20, "Konsentrasi & dominasi": 0.15, "Referensi transaksi (estimasi)": 0.15,
         "Respons harga terhadap flow": 0.20, "Konfirmasi volume & struktur harga": 0.15, "Risiko distribusi & pembatalan": 0.15}

K_TERKONFIRMASI = "AKUMULASI TERKONFIRMASI (perilaku harga)"
K_TERINDIKASI = "AKUMULASI TERINDIKASI"
K_DISTRIBUSI = "DISTRIBUSI TERINDIKASI"
K_ABSORPSI = "ABSORPSI SUPPLY"
K_TIDAK = "TIDAK KONKLUSIF"
KATEGORI = (K_TERKONFIRMASI, K_TERINDIKASI, K_DISTRIBUSI, K_ABSORPSI, K_TIDAK)
CATATAN_BOBOT = ("Bobot = usulan desain (20/15/15/20/15/15), BUKAN hasil optimasi dan BUKAN probabilitas keberhasilan. Tingkat 'terkonfirmasi "
                 "statistik' tidak diberikan: pola belum diuji out-of-sample.")


# ------------------------------------------------------------------- parsing
def parse_inventori(raw) -> dict:
    """Respons /analysis/inventory-chart/stock -> price df, cum df (tanggal x broker, net kumulatif lembar), nama broker."""
    if isinstance(raw, dict) and "price" not in raw and isinstance(raw.get("data"), dict):
        raw = raw["data"]
    if not isinstance(raw, dict) or not raw.get("price"):
        raise ad.AdimologyError("Inventory broker kosong untuk periode ini.")
    price = pd.DataFrame(raw["price"])
    price["date"] = pd.to_datetime(price["date"]).dt.date
    for c in ("open", "high", "low", "close", "volume"):
        price[c] = pd.to_numeric(price[c], errors="coerce")
    price = price.dropna(subset=["close"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    cols, names = {}, {}
    for b in raw.get("broker") or []:
        kode = str(b.get("broker"))
        names[kode] = b.get("name") or kode
        s = pd.Series({pd.to_datetime(x["date"]).date(): float(x["value"]) for x in b.get("data") or []})
        cols[kode] = s
    cum = pd.DataFrame(cols).reindex(price["date"]).ffill().fillna(0.0)
    return dict(price=price, cum=cum, names=names)


def pilih_grup(cum: pd.DataFrame, excl, n: int = 3) -> tuple[list, list]:
    end = cum.iloc[-1]
    cand = end[~end.index.isin(set(excl))]
    return (list(cand[cand > 0].sort_values(ascending=False).head(n).index), list(cand[cand < 0].sort_values().head(n).index))


def _harian(cum: pd.DataFrame) -> pd.DataFrame:
    net = cum.diff()
    net.iloc[0] = cum.iloc[0]
    return net


# ------------------------------------------------------------------- metrik
def metrik_jendela(price: pd.DataFrame, net_grup: pd.Series, jendela=JENDELA) -> pd.DataFrame:
    """Net kelompok (lot, perkiraan Rp), porsi terhadap volume, konsistensi harian, return harga, respons per 1% volume."""
    rows = []
    n = len(price)
    for w in jendela:
        if w > n:
            continue
        w_ = w
        ga = net_grup.tail(w_)
        vol = float(price["volume"].tail(w_).sum())
        net = float(ga.sum())
        pct = net / vol if vol > 0 else np.nan
        base = price["close"].iloc[-w_ - 1] if n > w_ else price["open"].iloc[0]
        ret = float(price["close"].iloc[-1] / base - 1)
        rp = float((ga * price["close"].tail(w_).to_numpy()).sum())
        rows.append(dict(jendela=w, net_lot=net / 100, net_rp_perkiraan=rp, pct_volume=pct, hari_net_positif=float((ga > 0).mean()),
                         ret_harga=ret, respons_per_1pct=(ret / pct) if pd.notna(pct) and abs(pct) >= 0.005 else np.nan))
    return pd.DataFrame(rows)


def konsistensi(net_grup: pd.Series, w: int = 20) -> dict:
    g = net_grup.tail(w)
    pos = (g > 0).to_numpy()
    run = best = 0
    for v in pos:
        run = run + 1 if v else 0
        best = max(best, run)
    sm = net_grup.rolling(5).sum().dropna().tail(w)
    tanda = np.sign(sm.to_numpy())
    flips = int((np.diff(tanda[tanda != 0]) != 0).sum()) if len(tanda) else 0
    balik = bool(len(sm) >= 6 and sm.iloc[0] < 0 < sm.iloc[-1])
    return dict(frac_pos=float(pos.mean()) if len(pos) else np.nan, run_terpanjang=best, pergantian_arah=flips,
                balik_dari_jual_ke_beli=balik, net5_terakhir_lot=float(net_grup.tail(5).sum()) / 100)


def respons_harga(price: pd.DataFrame, net_grup: pd.Series) -> dict:
    ret = price["close"].pct_change()
    ok = ret.notna()
    a, r_ = net_grup[ok.to_numpy()].to_numpy(), ret[ok].to_numpy()
    def corr(x, y):
        return float(np.corrcoef(x, y)[0, 1]) if len(x) >= 15 and np.std(x) > 0 and np.std(y) > 0 else np.nan
    return dict(corr_sama_hari=corr(a, r_), corr_hari_berikut=corr(a[:-1], r_[1:]) if len(a) > 1 else np.nan)


def supply(price: pd.DataFrame, net_acc: pd.Series, net_dist: pd.Series) -> dict:
    ret = price["close"].pct_change().fillna(0.0)
    vol = price["volume"]
    rvol = vol / vol.shift(1).rolling(20, min_periods=10).median()
    jual = net_dist[net_dist < 0]
    q = float(jual.abs().quantile(0.75)) if len(jual) >= 4 else np.inf
    # 'besar' = di kuartil atas sebarannya SENDIRI dan minimal 3% volume hari itu (aliran jual kecil yang konstan bukan penjualan besar)
    hari_jual_besar = (net_dist < 0) & (net_dist.abs() >= q) & (net_dist.abs() >= 0.03 * vol)
    absorb = hari_jual_besar & (ret >= -0.005)
    n_jual = int(hari_jual_besar.sum())
    sepi = (rvol >= 1.5) & (ret.abs() <= 0.01) & (net_acc <= 0)               # volume besar, harga diam, tanpa pembeli kelompok
    dist = (rvol >= 1.5) & (net_acc <= 0) & ((ret <= -0.01) | ((price["close"] - price["low"]) / (price["high"] - price["low"]).replace(0, np.nan) <= 0.3))
    return dict(n_jual_besar=n_jual, n_absorpsi=int(absorb.sum()), frac_absorpsi=float(absorb.sum() / n_jual) if n_jual else np.nan,
                n_volume_besar_harga_diam=int(sepi.sum()), n_hari_distribusi=int(dist.tail(20).sum()))


def breakout(price: pd.DataFrame, net_grup: pd.Series, w: int = 20, hari_gagal: int = 3) -> dict:
    out = []
    vol = price["volume"]
    rvol = vol / vol.shift(1).rolling(20, min_periods=10).median()
    for i in range(w, len(price)):
        level = price["high"].iloc[i - w:i].max()
        if price["close"].iloc[i] > level and rvol.iloc[i] >= 1.5:
            sesudah = price["close"].iloc[i + 1:i + 1 + hari_gagal]
            gagal = bool((sesudah < level).any()) if len(sesudah) else None
            out.append(dict(tgl=price["date"].iloc[i], level=float(level), gagal=gagal, flow_hari_itu=float(net_grup.iloc[i])))
    n = [b for b in out if b["gagal"] is not None]
    return dict(daftar=out[-5:], n=len(n), n_gagal=sum(1 for b in n if b["gagal"]), terakhir_valid=(out[-1] if out and out[-1]["gagal"] is False else None))


def estimasi_modal(summaries: dict, grup: list, harga: float) -> dict:
    """Estimasi modal transaksi BERSIH kelompok: rata-rata harga beli tertimbang volume per jendela summary; rentang antar jendela."""
    rows = []
    for w, bs in summaries.items():
        if bs is None or bs.empty:
            continue
        b = bs[bs["code"].isin(grup)].copy()
        for c in ("buy_avg", "buy_volume", "net_volume"):
            b[c] = pd.to_numeric(b[c], errors="coerce")
        if b.empty or b["net_volume"].sum() <= 0:
            continue
        wb = b["buy_volume"].where(b["buy_avg"].notna(), 0)
        if wb.sum() <= 0:
            continue
        bv = float(pd.to_numeric(b.get("buy_value"), errors="coerce").sum()) if "buy_value" in b else np.nan
        sv = float(pd.to_numeric(b.get("sell_value"), errors="coerce").sum()) if "sell_value" in b else np.nan
        rows.append(dict(jendela=w, avg_beli=float((b["buy_avg"].fillna(0) * wb).sum() / wb.sum()), net_lot=float(b["net_volume"].sum()) / 100,
                         buy_value=bv, sell_value=sv, imbalance=((bv - sv) / (bv + sv)) if pd.notna(bv) and pd.notna(sv) and (bv + sv) > 0 else np.nan))
    if not rows:
        return dict(ada=False, alasan="tidak ada jendela dengan net beli kelompok dan harga rata-rata")
    avg = [x["avg_beli"] for x in rows]
    pusat = next((x["avg_beli"] for x in rows if x["jendela"] == 20), float(np.median(avg)))
    lo, hi = min(avg), max(avg)
    return dict(ada=True, pusat=pusat, rendah=lo, tinggi=hi, lebar=(hi - lo) / pusat if pusat else np.nan, per_jendela=rows,
                harga_vs_pusat=harga / pusat - 1, harga_vs_rentang=("di bawah rentang" if harga < lo else "di atas rentang" if harga > hi else "di dalam rentang"))


# --------------------------------------------------------------------- dimensi
def _d(nama, skor, ya, tidak, ukuran=""):
    return dict(dimensi=nama, bobot=BOBOT[nama], skor=float(np.clip(skor, 0, 100)), mendukung=ya, menentang=tidak, ukuran=ukuran)


def dimensi(m: pd.DataFrame, kons: dict, resp: dict, sup: dict, bo: dict, modal: dict, konsentrasi: dict, pv: dict | None, price: pd.DataFrame) -> list[dict]:
    out = []
    pos_w = [int(r.net_lot > 0) for r in m.itertuples() if r.jendela >= 3]
    frac_w = float(np.mean(pos_w)) if pos_w else 0.0
    s1 = 100 * (0.5 * frac_w + 0.5 * (kons["frac_pos"] if pd.notna(kons["frac_pos"]) else 0.0))
    ya, tdk = [], []
    (ya if frac_w >= 0.75 else tdk).append(f"net kelompok positif di {sum(pos_w)} dari {len(pos_w)} jendela (3-60h)")
    (ya if kons["frac_pos"] >= 0.55 else tdk).append(f"{kons['frac_pos']:.0%} hari 20h net beli; run terpanjang {kons['run_terpanjang']} hari; {kons['pergantian_arah']} pergantian arah")
    if kons["balik_dari_jual_ke_beli"]:
        ya.append("arah berbalik dari net jual ke net beli (5h)")
    out.append(_d("Konsistensi aliran", s1, ya, tdk, "net positif antar jendela + proporsi hari net beli"))

    rasio = konsentrasi.get("rasio", np.nan)
    m20 = m[m["jendela"] == 20]
    porsi = float(m20["pct_volume"].iloc[0]) if len(m20) else np.nan
    s2 = 100 * (0.6 * min(1.0, (rasio if np.isfinite(rasio) else 1.0)) + 0.4 * min(1.0, max(porsi, 0) / 0.10)) if pd.notna(porsi) else 0.0
    ya, tdk = [], []
    (ya if rasio >= 1 else tdk).append(f"top-3 pembeli / top-15 penjual = {rasio:.2f}" if np.isfinite(rasio) else "tidak ada penjual bersih")
    if pd.notna(porsi):
        (ya if porsi >= 0.05 else tdk).append(f"net kelompok {porsi:.1%} dari volume 20h (denominator sama)")
    out.append(_d("Konsentrasi & dominasi", s2, ya, tdk, "rasio konsentrasi + porsi terhadap volume"))

    if modal.get("ada"):
        d = modal["harga_vs_pusat"]
        s3 = 100 if -0.03 <= d <= 0.05 else 60 if -0.08 <= d <= 0.10 else 30
        ya, tdk = [], []
        (ya if s3 >= 60 else tdk).append(f"harga {d:+.1%} dari referensi transaksi (estimasi) {modal['pusat']:,.0f} (rentang {modal['rendah']:,.0f}-{modal['tinggi']:,.0f}, {modal['harga_vs_rentang']})")
        if modal["lebar"] > 0.10:
            tdk.append(f"rentang estimasi lebar ({modal['lebar']:.0%}): ketidakpastian tinggi")
            s3 -= 15
    else:
        s3, ya, tdk = 0, [], [f"referensi transaksi (estimasi) tidak tersedia ({modal.get('alasan')})"]
    out.append(_d("Referensi transaksi (estimasi)", s3, ya, tdk, "posisi harga terhadap rata-rata beli bersih kelompok (proksi, bukan harga pokok)"))

    r20 = m[m["jendela"] == 20]
    ya, tdk = [], []
    if len(r20) and pd.notna(r20["ret_harga"].iloc[0]):
        ret20, net20 = float(r20["ret_harga"].iloc[0]), float(r20["net_lot"].iloc[0])
        if net20 > 0 and ret20 > 0.02:
            s4 = 85
            ya.append(f"flow beli 20h diikuti kenaikan harga {ret20:+.1%}")
        elif net20 > 0 and sup["frac_absorpsi"] >= 0.5:
            s4 = 70
            ya.append("penjualan besar diserap: harga bertahan")
        elif net20 > 0:
            s4 = 25
            tdk.append(f"unconfirmed flow: net beli 20h tetapi harga {ret20:+.1%}")
        else:
            s4 = 15
            tdk.append("tidak ada flow beli bersih 20h")
        c0 = resp.get("corr_sama_hari")
        if pd.notna(c0):
            (ya if c0 > 0.2 else tdk).append(f"korelasi flow-harga harian {c0:+.2f}")
            s4 += 10 if c0 > 0.2 else -10
    else:
        s4 = 0
        tdk.append("respons harga tidak dapat dihitung")
    out.append(_d("Respons harga terhadap flow", s4, ya, tdk, "return harga vs net flow, korelasi harian, absorpsi supply"))

    ya, tdk = [], []
    s5 = 50
    if pv:
        lab = pv["label_n"]
        s5 = {"SEHAT": 85, "KOREKSI_SEPI": 60, "NETRAL": 50, "NAIK_SEPI": 30, "DISTRIBUSI": 15}[lab]
        (ya if lab in ("SEHAT", "KOREKSI_SEPI") else tdk).append(f"harga-volume 5h: {lab} ({pv['chg_n']:+.1%}, vol {pv['rvol_n']:.1f}x)")
    if bo["n"]:
        gagal_r = bo["n_gagal"] / bo["n"]
        s5 += -25 if gagal_r >= 0.5 else 10 if bo["terakhir_valid"] else 0
        (tdk if gagal_r >= 0.5 else ya).append(f"breakout 60h: {bo['n_gagal']} gagal dari {bo['n']} (kembali di bawah level <=3 hari)")
    out.append(_d("Konfirmasi volume & struktur harga", s5, ya, tdk, "harga-volume, breakout valid vs gagal"))

    ya, tdk = [], []
    s6 = 100 - 18 * sup["n_hari_distribusi"] - 12 * sup["n_volume_besar_harga_diam"] - (25 * bo["n_gagal"] if bo["n"] else 0)
    if kons["net5_terakhir_lot"] < 0:
        s6 -= 20
        tdk.append(f"net 5h terakhir negatif ({kons['net5_terakhir_lot']:+,.0f} lot)")
    (tdk if sup["n_hari_distribusi"] else ya).append(f"{sup['n_hari_distribusi']} hari distribusi (volume besar, tanpa pembeli kelompok, harga lemah) dalam 20h")
    if sup["n_volume_besar_harga_diam"]:
        tdk.append(f"{sup['n_volume_besar_harga_diam']} hari volume besar tetapi harga diam tanpa pembeli kelompok (supply masih berat?)")
    out.append(_d("Risiko distribusi & pembatalan", s6, ya, tdk, "skor tinggi = risiko rendah"))
    return out


# ---------------------------------------------------------------- klasifikasi
def klasifikasi(dims: list[dict], m: pd.DataFrame, sup: dict, bo: dict, n_hari: int) -> tuple[str, str]:
    sk = {d["dimensi"]: d["skor"] for d in dims}
    d1, d2, d4, d6 = (sk["Konsistensi aliran"], sk["Konsentrasi & dominasi"], sk["Respons harga terhadap flow"], sk["Risiko distribusi & pembatalan"])
    get = lambda w, c: float(m.loc[m["jendela"] == w, c].iloc[0]) if (m["jendela"] == w).any() else np.nan  # noqa: E731
    net20, net60, ret20 = get(20, "net_lot"), get(60, "net_lot"), get(20, "ret_harga")
    if n_hari < 20 or np.isnan(net20):
        return K_TIDAK, "data kurang dari 20 hari bursa"
    if (net20 < 0 and (np.isnan(net60) or net60 <= 0) and d6 <= 55) or (d6 <= 35 and (sup["n_hari_distribusi"] >= 2 or (bo["n"] and bo["n_gagal"] >= 1))):
        return K_DISTRIBUSI, "net kelompok negatif di 20h/60h atau risiko distribusi tinggi dengan breakout gagal / hari distribusi"
    if sup["n_jual_besar"] >= 5 and sup["frac_absorpsi"] >= 0.5 and net20 >= 0 and (np.isnan(ret20) or ret20 >= -0.03):
        return K_ABSORPSI, "penjualan besar tidak menekan harga (>=50% hari jual besar harga bertahan) dan kelompok tidak net jual"
    if d1 >= 60 and d2 >= 50 and d4 >= 70 and d6 >= 60:
        return K_TERKONFIRMASI, "aliran konsisten, dominan, harga merespons, risiko distribusi rendah"
    if d1 >= 45 and net20 > 0:
        return K_TERINDIKASI, "aliran beli ada tetapi respons harga/dominasi belum cukup mengonfirmasi"
    return K_TIDAK, "bukti saling bertentangan atau tidak cukup"


TINGKAT = {K_TERKONFIRMASI: "Konfirmasi perilaku harga", K_TERINDIKASI: "Indikasi awal", K_DISTRIBUSI: "Indikasi awal",
           K_ABSORPSI: "Indikasi awal", K_TIDAK: "Tidak konklusif"}


def analisa(data: dict, harga: float, excl, konsentrasi: dict, pv: dict | None, summaries: dict | None = None,
            inv200: dict | None = None) -> dict:
    """data = parse_inventori(...) (+ opsional cum200); summaries = {jendela: broker_summary df} untuk estimasi modal."""
    price, cum = data["price"], data["cum"]
    acc, dist = pilih_grup(cum, excl)
    n = len(price)
    if not acc or n < 10:
        return dict(kategori=K_TIDAK, tingkat=TINGKAT[K_TIDAK], alasan="kelompok akumulator non-ritel tidak ditemukan / data < 10 hari",
                    grup=acc, grup_dist=dist, n_hari=n, dimensi=[], mendukung=[], menentang=[], keyakinan="Rendah", pembatalan=[], tabel=pd.DataFrame())
    net = _harian(cum)
    ga, gd = net[acc].sum(axis=1), (net[dist].sum(axis=1) if dist else pd.Series(0.0, index=net.index))
    ga_arr = pd.Series(ga.to_numpy(), index=range(n))
    gd_arr = pd.Series(gd.to_numpy(), index=range(n))
    m = metrik_jendela(price, ga_arr)
    if inv200 is not None and set(acc) <= set(inv200["cum"].columns) and len(inv200["price"]) >= 100:
        n200 = _harian(inv200["cum"])[acc].sum(axis=1)
        m200 = metrik_jendela(inv200["price"], pd.Series(n200.to_numpy(), index=range(len(n200))), (len(inv200["price"]),))
        m200["jendela"] = 200
        m = pd.concat([m, m200], ignore_index=True)
    kons, resp = konsistensi(ga_arr), respons_harga(price, ga_arr)
    sup, bo = supply(price, ga_arr, gd_arr), breakout(price, ga_arr)
    modal = estimasi_modal(summaries or {}, acc, harga)
    dims = dimensi(m, kons, resp, sup, bo, modal, konsentrasi, pv, price)
    kat, alasan = klasifikasi(dims, m, sup, bo, n)
    total = sum(d["skor"] * d["bobot"] for d in dims)
    mendukung = [x for d in dims for x in d["mendukung"]]
    menentang = [x for d in dims for x in d["menentang"]]
    low20 = float(price["low"].tail(20).min())
    batal = {K_TERKONFIRMASI: [f"kelompok net jual 3 hari berturut-turut atau net 5h berbalik negatif", f"close di bawah low 20h {low20:,.0f}",
                               "breakout gagal (close kembali di bawah level <= 3 hari)"],
             K_TERINDIKASI: ["net kelompok 20h berbalik negatif", f"close di bawah low 20h {low20:,.0f}", "tidak ada respons harga setelah 5 hari tambahan flow beli"],
             K_DISTRIBUSI: ["net beli konsisten 5 hari dengan harga merebut kembali level yang gagal", "hari distribusi berhenti 10 hari"],
             K_ABSORPSI: ["harga turun >3% saat penjualan besar berikutnya", "kelompok berbalik net jual"],
             K_TIDAK: ["data baru yang searah (flow konsisten + respons harga) atau pertentangan terselesaikan"]}[kat]
    keyakinan = "Sedang" if (kat != K_TIDAK and n >= 60 and len(m) >= 4) else "Rendah"
    return dict(kategori=kat, tingkat=TINGKAT[kat], alasan=alasan, grup=acc, grup_dist=dist, n_hari=n, tabel=m, konsistensi=kons, respons=resp,
                supply=sup, breakout=bo, modal=modal, dimensi=dims, skor_komposit=total, mendukung=mendukung, menentang=menentang,
                keyakinan=keyakinan, pembatalan=batal, nama={k: data["names"].get(k, k) for k in acc + dist},
                net_harian_10h=[float(x) / 100 for x in ga_arr.tail(10)], kum_10h=float(ga_arr.tail(10).sum()) / 100)


def posisi_tertekan(d: dict) -> tuple[str, str]:
    """'Indikasi posisi tertekan' hanya bila harga di bawah SELURUH rentang estimasi modal, aliran konsisten, dan respons harga negatif."""
    mo = d.get("modal") or {}
    if not mo.get("ada"):
        return "TIDAK DAPAT DISIMPULKAN", "referensi transaksi (estimasi) tidak tersedia"
    m = d["tabel"]
    r20 = float(m.loc[m["jendela"] == 20, "ret_harga"].iloc[0]) if (m["jendela"] == 20).any() else np.nan
    kons = d["konsistensi"]["frac_pos"]
    if mo["harga_vs_rentang"] == "di bawah rentang" and mo["harga_vs_pusat"] <= -0.05 and kons >= 0.6 and pd.notna(r20) and r20 < 0:
        return "INDIKASI POSISI TERTEKAN", (f"harga {mo['harga_vs_pusat']:+.1%} di bawah referensi transaksi (estimasi) ({mo['rendah']:,.0f}-{mo['tinggi']:,.0f}), "
                                           f"{kons:.0%} hari net beli, harga turun {r20:+.1%} sejak 20h; tetap estimasi, bukan posisi sebenarnya")
    if mo["harga_vs_pusat"] <= -0.05:
        return "TIDAK DAPAT DISIMPULKAN", "harga di bawah referensi transaksi (estimasi) tetapi konsistensi waktu/respons harga tidak mendukung"
    return "TIDAK TERINDIKASI", f"harga {mo['harga_vs_pusat']:+.1%} dari referensi transaksi (estimasi) ({mo['harga_vs_rentang']})"


def tujuh_pertanyaan(d: dict, tindakan: str) -> list[tuple[str, str]]:
    ada = bool(d["dimensi"])
    net20 = float(d["tabel"].loc[d["tabel"]["jendela"] == 20, "net_lot"].iloc[0]) if ada and (d["tabel"]["jendela"] == 20).any() else float("nan")
    pt = posisi_tertekan(d) if ada else ("-", "-")
    return [("1 Diamati", (f"net kelompok {','.join(d['grup'])} 20h {net20:+,.0f} lot; {d['konsistensi']['frac_pos']:.0%} hari net beli" if ada else "data tidak cukup")),
            ("2 Hipotesis", f"{d['kategori']} ({d['alasan']})"),
            ("3 Mendukung/menentang", f"{len(d['mendukung'])} bukti mendukung; {len(d['menentang'])} menentang"),
            ("4 Respons harga mengonfirmasi flow?", "ya" if d["kategori"] == K_TERKONFIRMASI else "belum / tidak" if ada else "tidak diketahui"),
            ("5 Probabilitas & payoff tervalidasi", "BELUM ADA (pola belum diuji out-of-sample)"),
            ("6 Kondisi pembatalan", "; ".join(d["pembatalan"]) or "-"),
            ("7 Tindakan", tindakan), ("(info) posisi tertekan", f"{pt[0]}: {pt[1]}")]


# ------------------------------------------------------------------- ambil
def ambil(client, kode: str, tgl_list: list, excl, investor: str = "all", market: str = "RG") -> dict:
    """4 panggilan: inventory 60h (harga + net kumulatif per broker), inventory 200h (kelompok tetap), summary 20h dan 60h (estimasi modal)."""
    t = list(tgl_list)
    if len(t) < 20:
        raise ad.AdimologyError("Data tanggal bursa kurang dari 20 hari untuk analisa bandar dinamis.")
    t60 = t[-60:]
    inv = parse_inventori(client.get(f"/analysis/inventory-chart/stock/{kode}", {
        "from": str(t60[0]), "to": str(t60[-1]), "scope": "vol", "investor": investor, "limit": 15, "market": market}))
    acc, dist = pilih_grup(inv["cum"], excl)
    inv200 = None
    if len(t) >= 100 and acc:
        try:
            inv200 = parse_inventori(client.get(f"/analysis/inventory-chart/stock/{kode}", {
                "from": str(t[-200:][0]), "to": str(t[-1]), "scope": "vol", "investor": investor, "limit": 15, "market": market,
                "filter": ",".join(acc + dist)}))
        except Exception:  # rentang panjang opsional
            inv200 = None
    summ = {}
    for w in (20, 60):
        try:
            summ[w] = ad.broker_summary(client, kode, t[-min(w, len(t))], t[-1], investor, market)
        except Exception:
            summ[w] = None
    return dict(inv=inv, inv200=inv200, summaries=summ, tgl_awal=t60[0], tgl_akhir=t60[-1])
