"""Penilaian bandarmologi berbasis bukti (menggantikan 'SETUP TERPENUHI' sebagai sinyal).

Backtest 4 saham (Jun 2025 - Okt 2026) menunjukkan sinyal Adimology muncul ~89% minggu dan tidak lebih
baik dari pembanding acak setelah fee. Prinsip di sini:
1. Default = tidak ada bukti. Status positif hanya bila beberapa bukti INDEPENDEN searah
   (materi: "bila semua analisa searah itulah hal yang menarik").
2. Jebakan didahulukan: distribusi terselubung, bandar keluar lewat nego, volume spike tanpa akumulasi,
   harga sudah dikejar, fase distribusi. Satu jebakan cukup untuk status WASPADA.
3. Semua aturan masih heuristik dan BELUM tervalidasi; app menampilkannya apa adanya.
Edukasi, bukan ajakan transaksi.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

YA, TIDAK, NETRAL, NA = "ya", "tidak", "netral", "tidak diukur"
KATEGORI = ("AKUMULASI KUAT", "AKUMULASI LEMAH / BELUM SEARAH", "TIDAK ADA BUKTI AKUMULASI", "WASPADA: TANDA DISTRIBUSI / JEBAKAN")

HASIL_BACKTEST = (
    "Backtest 4 saham (BBRI, ANTM, MDKA, ESSA), Jun 2025 - Okt 2026, 248 sinyal: target realistis Adimology tercapai "
    "23-32% (5-20 hari) sebelum SL; rata-rata hasil -0,3% s/d +0,1% per trade setelah fee, tidak berbeda nyata dari "
    "beli acak. Dengan orderbook asli (sejak Mar 2026) target terlalu jauh (median +15%) dan hanya tercapai 8-17%. "
    "Klaim 'offer tebal -> naik' tidak terbukti.")


# ------------------------------------------------------------------- fase pasar
def fase(df: pd.DataFrame) -> dict:
    """Fase heuristik dari harga harian: akumulasi / markup / distribusi / markdown / transisi."""
    d = df.dropna(subset=["close"]).tail(160).reset_index(drop=True)
    if len(d) < 60:
        return dict(fase="tidak diukur", alasan=f"data harga {len(d)} hari (< 60)")
    c = d["close"]
    ma20 = c.rolling(20).mean().iloc[-1]
    ma50 = c.rolling(50).mean().iloc[-1]
    last = float(c.iloc[-1])
    lo60, hi60 = float(d["low"].tail(60).min()), float(d["high"].tail(60).max())
    pos = (last - lo60) / (hi60 - lo60) if hi60 > lo60 else 0.5
    ret20 = last / float(c.iloc[-21]) - 1
    ret60 = last / float(c.iloc[-61]) - 1 if len(c) > 60 else ret20
    lebar20 = float(d["high"].tail(20).max()) / float(d["low"].tail(20).min()) - 1
    info = dict(posisi_60h=pos, ret20=ret20, ret60=ret60, lebar20=lebar20, ma20=float(ma20), ma50=float(ma50))
    if pos >= 0.8 and ret60 >= 0.25:
        return dict(info, fase="distribusi?", alasan=f"dekat puncak 60 hari setelah naik {ret60:+.0%}")
    if last < ma50 and ma20 < ma50 and ret20 < -0.05:
        return dict(info, fase="markdown", alasan=f"di bawah MA50, MA20 < MA50, turun {ret20:+.0%} 20 hari")
    ma50_naik = len(c) >= 70 and ma50 > c.rolling(50).mean().iloc[-21] * 1.02
    if last > ma50 and ma20 > ma50 and ret60 >= 0.10 and ma50_naik:
        return dict(info, fase="markup", alasan=f"tren naik: di atas MA50 yang menanjak, naik {ret60:+.0%} 60 hari")
    if (lebar20 <= 0.15 and abs(ret60) < 0.10) or (last < ma50 and ret20 > -0.05):
        return dict(info, fase="akumulasi?", alasan=f"konsolidasi / turun melambat (rentang 20 hari {lebar20:.0%}, "
                                                     f"{ret60:+.0%} 60 hari, posisi {pos:.0%} dari rentang 60 hari)")
    return dict(info, fase="transisi", alasan="belum ada pola jelas")


# ---------------------------------------------------------------- konsentrasi jual
def konsentrasi_jual(bs: pd.DataFrame, excl=()) -> dict:
    """Cermin konsentrasi beli: top 3 penjual vs top 15 pembeli (distribusi ke banyak 'semut')."""
    b = bs[~bs["code"].isin(set(excl))]
    jual = b[b["net_volume"] < 0].sort_values("net_volume")
    beli = b[b["net_volume"] > 0].sort_values("net_volume", ascending=False)
    top3 = float(-jual.head(3)["net_volume"].sum()) / 100
    top15 = float(beli.head(15)["net_volume"].sum()) / 100
    return dict(top3_jual_lot=top3, top15_beli_lot=top15, rasio=top3 / top15 if top15 > 0 else np.inf if top3 > 0 else np.nan,
                n_beli=int(len(beli)), n_jual=int(len(jual)), top3_kode=list(jual.head(3)["code"]))


# ------------------------------------------------------------------- penilaian
def _k(nama, status, poin, bukti, inti=False):
    return dict(kriteria=nama, status=status, poin=poin, bukti=bukti, inti=inti)


def penilaian(r, a: dict | None = None, rk: dict | None = None) -> dict:
    """r = adimology.Result, a = aliran (opsional), rk = teknikal.rekomendasi (opsional)."""
    excl = set(r.params.broker_dikecualikan)
    ks = []
    jebakan = []

    # 1. konsentrasi beli (raksasa menelan barang dari banyak semut)
    k = r.konsentrasi or {}
    st = k.get("status")
    poin = {"AKUMULASI TERKONSENTRASI": 1, "AKUMULASI SEBAGIAN": 0}.get(st, -1)
    ks.append(_k("Konsentrasi beli (top 3 vs top 15 penjual)", YA if poin > 0 else NETRAL if poin == 0 else TIDAK, poin,
                 f"{st}: {k.get('top3_beli_lot', 0):,.0f} vs {k.get('top15_jual_lot', 0):,.0f} lot; "
                 f"{k.get('n_beli', 0)} broker beli vs {k.get('n_jual', 0)} jual", inti=True))

    # 2. persistensi bandar lintas window
    if a and a.get("konsistensi") is not None and not a["konsistensi"].empty:
        kc = a["konsistensi"]
        sama, n = int(kc["sama_dengan_bandar"].sum()), len(kc)
        poin = 1 if sama >= 3 else -1 if sama <= 1 else 0
        ks.append(_k("Bandar konsisten 1/3/7/20 hari", YA if poin > 0 else NETRAL if poin == 0 else TIDAK, poin,
                     f"{sama} dari {n} window akumulator teratasnya sama", inti=True))
    else:
        ks.append(_k("Bandar konsisten 1/3/7/20 hari", NA, 0, "aliran lanjutan tidak aktif / gagal", inti=True))

    # 3. tidak sedang distribusi (hari terakhir & 3 hari terakhir)
    h = r.harian["net_lot"].astype(float)
    tiga_jual = len(h) >= 3 and bool((h.tail(3) < 0).all())
    if r.distribusi or tiga_jual:
        ks.append(_k("Tidak ada distribusi bandar", TIDAK, -1,
                     "net bandar negatif 3 hari berturut-turut" if tiga_jual else f"jual hari terakhir {r.net_terakhir:,.0f} lot "
                     f"(≥ {r.params.distribusi_pct:.0%} barang)", inti=True))
    elif r.bandar.get("net_lot", 0) <= 0:
        ks.append(_k("Tidak ada distribusi bandar", TIDAK, -1, "tidak ada akumulator bersih di window", inti=True))
    else:
        ks.append(_k("Tidak ada distribusi bandar", YA, 1, f"net hari terakhir {r.net_terakhir:+,.0f} lot", inti=True))

    # 4. harga belum dikejar (dekat modal bandar)
    fl = r.floating
    poin = 1 if fl <= 0.05 else -1 if fl > 0.15 else 0
    ks.append(_k("Harga dekat modal bandar", YA if poin > 0 else NETRAL if poin == 0 else TIDAK, poin,
                 f"harga {fl:+.1%} dari avg bandar {r.bandar['avg']:,.1f}"))
    if fl > 0.15:
        jebakan.append(f"Harga sudah {fl:+.0%} di atas modal bandar: membeli sekarang = memberi exit untuk bandar.")

    # 5. asing searah
    if a and a.get("asing"):
        s = a["asing"]
        poin = 1 if s["net_value"] > 0 else -1 if s["net_value"] < 0 else 0
        ks.append(_k("Asing searah (net buy)", YA if poin > 0 else NETRAL if poin == 0 else TIDAK, poin,
                     f"{s['status']} {s['net_value'] / 1e9:+,.2f} M"))
    else:
        ks.append(_k("Asing searah (net buy)", NA, 0, "aliran lanjutan tidak aktif / gagal"))

    # 6. nego: bandar keluar lewat crossing?
    if a and a.get("nego"):
        n = a["nego"]
        if n["status"] == "BANDAR JUAL DI NEGO":
            ks.append(_k("Tidak keluar lewat nego", TIDAK, -1, f"net bandar di nego {n['bandar_net_lot']:+,.0f} lot"))
            jebakan.append(f"Bandar menjual {abs(n['bandar_net_lot']):,.0f} lot di pasar negosiasi: akumulasi di pasar reguler "
                           "bisa jadi kamuflase.")
        else:
            ks.append(_k("Tidak keluar lewat nego", YA if n["vol_lot"] == 0 or n["bandar_net_lot"] >= 0 else NETRAL, 0,
                         f"{n['status']} ({n['porsi']:.0%} volume RG)"))
    else:
        ks.append(_k("Tidak keluar lewat nego", NA, 0, "aliran lanjutan tidak aktif / gagal"))

    # 7. fase struktur harga
    src = rk["ind"] if rk and rk.get("ind") is not None else r.chart
    f = fase(src)
    poin = {"akumulasi?": 1, "markup": 0, "transisi": 0, "distribusi?": -1, "markdown": -1}.get(f["fase"], 0)
    ks.append(_k("Fase harga mendukung", NA if f["fase"] == "tidak diukur" else YA if poin > 0 else NETRAL if poin == 0 else TIDAK,
                 poin, f"{f['fase']}: {f['alasan']}"))

    # 8. distribusi terselubung: harga naik di window tapi penjual terkonsentrasi
    w = r.chart[r.chart["tgl"] >= r.tgl_awal]
    naik_window = float(w["close"].iloc[-1] / w["open"].iloc[0] - 1) if len(w) else 0.0
    kj = konsentrasi_jual(r.bs, excl)
    if naik_window > 0.05 and kj["rasio"] >= 1:
        jebakan.append(f"Distribusi terselubung: harga naik {naik_window:+.0%} di window, tetapi 3 penjual teratas "
                       f"({', '.join(kj['top3_kode'])}) melepas {kj['top3_jual_lot']:,.0f} lot ≥ beli 15 pembeli teratas.")

    # 9. volume spike tanpa akumulasi (pancingan / pump)
    v = r.chart["volume"].astype(float)
    if len(v) >= 21:
        rv = float(v.iloc[-1] / v.iloc[-21:-1].median()) if v.iloc[-21:-1].median() > 0 else np.nan
        if rv >= 3 and r.net_terakhir <= 0:
            jebakan.append(f"Volume hari terakhir {rv:.1f}x median 20 hari tetapi bandar tidak menambah barang "
                           f"({r.net_terakhir:+,.0f} lot): ciri pancingan ritel.")
    if f["fase"] == "distribusi?" and r.net_terakhir < 0:
        jebakan.append("Harga di area puncak setelah reli besar dan bandar mulai menjual: pola fase distribusi.")

    # 10. afiliasi (informasi, bukan poin)
    info = []
    if a and a.get("afiliasi") and a["afiliasi"]["cocok"]:
        x = a["afiliasi"]["cocok"][0]
        info.append(f"Broker bandar dipakai pemegang saham ≥5% ({x['name']}, {x['next_pct']:.1f}%): bisa insider akumulasi "
                    "atau grup yang mengatur harga. Cek keterbukaan informasi.")

    skor = sum(x["poin"] for x in ks)
    diukur = [x for x in ks if x["status"] != NA]
    inti_gagal = any(x["inti"] and x["poin"] < 0 for x in ks)
    inti_na = any(x["inti"] and x["status"] == NA for x in ks)
    if jebakan:
        kategori = KATEGORI[3]
    elif skor >= 4 and len(diukur) >= 6 and not inti_gagal and not inti_na:
        kategori = KATEGORI[0]
    elif skor >= 2 and not inti_gagal:
        kategori = KATEGORI[1]
    else:
        kategori = KATEGORI[2]
    return dict(kategori=kategori, skor=skor, skor_maks=len(diukur), kriteria=ks, jebakan=jebakan, info=info,
                fase=f, naik_window=naik_window, konsentrasi_jual=kj,
                lengkap=not inti_na and len(diukur) == len(ks))


def tabel(p: dict) -> pd.DataFrame:
    ikon = {YA: "🟢 ya", TIDAK: "🔴 tidak", NETRAL: "⚪ netral", NA: "– tidak diukur"}
    return pd.DataFrame([{"Kriteria": x["kriteria"] + (" *" if x["inti"] else ""), "Status": ikon[x["status"]],
                          "Bukti": x["bukti"]} for x in p["kriteria"]])
