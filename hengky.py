"""Aturan disiplin trading dari materi Ko Hengky (Remora), diterjemahkan jadi pemeriksaan otomatis.

Sumber: file 'Nasihat_Hengki_Remora.md' (transkrip 4 sesi; kode [A] [B] [D2] [D5] mengacu ke sesi di file itu).
Catatan kritis dari file itu berlaku: transkrip hasil suara-ke-teks (angka & kode broker belum diverifikasi), klaim
akurasi pengajar bukan hasil uji, ada konflik kepentingan (kelas berbayar) dan survivorship bias. Semua ambang di sini
adalah ENCODING HEURISTIK kami dan BELUM divalidasi backtest. Edukasi, bukan ajakan transaksi.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

import adimology as ad
import bandar as bd

YA, TIDAK, NETRAL, NA = bd.YA, bd.TIDAK, bd.NETRAL, bd.NA

CATATAN_KRITIS = (
    "Materi bersumber dari transkrip suara-ke-teks: angka, kode broker, dan jam bisa keliru. Klaim akurasi pengajar "
    "(price-volume ~55%, + analisa transaksi 70-80%) adalah klaim pribadi, bukan hasil uji. Ada konflik kepentingan "
    "(pengajar menjual kelas) dan survivorship bias pada contoh saham. Ambang di app ini adalah penerjemahan heuristik "
    "kami dan belum divalidasi backtest.")

ATURAN_DISIPLIN = [
    "Pasar selalu benar: kamu yang menyesuaikan, bukan sebaliknya. Pola berubah? Cari pola baru, jangan menyalahkan bandar.",
    "Entry reason = exit reason: jual saat alasan beli gugur (level patah), bukan saat 'berharap'.",
    "Jangan average down kecuali sudah ada rencana; jangan beli saham turun hanya karena orang menyuruh beli.",
    "Hanya trade saat uptrend dan hanya saham likuid. Beli di support atau breakout, jangan di resistance.",
    "Probabilitas, bukan kemungkinan: jangan main 50:50. Tanpa setup, jangan trade (hindari overtrading).",
    "Jangan kembalikan profit. Fee ikut dihitung: scalping tipis bisa habis di fee.",
    "Info bagus + transaksi jelek = ada yang mencari exit liquidity. Info orang dalam tidak boleh menimpa setup.",
]

PV_TEKS = {"SEHAT": "Sehat: harga naik + volume naik",
           "NAIK_SEPI": "Waspada: harga naik tanpa volume (tidak likuid / gorengan)",
           "KOREKSI_SEPI": "Koreksi tanpa volume: belum ada distribusi",
           "DISTRIBUSI": "Distribusi: harga turun + volume besar",
           "NETRAL": "Netral"}


# ------------------------------------------------------------------ harga-volume
def klasifikasi_pv(chg, rvol, naik: float = 0.01, tinggi: float = 1.5, rendah: float = 0.7) -> str:
    if chg is None or rvol is None or pd.isna(chg) or pd.isna(rvol):
        return "NETRAL"
    if chg >= naik:
        return "SEHAT" if rvol >= tinggi else "NAIK_SEPI" if rvol <= rendah else "NETRAL"
    if chg <= -naik:
        return "DISTRIBUSI" if rvol >= tinggi else "KOREKSI_SEPI" if rvol <= rendah else "NETRAL"
    return "NETRAL"


def pv_baca(df: pd.DataFrame, n: int = 5) -> dict | None:
    """Wyckoff ala materi: harga-volume per hari (n terakhir), selling climax, dan hari 'candle kecil volume besar'."""
    d = df.dropna(subset=["close", "volume"]).tail(80).reset_index(drop=True)
    if len(d) < 30:
        return None
    med = d["volume"].shift(1).rolling(20, min_periods=15).median()
    d["rvol"] = d["volume"] / med.replace(0, np.nan)
    d["chg"] = d["close"].pct_change()
    d["label"] = [klasifikasi_pv(c, v) for c, v in zip(d["chg"], d["rvol"])]
    harian = d.tail(n)[["tgl", "close", "chg", "rvol", "label"]].reset_index(drop=True)
    chg_n = float(d["close"].iloc[-1] / d["close"].iloc[-n - 1] - 1)
    rvol_n = float(d["rvol"].tail(n).mean())
    label_n = klasifikasi_pv(chg_n, rvol_n, naik=0.02)

    climax = None
    for i in range(21, len(d)):
        ret20 = d["close"].iloc[i - 1] / d["close"].iloc[i - 21] - 1
        if d["rvol"].iloc[i] >= 3 and d["chg"].iloc[i] <= -0.03 and ret20 <= -0.15:
            low_setelah = float(d["low"].iloc[i + 1:].min()) if i + 1 < len(d) else np.nan
            climax = dict(tgl=d["tgl"].iloc[i], rvol=float(d["rvol"].iloc[i]), chg=float(d["chg"].iloc[i]),
                          ret20_sebelum=float(ret20), hari_sejak=len(d) - 1 - i, low=float(d["low"].iloc[i]),
                          low_setelah=low_setelah,
                          low_bertahan=None if np.isnan(low_setelah) else bool(low_setelah >= d["low"].iloc[i] * 0.995))
    t = d.tail(20)
    kecil = t[(t["rvol"] >= 3) & ((t["high"] - t["low"]) / t["close"] <= 0.02)]
    return dict(harian=harian, label_n=label_n, chg_n=chg_n, rvol_n=rvol_n, climax=climax,
                crossing=[x for x in kecil["tgl"]], n=n)


# ------------------------------------------------------------------- bid-offer
def _sisi(lots: dict, freq: dict, bid: bool) -> dict:
    harga = sorted(lots, reverse=bid)
    df = pd.DataFrame({"harga": harga, "lot": [lots[h] for h in harga], "freq": [freq.get(h, 0.0) for h in harga]})
    df["lot_per_antrean"] = np.where(df["freq"] > 0, df["lot"] / df["freq"].replace(0, np.nan), np.nan)
    med = float(df["lot"].median())
    dinding = [dict(harga=float(r.harga), lot=float(r.lot), kali=float(r.lot / med)) for r in df.itertuples()
               if len(df) >= 4 and med > 0 and r.lot >= 5 * med]
    vc = df["lot"].value_counts()
    ulang = set(vc[vc >= 3].index)
    bulat = set(df.loc[(df["lot"] >= 1000) & (df["lot"] % 1000 == 0), "lot"])
    cantik = [dict(harga=float(r.harga), lot=float(r.lot)) for r in df.itertuples()
              if r.lot in ulang or (len(df.loc[df["lot"].isin(bulat)]) >= 3 and r.lot in bulat)]
    tf = float(df["freq"].sum())
    return dict(df=df, n=len(df), total_lot=float(df["lot"].sum()), total_freq=tf,
                lpa=float(df["lot"].sum() / tf) if tf > 0 else np.nan, dinding=dinding, cantik=cantik)


def baca_orderbook(ob: dict | None) -> dict | None:
    """Bid-offer ala materi: lot per antrean, angka cantik (algo), dinding, likuiditas '3 papan' (detail: verifikasi)."""
    if not ob:
        return None
    b = _sisi(ob["bids"], ob.get("bids_freq", {}), True)
    o = _sisi(ob["offers"], ob.get("offers_freq", {}), False)
    return dict(bid=b, offer=o, rasio_offer_bid=ob["total_offer"] / ob["total_bid"] if ob["total_bid"] else np.nan,
                likuid_3_papan=b["n"] >= 3 and o["n"] >= 3, tutup=ob.get("n_levels", 0) <= 4,
                freq_tersedia=b["total_freq"] > 0 and o["total_freq"] > 0)


# ------------------------------------------------------------------ rencana entry
def rencana(rk: dict | None, fee_b: float, fee_j: float) -> dict | None:
    """Dua rencana: beli di support, atau breakout. Stop 1 tick di bawah level (entry reason = exit reason)."""
    if not rk:
        return None
    harga = rk["harga"]
    sup = rk["sup"][0]["harga"] if rk["sup"] else None
    res = [c["harga"] for c in rk["res"]]
    out = dict(harga=harga, jarak_res=(res[0] / harga - 1) if res else None,
               impas=harga * (1 + fee_b) / (1 - fee_j), support=None, breakout=None)
    out["impas_pct"] = out["impas"] / harga - 1
    if sup:
        stop = sup - ad.fraksi(sup)
        e = (rk["entry_lo"] + rk["entry_hi"]) / 2 if rk["zona_valid"] else sup
        out["support"] = dict(level=sup, stop_tick=stop, entry_ref=e, target=rk["target"], target_nama=rk["target_nama"],
                              rr=(rk["target"] - e) / max(e - stop, 1e-9), stop_atr=sup - rk["atr"],
                              rr_atr=(rk["target"] - e) / max(e - (sup - rk["atr"]), 1e-9))
    if res:
        lv = res[0]
        entry, stop = lv + ad.fraksi(lv), lv - ad.fraksi(lv)
        tgt = res[1] if len(res) > 1 else (lv + (lv - sup) if sup else lv * 1.05)
        if tgt <= entry:
            tgt = entry * 1.05
        vol = rk["ind"]["volume"].tail(20).median() if rk.get("ind") is not None else np.nan
        out["breakout"] = dict(level=lv, entry=entry, stop_tick=stop, target=tgt, rr=(tgt - entry) / max(entry - stop, 1e-9),
                               stop_atr=lv - rk["atr"], rr_atr=(tgt - entry) / max(entry - (lv - rk["atr"]), 1e-9),
                               volume_konfirmasi_lot=float(vol) * 1.5 / 100 if pd.notna(vol) else np.nan)
    return out


# ---------------------------------------------------------------------- SID & pra-open
def sid_trend(client, kode: str, bulan: int = 3) -> dict:
    """Jumlah investor (SID) bulanan: SID turun saat harga sideways = ritel keluar (contoh BSBK, sesi B)."""
    raw = client.get(f"/analysis/shareholder/number/{kode}", {})
    df = pd.DataFrame(raw)
    if df.empty or "value" not in df:
        return dict(status="TIDAK DIUKUR", n=0)
    df["tgl"] = pd.to_datetime(df["date"], utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["price"] = pd.to_numeric(df.get("price"), errors="coerce")
    df = df.dropna(subset=["value"]).sort_values("tgl")
    if len(df) < bulan + 1:
        return dict(status="TIDAK DIUKUR", n=len(df))
    a, b = df.iloc[-bulan - 1], df.iloc[-1]
    chg = float(b["value"] / a["value"] - 1)
    pchg = float(b["price"] / a["price"] - 1) if pd.notna(a["price"]) and a["price"] > 0 and pd.notna(b["price"]) else np.nan
    if chg <= -0.03 and (np.isnan(pchg) or abs(pchg) <= 0.10):
        st = "SID TURUN SAAT SIDEWAYS"
    elif chg <= -0.03:
        st = "SID TURUN, HARGA NAIK"
    elif chg >= 0.05 and not np.isnan(pchg) and pchg < -0.05:
        st = "SID NAIK SAAT HARGA TURUN"
    else:
        st = "SID STABIL"
    return dict(status=st, n=len(df), dari=a["tgl"].date(), sampai=b["tgl"].date(), sid_awal=float(a["value"]),
                sid_akhir=float(b["value"]), chg=chg, price_chg=pchg)


def pra_open(client, kode: str) -> dict:
    d = client.get(f"/analysis/intraday-data/{kode}", {"market": "RG"})
    if isinstance(d, dict) and isinstance(d.get("data"), dict):
        d = d["data"]
    if not isinstance(d, dict):
        raise ad.AdimologyError("Data intraday tidak tersedia.")
    n = lambda k: float(d[k]) if d.get(k) not in (None, "") else np.nan  # noqa: E731
    iep, prev = n("iep"), n("prev")
    return dict(iep=iep, iev_lot=n("iev") / 100 if pd.notna(n("iev")) else np.nan, prev=prev,
                gap=(iep / prev - 1) if iep > 0 and prev > 0 else np.nan, open=n("open"), last=n("close"),
                bid=n("bid_price"), bid_lot=n("bid_lot"), offer=n("offer_price"), offer_lot=n("offer_lot"),
                waktu=dt.datetime.now(dt.timezone(dt.timedelta(hours=7))).strftime("%H:%M:%S WIB"), valid=bool(iep > 0))


# ----------------------------------------------------------------------- profil
def uptrend(rk: dict | None) -> tuple[str, str, bool | None]:
    if not rk:
        return NA, "data teknikal tidak tersedia", None
    ma50 = rk["ma"].get(50)
    if ma50 is None:
        return NA, "MA50 belum tersedia (data < 50 hari)", None
    s = rk["ind"]["ma50"].dropna()
    slope = float(s.iloc[-1] / s.iloc[-11] - 1) if len(s) >= 11 else np.nan
    ok = rk["harga"] > ma50 and (np.isnan(slope) or slope > 0)
    teks = (f"harga {rk['harga']:,.0f} {'di atas' if rk['harga'] > ma50 else 'di bawah'} MA50 {ma50:,.0f}; "
            f"MA50 {'naik' if slope > 0 else 'turun'} {slope:+.1%} dalam 10 hari" if not np.isnan(slope) else
            f"harga {rk['harga']:,.0f} vs MA50 {ma50:,.0f}")
    return (YA if ok else TIDAK), teks, ok


def profil(rk: dict | None, pv: dict | None, kons: str | None, sid: dict | None) -> dict:
    """Trading (uptrend) vs kandidat moonstock (selling climax -> akumulasi, tahan 3-6 bulan; sesi A, D5)."""
    _, teks_up, up = uptrend(rk)
    cl = pv["climax"] if pv else None
    syarat = {"selling climax terdeteksi": bool(cl), "low climax bertahan": bool(cl and cl["low_bertahan"]),
              "beli terkonsentrasi": kons in ("AKUMULASI TERKONSENTRASI", "AKUMULASI SEBAGIAN"),
              "SID turun saat sideways": bool(sid and sid.get("status") in ("SID TURUN SAAT SIDEWAYS",))}
    if up:
        return dict(profil="TRADING (uptrend)", syarat=syarat, alasan=teks_up)
    kurang = [k for k, v in syarat.items() if not v]
    if not kurang:
        return dict(profil="KANDIDAT MOONSTOCK", syarat=syarat,
                    alasan="selling climax, low bertahan, beli terkonsentrasi, SID turun: pantau, tahan 3-6 bulan; "
                           "agresif hanya setelah institusi mengambil barang ritel")
    return dict(profil="BUKAN SETUP TRADING", syarat=syarat,
                alasan=f"tidak uptrend ({teks_up}); moonstock belum terkonfirmasi, kurang: {', '.join(kurang)}")


# --------------------------------------------------------------------- checklist
def _c(nama, status, bukti, inti=False, sumber=""):
    return dict(kriteria=nama, status=status, bukti=bukti, inti=inti, sumber=sumber)


def checklist(r, rk, pv, obb, rn, prof, nilai, min_nilai_m: float = 5.0) -> dict:
    ks = []
    st_up, teks_up, _ = uptrend(rk)
    ks.append(_c("Uptrend (hanya trade saat uptrend)", st_up, teks_up, True, "D2, D5"))

    df = rk["ind"] if rk and rk.get("ind") is not None else r.chart
    nilai_hari = float((df["close"] * df["volume"]).tail(20).median()) if len(df) >= 10 else np.nan
    lot_pos, lpp = r.plan.get("lot", 0), r.lot_per_papan
    frac = lot_pos / lpp if lpp and lpp > 0 else np.nan
    if pd.isna(nilai_hari):
        ks.append(_c("Likuid (trade hanya saham likuid)", NA, "data volume kurang", True, "D5"))
    elif nilai_hari < min_nilai_m * 1e9:
        ks.append(_c("Likuid (trade hanya saham likuid)", TIDAK,
                     f"nilai transaksi median {nilai_hari / 1e9:,.2f} M/hari < batas {min_nilai_m:g} M", True, "D5"))
    elif pd.notna(frac) and frac > 0.2:
        ks.append(_c("Likuid (trade hanya saham likuid)", NETRAL,
                     f"nilai {nilai_hari / 1e9:,.1f} M/hari cukup, tetapi posisi {lot_pos:,} lot = {frac:.0%} lot per papan "
                     f"({lpp:,.0f}): sulit keluar cepat", True, "D5"))
    else:
        ks.append(_c("Likuid (trade hanya saham likuid)", YA, f"nilai transaksi median {nilai_hari / 1e9:,.1f} M/hari", True, "D5"))

    if not rk or not rn:
        ks.append(_c("Beli di support / breakout, bukan di resistance", NA, "data teknikal tidak tersedia", True, "A, D2, D5"))
    else:
        di_area = rk["zona_valid"] and rk["entry_lo"] <= rk["harga"] <= rk["entry_hi"]
        if rn["jarak_res"] is not None and rn["jarak_res"] <= 0.015:
            ks.append(_c("Beli di support / breakout, bukan di resistance", TIDAK,
                         f"harga hanya {rn['jarak_res']:.1%} di bawah resistance: jangan beli di resistance", True, "A, D2, D5"))
        elif di_area:
            ks.append(_c("Beli di support / breakout, bukan di resistance", YA,
                         f"harga di area support {rk['entry_lo']:,.0f}-{rk['entry_hi']:,.0f}", True, "A, D2, D5"))
        else:
            b = f"; breakout di atas {rn['breakout']['level']:,.0f}" if rn["breakout"] else ""
            s_ = f"support {rn['support']['level']:,.0f}" if rn["support"] else "support belum jelas"
            ks.append(_c("Beli di support / breakout, bukan di resistance", NETRAL,
                         f"harga di tengah: tunggu, pasang alert di {s_}{b}", True, "A, D2, D5"))

    if rn and rn["support"]:
        s_ = rn["support"]
        ks.append(_c("Level pembatal jelas (entry reason = exit reason)", YA,
                     f"jual bila close di bawah support {s_['level']:,.0f} (stop 1 tick: {s_['stop_tick']:,.0f}); "
                     f"R/R {s_['rr']:.2f} (stop ATR: {s_['rr_atr']:.2f})", True, "D2, D5"))
    else:
        ks.append(_c("Level pembatal jelas (entry reason = exit reason)", TIDAK if rk else NA, "tidak ada support terdekat", True, "D2, D5"))

    if pv:
        n_dist = int((pv["harian"]["label"] == "DISTRIBUSI").sum())
        st = TIDAK if pv["label_n"] in ("NAIK_SEPI", "DISTRIBUSI") or n_dist >= 2 else YA if pv["label_n"] == "SEHAT" else NETRAL
        ks.append(_c("Harga-volume sehat (5 hari)", st, f"{PV_TEKS[pv['label_n']]} ({pv['chg_n']:+.1%}, volume {pv['rvol_n']:.1f}x); "
                                                       f"{n_dist} hari distribusi", False, "D2"))
    else:
        ks.append(_c("Harga-volume sehat (5 hari)", NA, "data harga kurang", False, "D2"))

    if not obb or obb["tutup"]:
        ks.append(_c("Bid-offer terbaca (likuid 3 papan, tanpa dinding)", NA, "orderbook ≤ 2 level (pasar tutup?) / tidak tersedia", False, "D5"))
    else:
        dinding = obb["offer"]["dinding"][:1] + obb["bid"]["dinding"][:1]
        if not obb["likuid_3_papan"]:
            ks.append(_c("Bid-offer terbaca (likuid 3 papan, tanpa dinding)", TIDAK, "kurang dari 3 papan di salah satu sisi", False, "D5"))
        elif dinding:
            d = dinding[0]
            ks.append(_c("Bid-offer terbaca (likuid 3 papan, tanpa dinding)", NETRAL,
                         f"dinding {d['lot']:,.0f} lot di {d['harga']:,.0f} ({d['kali']:.0f}x median): cek apakah di-reload / ditarik", False, "A, D2"))
        else:
            ks.append(_c("Bid-offer terbaca (likuid 3 papan, tanpa dinding)", YA, "3 papan di kedua sisi, tidak ada dinding mencolok", False, "D5"))

    kat = nilai["kategori"]
    ks.append(_c("Analisa transaksi (bandar) tidak melawan", YA if kat == "AKUMULASI KUAT" else TIDAK if kat.startswith("WASPADA") else NETRAL,
                 f"penilaian bandarmologi: {kat}", False, "A, B"))
    if rk and rn:
        tp_pct = rk["tp1"] / rk["harga"] - 1
        batas = 3 * (r.params.fee_beli + r.params.fee_jual)
        ks.append(_c("Target menutup fee (≥ 3x biaya pulang-pergi)", YA if tp_pct >= batas else TIDAK,
                     f"TP1 {tp_pct:+.1%} vs 3x fee {batas:.1%}; harga impas {rn['impas']:,.0f} ({rn['impas_pct']:+.2%})", False, "A"))
    if pv:
        spike = pv["chg_n"] >= 0.15 and kat != "AKUMULASI KUAT"
        ks.append(_c("Tidak ada tanda exit liquidity (info bagus + transaksi jelek)", TIDAK if spike else YA,
                     f"harga {pv['chg_n']:+.0%} dalam {pv['n']} hari; bukti bandar {kat}" if spike else
                     f"harga {pv['chg_n']:+.0%} dalam {pv['n']} hari", False, "A, B"))
    inti = [k for k in ks if k["inti"]]
    lolos = all(k["status"] == YA for k in inti)
    n_na = sum(k["status"] == NA for k in inti)
    if lolos:
        hasil = "ATURAN INTI TERPENUHI"
    elif n_na:
        hasil = "BELUM BISA DINILAI (data aturan inti kurang)"
    else:
        hasil = f"TAHAN: {sum(k['status'] != YA for k in inti)} dari {len(inti)} aturan inti belum terpenuhi"
    return dict(kriteria=ks, lolos_inti=lolos, hasil=hasil)


def analisa(client, r, rk: dict | None, nilai: dict, min_nilai_m: float = 5.0) -> dict:
    df = rk["ind"] if rk and rk.get("ind") is not None else r.chart
    pv = pv_baca(df)
    obb = baca_orderbook(r.ob)
    rn = rencana(rk, r.params.fee_beli, r.params.fee_jual)
    try:
        sid = sid_trend(client, r.params.kode) if client is not None else None
    except Exception:  # endpoint SID bisa tidak tersedia di paket API tertentu
        sid = None
    prof = profil(rk, pv, (r.konsentrasi or {}).get("status"), sid)
    cl = checklist(r, rk, pv, obb, rn, prof, nilai, min_nilai_m)
    return dict(pv=pv, ob=obb, rencana=rn, sid=sid, profil=prof, checklist=cl)


def tabel(cl: dict) -> pd.DataFrame:
    ikon = {YA: "🟢 ya", TIDAK: "🔴 tidak", NETRAL: "⚪ netral", NA: "– tidak diukur"}
    return pd.DataFrame([{"Aturan": k["kriteria"] + (" *" if k["inti"] else ""), "Status": ikon[k["status"]],
                          "Bukti": k["bukti"], "Sumber": k["sumber"]} for k in cl["kriteria"]])
