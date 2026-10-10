"""Adimology Calculator — Streamlit. Edukasi, bukan ajakan transaksi."""
from __future__ import annotations

import datetime as dt
import hmac
import os
from dataclasses import replace
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlalchemy as sa
import streamlit as st

import adimology as ad
import aliran as al
import bandar as bd
import fundamental as fd
import intraday as idt
import laporan as lp
import teknikal as tk

st.set_page_config(page_title="Adimology Calculator", page_icon="📈", layout="wide")


def get_secret(name: str) -> str | None:
    try:
        v = st.secrets.get(name)
        if v:
            return str(v)
    except Exception:
        pass
    return os.environ.get(name) or None


# ------------------------------------------------------------------------------ login
def login_gate() -> None:
    """Halaman login. Email & password diatur di Secrets (APP_LOGIN_EMAIL, APP_LOGIN_PASSWORD)."""
    if st.session_state.get("_auth"):
        return
    email_ok, pw_ok = get_secret("APP_LOGIN_EMAIL"), get_secret("APP_LOGIN_PASSWORD")
    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        if not (email_ok and pw_ok):
            st.error("Login belum diatur. Isi APP_LOGIN_EMAIL dan APP_LOGIN_PASSWORD di Secrets.")
            st.stop()
        with st.form("login"):
            email = st.text_input("Email")
            pw = st.text_input("Password", type="password")
            masuk = st.form_submit_button("Masuk", type="primary", width="stretch")
        if masuk:
            if (hmac.compare_digest(email.strip().lower().encode(), email_ok.strip().lower().encode())
                    and hmac.compare_digest(pw.encode(), pw_ok.encode())):
                st.session_state["_auth"] = True
                st.rerun()
            st.error("Email atau password salah.")
    st.stop()


login_gate()


# ------------------------------------------------------------------ API key & cache
def get_api_key() -> str | None:
    return get_secret("INVEZGO_API_KEY") or st.session_state.get("_key_input") or None


@st.cache_data(ttl=300, show_spinner=False)
def _cached_fetch(path: str, params_items: tuple, _key: str):
    return ad.fetch_json(_key, path, params_items)


@st.cache_data(ttl=15, show_spinner=False)
def _cached_fetch_rt(path: str, params_items: tuple, _key: str):
    return ad.fetch_json(_key, path, params_items)


class CachedClient(ad.InvezgoClient):
    """Cache 5 menit untuk data EOD, 15 detik untuk orderbook (realtime)."""

    def _fetch(self, path, params_items):
        fn = _cached_fetch_rt if "order-book" in path else _cached_fetch
        return fn(path, params_items, _key=self.api_key)


# --------------------------------------------------------------------------- helpers
def fmt(x, d=0):
    return "-" if x is None or pd.isna(x) else f"{x:,.{d}f}"


def make_chart(r: ad.Result) -> go.Figure:
    df = r.chart.tail(40)
    x = df["tgl"].astype(str)
    fig = go.Figure(go.Candlestick(x=x, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                                   name=r.params.kode, increasing_line_color="#16a34a", decreasing_line_color="#dc2626"))
    lines = [(r.bandar["avg"], f"Avg bandar {r.bandar['code']}", "#2563eb", "dash"),
             (r.calc["t_real_r"], "Target realistis", "#d97706", "dot"),
             (r.calc["t_max_r"], "Target max", "#9333ea", "dot"),
             (r.plan["sl"], "SL (low window)", "#6b7280", "dashdot")]
    for y, name, color, dash in lines:
        fig.add_hline(y=y, line_color=color, line_dash=dash, line_width=1,
                      annotation_text=f"{name} {y:,.0f}", annotation_position="top left",
                      annotation_font_size=11)
    fig.update_layout(height=470, margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False,
                      xaxis_type="category", showlegend=False)
    return fig


def harian_chart(r: ad.Result) -> go.Figure:
    h = r.harian
    colors = ["#16a34a" if v >= 0 else "#dc2626" for v in h["net_lot"]]
    fig = go.Figure()
    fig.add_bar(x=h["tgl"].astype(str), y=h["net_lot"], marker_color=colors, name="Net harian (lot)")
    fig.add_scatter(x=h["tgl"].astype(str), y=h["net_kumulatif_lot"], mode="lines+markers",
                    name="Kumulatif (lot)", line=dict(color="#2563eb"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), xaxis_type="category",
                      legend=dict(orientation="h", y=1.12))
    return fig


# ---------------------------------------------------------------------------- histori
# Database diatur di Secrets: [connections.histori] url = "postgresql://..." (Neon).
# Tanpa itu, histori hanya tersimpan selama sesi (session_state).
HIST_META = sa.MetaData()
HIST_TABLE = sa.Table(
    "histori", HIST_META,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("emiten", sa.String(10), nullable=False),
    sa.Column("waktu", sa.DateTime(timezone=True), nullable=False),
    sa.Column("target", sa.Float),
    sa.Column("stop_loss", sa.Float),
    sa.Column("harga", sa.Float))


def histori_url() -> str | None:
    try:
        u = st.secrets["connections"]["histori"]["url"]
        if u:
            return str(u)
    except Exception:
        pass
    return os.environ.get("HISTORI_DB_URL") or None


def _sa_url(url: str) -> str:
    """URL Neon (postgres:// atau postgresql://) -> driver psycopg 3."""
    for pre in ("postgres://", "postgresql://"):
        if url.startswith(pre):
            return "postgresql+psycopg://" + url[len(pre):]
    return url


@st.cache_resource(show_spinner=False)
def histori_engine(url: str) -> sa.Engine:
    eng = sa.create_engine(_sa_url(url), pool_pre_ping=True)
    HIST_META.create_all(eng)
    return eng


def _num(x):
    return None if x is None or pd.isna(x) else float(x)


def catat_histori(r: ad.Result) -> None:
    row = dict(emiten=r.params.kode, waktu=dt.datetime.now(dt.timezone.utc),
               target=_num(r.calc["t_real_r"]), stop_loss=_num(r.plan["sl"]), harga=_num(r.harga))
    url = histori_url()
    if url:
        try:
            with histori_engine(url).begin() as c:
                c.execute(HIST_TABLE.insert().values(**row))
            st.session_state.pop("_db_err", None)
            return
        except Exception as e:  # jangan sampai DB mematikan kalkulator
            st.session_state["_db_err"] = type(e).__name__
    st.session_state.setdefault("histori", []).append(row)


def load_histori(limit: int = 1000) -> pd.DataFrame:
    rows = list(st.session_state.get("histori", []))
    url = histori_url()
    if url:
        try:
            with histori_engine(url).connect() as c:
                q = sa.select(HIST_TABLE).order_by(HIST_TABLE.c.id.desc()).limit(limit)
                rows += [dict(m) for m in c.execute(q).mappings()]
        except Exception as e:
            st.session_state["_db_err"] = type(e).__name__
    cols = ["Nama Emiten", "Tanggal & Waktu Pengecekan", "Target Adimologi", "Stop Loss", "Current Price"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    df["waktu"] = pd.to_datetime(df["waktu"], utc=True).dt.tz_convert(ZoneInfo("Asia/Jakarta"))
    df = df.sort_values("waktu", ascending=False)
    df["waktu"] = df["waktu"].dt.strftime("%Y-%m-%d %H:%M:%S WIB")
    df = df.rename(columns={"emiten": cols[0], "waktu": cols[1], "target": cols[2],
                            "stop_loss": cols[3], "harga": cols[4]})
    return df[cols]


def hapus_histori() -> None:
    st.session_state.pop("histori", None)
    url = histori_url()
    if url:
        with histori_engine(url).begin() as c:
            c.execute(HIST_TABLE.delete())


def ringkas(r: ad.Result) -> dict:
    return {"Kode": r.params.kode, "Penilaian (tanpa aliran)": bd.penilaian(r)["kategori"],
            "Skenario Adimology": r.verdict, "Bandar": r.bandar["code"],
            "Net lot": r.bandar["net_lot"], "Avg bandar": r.bandar["avg"], "Harga": r.harga,
            "Target realistis": r.calc["t_real_r"], "Target max": r.calc["t_max_r"],
            "Upside realistis %": r.calc["upside_real"] * 100, "Alasan": "; ".join(r.alasan[:2])}


def tampil_penilaian(r: ad.Result, nilai: dict) -> None:
    kat = nilai["kategori"]
    box, ikon = {"AKUMULASI KUAT": (st.success, "🟢"), "AKUMULASI LEMAH / BELUM SEARAH": (st.info, "🔵"),
                 "TIDAK ADA BUKTI AKUMULASI": (st.warning, "⚪")}.get(kat, (st.error, "🚨"))
    box(f"**{r.params.kode} · Penilaian bandarmologi: {kat}**  \nSkor {nilai['skor']:+d} dari {nilai['skor_maks']} "
        "kriteria terukur. Status positif hanya bila bukti independen searah; satu jebakan cukup untuk WASPADA.", icon=ikon)
    for j in nilai["jebakan"]:
        st.error(j, icon="⚠️")
    for i_ in nilai["info"]:
        st.info(i_, icon="ℹ️")
    st.dataframe(bd.tabel(nilai), hide_index=True, width="stretch",
                 column_config={"Bukti": st.column_config.TextColumn(width="large")})
    al_ = st.session_state.get("aliran")
    catatan = "* = kriteria inti (wajib terukur dan tidak negatif untuk AKUMULASI KUAT)."
    if not nilai["lengkap"]:
        catatan += " Sebagian kriteria tidak diukur: aktifkan *Aliran lanjutan* di sidebar agar penilaian lengkap."
    if al_ and al_[0] == "err":
        catatan += f" Aliran lanjutan gagal: {al_[1]}"
    st.caption(catatan)


def tampil_volume_dan_b200(r: ad.Result) -> None:
    v = st.session_state.get("vol")
    if v:
        rows = [{"Periode": f"{n} hari bursa", "Volume rata-rata (lot/hari)": v[n]["lot_rata"], "Median (lot/hari)": v[n]["lot_median"],
                 "Nilai transaksi rata-rata (Rp miliar/hari)": v[n]["nilai_rata"] / 1e9} for n in (20, 60, 200) if n in v]
        ht = v["hari_terakhir"]
        st.markdown(f"**Volume rata-rata harian** · hari terakhir {fmt(ht['lot'])} lot"
                    + (f" ({ht['rvol']:.1f}x median 20 hari)" if pd.notna(ht["rvol"]) else ""))
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="content", column_config={
            "Volume rata-rata (lot/hari)": st.column_config.NumberColumn(format="%,.0f"),
            "Median (lot/hari)": st.column_config.NumberColumn(format="%,.0f"),
            "Nilai transaksi rata-rata (Rp miliar/hari)": st.column_config.NumberColumn(format="%,.2f")})
    b = st.session_state.get("b200")
    if not b:
        return
    st.markdown("#### 🏦 Bandar jangka panjang: akumulasi/distribusi dan nyangkut atau tidak")
    if b[0] == "err":
        st.warning(f"Info bandar jangka panjang gagal: {b[1]}")
        return
    p = b[1]
    box = {"NYANGKUT": st.warning, "IMPAS": st.info, "TIDAK NYANGKUT": st.success}.get(p["nyangkut"], st.error)
    box(f"**{r.params.kode} · {p['dari']} s/d {p['sampai']} ({p['hari']} hari bursa): {p['status']} · {p['nyangkut']}**  \n"
        f"{p['keterangan']}.  \nNet top-5 akumulator vs top-5 distributor: {p['net_acc5_value'] / 1e9:+,.1f} M vs "
        f"{p['net_dist5_value'] / 1e9:+,.1f} M. Konsentrasi: {p['konsentrasi']['status']}.", icon="🏦")
    cfg = {"net_lot": st.column_config.NumberColumn("Net periode (lot)", format="%,.0f"),
           "avg": st.column_config.NumberColumn("Avg", format="%,.1f"),
           "floating": st.column_config.NumberColumn("Harga vs avg", format="percent"),
           "net_window_lot": st.column_config.NumberColumn("Net window sekarang (lot)", format="%,.0f"),
           "code": "Kode", "broker": "Broker"}
    a1, a2 = st.columns(2)
    a1.markdown("**Top akumulator** (avg = harga beli rata-rata)")
    a1.dataframe(pd.DataFrame(p["akumulator"]), hide_index=True, column_config=cfg, width="stretch")
    a2.markdown("**Top distributor** (avg = harga jual rata-rata)")
    a2.dataframe(pd.DataFrame(p["distributor"]), hide_index=True, column_config=cfg, width="stretch")
    st.caption("Modal = rata-rata harga beli tertimbang volume selama periode; tidak tahu kapan barang dibeli dan tidak "
               "melihat penjualan di pasar nego, jadi 'nyangkut' adalah perkiraan kasar, bukan posisi sebenarnya.")


MA_WARNA = {5: "#0ea5e9", 20: "#f59e0b", 50: "#10b981", 100: "#8b5cf6", 200: "#ef4444"}


def rekomendasi_chart(rk: dict) -> go.Figure:
    d = rk["ind"].tail(160)
    x = d["tgl"].astype(str)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=x, open=d["open"], high=d["high"], low=d["low"], close=d["close"], name="Harga",
                                 increasing_line_color="#16a34a", decreasing_line_color="#dc2626"), row=1, col=1)
    for n, c in MA_WARNA.items():
        if d[f"ma{n}"].notna().any():
            fig.add_scatter(x=x, y=d[f"ma{n}"], mode="lines", name=f"MA{n}", line=dict(color=c, width=1.2), row=1, col=1)
    if rk["zona_valid"]:
        fig.add_hrect(y0=rk["entry_lo"], y1=rk["entry_hi"], fillcolor="#2563eb", opacity=0.15, line_width=0,
                      annotation_text="Area beli", annotation_position="top left", row=1, col=1)
    fig.add_hrect(y0=rk["tp1"], y1=rk["tp2"], fillcolor="#16a34a", opacity=0.10, line_width=0,
                  annotation_text="Area jual", annotation_position="top left", row=1, col=1)
    for y, nama, warna in [(rk["tp1"], "TP1", "#16a34a"), (rk["tp2"], "TP2", "#15803d"), (rk["sl"], "SL", "#dc2626")]:
        fig.add_hline(y=y, line_color=warna, line_dash="dash", line_width=1, row=1, col=1,
                      annotation_text=f"{nama} {y:,.0f}", annotation_position="top right", annotation_font_size=11)
    for c in rk["sup"][:3] + rk["res"][:3]:
        fig.add_hline(y=c["harga"], line_color="#9ca3af", line_dash="dot", line_width=1, row=1, col=1)
    fig.add_scatter(x=x, y=d["rsi"], mode="lines", name="RSI 14", line=dict(color="#7c3aed"), row=2, col=1)
    for y in (30, 70):
        fig.add_hline(y=y, line_color="#9ca3af", line_dash="dot", line_width=1, row=2, col=1)
    fig.add_bar(x=x, y=d["hist"], name="Histogram", row=3, col=1,
                marker_color=["#16a34a" if v >= 0 else "#dc2626" for v in d["hist"].fillna(0)])
    fig.add_scatter(x=x, y=d["macd"], mode="lines", name="MACD", line=dict(color="#2563eb"), row=3, col=1)
    fig.add_scatter(x=x, y=d["signal"], mode="lines", name="Signal", line=dict(color="#f97316"), row=3, col=1)
    fig.update_layout(height=720, margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.04))
    fig.update_xaxes(type="category", nticks=10)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    return fig


def ambil_fundamental(client: ad.InvezgoClient, kode: str, harga: float) -> dict:
    def opsional(path, params=None):
        try:
            return client.get(path, params)
        except Exception:
            return None
    def laporan(statement):
        # Invezgo membalas 422 bila limit > jumlah periode yang tersedia (mis. emiten baru IPO)
        for limit in (fd.N_KUARTAL, 8, 5, 4, 2, 1):
            try:
                return client.get(f"/analysis/financial-statement/{kode}",
                                  {"statement": statement, "type": "Q", "limit": limit})
            except ad.ParamDitolak:
                continue
        raise ad.AdimologyError(f"Laporan keuangan {kode} ({statement}) belum tersedia di Invezgo.")
    lap = {s: laporan(s) for s in ("IS", "BS", "CF")}
    return fd.analisa(opsional(f"/analysis/information/{kode}"), lap["IS"], lap["BS"], lap["CF"],
                      opsional(f"/analysis/keystat/{kode}", {"type": "Q", "limit": 8}), harga)


def ambil_aliran(client: ad.InvezgoClient, r: ad.Result) -> dict:
    kode, codes = r.params.kode, r.bandar["codes"]
    hari = r.chart["tgl"].tolist()
    out = {"asing": al.asing(client, kode, r.tgl_awal, r.tgl_akhir),
           "nego": al.negosiasi(client, kode, r.tgl_awal, r.tgl_akhir, r.total_vol_lot, codes),
           "konsistensi": al.konsistensi(client, kode, hari, set(r.params.broker_dikecualikan), r.params.investor,
                                         r.params.market, codes)}
    try:
        out["afiliasi"] = al.afiliasi(client, kode, codes)
    except Exception:  # endpoint insider bisa tidak tersedia di paket API tertentu
        out["afiliasi"] = None
    return out


PETUNJUK = """
**Cara membaca (penting)**
- Mulai dari **Penilaian bandarmologi** di Ringkasan: AKUMULASI KUAT hanya bila konsentrasi beli, konsistensi
  bandar, tidak ada distribusi, harga dekat modal bandar, asing, nego, dan fase harga **searah**. Satu **jebakan**
  (distribusi terselubung, bandar jual di nego, volume spike tanpa akumulasi, harga sudah dikejar, fase distribusi)
  = WASPADA.
- **Target Adimology adalah skenario**, bukan prediksi: di backtest tidak lebih baik dari beli acak.
- Keputusan di PDF: BELI hanya bila AKUMULASI KUAT **dan** teknikal di area beli.

**Bandar jangka panjang** (Ringkasan): akumulasi/distribusi selama 200 hari bursa (bisa diubah di sidebar) dan
apakah bandar **nyangkut**: harga ≥ 5% di bawah modal rata-rata top-3 akumulator = NYANGKUT; ±5% = IMPAS; di atas =
TIDAK NYANGKUT. Perkiraan kasar dari broker summary, bukan posisi sebenarnya. Di bawahnya ada volume rata-rata harian.

**Langkah cepat**
1. Isi **Kode saham** (4 huruf, mis. BBCA) di sidebar.
2. Atur **Window broker summary** (jumlah hari bursa yang dianalisa; default 5).
3. Klik **Hitung**. Hasil muncul di tab-tab di kanan dan otomatis tercatat di **Histori**.
4. Di bawah tombol Hitung muncul **Unduh Excel** (semua data per sheet) dan **Unduh PDF** (dokumen riset:
   keputusan BELI / TUNGGU / HINDARI-JUAL, rencana trading, grafik, teknikal, bandar, fundamental).

**Tab**
- **Ringkasan**: verdict Adimology, harga, avg bandar, target realistis & max, SL, grafik, kalkulator posisi.
- **Rekomendasi**: analisa teknikal lanjutan: area entry aman, SL, target jual (TP1/TP2) dari support/resistance,
  MA 5/20/50/100/200, RSI 14, MACD (12, 26, 9), dan nilai bandar (net value top-5 broker).
- **Fundamental**: highlight laba rugi, neraca, arus kas per kuartal (8 kuartal terakhir) dan valuasi
  (PER, PBV, PS, ROE, ROA, yield). **Hijau** = tumbuh/naik, **kuning** = perlu dicek (mis. utang > aset, rugi,
  arus kas negatif, current ratio < 1).
- **Statistik BPJP / BSJP** (bawah tab Rekomendasi): dari candle 30 menit 6 bulan, seberapa sering saham ini naik
  besok pagi setelah volume sore tinggi (BSJP), atau naik di sesi 1 setelah volume 30 menit pertama tinggi (BPJP);
  target jual (sudah termasuk fee) dan jam puncak yang paling sering = jendela jual. Ambang volume dalam lembar.
- **Konfirmasi aliran** (Ringkasan): konsentrasi top 3 pembeli vs top 15 penjual, asing, nego (bandar bisa keluar
  lewat crossing), konsistensi bandar 1/3/7/20 hari, afiliasi broker bandar dengan pemegang saham ≥5% (KSEI).
- **Mode bandar** di sidebar: top 1 broker atau gabungan top 3 akumulator. **Median 3 snapshot** orderbook
  (expander Orderbook historis) meredam fake bid/offer.
- **Broker**: periode, bandar terpilih, konsistensi 1/3/7/20 hari, pasar nego, top net buy/sell, aliran harian.
- **Orderbook**: antrean bid/offer dan lot per papan yang dipakai rumus.
- **Sensitivitas**: seberapa berubah target bila barang bandar / lot per papan berbeda.
- **Multi-saham**: jalankan analisa untuk banyak kode sekaligus (maks 15).
- **Histori**: catatan semua pengecekan (emiten, waktu WIB, target, SL, harga); bisa diunduh CSV.
- **Rumus & catatan**: rumus Adimology dan keterbatasannya.

**Parameter penting**
- **Broker dikecualikan**: broker ritel yang bukan bandar (mis. XL, YP, XC).
- **Broker bandar manual**: paksa broker tertentu sebagai bandar.
- **Lot per papan manual / Orderbook historis**: dipakai saat pasar tutup (orderbook hanya 1 level).
- **Parameter lanjutan**: margin, faktor target, ambang distribusi, modal & risiko per trade.

**Membaca tab Rekomendasi**
- **BELI DI AREA ENTRY**: harga sudah di area entry, indikator & bandar mendukung, R/R cukup.
- **TUNGGU KE AREA ENTRY**: setup bagus tapi harga di atas entry maksimal. Pasang antrean di area beli, jangan kejar.
- **TUNGGU**: tidak ada harga di zona support yang memberi R/R 1,5 (target terlalu dekat).
- **Area beli** = support s/d **entry maksimal** (harga tertinggi yang masih memberi R/R ≥ 1,5 ke target).
  **Area jual** = TP1 s/d TP2 (resistance). Stock split terdeteksi otomatis dan harga lama disesuaikan.
- Tab **Broker** menampilkan periode broker summary (tanggal awal s/d akhir window).
- **HINDARI**: mayoritas indikator/aliran bandar melawan, atau harga sudah tembus SL.
- **TIDAK TERUKUR**: data harga kurang dari 60 hari bursa (mis. saham baru IPO).

Data cache 5 menit (orderbook 15 detik). Semua hasil adalah edukasi, bukan ajakan transaksi.
"""


@st.dialog("Petunjuk penggunaan", width="large")
def petunjuk():
    st.markdown(PETUNJUK)


# ------------------------------------------------------------------------------ sidebar
st.sidebar.title("📈 Adimology")
sb1, sb2 = st.sidebar.columns(2)
if sb1.button("Keluar", width="stretch"):
    st.session_state.pop("_auth", None)
    st.session_state.pop("histori", None)
    st.rerun()
if sb2.button("ℹ️ Petunjuk", width="stretch"):
    petunjuk()
key = get_api_key()
if not key:
    st.sidebar.text_input("Invezgo API key", type="password", key="_key_input",
                          help="Untuk deploy, simpan di st.secrets (INVEZGO_API_KEY). Key tidak disimpan.")
    key = get_api_key()

with st.sidebar.form("param"):
    kode = st.text_input("Kode saham", value="MCAS", max_chars=4).upper()
    window = st.slider("Window broker summary (hari bursa)", 1, 20, 5)
    pakai_tgl = st.checkbox("Pilih tanggal akhir sendiri")
    tgl_akhir = st.date_input("Tanggal akhir", value=dt.date.today(), disabled=not pakai_tgl)
    investor = st.selectbox("Investor", ["all", "f", "d"], format_func={"all": "Semua", "f": "Asing", "d": "Domestik"}.get)
    broker_manual = st.text_input("Broker bandar manual (kosong = otomatis)", max_chars=2).upper()
    mode_bandar = st.selectbox("Mode bandar", ["top1", "top3"],
                               format_func={"top1": "Top 1 broker", "top3": "Gabungan top 3 akumulator"}.get,
                               help="Top 3: barang = total net 3 akumulator teratas, avg = rata-rata tertimbang volume beli.")
    excl = st.text_input("Broker dikecualikan (pisah koma)", value=", ".join(al.RITEL_DEFAULT),
                         help="Broker dominan ritel yang bukan bandar: XL (Stockbit), YP (Mirae), XC (Ajaib), PD (Indo Premier).")
    b200_on = st.checkbox("Info bandar jangka panjang (akumulasi/distribusi, nyangkut atau tidak)", value=True,
                          help="Satu panggilan API untuk broker summary seluruh periode di bawah.")
    b200_hari = st.number_input("Jumlah hari bursa jangka panjang", 20, 250, 200, 10)
    aliran_on = st.checkbox("Aliran lanjutan (asing, nego, konsistensi 1/3/7/20 hari, afiliasi ≥5%)", value=True,
                            help="Menambah ±7 panggilan API per Hitung.")
    lpp_manual = st.number_input("Lot per papan manual (0 = dari orderbook)", min_value=0.0, value=0.0, step=100.0,
                                 help="Berguna saat pasar tutup dan orderbook hanya 1 level.")
    with st.expander("Orderbook historis"):
        ob_hist = st.checkbox("Pakai snapshot historis")
        ob_date = st.date_input("Tanggal", value=dt.date.today() - dt.timedelta(days=1), min_value=dt.date(2026, 3, 25))
        ob_time = st.text_input("Jam WIB (HH:mm, 09:00-15:59)", value="10:30")
        ob_multi = st.checkbox("Median 3 snapshot (10:00 / 11:30 / 14:30)",
                               help="Lot per papan dari median 3 snapshot hari bursa terakhir, meredam fake bid/offer. +3 panggilan API.")
    with st.expander("Parameter lanjutan"):
        margin_a = st.number_input("Margin a (%)", 0.0, 50.0, 5.0, 0.5) / 100
        faktor = st.number_input("Faktor target realistis", 0.1, 1.0, 0.5, 0.05)
        min_up = st.number_input("Min upside (%)", 0.0, 50.0, 3.0, 0.5) / 100
        maks_kejar = st.number_input("Maks di atas avg bandar (%)", 0.0, 50.0, 5.0, 0.5) / 100
        dist = st.number_input("Ambang distribusi (% dari net window)", 1.0, 100.0, 25.0, 5.0) / 100
        modal = st.number_input("Modal (Rp)", 100_000, 10_000_000_000, 10_000_000, 1_000_000)
        risiko = st.number_input("Risiko per trade (%)", 0.1, 10.0, 1.0, 0.1) / 100
        fee_b = st.number_input("Fee beli (%)", 0.0, 2.0, 0.15, 0.01) / 100
        fee_j = st.number_input("Fee jual (%)", 0.0, 2.0, 0.25, 0.01) / 100
    submitted = st.form_submit_button("Hitung", type="primary", width="stretch")

params = ad.Params(
    kode=kode, tanggal_akhir=tgl_akhir if pakai_tgl else None, window_hari=window, investor=investor,
    broker_manual=broker_manual or None,
    broker_dikecualikan=tuple(c.strip().upper() for c in excl.split(",") if c.strip()),
    margin_a=margin_a, faktor_realistis=faktor,
    ob_date=ob_date.isoformat() if ob_hist else None, ob_time=ob_time if ob_hist else None,
    lot_per_papan_override=lpp_manual, min_upside=min_up, maks_kejar=maks_kejar, distribusi_pct=dist,
    modal=modal, risiko_pct=risiko, fee_beli=fee_b, fee_jual=fee_j, mode_bandar=mode_bandar, ob_multi=ob_multi)

if submitted:
    st.session_state.pop("tek", None)
    st.session_state.pop("fund", None)
    st.session_state.pop("export", None)
    st.session_state.pop("ovn", None)
    st.session_state.pop("aliran", None)
    st.session_state.pop("nilai", None)
    st.session_state.pop("vol", None)
    st.session_state.pop("b200", None)
    if not key:
        st.session_state["res"] = ("err", "Isi API key Invezgo di sidebar atau di st.secrets.")
    else:
        try:
            with st.spinner(f"Mengambil data {kode} dari Invezgo..."):
                st.session_state["res"] = ("ok", ad.analyze(CachedClient(key), params))
            catat_histori(st.session_state["res"][1])
        except ad.AdimologyError as e:
            st.session_state["res"] = ("err", str(e))
        except Exception as e:  # jaga UI tetap hidup
            st.session_state["res"] = ("err", f"Error tak terduga: {e}")
        if st.session_state["res"][0] == "ok":
            r0 = st.session_state["res"][1]
            try:
                with st.spinner("Menghitung analisa teknikal..."):
                    df_tk = tk.get_chart_panjang(CachedClient(key), r0.params.kode, r0.params.tanggal_akhir)
                    bv = tk.bandar_value(r0.bs, r0.bandar["codes"], r0.params.broker_dikecualikan)
                    st.session_state["tek"] = ("ok", tk.rekomendasi(df_tk, bv))
            except Exception as e:  # teknikal gagal tidak boleh mematikan kalkulator
                st.session_state["tek"] = ("err", str(e) if isinstance(e, ad.AdimologyError) else f"Error tak terduga: {e}")
            if aliran_on:
                try:
                    with st.spinner("Mengambil aliran asing, nego, konsistensi, afiliasi..."):
                        st.session_state["aliran"] = ("ok", ambil_aliran(CachedClient(key), r0))
                except Exception as e:  # aliran gagal tidak boleh mematikan kalkulator
                    st.session_state["aliran"] = ("err", str(e) if isinstance(e, ad.AdimologyError) else f"Error tak terduga: {e}")
            ali_, tek_n = st.session_state.get("aliran"), st.session_state.get("tek")
            st.session_state["nilai"] = bd.penilaian(r0, ali_[1] if ali_ and ali_[0] == "ok" else None,
                                                     tek_n[1] if tek_n and tek_n[0] == "ok" else None)
            rk_n = tek_n[1] if tek_n and tek_n[0] == "ok" else None
            src_n = rk_n["ind"] if rk_n else r0.chart
            st.session_state["vol"] = bd.volume_rata(src_n)
            if b200_on:
                try:
                    with st.spinner(f"Menganalisa bandar {int(b200_hari)} hari terakhir..."):
                        st.session_state["b200"] = ("ok", bd.bandar_panjang(
                            CachedClient(key), r0.params.kode, src_n["tgl"].tolist(), r0.harga, r0.bs,
                            r0.params.broker_dikecualikan, r0.params.investor, r0.params.market, hari=int(b200_hari)))
                except Exception as e:  # bandar jangka panjang gagal tidak boleh mematikan kalkulator
                    st.session_state["b200"] = ("err", str(e) if isinstance(e, ad.AdimologyError) else f"Error tak terduga: {e}")
            try:
                with st.spinner("Mengambil laporan keuangan..."):
                    st.session_state["fund"] = ("ok", ambil_fundamental(CachedClient(key), r0.params.kode, r0.harga))
            except Exception as e:  # fundamental gagal tidak boleh mematikan kalkulator
                st.session_state["fund"] = ("err", str(e) if isinstance(e, ad.AdimologyError) else f"Error tak terduga: {e}")
            try:
                with st.spinner("Menghitung statistik BPJP/BSJP (candle 30 menit)..."):
                    bars_ = idt.get_bars(CachedClient(key), r0.params.kode, r0.params.tanggal_akhir)
                    st.session_state["ovn"] = ("ok", idt.statistik(bars_, fee=r0.params.fee_beli + r0.params.fee_jual))
            except Exception as e:  # statistik intraday gagal tidak boleh mematikan kalkulator
                st.session_state["ovn"] = ("err", str(e) if isinstance(e, ad.AdimologyError) else f"Error tak terduga: {e}")
            tek_, fund_ = st.session_state.get("tek"), st.session_state.get("fund")
            rk_ = tek_[1] if tek_ and tek_[0] == "ok" else None
            f_ = fund_[1] if fund_ and fund_[0] == "ok" else None
            ovn_ = st.session_state.get("ovn")
            o_ = ovn_[1] if ovn_ and ovn_[0] == "ok" else None
            alr_ = st.session_state.get("aliran")
            a_ = alr_[1] if alr_ and alr_[0] == "ok" else None
            now = dt.datetime.now(ZoneInfo("Asia/Jakarta"))
            waktu = now.strftime("%Y-%m-%d %H:%M WIB")
            nama_file = f"riset_{r0.params.kode}_{now:%Y%m%d_%H%M}"
            exp = {"nama": nama_file}
            with st.spinner("Menyiapkan dokumen ekspor..."):
                for jenis, fn in (("xlsx", lp.excel_bytes), ("pdf", lp.pdf_bytes)):
                    try:
                        exp[jenis] = fn(r0, rk_, f_, waktu, o_, a_, st.session_state.get("b200", (None, None))[1]
                                        if (st.session_state.get("b200") or (None,))[0] == "ok" else None)
                    except Exception as e:  # ekspor gagal tidak boleh mematikan kalkulator
                        exp[f"{jenis}_err"] = f"{type(e).__name__}: {e}"
            st.session_state["export"] = exp

exp = st.session_state.get("export")
if exp:
    st.sidebar.markdown("**Export riset**")
    if "xlsx" in exp:
        st.sidebar.download_button("📊 Unduh Excel", exp["xlsx"], f"{exp['nama']}.xlsx", width="stretch",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    if "pdf" in exp:
        st.sidebar.download_button("📄 Unduh PDF", exp["pdf"], f"{exp['nama']}.pdf", mime="application/pdf", width="stretch")
    for jenis in ("xlsx", "pdf"):
        if f"{jenis}_err" in exp:
            st.sidebar.error(f"Gagal membuat {jenis.upper()}: {exp[f'{jenis}_err']}")

# ------------------------------------------------------------------------------- main
st.title("Adimology Calculator")
st.caption("Penilaian bandarmologi berbasis bukti (aliran broker, asing, nego, fase harga) plus skenario target "
           "Adimology dari data Invezgo. Edukasi, bukan ajakan transaksi. Semua aturan heuristik; target Adimology "
           "tidak terbukti unggul di backtest.")

tab_main, tab_rek, tab_fund, tab_broker, tab_ob, tab_sens, tab_multi, tab_hist, tab_rumus = st.tabs(
    ["Ringkasan", "Rekomendasi", "Fundamental", "Broker", "Orderbook", "Sensitivitas", "Multi-saham", "Histori", "Rumus & catatan"])

state = st.session_state.get("res")
with tab_main:
    if not state:
        st.info("Atur parameter di sidebar lalu klik **Hitung**.")
    elif state[0] == "err":
        st.error(state[1])
    else:
        r: ad.Result = state[1]
        tampil_penilaian(r, st.session_state.get("nilai") or bd.penilaian(r))
        tampil_volume_dan_b200(r)
        st.markdown("#### 🧮 Skenario target Adimology (belum terbukti)")
        st.info(f"**{r.params.kode} → {r.verdict}** (skenario, bukan sinyal)" + "".join(f"\n\n- {a}" for a in r.alasan),
                icon="🧮")
        c = st.columns(6)
        c[0].metric("Harga", fmt(r.harga))
        c[1].metric(f"Avg bandar ({r.bandar['code']})", fmt(r.bandar["avg"], 1), f"{r.floating:+.1%} floating", delta_color="normal")
        c[2].metric("Dasar (avg + a)", fmt(r.calc["dasar"], 1))
        c[3].metric("Target realistis", fmt(r.calc["t_real_r"]), f"{r.calc['upside_real']:+.1%}")
        c[4].metric("Target max", fmt(r.calc["t_max_r"]), f"{r.calc['upside_max']:+.1%}")
        c[5].metric("SL (low window)", fmt(r.plan["sl"]), f"R/R {r.plan['rr']:.2f}" if pd.notna(r.plan["rr"]) else None, delta_color="off")
        st.markdown(f"**Rentang skenario:** {fmt(r.calc['t_real_r'])} – {fmt(r.calc['t_max_r'])} "
                    f"({r.calc['upside_real']:+.1%} s/d {r.calc['upside_max']:+.1%}). Perlakukan sebagai *potensi*, bukan "
                    "probabilitas.")
        with st.expander("Hasil backtest metode Adimology"):
            st.write(bd.HASIL_BACKTEST)
            st.caption("Uji masih terbatas (4 saham, satu periode). Detail: backtest/README.md di repo.")
        st.plotly_chart(make_chart(r), width="stretch")
        st.caption(f"Window {r.tgl_awal} s/d {r.tgl_akhir} ({r.params.window_hari} hari bursa) · "
                   f"barang bandar {fmt(r.calc['barang'])} lot · lot per papan {fmt(r.lot_per_papan)} ({r.lpp_sumber}) · "
                   f"p = {r.calc['p']:.2f} papan · level pembatal: close di bawah {fmt(r.plan['sl'])}")
        st.subheader("Kalkulator posisi")
        if r.plan["lot"] > 0:
            st.dataframe(pd.DataFrame([
                {"Keterangan": "Ukuran posisi", "Nilai": f"{r.plan['lot']:,} lot"},
                {"Keterangan": "Modal terpakai", "Nilai": f"Rp{r.plan['modal_pakai']:,.0f}"},
                {"Keterangan": "Jika kena target realistis", "Nilai": f"Rp{r.plan['laba_target']:+,.0f}"},
                {"Keterangan": "Jika kena SL", "Nilai": f"Rp{r.plan['rugi_sl']:+,.0f}"}]),
                hide_index=True, width="content")
            st.caption(f"Risiko {r.params.risiko_pct:.1%} dari Rp{r.params.modal:,.0f}; fee perkiraan. "
                       "Untuk SL di bawah harga saja. Ukuran posisi hanya hitungan mekanis, bukan rekomendasi.")
        else:
            st.write("Harga tidak di atas SL atau modal tidak cukup, ukuran posisi tidak dihitung.")


def baris_rencana(rk: dict) -> list[dict]:
    mr = f"{rk['min_rr']:.1f}".replace(".", ",")
    area_beli = (f"{rk['entry_lo']:,.0f} – {rk['entry_hi']:,.0f}" if rk["zona_valid"] else "tidak ada")
    return [
        {"Keterangan": "Harga terakhir", "Nilai": fmt(rk["harga"]),
         "Catatan": f"data s/d {rk['tgl']} · R/R jika beli di harga ini: {fmt(rk['rr_harga'], 2)}"},
        {"Keterangan": "Area beli (entry)", "Nilai": area_beli,
         "Catatan": (f"support {rk['entry_lo']:,.0f} s/d entry maksimal; R/R rata-rata {rk['rr']:.2f} ke {rk['target_nama']}"
                     if rk["zona_valid"] else f"zona support {rk['entry_lo']:,.0f} – {rk['entry_hi_sup']:,.0f} tidak memberi R/R {mr}")},
        {"Keterangan": f"Entry maksimal (R/R {mr})", "Nilai": fmt(rk["entry_maks"]),
         "Catatan": f"beli di atas harga ini R/R ke {rk['target_nama']} {rk['target']:,.0f} < {mr}"},
        {"Keterangan": "Stop loss", "Nilai": fmt(rk["sl"]), "Catatan": f"support − 1 ATR · {rk['sl'] / rk['entry_lo'] - 1:+.1%} dari support"},
        {"Keterangan": "Area jual (TP1 – TP2)", "Nilai": f"{rk['tp1']:,.0f} – {rk['tp2']:,.0f}",
         "Catatan": f"{rk['tp1'] / rk['harga'] - 1:+.1%} s/d {rk['tp2'] / rk['harga'] - 1:+.1%} dari harga · {rk['tp_basis']}"},
        {"Keterangan": "Target utama", "Nilai": f"{rk['target_nama']} {rk['target']:,.0f}",
         "Catatan": "TP1 terlalu dekat, target utama dipindah ke TP2" if rk["target_nama"] == "TP2" else "resistance terdekat"},
    ]


def _p(v):
    return "-" if v is None or pd.isna(v) else f"{v:.0%}"


def tampil_bpjp_bsjp():
    ovn = st.session_state.get("ovn")
    if not ovn:
        return
    st.divider()
    st.subheader("Statistik BPJP / BSJP")
    if ovn[0] == "err":
        st.error(f"Statistik BPJP/BSJP gagal: {ovn[1]}")
        return
    o = ovn[1]
    h, amb = o["hari_ini"], o["ambang"]
    st.caption(f"Candle 30 menit {o['dari']} s/d {o['sampai']} ({o['n_hari']} hari bursa). Volume tinggi = ≥ "
               f"{idt.AMBANG_RVOL:g}x median 20 hari pada jam yang sama. Target sudah termasuk fee pulang-pergi "
               f"{o['fee']:.2%}. Heuristik dari data historis, bukan jaminan.")
    if not h["lengkap"]:
        st.info(f"Data {h['tgl']} belum lengkap (pasar masih buka); volume sore hari ini belum final.", icon="⏳")

    st.markdown("#### 🌇 BSJP: beli sore (close), jual besok pagi")
    sg = o["sinyal"]["bsjp"]
    if sg:
        box = st.success if sg["layak"] else st.warning
        box(f"**{h['tgl']}: volume jam terakhir {sg['rvol']:.1f}x median** → kelompok *{sg['kelompok']}* ({sg['n']} hari).  \n"
            f"Peluang gap naik besok **{_p(sg['p_gap'])}**, peluang menyentuh +1% bersih fee **{_p(sg['p_target'])}**, "
            f"peluang rugi bila ditahan s/d akhir sesi 1 **{_p(sg['p_rugi'])}**.  \n"
            f"**Target jual {sg['target']:,.0f}** ({sg['naik']:+.1%} dari close {sg['ref']:,.0f}). Puncak besok paling sering di "
            f"**{idt.label_jam(sg['jam_jual'])}** ({_p(sg['p_jam'])}) → itu jendela jual; lewat jam itu peluang naik mengecil."
            + ("" if sg["layak"] else "  \n**Peluang historis < 50% atau median rugi setelah fee: BSJP tidak disarankan.**"),
            icon="✅" if sg["layak"] else "⚠️")
    st.caption(f"Ambang volume sore tinggi: ≥ **{amb['sore']:,.0f} lembar** di 15:00–16:00 (termasuk lelang penutupan).")
    st.dataframe(idt.tabel(o["bsjp"], bsjp=True), width="stretch")

    st.markdown("#### 🌅 BPJP: baca 30 menit pertama, beli ~09:30, jual sebelum sesi 1 tutup")
    tinggi = o["bpjp"][f"Volume tinggi (≥{idt.AMBANG_RVOL:g}x)"]
    normal = o["bpjp"]["Volume normal"]
    if tinggi.get("n"):
        lebih = tinggi["p_1"] > normal.get("p_1", 0)
        (st.success if lebih and tinggi["p_1"] >= 0.5 else st.warning)(
            f"**Aturan besok pagi:** jika volume 08:30–09:30 ≥ **{amb['buka']:,.0f} lembar**, historis ({tinggi['n']} hari): "
            f"peluang menyentuh +1% bersih fee **{_p(tinggi['p_1'])}** (hari biasa {_p(normal.get('p_1'))}), median kenaikan "
            f"tertinggi **{tinggi['median_naik']:+.1%}**, puncak paling sering **{idt.label_jam(tinggi['jam_puncak'])}** "
            f"({_p(tinggi['p_jam_puncak'])}). Peluang rugi bila ditahan s/d 11:30: {_p(tinggi['p_rugi_akhir'])}."
            + ("" if lebih else "  \nVolume pagi tinggi **tidak** terbukti menambah peluang naik untuk saham ini."),
            icon="✅" if lebih and tinggi["p_1"] >= 0.5 else "⚠️")
    sb = o["sinyal"]["bpjp"]
    if sb:
        st.caption(f"Pagi {h['tgl']}: volume 30 menit pertama {sb['rvol']:.1f}x median, harga ~09:30 {sb['ref']:,.0f}.")
    st.dataframe(idt.tabel(o["bpjp"], bsjp=False), width="stretch")

    sebaran = pd.DataFrame({"BSJP (besok)": o["bsjp"]["Semua hari"].get("sebaran_puncak", pd.Series(dtype=float)),
                            "BPJP (hari sama)": o["bpjp"]["Semua hari"].get("sebaran_puncak", pd.Series(dtype=float))}).fillna(0) * 100
    if not sebaran.empty:
        fig_p = go.Figure()
        for c, w in zip(sebaran.columns, ["#f59e0b", "#2563eb"]):
            fig_p.add_bar(x=[idt.label_jam(j) for j in sebaran.index], y=sebaran[c], name=c, marker_color=w)
        fig_p.update_layout(height=260, barmode="group", margin=dict(l=10, r=10, t=30, b=10), yaxis_title="% hari",
                            title=dict(text="Jam puncak harga sesi 1 (kapan sebaiknya jual)", font=dict(size=13)),
                            legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig_p, width="stretch")


def rp_milyar(x):
    return "-" if x is None or pd.isna(x) else f"Rp{x / 1e9:+,.2f} M"


with tab_rek:
    tek = st.session_state.get("tek")
    if not tek:
        st.info("Klik **Hitung** di sidebar. Analisa teknikal dihitung bersamaan dengan Adimology.")
    elif tek[0] == "err":
        st.error(f"Analisa teknikal gagal: {tek[1]}")
    else:
        rk = tek[1]
        kode_rk = state[1].params.kode if state and state[0] == "ok" else ""
        box = {"BELI DI AREA ENTRY": st.success, "TUNGGU KE AREA ENTRY": st.info,
               "TUNGGU": st.warning, "HINDARI": st.error, "TIDAK TERUKUR": st.warning}[rk["verdict"]]
        box(f"**{kode_rk} → {rk['verdict']}**: {rk['alasan']} · skor {rk['skor']:+d} dari ±{rk['skor_max']}")
        for ev in rk.get("split", []):
            st.info(f"Harga sebelum {ev['tgl']} disesuaikan untuk {ev['jenis']} supaya MA, support/resistance tidak rusak.", icon="ℹ️")
        st.dataframe(pd.DataFrame(baris_rencana(rk)), hide_index=True, width="content")
        st.plotly_chart(rekomendasi_chart(rk), width="stretch")

        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown("**Moving average**")
            st.dataframe(pd.DataFrame([{"MA": f"MA{n}", "Nilai": v,
                                        "Posisi harga": "-" if v is None else ("di atas" if rk["harga"] > v else "di bawah")}
                                       for n, v in rk["ma"].items()]),
                         hide_index=True, width="stretch", column_config={"Nilai": st.column_config.NumberColumn(format="%,.1f")})
            st.markdown(f"**RSI 14:** {fmt(rk['rsi'], 1)} · **MACD:** {fmt(rk['macd'], 2)} / signal {fmt(rk['signal'], 2)}")
        with k2:
            st.markdown("**Support & resistance**")
            sr = ([{"Jenis": "Resistance", "Harga": x["harga"], "Sentuhan": x["n"]} for x in rk["res"][:3][::-1]] +
                  [{"Jenis": "Support", "Harga": x["harga"], "Sentuhan": x["n"]} for x in rk["sup"][:3]])
            if sr:
                st.dataframe(pd.DataFrame(sr), hide_index=True, width="stretch",
                             column_config={"Harga": st.column_config.NumberColumn(format="%,.0f")})
            else:
                st.write("Belum ada swing high/low yang jelas.")
        with k3:
            bv = rk["bandar"]
            st.markdown(f"**Bandar value** (window broker summary) · **{bv.get('status', '-')}**")
            st.metric("Net value bandar terpilih", rp_milyar(bv.get("bandar")))
            st.metric("Net top-5 akumulator vs distributor", rp_milyar(bv.get("net_top5")),
                      f"beli {rp_milyar(bv.get('top5_beli'))} · jual {rp_milyar(bv.get('top5_jual'))}", delta_color="off")
        st.markdown("**Rincian sinyal**")
        st.markdown("\n".join(f"- {'🟢' if p > 0 else '🔴' if p < 0 else '⚪'} {t}" for p, t in rk["poin"]))
        catatan = ("MA200 belum tersedia (data < 200 hari bursa). " if rk["ma"][200] is None else "")
        st.caption(f"{catatan}Data {rk['n_bar']} hari bursa s/d {rk['tgl']}. Support/resistance dari swing high/low "
                   "±5 hari (250 hari terakhir), SL = support − 1 ATR14. Edukasi, bukan ajakan transaksi.")
    tampil_bpjp_bsjp()


with tab_fund:
    fund = st.session_state.get("fund")
    if not fund:
        st.info("Klik **Hitung** di sidebar. Laporan keuangan diambil bersamaan dengan Adimology.")
    elif fund[0] == "err":
        st.error(f"Data fundamental gagal diambil: {fund[1]}")
    else:
        f = fund[1]
        kode_f = state[1].params.kode if state and state[0] == "ok" else ""
        st.subheader(f"{kode_f} · {f['nama']}" if f["nama"] else kode_f)
        st.caption(" · ".join(x for x in [f["sektor"], f["subsektor"], f"laporan terakhir {f['periode']}",
                                          "nominal tabel dalam Rp miliar, per kuartal (bukan kumulatif)"] if x))
        for p_ in f["peringatan"]:
            st.warning(p_, icon="⚠️")
        for s_ in f["sorotan"]:
            st.success(s_, icon="📈")
        ch = f["chart"]
        fig_f = go.Figure()
        fig_f.add_bar(x=ch["periode"], y=ch["pendapatan"] / 1e9, name="Pendapatan", marker_color="#93c5fd")
        fig_f.add_bar(x=ch["periode"], y=ch["laba_bersih"] / 1e9, name="Laba bersih", marker_color="#2563eb")
        fig_f.update_layout(height=320, barmode="group", margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Rp miliar",
                            legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig_f, width="stretch")
        for judul, kunci in [("Laba rugi (PnL)", "pnl"), ("Neraca (balance sheet)", "bs"), ("Arus kas (cash flow)", "cf")]:
            st.markdown(f"**{judul}**")
            st.dataframe(fd.style(f[kunci], f["mark"], kunci), width="stretch")
        st.markdown("**Valuasi** (harga terakhir × saham beredar; laba & pendapatan 12 bulan terakhir)")
        v_ = f["val"].set_index("Metrik")
        st.dataframe(fd.style(v_, {("val", m, "Nilai"): c for m, c in f["vmark"].items()}, "val"), width="content")
        st.caption("🟩 hijau = tumbuh/naik (YoY untuk pendapatan, laba, CFO; dibanding kuartal sebelumnya untuk aset, "
                   "ekuitas, kas). 🟨 kuning = perlu dicek (liabilitas > aset, ekuitas/laba/arus kas negatif, DER > 2, "
                   "current ratio < 1, neraca tidak seimbang). Sumber: laporan keuangan Invezgo; DER & current ratio "
                   "tidak dinilai untuk bank. Edukasi, bukan ajakan transaksi.")


if state and state[0] == "ok":
    r = state[1]
    with tab_broker:
        st.write(f"**Bandar terpilih:** {r.bandar['code']} ({r.bandar['name']}) · net {fmt(r.bandar['net_lot'])} lot · "
                 f"{r.bandar['porsi']:.1f}% dari volume window ({fmt(r.total_vol_lot)} lot)")
        st.plotly_chart(harian_chart(r), width="stretch")
        st.info(f"Periode broker summary: **{r.tgl_awal} s/d {r.tgl_akhir}** ({r.params.window_hari} hari bursa, "
                "atur lewat slider *Window broker summary*). Top net buy/sell dan rincian harian memakai periode ini.", icon="📅")
        st.caption(f"Lot per papan: {fmt(r.lot_per_papan)} ({r.lpp_sumber}).")
        al_b = st.session_state.get("aliran")
        if al_b and al_b[0] == "ok":
            kc = al_b[1]["konsistensi"]
            if not kc.empty:
                st.markdown("**Konsistensi akumulator teratas (materi: cek 1, 3, 7, maks 20 hari)**")
                st.dataframe(kc, hide_index=True, width="stretch", column_config={
                    "net_lot": st.column_config.NumberColumn("Net (lot)", format="%,.0f"),
                    "avg_beli": st.column_config.NumberColumn("Avg beli", format="%,.1f"),
                    "sama_dengan_bandar": st.column_config.CheckboxColumn("Sama dengan bandar")})
            top_ng = al_b[1]["nego"]["top"]
            if top_ng:
                st.markdown("**Pasar negosiasi (window)**")
                st.dataframe(pd.DataFrame(top_ng), hide_index=True, width="content",
                             column_config={"net_lot": st.column_config.NumberColumn("Net (lot)", format="%,.0f")})
        cols = ["code", "name", "net_lot", "porsi_net_%", "buy_avg", "sell_avg", "net_value"]
        cfg = {"code": st.column_config.TextColumn("Kode", width=60),
               "name": st.column_config.TextColumn("Broker", width=170),
               "net_lot": st.column_config.NumberColumn("Net (lot)", format="%,.0f"),
               "porsi_net_%": st.column_config.NumberColumn("Porsi %", format="%.2f"),
               "buy_avg": st.column_config.NumberColumn("Avg beli", format="%.1f"),
               "sell_avg": st.column_config.NumberColumn("Avg jual", format="%.1f"),
               "net_value": st.column_config.NumberColumn("Net value (Rp)", format="%,.0f")}
        a1, a2 = st.columns(2)
        a1.markdown("**Top net buy**")
        a1.dataframe(r.bs.sort_values("net_volume", ascending=False).head(10)[cols], hide_index=True, column_config=cfg, width="stretch")
        a2.markdown("**Top net sell**")
        a2.dataframe(r.bs.sort_values("net_volume").head(10)[cols], hide_index=True, column_config=cfg, width="stretch")
        st.markdown(f"**Rincian harian bandar {r.bandar['code']}** ({r.tgl_awal} s/d {r.tgl_akhir})")
        hr = r.harian.copy()
        for c in ["net_lot", "avg_beli", "avg_jual", "net_kumulatif_lot"]:
            hr[c] = pd.to_numeric(hr[c], errors="coerce")
        st.dataframe(hr, hide_index=True, width="stretch", column_config={
            "tgl": st.column_config.TextColumn("Tanggal"),
            "net_lot": st.column_config.NumberColumn("Net (lot)", format="%,.0f"),
            "avg_beli": st.column_config.NumberColumn("Avg beli", format="%,.1f"),
            "avg_jual": st.column_config.NumberColumn("Avg jual", format="%,.1f"),
            "net_kumulatif_lot": st.column_config.NumberColumn("Kumulatif (lot)", format="%,.0f")})
        st.caption("Round trip (beli lalu jual habis dalam window) tidak terdeteksi otomatis. Lihat tabel top net buy/sell "
                   "atau pilih broker manual.")
    with tab_ob:
        if r.ob:
            o = r.ob
            m = st.columns(4)
            m[0].metric("Total bid (lot)", fmt(o["total_bid"]))
            m[1].metric("Total offer (lot)", fmt(o["total_offer"]))
            m[2].metric("Total papan", f"{o['papan']:,.1f}")
            m[3].metric("Lot per papan", fmt(o["lot_per_papan"]))
            if o["n_levels"] <= 4:
                st.warning("Orderbook hanya 1-2 level (umumnya saat pasar tutup). Gunakan snapshot historis atau "
                           "'lot per papan manual', dan cek tab Sensitivitas.")
            lv = pd.DataFrame([{"Sisi": "Offer", "Harga": k, "Lot": v} for k, v in sorted(o["offers"].items(), reverse=True)] +
                              [{"Sisi": "Bid", "Harga": k, "Lot": v} for k, v in sorted(o["bids"].items(), reverse=True)])
            st.dataframe(lv, hide_index=True, width="stretch")
        else:
            st.info("Orderbook kosong. Hitungan memakai 'lot per papan manual'.")
    with tab_sens:
        st.caption("Orderbook adalah foto sesaat dan bisa palsu (fake bid/offer). Lihat seberapa sensitif target terhadap asumsi.")
        piv = r.sens.pivot_table(index="barang_bandar_lot", columns="lot_per_papan", values="max")
        st.markdown("**Target max** (baris: barang bandar lot · kolom: lot per papan)")
        st.dataframe(piv.style.format("{:,.0f}"), width="stretch")
        piv2 = r.sens.pivot_table(index="barang_bandar_lot", columns="lot_per_papan", values="realistis")
        st.markdown("**Target realistis**")
        st.dataframe(piv2.style.format("{:,.0f}"), width="stretch")

with tab_multi:
    st.caption("Jalankan analisa yang sama untuk beberapa saham sekaligus (maks 15), memakai parameter di sidebar "
               "(broker manual & lot per papan manual diabaikan).")
    with st.form("multi"):
        daftar = st.text_area("Kode saham (pisah koma / baris)", value="MCAS, BBCA, BBRI")
        go_multi = st.form_submit_button("Jalankan batch")
    if go_multi:
        if not key:
            st.error("Isi API key Invezgo terlebih dahulu.")
        else:
            codes = [c for c in dict.fromkeys(x.strip().upper() for x in daftar.replace("\n", ",").split(",")) if c][:15]
            rows, prog = [], st.progress(0.0)
            client = CachedClient(key)
            for i, cd in enumerate(codes):
                try:
                    rr_ = ad.analyze(client, replace(params, kode=cd, broker_manual=None, lot_per_papan_override=0.0))
                    rows.append(ringkas(rr_))
                    catat_histori(rr_)
                except Exception as e:
                    rows.append({"Kode": cd, "Skenario Adimology": "ERROR", "Alasan": str(e)})
                prog.progress((i + 1) / len(codes))
            prog.empty()
            st.session_state["multi_df"] = pd.DataFrame(rows)
    if "multi_df" in st.session_state:
        st.dataframe(st.session_state["multi_df"], hide_index=True, width="stretch",
                     column_config={"Net lot": st.column_config.NumberColumn(format="%,.0f"),
                                    "Avg bandar": st.column_config.NumberColumn(format="%.1f"),
                                    "Upside realistis %": st.column_config.NumberColumn(format="%+.1f")})

with tab_hist:
    hdf = load_histori()
    if not histori_url():
        st.warning("Database histori belum diatur, histori hanya tersimpan selama sesi ini "
                   "(hilang saat refresh/Keluar). Isi `[connections.histori] url` di Secrets.")
    elif st.session_state.get("_db_err"):
        st.error(f"Gagal mengakses database histori ({st.session_state['_db_err']}). "
                 "Hasil terbaru disimpan sementara di sesi ini.")
    if hdf.empty:
        st.info("Belum ada histori. Setiap hasil **Hitung** dan batch Multi-saham tercatat di sini.")
    else:
        st.dataframe(hdf, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%,.0f")
                                    for c in ["Target Adimologi", "Stop Loss", "Current Price"]})
        h1, h2, h3 = st.columns([1, 1, 1])
        h1.download_button("Unduh CSV", hdf.to_csv(index=False).encode(), "histori_adimology.csv", "text/csv")
        yakin = h3.checkbox("Yakin hapus semua")
        if h2.button("Hapus histori", disabled=not yakin):
            hapus_histori()
            st.rerun()
        st.caption("Target Adimologi = target realistis. Menampilkan 1.000 catatan terbaru.")

with tab_rumus:
    st.markdown("""
**Rumus** (slide Adimology + Excel *Rumus Adimology*)
- `a` = avg beli bandar × 5%
- `p` = barang bandar (lot) ÷ lot per papan; lot per papan = (total bid + total offer) ÷ jumlah papan
- Target realistis = avg + a + (p × 0,5 papan) · Target max = avg + a + (p papan)

**Perbaikan dari Excel:** jumlah papan dan langkah harga dihitung **per rentang fraksi** (1/2/5/10/25), bukan satu fraksi tetap.

**Verdict** memakai aturan eksplisit (ubah di Parameter lanjutan): bandar harus net beli, tidak sedang distribusi di hari terakhir,
harga belum melewati target realistis, upside minimal sekian persen, dan harga tidak mengejar jauh di atas avg bandar.

**Keterbatasan**
- Orderbook = foto sesaat, bisa palsu. Satu broker ≠ satu bandar. Hasil bergantung pada window.
- Angka 5% dan faktor 0,5 adalah asumsi, bukan hasil riset. Belum ada backtest.
- Fraksi mengikuti aturan BEI standar; verifikasi bila aturan berubah.
""")
