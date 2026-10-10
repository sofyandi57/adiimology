"""Adimology Calculator — Streamlit. Edukasi, bukan ajakan transaksi."""
from __future__ import annotations

import datetime as dt
import hmac
import os
from dataclasses import replace
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import sqlalchemy as sa
import streamlit as st

import adimology as ad

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


@st.cache_resource(show_spinner=False)
def histori_engine(url: str) -> sa.Engine:
    eng = sa.create_engine(url, pool_pre_ping=True)
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
    return {"Kode": r.params.kode, "Verdict": r.verdict, "Bandar": r.bandar["code"],
            "Net lot": r.bandar["net_lot"], "Avg bandar": r.bandar["avg"], "Harga": r.harga,
            "Target realistis": r.calc["t_real_r"], "Target max": r.calc["t_max_r"],
            "Upside realistis %": r.calc["upside_real"] * 100, "Alasan": "; ".join(r.alasan[:2])}


# ------------------------------------------------------------------------------ sidebar
st.sidebar.title("📈 Adimology")
if st.sidebar.button("Keluar"):
    st.session_state.pop("_auth", None)
    st.session_state.pop("histori", None)
    st.rerun()
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
    excl = st.text_input("Broker dikecualikan (pisah koma)", value="XL",
                         help="Platform ritel yang bukan bandar, mis. XL (Stockbit), YP (Mirae), XC (Ajaib).")
    lpp_manual = st.number_input("Lot per papan manual (0 = dari orderbook)", min_value=0.0, value=0.0, step=100.0,
                                 help="Berguna saat pasar tutup dan orderbook hanya 1 level.")
    with st.expander("Orderbook historis"):
        ob_hist = st.checkbox("Pakai snapshot historis")
        ob_date = st.date_input("Tanggal", value=dt.date.today() - dt.timedelta(days=1), min_value=dt.date(2026, 3, 25))
        ob_time = st.text_input("Jam WIB (HH:mm, 09:00-15:59)", value="10:30")
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
    modal=modal, risiko_pct=risiko, fee_beli=fee_b, fee_jual=fee_j)

if submitted:
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

# ------------------------------------------------------------------------------- main
st.title("Adimology Calculator")
st.caption("Target harga metode Adimology dari data broker summary dan orderbook Invezgo. "
           "Edukasi, bukan ajakan transaksi. Metode heuristik, belum teruji statistik.")

tab_main, tab_broker, tab_ob, tab_sens, tab_multi, tab_hist, tab_rumus = st.tabs(
    ["Ringkasan", "Broker", "Orderbook", "Sensitivitas", "Multi-saham", "Histori", "Rumus & catatan"])

state = st.session_state.get("res")
with tab_main:
    if not state:
        st.info("Atur parameter di sidebar lalu klik **Hitung**.")
    elif state[0] == "err":
        st.error(state[1])
    else:
        r: ad.Result = state[1]
        box = st.success if r.setup_ok else st.warning
        box(f"**{r.params.kode} → {r.verdict}**" + "".join(f"\n\n- {a}" for a in r.alasan))
        c = st.columns(6)
        c[0].metric("Harga", fmt(r.harga))
        c[1].metric(f"Avg bandar ({r.bandar['code']})", fmt(r.bandar["avg"], 1), f"{r.floating:+.1%} floating", delta_color="normal")
        c[2].metric("Dasar (avg + a)", fmt(r.calc["dasar"], 1))
        c[3].metric("Target realistis", fmt(r.calc["t_real_r"]), f"{r.calc['upside_real']:+.1%}")
        c[4].metric("Target max", fmt(r.calc["t_max_r"]), f"{r.calc['upside_max']:+.1%}")
        c[5].metric("SL (low window)", fmt(r.plan["sl"]), f"R/R {r.plan['rr']:.2f}" if pd.notna(r.plan["rr"]) else None, delta_color="off")
        st.plotly_chart(make_chart(r), width="stretch")
        st.caption(f"Window {r.tgl_awal} s/d {r.tgl_akhir} ({r.params.window_hari} hari bursa) · "
                   f"barang bandar {fmt(r.calc['barang'])} lot · lot per papan {fmt(r.lot_per_papan)} ({r.lpp_sumber}) · "
                   f"p = {r.calc['p']:.2f} papan · level pembatal: close di bawah {fmt(r.plan['sl'])}")
        st.subheader("Kalkulator posisi")
        if r.plan["lot"] > 0:
            k = st.columns(4)
            k[0].metric("Ukuran posisi", f"{r.plan['lot']:,} lot")
            k[1].metric("Modal terpakai", f"Rp{r.plan['modal_pakai']:,.0f}")
            k[2].metric("Jika kena target realistis", f"Rp{r.plan['laba_target']:+,.0f}")
            k[3].metric("Jika kena SL", f"Rp{r.plan['rugi_sl']:+,.0f}")
            st.caption(f"Risiko {r.params.risiko_pct:.1%} dari Rp{r.params.modal:,.0f}; fee perkiraan. "
                       "Untuk SL di bawah harga saja. Ukuran posisi hanya hitungan mekanis, bukan rekomendasi.")
        else:
            st.write("Harga tidak di atas SL atau modal tidak cukup, ukuran posisi tidak dihitung.")


if state and state[0] == "ok":
    r = state[1]
    with tab_broker:
        st.write(f"**Bandar terpilih:** {r.bandar['code']} ({r.bandar['name']}) · net {fmt(r.bandar['net_lot'])} lot · "
                 f"{r.bandar['porsi']:.1f}% dari volume window ({fmt(r.total_vol_lot)} lot)")
        st.plotly_chart(harian_chart(r), width="stretch")
        cols = ["code", "name", "net_lot", "porsi_net_%", "buy_avg", "sell_avg", "net_value"]
        cfg = {"net_lot": st.column_config.NumberColumn("Net (lot)", format="%,.0f"),
               "porsi_net_%": st.column_config.NumberColumn("Porsi %", format="%.2f"),
               "buy_avg": st.column_config.NumberColumn("Avg beli", format="%.1f"),
               "sell_avg": st.column_config.NumberColumn("Avg jual", format="%.1f"),
               "net_value": st.column_config.NumberColumn("Net value (Rp)", format="%,.0f")}
        a1, a2 = st.columns(2)
        a1.markdown("**Top net buy**")
        a1.dataframe(r.bs.sort_values("net_volume", ascending=False).head(10)[cols], hide_index=True, column_config=cfg, width="stretch")
        a2.markdown("**Top net sell**")
        a2.dataframe(r.bs.sort_values("net_volume").head(10)[cols], hide_index=True, column_config=cfg, width="stretch")
        st.markdown("**Rincian harian bandar**")
        st.dataframe(r.harian, hide_index=True, width="stretch")
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
                    rows.append({"Kode": cd, "Verdict": "ERROR", "Alasan": str(e)})
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
