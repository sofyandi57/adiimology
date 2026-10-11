"""Tab 'Ledger Broker': pelacak akumulasi/distribusi broker harian yang tersimpan (database), bukan hitungan sesaat."""
from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import adimology as ad
import apicache
import ledger as lg
import ledger_calc as lc
import ledger_score as ls

WARNA = {lc.ACC: "🟢", lc.DIST: "🔴", lc.MIXED: "🟡", lc.NEUTRAL: "⚪", lc.INSUF: "⚫"}


def ledger_url() -> str | None:
    for sec in ("ledger", "histori"):
        try:
            u = st.secrets["connections"][sec]["url"]
            if u:
                return str(u)
        except Exception:
            pass
    return os.environ.get("LEDGER_DB_URL") or os.environ.get("HISTORI_DB_URL") or None


@st.cache_resource(show_spinner=False)
def _eng(url: str, versi: int = 3):      # naikkan versi bila ada tabel baru (cache resource bertahan antar-deploy)
    return apicache.siapkan(lg.engine(url))


def _engine():
    u = ledger_url()
    return _eng(u or "sqlite:///ledger.sqlite"), bool(u)


def tampil(kode: str, api_key: str | None) -> None:
    if not kode:
        st.info("Isi kode saham di sidebar lalu klik **Hitung cepat** atau **Hitung lengkap**. Ledger memakai saham hasil perhitungan terakhir.")
        return
    st.subheader(f"Ledger Broker — {kode}")
    st.caption("Catatan harian broker per saham yang DISIMPAN di database dan diperbarui bertahap. Penanda: **[OBS]** teramati, **[EST]** estimasi model, "
               "**[INF]** inferensi, **[DEC]** keputusan. Broker adalah perantara banyak nasabah: angka di sini adalah *proksi aliran* (NET_FLOW_PROXY), "
               "bukan kepemilikan/harga pokok aktual, dan bukan identitas 'bandar'.")
    try:
        eng, persisten = _engine()
    except Exception as e:
        st.error(f"Database ledger tidak dapat dibuka: {e}")
        return
    if not persisten:
        st.warning("Database persisten belum diatur: ledger memakai file lokal sementara yang HILANG saat aplikasi di-restart/redeploy. "
                   "Isi `[connections.ledger] url` (atau pakai `[connections.histori]`) di Secrets.")
    meta = lg.muat_meta(eng, kode)
    if meta and meta.get("sessions_loaded"):
        sisa = max(int(meta.get("target_sessions") or lg.SESI_DEFAULT) - int(meta["sessions_loaded"]), 0)
        st.markdown(f"**Cakupan riwayat:** {meta['sessions_loaded']}/{meta['target_sessions']} sesi · {meta['first_available_trade_date']} → {meta['last_available_trade_date']} · "
                    f"history_complete={'ya' if meta['history_complete'] else 'TIDAK'} · sinkron terakhir {str(meta.get('last_successful_sync') or '-')[:16]} UTC")
        st.caption(f"Sisa ≈ {sisa} panggilan API untuk melengkapi target (bila pasar menyediakan sesi sebanyak itu; saham baru IPO punya riwayat lebih pendek). "
                   "Pembaruan harian hanya mengambil sesi baru.")
    else:
        st.caption("Ledger saham ini kosong. Klik **Perbarui / backfill**: pertama kali ≈ 1 panggilan per sesi (≤ target), dilakukan bertahap sesuai batas per klik. "
                   "Setelah itu pembaruan hanya sesi baru, dan **Hitung** memakai ledger tanpa panggilan API.")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    hari = c1.number_input("Sesi bursa dimuat (target)", 20, 250, 200, step=10, help="Default 200 SESI BURSA (bukan hari kalender). 1 panggilan API per sesi yang belum tersimpan; sesi final tidak diambil ulang.")
    mx = c2.number_input("Maks panggilan API / klik", 5, 250, 60, step=5, help="Menjaga kuota/rate limit. Klik lagi untuk melanjutkan backfill.")
    upd = c3.button("⬇️ Perbarui / backfill", width="stretch", disabled=not api_key)
    rek = c4.button("🔁 Rekonsiliasi 3 hari", width="stretch", disabled=not api_key,
                    help="Ambil ulang 3 hari terakhir dan koreksi bila sumber merevisi data.")
    if not api_key:
        st.info("Isi API key untuk memperbarui. Data yang sudah tersimpan tetap dapat dianalisis.")
    if upd or rek:
        client = apicache.DbClient(api_key, eng)
        try:
            if upd:
                bar = st.progress(0.0, text="Mengambil data broker harian…")
                r = lg.perbarui(client, eng, kode, int(hari), int(mx), progress=lambda i, n, d: bar.progress(i / max(n, 1), text=f"{d} ({i}/{n})"))
                bar.empty()
                st.success(f"Selesai: {r['diambil']} panggilan API, {r['sudah']} hari sudah final di database, {r['tersisa']} hari tersisa."
                           + (f" Gagal: {len(r['gagal'])} (lihat Log API)." if r["gagal"] else "")
                           + (f" Kosong: {len(r['kosong'])} hari." if r["kosong"] else ""))
            else:
                out = lg.rekonsiliasi(client, eng, kode, 3)
                st.success("Rekonsiliasi: " + "; ".join(f"{o['tgl']} {'DIKOREKSI' if o['dikoreksi'] else 'sama'}" for o in out))
        except Exception as e:
            st.error(f"Pembaruan gagal: {str(e)[:200]}")
    b, m = lg.muat(eng, kode, int(hari))
    if b.empty or m.empty:
        st.info("Belum ada data tersimpan untuk saham ini. Klik **Perbarui / backfill**.")
        return
    with st.expander("Ambang analisis (dapat diubah)"):
        a, bb, c = st.columns(3)
        cfg = lc.Cfg(min_hari_aktif=a.number_input("Min hari aktif", 5, 100, 20), persist_frac=a.slider("Porsi sesi net beli (persisten)", 0.3, 0.9, 0.6),
                     min_partisipasi=bb.number_input("Partisipasi min (net/volume)", 0.0, 0.5, 0.02, step=0.005, format="%.3f"),
                     episode_min_hari=bb.number_input("Episode min (sesi)", 3, 60, 5), dist_beruntun=int(c.number_input("Net jual beruntun", 2, 10, 3)),
                     turun_dari_puncak=c.slider("Turun dari puncak (alert)", 0.1, 0.8, 0.3), dist_nilai_mult=bb.number_input("Pengali nilai jual", 1.0, 5.0, 1.5, step=0.1))
        oi = st.number_input("Persediaan awal broker terpilih (lot, opsional; 0 = tidak diketahui)", 0, 10**9, 0)
        if oi > 0:
            cfg.opening_inventory = float(oi)
    res = lc.jalankan(b, m, cfg, lg.log_api(eng, kode))
    val = res["validasi"]
    if not res["ok"]:
        st.error(res.get("alasan", "ledger kosong"))
        return
    ik = {"OK": "✅", "PERINGATAN": "⚠️", "GAGAL": "⛔"}
    st.markdown(f"**Cakupan** {res['tgl_awal']} → {res['tgl_akhir']} · {val['cakupan']['hari']} hari · {val['cakupan']['n_broker']} broker · "
                f"**Validasi {ik.get(val['status'], '')} {val['status']}**")
    with st.expander("Validasi data & log API"):
        st.dataframe(pd.DataFrame(val["cek"]), hide_index=True, width="stretch")
        lg_df = lg.log_api(eng, kode)
        if not lg_df.empty:
            st.caption(f"Pemakaian API 7 hari: {len(lg_df)} panggilan, {int((lg_df.status != 'OK').sum())} error, rata-rata {lg_df.ms.mean():.0f} ms")
            st.dataframe(lg_df.head(50), hide_index=True, width="stretch")
    if val["status"] == "GAGAL":
        st.error("Validasi GAGAL: status dibatasi ke INSUFFICIENT_DATA sampai data diperbaiki.")

    sinyal, dihitung = ls.hitung_dan_simpan(eng, res, lg.muat_meta(eng, kode))
    st.markdown("### Bandar Score (0–10)")
    if sinyal.get("bandar_score") is None:
        st.info(f"N/A · {sinyal['episode_status']}: {'; '.join(sinyal['confidence_reason'])}")
    else:
        sc = sinyal["bandar_score"]
        box = st.success if sc >= 8 else st.info if sc >= 6 else st.warning if sc >= 4 else st.error
        box(f"**{sc}/10 · {sinyal['label']}**  \nStatus episode: **{sinyal['episode_status']}** · confidence **{sinyal['confidence_score']}** ({'; '.join(sinyal['confidence_reason'])})  \n"
            f"Observasi {sinyal['observation_start_date']} → {sinyal['observation_end_date']} · kelompok kunci: {', '.join(sinyal['key_brokers']) or '-'} · formula {sinyal['formula_version']}")
        st.caption("Lapisan positioning/timing berbasis aturan, belum terbukti punya keunggulan, dan BUKAN identitas/niat pihak tertentu. Bobot maksimum dalam kerangka keputusan: 10%.")
        cc1, cc2 = st.columns(2)
        with cc1:
            st.dataframe(pd.DataFrame(sinyal["score_components"]).rename(columns={"dimensi": "Dimensi", "poin": "Poin", "maks": "Maks", "detail": "Detail"}), hide_index=True, width="stretch")
        with cc2:
            st.markdown("**Bukti mendukung**\n" + "\n".join(f"- {x}" for x in sinyal["supporting_evidence"][:8]))
            st.markdown("**Bukti berlawanan**\n" + "\n".join(f"- {x}" for x in sinyal["contradicting_evidence"][:8]))
        st.caption("Sinyal disimpan di database (ledger_signal) dan hanya dihitung ulang bila data ledger atau konfigurasi berubah." + (" Dihitung ulang sekarang." if dihitung else " Memakai sinyal tersimpan."))
    if sinyal.get("distribusi_ritel") or sinyal.get("zona_pasokan"):
        with st.expander("Distribusi ke ritel & zona pasokan (broker ritel XL/YP/XC/PD)", expanded=bool(sinyal.get("zona_pasokan"))):
            if sinyal.get("distribusi_ritel"):
                st.markdown("**Hari volume ekstrem di dekat puncak ketika broker ritel net beli besar** (kandidat distribusi ke ritel; `turun` = harga turun ≥ 5% dalam 5 sesi)")
                st.dataframe(pd.DataFrame(sinyal["distribusi_ritel"]), hide_index=True, width="stretch")
            if sinyal.get("zona_pasokan"):
                st.markdown("**Zona pasokan di atas harga sekarang** (ritel menumpuk net beli; pemegang yang masih rugi cenderung menjual saat harga kembali ke sana; memotong target jual)")
                st.dataframe(pd.DataFrame(sinyal["zona_pasokan"]).rename(columns={"lo": "Dari", "hi": "Sampai", "lot": "Lot ritel net beli", "pct": "Porsi"}), hide_index=True, width="stretch",
                             column_config={"Porsi": st.column_config.NumberColumn(format="percent")})
    with st.expander("Rolling flow per broker (sesi bursa: 1D/5D/20D/60D/120D/YTD)"):
        Rr = ls.rolling_broker(res["ledger"])
        tl = Rr["trade_date"].max()
        snap = Rr[Rr["trade_date"] == tl].copy()
        snap["abs60"] = snap["net_lots_60d"].abs().fillna(snap["net_lots_5d"].abs())
        tampil = snap.sort_values("abs60", ascending=False).head(15)[["broker_code", "net_lots_1d", "net_lots_5d", "net_lots_20d", "net_lots_60d", "net_lots_120d", "net_lots_ytd",
                                                                       "net_flow_pct_volume_20d", "net_buy_days_20d", "net_sell_days_20d", "net_sell_streak"]]
        st.caption(f"Per {tl}; jendela dihitung dari sesi bursa termuat (kosong = riwayat belum cukup). 1D = satu sesi. Top 15 berdasarkan |net 60D|.")
        st.dataframe(tampil, hide_index=True, width="stretch", column_config={"net_flow_pct_volume_20d": st.column_config.NumberColumn(format="percent")})
        st.download_button("⬇️ Unduh rolling snapshot (CSV)", snap.drop(columns=["abs60"]).to_csv(index=False).encode(), f"ledger_derived_{kode}_{tl}.csv", "text/csv")
    j = lc.jawaban(res, kode)
    st.markdown("### Ringkasan")
    st.table(pd.DataFrame(j["ringkasan"], columns=["Butir", "Isi"]).set_index("Butir"))
    if res["alert"]:
        st.markdown("### Peringatan")
        st.dataframe(pd.DataFrame(res["alert"]).rename(columns={"tipe": "Peringatan", "level": "Level", "broker": "Broker", "ts": "Waktu", "jendela": "Jendela",
                                                                "bukti": "Data pemicu", "confidence": "Confidence"}), hide_index=True, width="stretch")

    H, pr = res["hasil"], res["peringkat"]
    st.markdown("### Peringkat broker")
    t1, t2, t3, t4, t5 = st.tabs(["Net buyer", "Net seller", "Gross (ex-crossing)", "Akumulator persisten", "Geser akumulasi → distribusi"])
    cols = {"broker": "Broker", "nama": "Nama", "net_flow_lots": "Net lot", "gross_buy_ex_crossing": "Gross beli ex-cross", "gross_sell_ex_crossing": "Gross jual ex-cross",
            "status": "Status", "confidence": "Confidence"}
    show = lambda df: st.dataframe(df[list(cols)].rename(columns=cols), hide_index=True, width="stretch")  # noqa: E731
    with t1:
        show(pr["top_net_buyer"])
    with t2:
        show(pr["top_net_seller"])
    with t3:
        a, bb = st.columns(2)
        a.caption("Gross beli terbesar")
        with a:
            show(pr["gross_buyer"])
        bb.caption("Gross jual terbesar")
        with bb:
            show(pr["gross_seller"])
    with t4:
        show(pr["persisten"]) if not pr["persisten"].empty else st.caption("N/A: tidak ada.")
    with t5:
        show(pr["bergeser"]) if not pr["bergeser"].empty else st.caption("N/A: tidak ada.")

    st.markdown("### Timeline broker")
    urut = sorted(H, key=lambda k: -abs(H[k]["net_flow_lots"]))
    pilih = st.selectbox("Broker", urut, format_func=lambda k: f"{WARNA[H[k]['status']]} {k} · {H[k]['nama']} · {H[k]['status']} · net {H[k]['net_flow_lots']:,.0f} lot")
    g = res["ledger"][res["ledger"]["broker_code"] == pilih].sort_values("trade_date")
    v = H[pilih]
    fig = go.Figure()
    fig.add_bar(x=g["trade_date"], y=g["net_lots"], name="Net lot harian", marker_color=["#2e7d32" if x >= 0 else "#c62828" for x in g["net_lots"]], opacity=0.6)
    fig.add_scatter(x=g["trade_date"], y=g["cumulative_net_lots"], name="Net kumulatif (lot)", line=dict(color="#1565c0", width=2))
    fig.add_scatter(x=g["trade_date"], y=g["close_price"], name="Close", yaxis="y2", line=dict(color="#555", width=1.5))
    if v.get("episode") and pd.notna(v["episode"].get("avg_buy_ex_crossing")):
        fig.add_hline(y=v["episode"]["avg_buy_ex_crossing"], yref="y2", line_dash="dot", line_color="#ef6c00", annotation_text="proksi biaya (ex-crossing)")
        fig.add_vline(x=pd.Timestamp(v["episode"]["mulai"]).timestamp() * 1000, line_dash="dash", line_color="#2e7d32", annotation_text="mulai episode")
    for key, c in (("early", "#fbc02d"), ("major", "#c62828"), ("breakout", "#6a1b9a")):
        z = res["zona"].get("zona", {}).get(key)
        if z:
            fig.add_hrect(y0=z["harga_rendah"], y1=z["harga_tinggi"], yref="y2", fillcolor=c, opacity=0.12, line_width=0, annotation_text=z["nama"].split(" (")[0],
                          annotation_position="top left")
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"), yaxis=dict(title="lot"), yaxis2=dict(title="harga", overlaying="y", side="right"))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Status {v['status']} (confidence {v['confidence']}). " + (v.get("catatan") or "") +
               " Sinyal akumulasi: " + ", ".join(f"{'✔' if ok else '✘'} {s}" for s, ok in v["sinyal_acc"].items()))

    st.markdown("### Estimasi posisi (NET_FLOW_PROXY)")
    tp = lc.tabel_posisi(res)
    st.dataframe(tp, hide_index=True, width="stretch", column_config={
        c: st.column_config.NumberColumn(format="%,.0f") for c in ("Gross Buy Lots", "Gross Sell Lots", "Net Flow Lots", "Est. Remaining Low", "Est. Remaining High", "Est. Remaining Base")} |
        {c: st.column_config.NumberColumn(format="%.1f") for c in ("Avg Buy (gross)", "Avg Buy (ex-crossing)", "Net-flow cost proxy")} |
        {"Crossing % gross buy": st.column_config.NumberColumn(format="percent")})
    st.caption("Avg Buy (gross) = total nilai beli / total lembar beli. Ex-crossing membuang hari yang beli≈jual dalam jumlah besar. Net-flow cost proxy = selisih nilai / "
               "selisih lembar bersih (hanya bila net beli positif). Est. Remaining: rendah = persediaan awal 0; tinggi = persediaan awal minimal agar posisi tidak negatif; "
               "Base = N/A kecuali Anda mengisi persediaan awal. Bukan kepemilikan aktual.")

    st.markdown("### Zona distribusi yang diamati")
    z = res["zona"].get("zona", {})
    if not z:
        st.caption("N/A: belum ada penjualan teramati oleh broker kunci.")
    for key in ("early", "major", "breakout"):
        zz = z.get(key)
        if zz:
            with st.container(border=True):
                st.markdown(f"**{zz['nama']}** · {zz['harga_rendah']:,.0f}–{zz['harga_tinggi']:,.0f} · confidence {zz['confidence']}")
                st.caption(f"[OBS] tanggal: {', '.join(zz['tanggal'])} · broker: {', '.join(zz['broker'])} · gross jual {zz['gross_sell_lots']:,.0f} lot · "
                           f"net {zz['net_lots']:,.0f} lot · nilai jual Rp {zz['sell_value']:,.0f} · reaksi harga "
                           + (f"rata-rata {zz['reaksi_rata']:+.1%} dalam {res['cfg'].reaksi_hari} sesi" if pd.notna(zz['reaksi_rata']) else "N/A")
                           + f"\n\nInvalidasi: {zz['invalidasi']}. Ini area penjualan teramati, bukan target harga broker.")
        elif key == "breakout":
            st.caption("Breakout continuation: N/A (tidak ada penjualan yang diikuti kenaikan harga ≥ ambang).")

    st.markdown("### Jawaban terstruktur")
    for q, a in j["jawab"]:
        with st.expander(q):
            st.write(a)

    st.download_button("⬇️ Unduh ledger (CSV)", lc.csv_ledger(res), f"ledger_{kode}_{res['tgl_akhir']}.csv", "text/csv", width="stretch")
