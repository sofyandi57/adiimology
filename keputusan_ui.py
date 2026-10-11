"""Kartu keputusan tunggal di atas aplikasi: satu aksi, dasar, rencana angka, konflik sinyal, kualitas data."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import audit as au
import bandar as bd
import laporan as lp

MODUL = [("tek", "Teknikal"), ("aliran", "Aliran asing/nego"), ("dinamis", "Dinamika bandar"), ("b200", "Bandar 200 hari"),
         ("hengky", "Aturan Hengky"), ("fund", "Fundamental"), ("ovn", "BPJP/BSJP")]


def _ok(st_):
    v = st.session_state.get(st_)
    return v[1] if v and v[0] == "ok" else None


def status_modul() -> list[tuple[str, str]]:
    out = []
    for k, nama in MODUL:
        v = st.session_state.get(k)
        out.append((nama, "ok" if v and v[0] == "ok" else "gagal" if v and v[0] == "err" else "belum"))
    return out


def konflik(rk, nilai, hk_, f, r) -> list[str]:
    k = []
    gate = bd.gate(nilai)
    if rk and rk["verdict"] == "BELI DI AREA ENTRY" and gate != "kuat":
        k.append("Teknikal di area beli, tetapi bukti bandarmologi belum cukup kuat")
    if rk and rk["verdict"] == "HINDARI" and gate == "kuat":
        k.append("Bandar terindikasi akumulasi, tetapi teknikal HINDARI (harga lemah/tembus SL)")
    if hk_ is not None and not hk_["checklist"]["lolos_inti"] and rk and rk["verdict"] == "BELI DI AREA ENTRY":
        k.append("Aturan inti Hengky belum terpenuhi")
    if r.verdict and "BELI" in str(r.verdict).upper() and (gate != "kuat" or (rk and rk["verdict"] != "BELI DI AREA ENTRY")):
        k.append(f"Skenario Adimology '{r.verdict}' tidak didukung sinyal lain (skenario ini tidak terbukti unggul di backtest)")
    return k


def hitung(r, mode: str):
    """(kep, alasan, rk, f, alr, hk_, nilai). Mode cepat tidak boleh menghasilkan BELI."""
    rk, f, alr, hk_ = _ok("tek"), _ok("fund"), _ok("aliran"), _ok("hengky")
    nilai = st.session_state.get("nilai") or bd.penilaian(r, alr, rk)
    kep, alasan = lp.keputusan(r, rk, f, alr, nilai, hk_)
    if mode == "cepat" and kep == "BELI":
        kep = "TUNGGU"
        alasan.append("BELI hanya dapat ditetapkan pada analisis LENGKAP (dinamika bandar & aturan Hengky belum dihitung)")
    return kep, alasan, rk, f, alr, hk_, nilai


def tampil(r, mode: str) -> None:
    kep, alasan, rk, f, alr, hk_, nilai = hitung(r, mode)
    box = {"BELI": st.success, "TUNGGU": st.warning}.get(kep, st.error)
    sub = " · ".join(a for a in alasan[:3])
    box(f"### {r.params.kode} → {kep}\n{sub}", icon={"BELI": "🟢", "TUNGGU": "🟡"}.get(kep, "🔴"))
    if mode == "cepat":
        st.caption("Mode **cepat**: keputusan sementara dari data dasar (harga, broker window, teknikal). Fundamental, dinamika bandar, Hengky, ritel belum dihitung.")

    if rk:
        c = st.columns(5)
        c[0].metric("Harga", f"{rk['harga']:,.0f}")
        c[1].metric("Area beli", f"{rk['entry_lo']:,.0f}–{rk['entry_hi']:,.0f}" if rk["zona_valid"] else "tidak ada", help="Zona support dengan R/R ≥ 1,5")
        c[2].metric("Entry maksimal", f"{rk['entry_maks']:,.0f}")
        c[3].metric("Stop loss", f"{rk['sl']:,.0f}", f"{rk['sl'] / rk['harga'] - 1:+.1%}", delta_color="off")
        c[4].metric("Area jual", f"{rk['tp1']:,.0f}–{rk['tp2']:,.0f}", f"{rk['tp1'] / rk['harga'] - 1:+.1%} s/d {rk['tp2'] / rk['harga'] - 1:+.1%}", delta_color="off")
    else:
        st.warning("Rencana angka (entry/SL/target) tidak tersedia: analisa teknikal gagal atau data harga kurang.")

    kf = konflik(rk, nilai, hk_, f, r)
    if kf:
        st.warning("**Sinyal tidak searah:** " + "; ".join(kf))
    ubah = [x["kriteria"] for x in nilai["kriteria"] if x["status"] in ("tidak", "netral")] + \
           ([x["kriteria"] for x in hk_["checklist"]["kriteria"] if x["status"] in ("tidak", "netral") and x["inti"]] if hk_ else [])
    if kep != "BELI":
        st.markdown("**Keputusan berubah menjadi BELI jika:** " + ("; ".join(f"'{u}' terpenuhi" for u in ubah[:6]) or "tidak ada syarat tersisa") + " (dan tidak ada jebakan).")
    if rk:
        sup = rk["sup"][0]["harga"] if rk["sup"] else None
        st.markdown(f"**Batal/keluar:** close di bawah support {sup:,.0f} (SL {rk['sl']:,.0f})" if sup else f"**Batal/keluar:** close di bawah SL {rk['sl']:,.0f}")

    with st.expander("Dasar keputusan & kualitas data"):
        rows = [("Bandarmologi", f"{nilai['kategori']}" + (f" · bukti: {nilai['tingkat']}" if nilai.get("tingkat") else "")),
                ("Teknikal", f"{rk['verdict']} · skor {rk['skor']:+d}/±{rk['skor_max']}" if rk else "tidak terukur"),
                ("Aturan Hengky", hk_["checklist"]["hasil"] if hk_ else "belum dihitung"),
                ("Fundamental", ("ada tanda bahaya besar" if (f and lp._fatal(f)) else f"tidak ada tanda bahaya besar ({len(f.get('peringatan', []))} catatan)") if f else "belum dihitung"),
                ("Skenario Adimology", f"{r.verdict} (bukan dasar keputusan)")]
        st.dataframe(pd.DataFrame(rows, columns=["Sumber", "Hasil"]), hide_index=True, width="stretch")
        sm = status_modul()
        st.caption("Modul: " + " · ".join(f"{'✅' if s == 'ok' else '❌' if s == 'gagal' else '➖'} {n}" for n, s in sm) + "   (✅ terhitung, ❌ gagal, ➖ belum)")
        st.caption("Semua aturan heuristik dan **belum terbukti** secara statistik; target Adimology tidak punya keunggulan di backtest. Keputusan = disiplin checklist, bukan prediksi/probabilitas. Bukan ajakan transaksi.")
    if mode == "cepat":
        if st.button("➕ Lengkapi analisis (fundamental, dinamika bandar, Hengky, ritel)", type="primary"):
            st.session_state["_lengkapi"] = True
            st.rerun()
