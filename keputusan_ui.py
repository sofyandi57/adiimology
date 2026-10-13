"""Kartu keputusan tunggal di atas aplikasi: satu aksi, dasar, rencana angka, konflik sinyal, kualitas data."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import audit as au
import bandar as bd
import konteks as ktx
import laporan as lp
import skor as sk

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
    if rk and rk["verdict"] == "BELI DI AREA ENTRY" and gate == "lain" and nilai["kategori"] not in lp._BANDAR_POSITIF:
        k.append("Teknikal di area beli, tetapi tidak ada jejak akumulasi bandar")
    if rk and rk["verdict"] == "BELI DI AREA ENTRY" and gate == "waspada":
        k.append("Teknikal di area beli, tetapi ada jebakan bandar (lihat tab Bandar)")
    if rk and rk["verdict"] == "HINDARI" and gate == "kuat":
        k.append("Bandar terindikasi akumulasi, tetapi teknikal HINDARI (harga lemah/tembus SL)")
    if hk_ is not None and not hk_["checklist"]["lolos_inti"] and rk and rk["verdict"] == "BELI DI AREA ENTRY":
        k.append("Aturan inti Hengky belum terpenuhi")
    if r.verdict and "BELI" in str(r.verdict).upper() and (gate != "kuat" or (rk and rk["verdict"] != "BELI DI AREA ENTRY")):
        k.append(f"Skenario Adimology '{r.verdict}' tidak didukung sinyal lain (skenario ini tidak terbukti unggul di backtest)")
    return k


def situasi_sesi():
    """Situasi khusus (backdoor / perubahan pengendali) dari data sesi: konteks (insider 365 hari), kalender aksi, harga, catatan manual."""
    import situasi_khusus as sx_
    res = st.session_state.get("res")
    if not res or res[0] != "ok":
        return None
    r = res[1]
    rk, f = _ok("tek"), _ok("fund")
    try:
        return sx_.deteksi(st.session_state.get("konteks"), (f or {}).get("aksi"), rk["ind"] if rk else r.chart, st.session_state.get("catatan_khusus") or [],
                           r.harga, r.params.modal, float(st.session_state.get("maks_spek_pct", 0.02)), r.params.fee_beli)
    except Exception:
        return None


def ctx_sesi(mode: str) -> dict:
    return dict(mode=mode, konteks=st.session_state.get("konteks"), ledger=st.session_state.get("ledger_sinyal"), flow=st.session_state.get("flow"),
                validasi=st.session_state.get("_validasi_status"), situasi=situasi_sesi())


def hitung(r, mode: str):
    """(kep, alasan, rk, f, alr, hk_, nilai). Mode cepat tidak boleh menghasilkan BELI."""
    rk, f, alr, hk_ = _ok("tek"), _ok("fund"), _ok("aliran"), _ok("hengky")
    nilai = st.session_state.get("nilai") or bd.penilaian(r, alr, rk)
    kep, alasan = lp.keputusan(r, rk, f, alr, nilai, hk_, ctx_sesi(mode))
    return kep, alasan, rk, f, alr, hk_, nilai


def tampil(r, mode: str) -> None:
    kep, alasan, rk, f, alr, hk_, nilai = hitung(r, mode)
    sc = sk.hitung(r, rk, f, alr, nilai, hk_, ctx_sesi(mode))
    kolom = st.columns(5)
    for c_, d in zip(kolom[:4], sc["dims"]):
        c_.metric(f"{d['kode']} · {d['nama'].split(' (')[0]}", f"{d['nilai']:.2f}" if d["nilai"] is not None else "NULL",
                  (f"{d['nilai'] * 10:.1f}/10 · {d['arti']}" if d["nilai"] is not None else "belum dapat dinilai"), delta_color="off")
        if d["nilai"] is not None:
            c_.progress(d["nilai"])
    kolom[4].metric("Total", f"{sc['total']:.2f}" if sc["total"] is not None else "N/A", f"coverage {sc['coverage']:.0%}", delta_color="off")
    if sc["total"] is not None:
        kolom[4].progress(min(sc["total"], 1.0))
    st.caption("Skor diskrit 0 · 0,25 · 0,50 · 0,75 · 1,00 (×10 = skala 1–10). NULL = belum dapat dinilai, tidak dianggap nol. Total = rata-rata komponen valid; Score_Coverage = porsi komponen valid. "
               "Heuristik internal yang BELUM terbukti. Hard gate di bawah dapat membatasi vonis walau Total tinggi.")
    for g in sc["gates"]:
        (st.error if g["berat"] == "AVOID" else st.warning)(f"**{g['gate']}** ({'membatasi ke HINDARI' if g['berat'] == 'AVOID' else 'membatasi ke TUNGGU'}): {g['alasan']}", icon="⛔" if g["berat"] == "AVOID" else "🚧")
    sx = sc.get("situasi")
    if sx and sx.get("aktif"):
        ps = sx["posisi"]
        st.warning(f"### ⚠️ SITUASI KHUSUS: {sx['jenis']} — SPEKULATIF\nTahap: {sx['tahap_teks']}.  \n"
                   f"**Posisi maksimal {ps['maks_pct']:.1%} modal ≈ {ps['lot']:,} lot (Rp{ps['nilai']:,.0f}).** {ps['aturan'].capitalize()}.  \n"
                   + ("Tanggal kunci: " + "; ".join(sx["tanggal_kunci"]) if sx["tanggal_kunci"] else "Tanggal kunci: belum ada (tambahkan catatan manual bila tahu tenggatnya)."), icon="🎲")
        with st.expander("Rincian situasi khusus (bukti, bendera, kondisi batal)"):
            st.markdown("\n".join(f"- {x}" for x in sx["bukti"]) or "-")
            for x_ in sx["flags"]:
                st.error(x_, icon="🚩")
            st.markdown("**Batal / keluar bila:** " + ("; ".join(sx["invalidasi"]) or "-"))
            st.caption(sx["catatan_data"] + " Skor TA/FA/BA kurang relevan untuk saham event-driven: harga digerakkan nasib transaksi.")
    box = {"BELI": st.success, "TUNGGU": st.warning}.get(kep, st.error)
    sub = sc["ringkas"] + " · " + "; ".join(sc["ket"])
    box(f"### {r.params.kode} → {kep}\n{sub}", icon={"BELI": "🟢", "TUNGGU": "🟡"}.get(kep, "🔴"))
    pos = sc["posisi"]
    cp_, pm_ = st.columns(2)
    cp_.info(f"**Calon pembeli:** {pos['calon_pembeli']}", icon="🛒")
    pm_.info(f"**Sudah punya saham ini:** {pos['pemegang']}", icon="📦")
    for fl in sc["flags"]:
        st.warning("Bendera: " + fl, icon="🚩")
    with st.expander("📐 Rincian skor (komponen, bukti pendukung dan berlawanan)"):
        for d in sc["dims"]:
            st.markdown(f"**{d['kode']} {d['nama']}: {('%.2f' % d['nilai']) if d['nilai'] is not None else 'NULL'} ({d['arti']})**")
            if d["catatan"]:
                st.caption(d["catatan"])
            if d["komponen"]:
                st.dataframe(pd.DataFrame(d["komponen"]).rename(columns={"nama": "Komponen", "poin": "Poin", "maks": "Maks", "detail": "Detail"}), hide_index=True, width="stretch")
            c1_, c2_ = st.columns(2)
            c1_.markdown("**Mendukung**\n" + ("\n".join(f"- {x}" for x in d["bukti"][:8]) or "- -"))
            c2_.markdown("**Berlawanan**\n" + ("\n".join(f"- {x}" for x in d["lawan"][:8]) or "- -"))
            for ln in d.get("lensa", []):
                st.caption("Lensa bandar: " + ln)
        st.markdown("**Kondisi invalidasi:** " + ("; ".join(sc["invalidasi"]) or "-"))
        st.caption(f"Versi skor {sc['versi']} · kualitas data: {sc['data_quality']}")
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

    kx = st.session_state.get("konteks")
    for w in ktx.catatan_risiko(kx):
        st.warning(w, icon="🚩" if "NOTASI" in w else "⚠️")
    try:
        val = au.validasi(r, rk, f, au.skenario_posisi(r, rk))
    except Exception:
        val = None
    if val and val["status"] == "GAGAL":
        st.error("**Validasi angka GAGAL, jangan dipakai:** " + " | ".join(c["detail"] for c in val["cek"] if c["status"] == "GAGAL"), icon="⛔")
    elif val and val["n_peringatan"]:
        st.caption("Peringatan validasi: " + " | ".join(c["detail"] for c in val["cek"] if c["status"] == "PERINGATAN")[:300])
    kf = konflik(rk, nilai, hk_, f, r)
    if kf:
        st.warning("**Sinyal tidak searah:** " + "; ".join(kf))
    ubah = [x["kriteria"] for x in nilai["kriteria"] if x["status"] in ("tidak", "netral")] + \
           ([x["kriteria"] for x in hk_["checklist"]["kriteria"] if x["status"] in ("tidak", "netral") and x["inti"]] if hk_ else [])
    if kep != "BELI":
        st.markdown(f"**Agar vonis menjadi BELI:** Total ≥ 0,70 (sekarang {('%.2f' % sc['total']) if sc['total'] is not None else 'N/A'}), coverage ≥ 75%, tidak ada komponen < 0,50, harga di area beli, tanpa hard gate, Hitung lengkap, aturan inti Hengky lolos. "
                    + ("**Kriteria yang BELUM terpenuhi dan menahan skor** (jika terpenuhi, skor naik): " + "; ".join(f"'{u}'" for u in ubah[:6]) + "." if ubah else "Tidak ada kriteria checklist yang tertinggal."))
    if rk:
        sup = rk["sup"][0]["harga"] if rk["sup"] else None
        st.markdown(f"**Batal/keluar:** close di bawah support {sup:,.0f} (SL {rk['sl']:,.0f})" if sup else f"**Batal/keluar:** close di bawah SL {rk['sl']:,.0f}")

    with st.expander("🌐 Konteks pasar & risiko (IHSG, notasi khusus, keterbukaan, insider, KSEI, peringkat)", expanded=False):
        for ln in ktx.baris_laporan(kx)[1:]:
            st.markdown(ln if ln.startswith("- ") else f"- {ln}")
        st.caption("Konteks hanya menahan BELI untuk notasi khusus; lainnya peringatan, tidak membalik vonis. Bagian yang 'N/A' berarti endpoint tidak memberi data (tidak ditebak).")
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


def alasan_ringkas(kep: str, alasan: list[str], maks: int = 4) -> str:
    """Alasan utama satu baris untuk tabel Histori: skor tiga dimensi + bendera; bila tidak ada, ringkasan penyebab."""
    for a in alasan:
        if a.startswith("RINGKAS: "):
            bend = [x[len("Bendera: "):] for x in alasan if x.startswith("Bendera: ")]
            return a[len("RINGKAS: "):] + ((" · bendera: " + ", ".join(bend)) if bend else "")
    out = []
    for a in alasan:
        t = a.strip()
        low = t.lower()
        if kep.startswith("HINDARI"):
            if "veto" in low and "bandar" in low and "ledger" not in low:
                out.append("Bandar distribusi/menjual")
            elif t.startswith("Ledger broker: status"):
                out.append("Ledger: " + t.split("status ")[1].split(" ")[0] + " (bandar keluar)")
            elif t.startswith("Fundamental: ") and "tidak ada tanda" not in low and "tidak terukur" not in low:
                out.append("Fundamental buruk: " + t[len("Fundamental: "):][:70])
            elif t.startswith("Teknikal: HINDARI"):
                out.append("Teknikal HINDARI")
            elif t.startswith("Konteks:") and "NOTASI" in t:
                out.append("Notasi khusus BEI")
        else:
            if t.startswith("Bandarmologi: "):
                out.append("Bandar: " + t[len("Bandarmologi: "):].split(" (")[0])
            elif t.startswith("Teknikal: "):
                out.append("Teknikal: " + t[len("Teknikal: "):].split(" (")[0])
            elif t.startswith("BELI ditahan") or t.startswith("TUNGGU:"):
                out.append(t.split(":")[0].replace("BELI ditahan", "BELI ditahan") + (": " + t.split(":", 1)[1].strip()[:60] if ":" in t else ""))
            elif t.startswith("Fundamental: ") and "tidak ada tanda" not in low and "tidak terukur" not in low:
                out.append("Fundamental: " + t[len("Fundamental: "):][:50])
            elif t.startswith("Konteks:") or "notasi khusus" in low:
                out.append(t[:60])
    seen, uniq = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return " · ".join(uniq[:maks]) or ("lihat laporan" if alasan else "-")
