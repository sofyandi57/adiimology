"""Smoke test UI Streamlit dengan requests.get dipalsukan."""
import adimology as ad
import bandar as bd
import numpy as np
import pandas as pd
import requests
from streamlit.testing.v1 import AppTest
from fakes import mcas

APP = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "app.py")
EMAIL, PW = "user@example.com", "rahasia-tes"

class R:
    status_code = 200
    def __init__(s, j): s.j = j
    def raise_for_status(s): pass
    def json(s): return s.j


def login_env(monkeypatch):
    monkeypatch.setenv("APP_LOGIN_EMAIL", EMAIL)
    monkeypatch.setenv("APP_LOGIN_PASSWORD", PW)


def open_app(monkeypatch, email=EMAIL, pw=PW):
    login_env(monkeypatch)
    at = AppTest.from_file(APP, default_timeout=60).run()
    [t for t in at.text_input if t.label == "Email"][0].set_value(email)
    [t for t in at.text_input if t.label == "Password"][0].set_value(pw)
    [b for b in at.button if b.label == "Masuk"][0].click().run()
    if at.text_input and [t for t in at.text_input if t.label == "Kode saham"]:
        [t for t in at.text_input if t.label == "Kode saham"][0].set_value("MCAS")
    return at


def test_login_required(monkeypatch):
    login_env(monkeypatch)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert [t.label for t in at.text_input] == ["Email", "Password"]
    assert not [b for b in at.button if b.label == "Hitung lengkap"]


def test_login_wrong_password(monkeypatch):
    at = open_app(monkeypatch, pw="salah")
    assert any("salah" in e.value for e in at.error)
    assert not [b for b in at.button if b.label == "Hitung lengkap"]

def test_app_runs_and_computes(monkeypatch, tmp_path):
    monkeypatch.setenv("HISTORI_DB_URL", f"sqlite:///{tmp_path / 'histori.db'}")
    fc = mcas()
    def fake_get(url, headers=None, params=None, timeout=None):
        path = url.replace(ad.BASE_URL, "")
        return R(fc.get(path, params))
    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setenv("INVEZGO_API_KEY", "dummy")
    at = open_app(monkeypatch)
    assert not at.exception
    [b for b in at.button if b.label == "Hitung lengkap"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("TIDAK ADA SINYAL" in w.value and "skenario" in w.value for w in at.info)   # Adimology = skenario
    labels = [m.label for m in at.metric]
    assert "Target realistis" in labels and "Target max" in labels
    vals = {m.label: m.value for m in at.metric}
    assert vals["Target realistis"] == "292" and vals["Target max"] == "296"
    pos = [d.value for d in at.dataframe if "Rugi di stop (Rp, incl. fee)" in d.value.columns]
    assert pos and len(pos[0]) >= 1 and (pos[0]["Rugi di stop (Rp, incl. fee)"].fillna(0) <= 100_000 + 1).all()   # <= anggaran risiko 1% dari 10 jt
    assert "histori" not in at.session_state  # tersimpan ke DB, bukan ke sesi
    # tab Rekomendasi terisi (data tiruan hanya 5 hari -> TIDAK TERUKUR, tapi tetap tampil tanpa error)
    tek = at.session_state["tek"]
    assert tek[0] == "ok" and tek[1]["verdict"] == "TIDAK TERUKUR"
    rek = [d.value for d in at.dataframe if "Catatan" in d.value.columns][0]
    assert {"Area beli (entry)", "Entry maksimal (R/R 1,5)", "Stop loss", "Area jual (TP1 – TP2)"} <= set(rek["Keterangan"])
    fund = at.session_state["fund"]
    assert fund[0] == "ok" and fund[1]["periode"] == "Q2 2026" and fund[1]["nama"] == "M Cash Integrasi"
    assert any("naik" in x.value for x in at.success)                 # sorotan hijau
    assert any("Laba rugi" in m.value for m in at.markdown)
    ovn = at.session_state["ovn"]
    assert ovn[0] == "ok" and ovn[1]["n_hari"] == 40 and ovn[1]["sinyal"]["bsjp"] is not None
    assert any("BSJP" in m.value for m in at.markdown)
    rit = at.session_state["ritel"]
    assert rit["minat"].startswith("MENURUN") and rit["status"] == "CAMPURAN: SID TURUN, RITEL NET BELI"   # SID 5000->4600; CC & XL net beli
    assert rit["window"]["codes"] == ["XL", "XC", "PD", "CC", "YP"]
    assert any("Minat ritel" in x.value for x in at.subheader)
    assert b"## MINAT RITEL" in at.session_state["export"]["txt"] and b"minat=MENURUN" in at.session_state["export"]["txt"]
    assert b"## LEDGER BROKER" in at.session_state["export"]["txt"] and b"## LEDGER BROKER" in at.session_state["export"]["md"] and "csv" in at.session_state["export"]
    hkr = at.session_state["hengky"]
    assert hkr[0] == "ok" and not hkr[1]["checklist"]["lolos_inti"]            # data tiruan: tidak uptrend / data kurang
    assert any("Aturan" in d.value.columns and "Status" in d.value.columns for d in at.dataframe)
    assert hkr[1]["sid"]["status"] == "SID TURUN SAAT SIDEWAYS"
    [b for b in at.button if b.label == "Cek IEP / IEV sekarang"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["pra_open"][1]["gap"] > 0
    b200 = at.session_state["b200"]
    assert b200[0] == "err" and "kurang dari 20 hari" in b200[1]      # data tiruan hanya 5 hari harga; sukses diuji di test_bandar_panjang
    assert "hari_terakhir" in at.session_state["vol"]
    assert any("Volume rata-rata harian" in m.value for m in at.markdown)
    din = at.session_state["dinamis"]
    assert din[0] == "err" and "kurang dari 20 hari" in din[1]                 # data tiruan 5 hari; sukses diuji di test_dinamika
    assert b"[GAGAL] kerangka bukti bandarmologi" in at.session_state["export"]["txt"] and b"TIDAK KONKLUSIF" in at.session_state["export"]["txt"]
    nilai = at.session_state["nilai"]
    assert nilai["kategori"] in bd.KATEGORI and len(nilai["kriteria"]) == 7
    assert any("Bandarmologi (" in x.value for x in [*at.success, *at.info, *at.warning, *at.error])
    assert any("Kriteria" in d.value.columns for d in at.dataframe)
    aliran = at.session_state["aliran"]
    assert aliran[0] == "ok", aliran
    import aliran as al_mod
    konf = pd.DataFrame(al_mod.ringkas(at.session_state["res"][1], aliran[1]))
    assert {"Konsentrasi (top 3 beli vs top 15 jual)", "Asing (window)", "Pasar negosiasi (window)",
            "Konsistensi bandar 1/3/7/20 hari", "Afiliasi pemegang saham ≥5%"} <= set(konf["Konfirmasi"])
    assert konf.set_index("Konfirmasi").loc["Afiliasi pemegang saham ≥5%", "Status"] == "TERAFILIASI"
    exp = at.session_state["export"]
    assert exp["pdf"][:4] == b"%PDF" and exp["xlsx"][:2] == b"PK" and exp["nama"].startswith("riset_MCAS_")
    assert exp["md"].startswith(b"# MCAS \xe2\x80\x94") and b"## B. KESIMPULAN" in exp["md"] and b"|---|" in exp["md"] and "md_err" not in exp
    assert exp["txt"].startswith(b"# MCAS") and len(exp["txt"]) < 24_000 and b"## B. KESIMPULAN" in exp["txt"]
    assert b"STATUS_VALIDASI=" in exp["txt"] and b"## BATASAN DATA & ENDPOINT" in exp["txt"] and b"[GAGAL] bandar jangka panjang" in exp["txt"]
    assert b"aksi_korporasi=" in exp["txt"] and b"INVESTMENT_THESIS" in exp["txt"]
    # tombol Petunjuk membuka dialog tanpa error
    [b for b in at.button if b.label == "ℹ️ Petunjuk"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    # sesi baru (login ulang) masih melihat histori dari DB -> persisten
    at2 = open_app(monkeypatch)
    hdf = [d.value for d in at2.dataframe if "Nama Emiten" in d.value.columns][0]
    assert [c for c in hdf.columns if c not in ("Vonis", "Mode", "Alasan utama", "Skor Bandar", "Skor Teknikal", "Skor Fundamental", "Rata-rata")] == ["Nama Emiten", "Tanggal & Waktu Pengecekan", "Target Adimologi", "Stop Loss", "Current Price"]
    assert hdf.iloc[0]["Vonis"] in ("BELI", "TUNGGU", "HINDARI / JUAL") and hdf.iloc[0]["Alasan utama"] not in ("", "-")
    assert len(hdf) == 1 and hdf.iloc[0]["Nama Emiten"] == "MCAS" and hdf.iloc[0]["Target Adimologi"] == 292
    assert hdf.iloc[0]["Tanggal & Waktu Pengecekan"].endswith("WIB")


def test_multi_batch(monkeypatch):
    fc = mcas()
    monkeypatch.setattr(requests, "get", lambda url, headers=None, params=None, timeout=None: R(fc.get(url.replace(ad.BASE_URL, ""), params)))
    monkeypatch.setenv("INVEZGO_API_KEY", "dummy")
    at = open_app(monkeypatch)
    at.text_area[0].set_value("MCAS").run()
    [b for b in at.button if b.label == "Jalankan batch"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    df = at.session_state["multi_df"]
    assert df.loc[0, "Kode"] == "MCAS" and df.loc[0, "Skenario Adimology"] == "TIDAK ADA SINYAL"
    assert df.loc[0, "Penilaian (tanpa aliran)"] in bd.KATEGORI
    # tanpa HISTORI_DB_URL: histori jatuh ke sesi, app tetap jalan
    ok = df.loc[df["Skenario Adimology"] != "ERROR", "Kode"].tolist()
    assert ok and [h["emiten"] for h in at.session_state["histori"]] == ok


def test_b200_success_renders(monkeypatch):
    fc = mcas()
    monkeypatch.setattr(requests, "get", lambda url, headers=None, params=None, timeout=None: R(fc.get(url.replace(ad.BASE_URL, ""), params)))
    monkeypatch.setenv("INVEZGO_API_KEY", "dummy")
    fake = {"dari": "2026-01-01", "sampai": "2026-10-09", "hari": 200, "n_broker": 80, "status": "AKUMULASI",
            "net_acc5_value": 9e9, "net_dist5_value": -4e9, "modal_top3": 300.0, "net_top3_lot": 5000.0,
            "floating_top3": -0.12, "nyangkut": "NYANGKUT", "keterangan": "harga -12,0% di bawah modal",
            "akumulator": [{"code": "AA", "broker": "X", "net_lot": 5000.0, "avg": 300.0, "floating": -0.12, "net_window_lot": 10.0}],
            "distributor": [{"code": "BB", "broker": "Y", "net_lot": -3000.0, "avg": 320.0, "floating": -0.17, "net_window_lot": float("nan")}],
            "konsentrasi": {"status": "TERSEBAR"}}
    monkeypatch.setattr(bd, "bandar_panjang", lambda *a, **k: fake)
    at = open_app(monkeypatch)
    [b for b in at.button if b.label == "Hitung lengkap"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["b200"][0] == "ok"
    assert any("harga di bawah estimasi modal" in w.value for w in at.warning)
    assert any({"code", "floating"} <= set(d.value.columns) for d in at.dataframe)
    assert at.session_state["export"]["pdf"][:4] == b"%PDF"


def test_dinamika_framework_renders_and_feeds_report(monkeypatch):
    import dinamika as dn
    from test_dinamika import _data, _summ
    fc = mcas()
    monkeypatch.setattr(requests, "get", lambda url, headers=None, params=None, timeout=None: R(fc.get(url.replace(ad.BASE_URL, ""), params)))
    monkeypatch.setenv("INVEZGO_API_KEY", "dummy")
    rng = np.random.default_rng(1)
    price = 1000 * np.cumprod(1 + 0.0016 + rng.normal(0, 0.003, 60))
    flows = {"AA": np.full(60, 30_000.0) + rng.normal(0, 3000, 60), "BB": np.full(60, 10_000.0), "SS": np.full(60, -8_000.0)}
    monkeypatch.setattr(dn, "ambil", lambda *a, **k: dict(inv=_data(price, flows), inv200=None, summaries={20: _summ(float(price[-20])), 60: _summ(float(price[0]))}))
    at = open_app(monkeypatch)
    [b for b in at.button if b.label == "Hitung lengkap"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    d = at.session_state["dinamis"]
    assert d[0] == "ok" and d[1]["kategori"] in dn.KATEGORI and at.session_state["nilai"]["dinamis"] is d[1]
    assert any("Enam dimensi bukti" in m.value for m in at.markdown)
    assert any({"Dimensi", "Bobot", "Skor"} <= set(df.value.columns) for df in at.dataframe)
    txt = at.session_state["export"]["txt"]
    assert b"## BANDARMOLOGI: KERANGKA BUKTI" in txt and b"bukti_mendukung=" in txt and b"kondisi_pembatalan=" in txt and b"terkonfirmasi_statistik=TIDAK" in txt
    assert at.session_state["export"]["pdf"][:4] == b"%PDF" and at.session_state["export"]["xlsx"][:2] == b"PK"


def _fake_api(monkeypatch):
    fc = mcas()
    calls = []

    def fake_get(url, headers=None, params=None, timeout=None):
        path = url.replace(ad.BASE_URL, "")
        calls.append(path)
        return R(fc.get(path, params))
    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setenv("INVEZGO_API_KEY", "dummy")
    return calls


def test_kode_kosong_ditolak(monkeypatch):
    _fake_api(monkeypatch)
    at = open_app(monkeypatch)
    [t for t in at.text_input if t.label == "Kode saham"][0].set_value("")
    [b for b in at.button if b.label == "⚡ Hitung cepat"][0].click().run()
    assert any("kode saham" in e.value.lower() for e in at.error)


def test_mode_cepat_hemat_dan_lengkapi(monkeypatch):
    calls = _fake_api(monkeypatch)
    at = open_app(monkeypatch)
    calls.clear()
    [b for b in at.button if b.label == "⚡ Hitung cepat"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    n_cepat = len(calls)
    assert n_cepat <= 8 and not any("financial-statement" in c for c in calls)
    assert at.session_state["mode"] == "cepat" and "fund" not in at.session_state
    assert any("Mode **cepat**" in c.value for c in at.caption)
    assert any("MCAS" in m.value and "Saham dihitung" in m.value for m in at.markdown)
    [b for b in at.button if b.label.startswith("➕ Lengkapi")][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["mode"] == "lengkap" and at.session_state["fund"][0] == "ok"
    assert any("financial-statement" in c for c in calls) and len(calls) > n_cepat


def test_kartu_keputusan_tampil(monkeypatch):
    _fake_api(monkeypatch)
    at = open_app(monkeypatch)
    [b for b in at.button if b.label == "Hitung lengkap"][0].click().run()
    assert not at.exception
    assert any(("→ TUNGGU" in x.value or "→ HINDARI" in x.value or "→ BELI" in x.value) for x in list(at.success) + list(at.warning) + list(at.error))
    assert {t.label for t in at.tabs} >= {"🎯 Keputusan", "📊 Teknikal & Rencana", "🏦 Bandar & Ledger", "📑 Fundamental", "🧰 Alat & Histori"}


def test_histori_snapshot_recall(monkeypatch, tmp_path):
    monkeypatch.setenv("HISTORI_DB_URL", f"sqlite:///{tmp_path / 'h.db'}")
    _fake_api(monkeypatch)
    at = open_app(monkeypatch)
    [b for b in at.button if b.label == "Hitung lengkap"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    at.run()
    sel = [s for s in at.selectbox if s.label == "Pilih laporan"]
    assert sel and "MCAS" in sel[0].options[0]
    assert any("Panggil ulang" in m.value for m in at.markdown)
    import sqlalchemy as sa
    e = sa.create_engine(f"sqlite:///{tmp_path / 'h.db'}")
    with e.connect() as c:
        n, = c.execute(sa.text("select count(*) from histori_snapshot")).first()
    assert n == 1


def test_rapor_keputusan_tercatat_dan_evaluasi(monkeypatch, tmp_path):
    monkeypatch.setenv("HISTORI_DB_URL", f"sqlite:///{tmp_path / 'h2.db'}")
    _fake_api(monkeypatch)
    at = open_app(monkeypatch)
    [b for b in at.button if b.label == "Hitung lengkap"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    import sqlalchemy as sa
    e = sa.create_engine(f"sqlite:///{tmp_path / 'h2.db'}")
    with e.connect() as c:
        row = c.execute(sa.text("select emiten, keputusan, mode from histori_keputusan")).first()
    assert row[0] == "MCAS" and row[1] in ("BELI", "TUNGGU", "HINDARI / JUAL") and row[2] == "lengkap"
    at.run()
    [b for b in at.button if b.label == "Evaluasi sekarang"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["rapor"][0]["status"] in ("BELUM_ADA_DATA", "TANPA_ZONA")
