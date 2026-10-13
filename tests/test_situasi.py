"""Situasi khusus: data meniru respons Invezgo nyata EPAC (insider Jun-Jul 2026, agenda RUPSLB 3 Nov 2026) dan GRPM (konversi waran)."""
import datetime as dt

import pandas as pd

import audit as au
import skor as sk
import situasi_khusus as sx

HARI_INI = dt.date(2026, 10, 14)
INSIDER_EPAC = [
    {"date": "2026-07-23T05:54:01.000Z", "code": "EPAC", "name": "TRIPLE BERKAH BERSAMA", "prev_percent": 20.6777, "next_percent": 38.8409, "change": 18.1632,
     "badge": "{}", "subrow": [{"date": "2026-07-23T00:00:00.000Z", "price": 12, "status": "Buy", "value": 250619400}]},
    {"date": "2026-07-23T05:16:01.000Z", "code": "EPAC", "name": "RYAN PERMANA", "prev_percent": 10.5765, "next_percent": 0, "change": -10.5765,
     "badge": "{PENGENDALI}", "subrow": [{"date": "2026-07-23T00:00:00.000Z", "price": 12, "status": "Sell", "value": -349384900}]},
    {"date": "2026-07-02T05:56:01.000Z", "code": "EPAC", "name": "TRIPLE BERKAH BERSAMA", "prev_percent": 23.2205, "next_percent": 20.6777, "change": -2.5428,
     "badge": "{}", "subrow": [{"date": "2026-06-26T00:00:00.000Z", "price": 94, "status": "Sell", "value": -84000000}]},
    {"date": "2026-06-24T12:06:01.000Z", "code": "EPAC", "name": "TRIPLE BERKAH BERSAMA", "prev_percent": 6.82, "next_percent": 23.2205, "change": 16.4005,
     "badge": "{}", "subrow": [{"date": "2026-06-24T00:00:00.000Z", "price": 12, "status": "Buy", "value": 226252100}]},
]
KALENDER_EPAC = {"data": [
    {"code": "EPAC", "type": "RUPS_SCHEDULE", "payload": {"Date": "2026-11-03T00:00:00", "Venue": "IBIS Style Jakarta Tanah Abang", "Remark": "EGM",
     "Result": "RUPSLB:  - Persetujuan perubahan anggota Komisaris Perseroan  - Persetujuan atas rencana peningkatan modal dasar Perseroan dan perubahan Anggaran Dasar Perseroan  "
               "- Persetujuan Rencana Penambahan Modal Dengan Hak Memesan Efek Terlebih Dahulu (PMHMETD)  - Persetujuan pemberian Kuasa kepada Direksi untuk rencana penghentian (discontinue) "
               "operasional segmen usaha kemasan fleksibel konvensional milik Perseroan  - Persetujuan Pembelian Kembali Saham Perusahaan Terbuka (Buyback)", "DateStr": "Selasa, 3 November 2026"}},
    {"code": "EPAC", "type": "RUPS_RESULT", "payload": {"Date": "2024-06-27T00:00:00", "Result": "Hasil RUPST: tidak ada pembagian dividen"}},
]}
KALENDER_GRPM = {"data": [
    {"code": "GRPM", "type": "CONVERTION", "payload": {"Remark": "Sisa Waran 127,833,531", "Shares": "4532895", "RecDate": "2026-07-14T00:00:00", "ShareFrom": "Konversi Waran",
                                                       "TradeDate": "2026-07-14T00:00:00", "TotalShare": "1571666469"}},
    {"code": "GRPM", "type": "CONVERTION", "payload": {"Remark": "Sisa Waran 150,300,287", "Shares": "49500", "RecDate": "2026-06-29T00:00:00", "ShareFrom": "Konversi Waran",
                                                       "TradeDate": "2026-06-29T00:00:00", "TotalShare": "1549199713"}},
]}
OHLC = pd.DataFrame({"tgl": pd.bdate_range("2026-06-01", "2026-10-09").date, "close": 60.0})
OHLC.loc[OHLC["tgl"] >= dt.date(2026, 6, 20), "close"] = 94.0
OHLC.loc[OHLC["tgl"] >= dt.date(2026, 7, 15), "close"] = 50.0


def test_parse_aksi_teks_utuh_dan_tanggal():
    a = au.parse_aksi(KALENDER_EPAC)
    assert a[0]["tgl"] == dt.date(2026, 11, 3) and "buyback" in a[0]["teks"].lower() and len(a[0]["ringkas"]) <= 140


def test_epac_backdoor_terdeteksi():
    s = sx.deteksi({"insider_raw": INSIDER_EPAC}, au.parse_aksi(KALENDER_EPAC), OHLC, [], harga=49.0, modal=10_000_000, maks_pct=0.02, today=HARI_INI)
    assert s["aktif"] and s["jenis"].startswith("BACKDOOR") and {2, 3, 4} <= set(s["tahap"])
    assert any("RYAN PERMANA" in b and "-76%" in b for b in s["bukti"])                            # jual Rp12 saat pasar Rp50
    assert any("TATA KELOLA" in f and "94" in f for f in s["flags"])                                # beli Rp12 lalu jual Rp94 dalam 2 hari
    assert any("2026-11-03" in t for t in s["tanggal_kunci"]) and any("RUPS 2026-11-03" in i for i in s["invalidasi"])
    assert {"rights issue (HMETD)", "peningkatan modal dasar", "perubahan / penghentian kegiatan usaha", "buyback"} <= set(s["kategori"])
    assert s["posisi"]["lot"] == int(10_000_000 * 0.02 // (100 * 49 * 1.0015)) and "SITUASI KHUSUS" in "\n".join(sx.baris(s))


def test_grpm_waran_dan_catatan_manual():
    s = sx.deteksi({}, au.parse_aksi(KALENDER_GRPM), OHLC, [], harga=120.0, modal=10_000_000, today=HARI_INI)
    assert not s["aktif"] and abs(s["dilusi_waran"] - 127833531 / 1571666469) < 1e-6        # konversi waran saja bukan backdoor
    cat = [dict(jenis="Term sheet / MoU akuisisi", teks="PMUI jual GRPM ke Tunas Binatama (Grup Rimau)", tenggat=dt.date(2026, 12, 31))]
    s2 = sx.deteksi({}, au.parse_aksi(KALENDER_GRPM), OHLC, cat, harga=120.0, modal=10_000_000, today=HARI_INI)
    assert s2["aktif"] and s2["tahap"] == [1] and any("2026-12-31" in i for i in s2["invalidasi"])


def test_ca_membaca_agenda_dan_gate_event():
    aksi = au.parse_aksi(KALENDER_EPAC)
    s = sx.deteksi({"insider_raw": INSIDER_EPAC}, aksi, OHLC, [], harga=49.0, modal=10_000_000, today=HARI_INI)
    k = dict(notasi=dict(status="ok", ada=False, teks=[]))
    ca = sk.skor_ca(k, dict(aksi=[dict(a, tgl=dt.date.today() + dt.timedelta(days=20)) for a in aksi]), s)
    assert ca["nilai"] <= 0.25 and any(g["gate"] == "EVENT_DRIVEN_WARNING" for g in ca["gates"])
    assert any("rights issue" in x for x in ca["lawan"]) and any("buyback" in x and "tidak dihitung positif" in x for x in ca["lawan"])
    lama = sk.skor_ca(k, dict(aksi=[dict(type="RIGHT", teks="rights issue HMETD", tgl=dt.date(2023, 1, 5))]), None)
    assert lama["nilai"] == 0.5                                                                     # aksi > 180 hari diabaikan
