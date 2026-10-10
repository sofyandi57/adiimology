import datetime as dt

import pytest

import audit as au
from test_laporan import _inputs


def test_posisi_indy_case_reconciles():
    # kasus feedback INDY: entry 2790, stop 2620, modal 10 jt, risiko 1%
    p = au.hitung_posisi(10_000_000, 0.01, 2790, 2620, 2870)
    per_lot = 100 * (2790 * 1.0015 - 2620 * 0.9975)
    assert p["lot"] == 5 and p["pengikat"] == "risiko" and p["lot_risiko"] == 5
    assert p["rugi_stop"] == pytest.approx(5 * per_lot) and p["rugi_stop"] <= p["anggaran_risiko"]
    harga_saja = 5 * 100 * (2790 - 2620)                                   # 85.000 + fee = rugi_stop
    assert p["rugi_stop"] - harga_saja == pytest.approx(5 * 100 * (2790 * 0.0015 + 2620 * 0.0025))
    assert p["laba_target"] == pytest.approx(5 * 100 * (2870 * 0.9975 - 2790 * 1.0015)) and p["rr_bersih"] < 1     # TP1 dekat: R/R bersih < 1


def test_posisi_meja_case_uses_displayed_stop():
    p = au.hitung_posisi(10_000_000, 0.01, 159, 135, 173)
    assert p["lot"] == 40 and p["rugi_stop"] == pytest.approx(40 * 100 * (159 * 1.0015 - 135 * 0.9975))
    assert 90_000 < p["rugi_stop"] <= 100_000                              # bukan 18 lot / Rp100.000 versi lama
    assert p["rugi_gap"] > p["rugi_stop"]
    assert au.hitung_posisi(1e7, 0.01, 100, 120, 130)["lot"] == 0         # stop di atas entry


def test_validasi_passes_and_fails_on_tampering():
    r, rk, f = _inputs()
    pos = au.skenario_posisi(r, rk)
    assert pos and all(x["stop"] == rk["sl"] for x in pos)
    ok = au.validasi(r, rk, f, pos, dt.datetime(2026, 10, 11))
    assert ok["n_gagal"] == 0 or {c["id"] for c in ok["cek"] if c["status"] == "GAGAL"} <= {"urutan_level"}
    bad = [dict(pos[-1], rugi_stop=pos[-1]["rugi_stop"] * 2)]
    gagal = au.validasi(r, rk, f, bad, dt.datetime(2026, 10, 11))
    assert gagal["status"] == "GAGAL" and any(c["id"].startswith("posisi") and c["status"] == "GAGAL" for c in gagal["cek"])
    rk2 = dict(rk, rr=rk["rr"] + 1)
    assert any(c["id"] == "rr_rekonsiliasi" and c["status"] == "GAGAL" for c in au.validasi(r, rk2, f, pos, dt.datetime(2026, 10, 11))["cek"])
    usang = au.validasi(r, rk, f, pos, dt.datetime(2026, 12, 1))
    assert any(c["id"] == "kesegaran_data" and c["status"] == "PERINGATAN" for c in usang["cek"])


def test_batasan_lists_failures_gaps_and_structural():
    r, rk, f = _inputs()
    status = {"tek": ("ok", rk), "fund": ("ok", f), "ovn": ("err", "Data intraday terlalu sedikit"), "aliran": None, "b200": ("err", "boom")}
    out = au.batasan(status, r, rk, f, None, {"afiliasi": None}, None, {"sid": None}, None)
    txt = "\n".join(out)
    assert "[GAGAL] statistik BPJP/BSJP" in txt and "[GAGAL] bandar jangka panjang: boom" in txt and "[DINONAKTIFKAN] aliran" in txt
    assert "[HILANG] SID" in txt and "[HILANG] Afiliasi" in txt and "Data tahunan (FY) IS tidak tersedia" in txt
    assert "[STRUKTURAL] Tidak tersedia di endpoint: proyeksi" in txt and "Backtest: 4 saham" in txt
    assert au.parse_aksi({"data": [{"type": "DIVIDEND", "payload": {"Amount": "10", "ExDate": "2026-11-01"}}]})[0]["ringkas"].startswith("Amount=10")
