import datetime as dt

import pandas as pd

import rapor as rp

REC = dict(zona_valid=True, entry_lo=95, entry_hi=100, sl=90, tp1=120, tgl_data=dt.date(2026, 10, 1))


def bars(rows):
    return pd.DataFrame([dict(tgl=dt.date(2026, 10, 2 + i), open=o, high=h, low=l, close=c) for i, (o, h, l, c) in enumerate(rows)])


def test_tp_sl_dan_urutan_konservatif():
    ok = rp.evaluasi(REC, bars([(102, 105, 99, 101), (101, 125, 100, 124)]))
    assert ok["status"] == "TP1" and ok["r"] > 1.5
    sl = rp.evaluasi(REC, bars([(102, 105, 99, 101), (100, 104, 89, 90)]))
    assert sl["status"] == "SL" and sl["r"] < -1
    sama = rp.evaluasi(REC, bars([(100, 130, 85, 100)]))      # satu bar sentuh SL dan TP -> SL
    assert sama["status"] == "SL"


def test_tidak_terisi_tanpa_zona_berjalan():
    assert rp.evaluasi(REC, bars([(110, 112, 105, 110)] * 6))["status"] == "TIDAK_TERISI"
    assert rp.evaluasi(dict(REC, zona_valid=False), bars([(100, 101, 99, 100)]))["status"] == "TANPA_ZONA"
    assert rp.evaluasi(REC, bars([]))["status"] == "BELUM_ADA_DATA"
    assert rp.evaluasi(REC, bars([(100, 105, 99, 103), (103, 106, 101, 105)]))["status"] == "BERJALAN"
    gap = rp.evaluasi(REC, bars([(97, 99, 96, 98), (98, 130, 97, 120)]))
    assert gap["isi"] == 97                                    # gap turun: isi di open, bukan entry_hi


def test_ringkas_dan_wilson():
    h = [dict(keputusan="BELI", status="TP1", r=2.0)] * 3 + [dict(keputusan="BELI", status="SL", r=-1.1), dict(keputusan="TUNGGU", status="TIDAK_TERISI")]
    t = rp.ringkas(h).set_index("Keputusan")
    assert t.loc["BELI", "TP1"] == 3 and t.loc["BELI", "SL"] == 1 and abs(t.loc["BELI", "Menang % (TP1/(TP1+SL))"] - 0.75) < 1e-9
    assert "belum" in t.loc["BELI", "Cukup data?"] and t.loc["TUNGGU", "Terisi"] == 0
    lo, hi = rp.wilson(3, 4)
    assert 0 < lo < 0.75 < hi < 1
