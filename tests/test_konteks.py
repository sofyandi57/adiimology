import datetime as dt

import konteks as k

IDX = [{"date": f"2026-09-{d:02d}T00:00:00.000Z", "close": 6300 - i * 10} for i, d in enumerate(range(1, 30))]
INSIDER = [
    {"date": "2026-08-05T07:46:01.000Z", "code": "BUVA", "name": "PT TIRTA", "change": -1.01, "badge": "{PENGENDALI}", "purpose": "Divestasi",
     "subrow": [{"date": "2026-07-30", "price": 1000, "status": "Sell", "value": -250000000}]},
    {"date": "2026-06-02T08:56:00.000Z", "code": "BUVA", "name": "SATRIO", "change": 0.01, "badge": "{DIREKSI}", "purpose": "Investasi",
     "subrow": [{"date": "2026-05-25", "price": 700, "status": "Buy", "value": 500000}]}]
KSEI = [{"code": "BUVA", "date": "2026-07-31T00:00:00.000Z", "foreign_cp": "1204695967", "foreign_ib": "86524300", "local_cp": "18672929971", "local_id": "3843231833"},
        {"code": "BUVA", "date": "2026-09-30T00:00:00.000Z", "foreign_cp": "1162134267", "foreign_ib": "71743500", "local_cp": "18750343571", "local_id": "3816575083"}]
TOP = {"accum": [{"code": "TOWR", "change": 5.5}, {"code": "BUVA", "change": 1.0}], "dist": [{"code": "KIJA", "change": -2}]}


def test_ihsg_status():
    assert k.ihsg(IDX)["kondisi"] == "LEMAH"
    naik = [{"date": "x", "close": 5000 + i * 20} for i in range(60)]
    assert k.ihsg(naik)["kondisi"] == "KONDUSIF"
    assert k.ihsg([])["status"] == "na" and k.ihsg(IDX[:5])["status"] == "na"


def test_insider_ksei_peringkat():
    i = k.insider(INSIDER, today=dt.date(2026, 9, 1))
    assert i["status"] == "ok" and i["n"] == 2 and i["pengendali_jual"] and i["nilai_jual"] == 250000000 and i["nilai_beli"] == 500000
    ks = k.ksei(KSEI)
    assert ks["status"] == "ok" and ks["arah"] in ("NAIK", "TURUN", "DATAR") and len(ks["rows"]) == 2
    pr = k.peringkat({"BDM": TOP}, "BUVA")
    assert pr["muncul"] == [dict(daftar="BDM", sisi="accum", urut=2, change=1.0)]
    assert k.peringkat({"BDM": TOP}, "ZZZZ")["muncul"] == []


def test_notasi_dan_kabar_defensif():
    assert k.notasi([], "BUVA")["status"] == "na"
    assert k.notasi({"data": [{"code": "AAAA", "notation": "E"}]}, "BUVA")["ada"] is False
    n = k.notasi({"data": [{"code": "BUVA", "notation": "X", "date": "2026-10-01"}]}, "BUVA")
    assert n["ada"] and "BUVA" in n["teks"][0]
    n2 = k.notasi({"BUVA": [{"kode": "X"}]}, "BUVA")
    assert n2["ada"]
    kb = k.kabar({"data": [{"title": "Rencana PKPU dan right issue", "date": "2026-10-08T01:00:00Z"}, {"title": "lama", "date": "2025-01-01T00:00:00Z"}]},
                 today=dt.date(2026, 10, 9))
    assert kb["n"] == 1 and kb["ada_kata_risiko"] and "pkpu" in kb["items"][0]["risiko"]
    assert k.kabar({"data": []})["n"] == 0


class _Klien:
    def get(self, path, params=None):
        if "chart/index" in path:
            return IDX
        if "notation" in path:
            return {"data": [{"code": "BUVA", "notation": "X"}]}
        if "insider" in path:
            return INSIDER
        if "ksei" in path:
            return KSEI
        if "/top/" in path:
            return TOP
        raise RuntimeError("endpoint down")


def test_ambil_gagal_aman_dan_terapkan():
    x = k.ambil(_Klien(), "BUVA", dt.date(2026, 10, 9), penuh=True)
    assert x["ihsg"]["status"] == "ok" and x["notasi"]["ada"] and x["disclosure"]["status"] == "na" and x["news"]["status"] == "na"
    kep, al = k.terapkan("BELI", ["dasar"], x, "lengkap")
    assert kep == "TUNGGU" and any("notasi" in a.lower() for a in al)
    assert k.terapkan("BELI", [], None, "cepat")[0] == "TUNGGU"
    assert k.terapkan("BELI", [], None, "lengkap")[0] == "BELI"
    assert k.terapkan("HINDARI / JUAL", [], x, "lengkap")[0] == "HINDARI / JUAL"
    cepat = k.ambil(_Klien(), "BUVA", dt.date(2026, 10, 9), penuh=False)
    assert set(cepat) == {"ihsg", "notasi"}
    L = "\n".join(k.baris_laporan(x))
    assert "## KONTEKS PASAR & RISIKO" in L and "Notasi khusus: ADA" in L and "N/A" in L
    assert "N/A: konteks tidak diambil" in "\n".join(k.baris_laporan(None))
