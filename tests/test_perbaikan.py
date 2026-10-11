import datetime as dt

import adimology as ad
import audit as au
from fakes import mcas
from test_laporan import _inputs


def _deep(side):
    return [{f"{side}{i}price": 100 + i, f"{side}{i}lot": 500 * i, f"{side}{i}freq": 0} for i in range(1, 6)]


class ObHist(type(mcas())):
    def get(self, path, params=None):
        if "/order-book/" in path and params and params.get("date"):
            return {"bid": _deep("bid"), "offer": _deep("offer")}
        return super().get(path, params)


def test_validasi_tanpa_zona_beli_tidak_gagal_palsu():
    """Kasus BUVA: harga di atas entry maksimal -> zona beli tidak valid -> RR dihitung dari support, bukan dari harga."""
    r, rk, f = _inputs()
    tp, sl, lo = 700.0, 600.0, 640.0
    rk2 = dict(rk, zona_valid=False, entry_lo=lo, entry_hi=lo - 5, entry_hi_sup=lo + 10, sl=sl, tp1=tp, tp2=tp + 40, target=tp, harga=665.0,
               rr=(tp - lo) / (lo - sl), tgl=dt.date(2026, 10, 9))
    v = au.validasi(r, rk2, f, [], dt.datetime(2026, 10, 11))
    ids = {c["id"]: c["status"] for c in v["cek"]}
    assert ids["rr_rekonsiliasi"] == "OK" and ids["urutan_level"] == "OK" and v["status"] != "GAGAL"


def test_orderbook_dangkal_pakai_snapshot_historis():
    c = ObHist.__new__(ObHist)
    base = mcas()
    c.__dict__.update(base.__dict__)
    r = ad.analyze(c, ad.Params(kode="MCAS"))
    assert r.lpp_sumber.startswith("snapshot historis") and r.ob["n_levels"] == 10
    r2 = ad.analyze(base, ad.Params(kode="MCAS"))               # snapshot historis juga dangkal -> tetap orderbook live
    assert r2.lpp_sumber == "orderbook"
    r3 = ad.analyze(c, ad.Params(kode="MCAS", ob_auto=False))
    assert r3.lpp_sumber == "orderbook"


def test_alasan_ringkas():
    import keputusan_ui as ku
    al = ["Bandarmologi: DISTRIBUSI TERINDIKASI (tingkat bukti: lemah)", "Jebakan: Bandar menjual 5.000 lot", "Teknikal: HINDARI (harga di bawah SL)",
          "Fundamental: Rugi bersih 3 kuartal beruntun", "HINDARI karena bandar menunjukkan distribusi/penjualan (veto): x", "Ledger broker: status DISTRIBUTING (skor 3.0, confidence 0.8) = veto, bandar sedang keluar"]
    t = ku.alasan_ringkas("HINDARI / JUAL", al)
    assert "Bandar distribusi/menjual" in t and "Fundamental buruk" in t and "Teknikal HINDARI" in t and "Ledger: DISTRIBUTING" in t
    assert ku.alasan_ringkas("TUNGGU", ["Bandarmologi: TIDAK KONKLUSIF (tingkat bukti: x)", "Teknikal: TUNGGU (jangan kejar)"]) == "Bandar: TIDAK KONKLUSIF · Teknikal: TUNGGU"
