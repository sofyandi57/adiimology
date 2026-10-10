import datetime as dt

import pandas as pd
import pytest

import adimology as ad
import aliran as al
from fakes import _row, mcas


def test_konsentrasi_slide_rule():
    bs = pd.DataFrame([_row("AA", 900_000, 100), _row("BB", 300_000, 101)] +
                      [_row(f"S{i}", -50_000, 99) for i in range(20)])
    for c in ["net_volume", "buy_volume"]:
        bs[c] = pd.to_numeric(bs[c])
    k = ad.konsentrasi(bs)
    assert k["top3_beli_lot"] == 12_000 and k["top15_jual_lot"] == 7_500
    assert k["status"] == "AKUMULASI TERKONSENTRASI" and k["n_jual"] > k["n_beli"]


def test_top3_mode_weighted_avg_and_default_unchanged():
    r1 = ad.analyze(mcas(), ad.Params(kode="MCAS"))
    assert r1.bandar["code"] == "CC" and r1.calc["t_real_r"] == 292      # default tetap
    r3 = ad.analyze(mcas(), ad.Params(kode="MCAS", mode_bandar="top3", broker_dikecualikan=()))
    assert r3.bandar["codes"] == ["CC", "XL"] and r3.bandar["code"] == "CC+XL"
    assert r3.bandar["net_lot"] == pytest.approx((1359000 + 873200) / 100)
    assert 274.0888 < r3.bandar["avg"] < 287.36                          # rata-rata tertimbang volume beli
    assert r1.konsentrasi["status"] in {"AKUMULASI TERKONSENTRASI", "AKUMULASI SEBAGIAN", "TERSEBAR"}


class F(ad.InvezgoClient):
    def __init__(self):
        super().__init__("x")
        self.calls = []

    def get(self, path, params=None):
        self.calls.append((path, dict(params or {})))
        if "/summary/stock/" in path:
            if params["investor"] == "f":
                return [_row("AK", 20000, 100) | {"net_value": "5000000000"}, _row("BK", -5000, 100) | {"net_value": "-1000000000"}]
            if params["market"] == "NG":
                return [_row("CC", -40000, 100), _row("ZP", 40000, 100)]
            w = (dt.date.fromisoformat(params["to"]) - dt.date.fromisoformat(params["from"])).days
            return [_row("CC" if w >= 2 else "YU", 50000, 100), _row("XL", 90000, 100)]
        if "/shareholder-above" in path:
            return {"data": [{"date": "2026-05-01T17:00:00Z", "code": "MCAS", "name": "PT INDUK", "format_securities": "cc",
                              "next_pct": 55.2}, {"date": "2026-05-01T17:00:00Z", "code": "MCAS", "name": "FUND X",
                              "format_securities": "AK", "next_pct": 6.1}]}
        raise AssertionError(path)


def test_asing_nego_konsistensi_afiliasi():
    c = F()
    d0 = dt.date(2026, 10, 9)
    a = al.asing(c, "MCAS", d0, d0)
    assert a["status"] == "NET BUY ASING" and a["net_value"] == 4e9 and a["top_beli"] == ["AK"]
    n = al.negosiasi(c, "MCAS", d0, d0, vol_rg_lot=10_000, kode_bandar=["CC"])
    assert n["status"] == "BANDAR JUAL DI NEGO" and n["bandar_net_lot"] == -400
    hari = [d0 - dt.timedelta(days=i) for i in range(25)][::-1]
    k = al.konsistensi(c, "MCAS", hari, excl={"XL"}, kode_bandar=["CC"])
    assert list(k["window"]) == ["1 hari", "3 hari", "7 hari", "20 hari"]
    assert list(k["sama_dengan_bandar"]) == [False, True, True, True]
    f = al.afiliasi(c, "MCAS", ["CC"])
    assert [x["name"] for x in f["cocok"]] == ["PT INDUK"]


def test_orderbook_multi_median_and_old_dates():
    class OB(ad.InvezgoClient):
        def __init__(self):
            super().__init__("x")

        def get(self, path, params=None):
            assert "/order-book/" in path
            k = {"10:00": 1, "11:30": 2, "14:30": 4}[params["time"]]
            return {"bid": [{f"bid{i}price": 100 - i, f"bid{i}lot": 1000 * k} for i in range(1, 4)],
                    "offer": [{f"offer{i}price": 100 + i, f"offer{i}lot": 1000 * k} for i in range(1, 4)]}

    m = ad.orderbook_multi(OB(), "TEST", ad.Params(kode="TEST"), dt.date(2026, 10, 9))
    assert [x["jam"] for x in m] == ["10:00", "11:30", "14:30"]
    lpps = sorted(x["lot_per_papan"] for x in m)
    assert lpps[1] == pytest.approx(2 * lpps[0]) and lpps[2] == pytest.approx(4 * lpps[0])
    assert ad.orderbook_multi(OB(), "TEST", ad.Params(kode="TEST"), dt.date(2026, 1, 5)) == []   # sebelum data historis
