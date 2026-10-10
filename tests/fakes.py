"""Data tiruan berbentuk respons Invezgo (angka MCAS 5-9 Okt 2026 dan skenario setup)."""
import adimology as ad


def _row(code, net, avg, name="X"):
    return dict(code=code, buy_freq="1", buy_volume=str(abs(net) * 2 + 1000), buy_value="1", sell_freq="1",
                sell_volume="1", sell_value="1", buy_avg=str(avg), sell_avg="290", net_value="1",
                net_volume=str(net), net_freq="1", name=name)


class FakeClient(ad.InvezgoClient):
    def __init__(self, chart, window_rows, daily, ob):
        super().__init__("x")
        self.chart, self.window_rows, self.daily, self.ob = chart, window_rows, daily, ob

    def get(self, path, params=None):
        if "/chart/stock/" in path:
            return [dict(date=d + "T00:00:00.000Z", open=o, high=h, low=l, close=c, volume=str(v))
                    for d, o, h, l, c, v in self.chart]
        if "/summary/stock/" in path:
            if params["from"] == params["to"]:
                n, avg = self.daily[params["from"]]
                return [_row("CC", n, avg, "MANDIRI")]
            return self.window_rows
        if "/order-book/" in path:
            return self.ob
        if "/financial-statement/" in path:
            return FIN[params["statement"]]
        if "/keystat/" in path:
            return FIN["KS"]
        if "/information/" in path:
            return {"code": "MCAS", "name": "M Cash Integrasi", "sector": "Teknologi", "subsector": "Perangkat lunak"}
        raise AssertionError(path)


def mcas():
    chart = [("2026-10-05", 230, 244, 220, 228, 2245200), ("2026-10-06", 228, 234, 228, 232, 556700),
             ("2026-10-07", 232, 242, 224, 228, 1360700), ("2026-10-08", 228, 284, 228, 284, 24698400),
             ("2026-10-09", 310, 326, 258, 264, 32877800)]
    window = [_row("CC", 1359000, 274.0888, "MANDIRI"), _row("XL", 873200, 287.36, "STOCKBIT")]
    daily = {"2026-10-05": (382000, 229.7), "2026-10-06": (75900, 233.5), "2026-10-07": (492800, 237.0),
             "2026-10-08": (1409800, 273.2), "2026-10-09": (-1001500, 290.8)}
    ob = {"code": "MCAS", "bid": [{"bid1price": 264, "bid1lot": 956, "bid1freq": 0}],
          "offer": [{"offer1price": 266, "offer1lot": 2450, "offer1freq": 0}]}
    return FakeClient(chart, window, daily, ob)


def setup_case():
    """Akumulasi bersih, tidak distribusi, harga dekat avg bandar, harga < 200 (fraksi 1)."""
    chart = [("2026-10-05", 98, 102, 96, 100, 1_000_000), ("2026-10-06", 100, 103, 97, 100, 1_000_000),
             ("2026-10-07", 100, 104, 98, 101, 1_000_000), ("2026-10-08", 101, 105, 99, 100, 1_000_000),
             ("2026-10-09", 100, 104, 98, 101, 1_000_000)]
    window = [_row("CC", 500000, 100.0, "MANDIRI"), _row("XL", 100000, 101, "STOCKBIT")]
    daily = {d: (1000, 100.0) for d in ["2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08", "2026-10-09"]}
    ob = {"bid": [{"bid1price": 100, "bid1lot": 4000}], "offer": [{"offer1price": 101, "offer1lot": 6000}]}
    return FakeClient(chart, window, daily, ob)


def panel(rows: dict) -> dict:
    """Bentuk respons /analysis/financial-statement & /keystat: {akun: {(tahun, kuartal): nilai}}."""
    return {"rows": [{"name": n, "level": 0,
                      "values": [{"col": f"Q{q} {y}", "year": y, "amount": v, "period": f"Q{q}"}
                                 for (y, q), v in sorted(d.items(), reverse=True)]} for n, d in rows.items()]}


_KS = [(2025, 1), (2025, 2), (2025, 3), (2025, 4), (2026, 1), (2026, 2)]
FIN = {
    "IS": panel({"Penjualan dan pendapatan usaha": dict(zip(_KS, [100e9, 210e9, 330e9, 460e9, 120e9, 250e9])),
                 "Laba (rugi) yang dapat diatribusikan ke entitas induk": dict(zip(_KS, [5e9, 11e9, 18e9, 26e9, 7e9, 15e9]))}),
    "BS": panel({"Jumlah aset": {k: 900e9 for k in _KS}, "Jumlah liabilitas": {k: 400e9 for k in _KS},
                 "Jumlah ekuitas": {k: 500e9 for k in _KS}}),
    "CF": panel({"Jumlah arus kas bersih yang diperoleh dari (digunakan untuk) aktivitas operasi":
                 dict(zip(_KS, [8e9, 15e9, 20e9, 30e9, 9e9, 19e9]))}),
    "KS": panel({"Saham": {k: 868e6 for k in _KS}}),
}
