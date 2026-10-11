import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pytest

import adimology as ad
import ledger as lg
import ledger_calc as lc

WIB = ZoneInfo("Asia/Jakarta")
NOW = dt.datetime(2026, 10, 12, 10, 0, tzinfo=WIB)
HARI = [d.date() for d in __import__("pandas").bdate_range(end="2026-10-09", periods=70)]


def _skenario():
    """AA: akumulasi hari 10-40 (harga datar), lalu distribusi; BB: pasif; CC/DD: crossing."""
    rng = np.random.default_rng(1)
    close = {}
    p = 1000.0
    for i, d in enumerate(HARI):
        p = p * (1.0 + (0.004 if i > 40 else 0.0) + rng.normal(0, 0.003))
        close[d] = round(p)
    vol = {d: 1_000_000 for d in HARI}
    daily = {}
    for i, d in enumerate(HARI):
        rows = []
        if 10 <= i <= 40:
            b, s = 80_000, 10_000
        elif i > 50:
            b, s = 5_000, 70_000
        else:
            b, s = 20_000, 20_000
        rows.append(dict(code="AA", name="BROKER A", buy_volume=b, sell_volume=s, buy_value=b * close[d] * 0.995,
                         sell_value=s * close[d] * 1.005, buy_avg=close[d] * 0.995, sell_avg=close[d] * 1.005, net_volume=b - s, net_value=0))
        rows.append(dict(code="BB", name="BROKER B", buy_volume=15_000, sell_volume=16_000, buy_value=15_000 * close[d], sell_value=16_000 * close[d],
                         buy_avg=close[d], sell_avg=close[d], net_volume=-1000, net_value=0))
        rows.append(dict(code="CC", name="CROSS", buy_volume=100_000, sell_volume=100_000, buy_value=100_000 * close[d], sell_value=100_000 * close[d],
                         buy_avg=close[d], sell_avg=close[d], net_volume=0, net_value=0))
        daily[d] = rows
    return close, vol, daily


class Fake(ad.InvezgoClient):
    def __init__(self, ubah=None):
        super().__init__("x")
        self.close, self.vol, self.daily = _skenario()
        self.calls = 0
        self.ubah = ubah or {}

    def get(self, path, params=None):
        self.calls += 1
        if "/chart/stock/" in path:
            return [dict(date=f"{d - dt.timedelta(days=1)}T17:00:00.000Z", open=c, high=c, low=c, close=c, volume=str(self.vol[d]))
                    for d, c in self.close.items()]
        d = dt.date.fromisoformat(params["from"])
        if d in self.ubah:
            return self.ubah[d]
        return self.daily.get(d, [])


@pytest.fixture
def eng(tmp_path):
    return lg.engine(f"sqlite:///{tmp_path}/l.db")


def test_hari_final():
    assert lg.hari_final(dt.date(2026, 10, 9), NOW)
    assert not lg.hari_final(dt.date(2026, 10, 12), NOW)
    assert lg.hari_final(dt.date(2026, 10, 12), dt.datetime(2026, 10, 12, 18, 30, tzinfo=WIB))


def test_backfill_idempoten_dan_resumable(eng):
    c = Fake()
    r1 = lg.perbarui(c, eng, "tst", hari_bursa=30, max_calls=10, now=NOW, jeda=0)
    assert r1["diambil"] == 10 and r1["tersisa"] == 20
    r2 = lg.perbarui(c, eng, "TST", hari_bursa=30, max_calls=100, now=NOW, jeda=0)
    assert r2["sudah"] == 10 and r2["diambil"] == 20 and not r2["gagal"]
    n_calls = c.calls
    r3 = lg.perbarui(c, eng, "TST", hari_bursa=30, now=NOW, jeda=0)
    assert r3["diambil"] == 0 and c.calls == n_calls + 1          # hanya chart
    b, m = lg.muat(eng, "TST")
    assert len(b) == 30 * 3 and b.duplicated(["trade_date", "broker_code"]).sum() == 0
    assert len(lg.log_api(eng, "TST")) > 0


def test_hari_ini_belum_final_diambil_ulang(eng):
    c = Fake()
    sore = dt.datetime(2026, 10, 9, 12, 0, tzinfo=WIB)
    lg.perbarui(c, eng, "TST", hari_bursa=3, now=sore, jeda=0)
    b, _ = lg.muat(eng, "TST")
    assert not b[b.trade_date == dt.date(2026, 10, 9)]["is_final"].any()
    lg.perbarui(c, eng, "TST", hari_bursa=3, now=NOW, jeda=0)
    b, _ = lg.muat(eng, "TST")
    assert b["is_final"].all()


def test_rekonsiliasi_koreksi(eng):
    c = Fake()
    lg.perbarui(c, eng, "TST", hari_bursa=5, now=NOW, jeda=0)
    d = HARI[-1]
    c.ubah[d] = [dict(code="AA", name="BROKER A", buy_volume=1, sell_volume=2, buy_value=1, sell_value=2, buy_avg=1, sell_avg=1, net_volume=-1, net_value=0)]
    out = lg.rekonsiliasi(c, eng, "TST", 2, NOW)
    assert any(o["tgl"] == d and o["dikoreksi"] for o in out)
    b, _ = lg.muat(eng, "TST")
    assert b[b.trade_date == d]["broker_code"].tolist() == ["AA"] or "AA" in b[b.trade_date == d]["broker_code"].tolist()


def test_validasi_dan_crossing(eng):
    c = Fake()
    lg.perbarui(c, eng, "TST", hari_bursa=60, now=NOW, jeda=0)
    b, m = lg.muat(eng, "TST")
    v = lg.validasi(b, m, lg.log_api(eng), NOW)
    ids = {x["id"]: x["status"] for x in v["cek"]}
    assert ids["duplikat"] == "OK" and ids["tanggal_hilang"] == "OK"
    x = lg.tandai_crossing(b, m)
    assert x[x.broker_code == "CC"]["crossing"].all() and not x[x.broker_code == "AA"]["crossing"].any()


def test_validasi_tanggal_hilang(eng):
    c = Fake()
    lg.perbarui(c, eng, "TST", hari_bursa=10, now=NOW, jeda=0)
    b, m = lg.muat(eng, "TST")
    b = b[b.trade_date != HARI[-3]]
    v = lg.validasi(b, m, None, NOW)
    assert {x["id"]: x["status"] for x in v["cek"]}["tanggal_hilang"] != "OK"


@pytest.fixture
def hasil(eng):
    c = Fake()
    lg.perbarui(c, eng, "TST", hari_bursa=70, max_calls=100, now=NOW, jeda=0)
    b, m = lg.muat(eng, "TST")
    return lc.jalankan(b, m, lc.Cfg(), lg.log_api(eng), NOW)


def test_episode_bukan_hari_pertama_dan_label(hasil):
    H = hasil["hasil"]
    ep = H["AA"]["episode"]
    assert ep and ep["mulai"] == HARI[10] and ep["puncak"] <= HARI[41]
    assert H["AA"]["posisi"]["label"] == "NET_FLOW_PROXY" and H["AA"]["posisi"]["dasar"] is None
    assert H["AA"]["status"] in (lc.DIST, lc.MIXED)
    assert H["BB"]["status"] in (lc.NEUTRAL, lc.INSUF) and H["BB"]["status"] != lc.ACC
    assert H["AA"]["avg_buy_ex_crossing"] < H["AA"]["avg_buy_gross"] * 1.01


def test_crossing_dipisah(hasil):
    cc = hasil["hasil"]["CC"]
    assert cc["gross_buy_ex_crossing"] == 0 and cc["gross_buy_lots"] > 0 and cc["status"] != lc.ACC


def test_zona_alert_posisi_dan_jawaban(hasil):
    z = hasil["zona"]["zona"]
    assert "early" in z and "major" in z and z["major"]["invalidasi"]
    assert all(a["ts"] and a["bukti"] for a in hasil["alert"])
    t = lc.tabel_posisi(hasil)
    assert {"Broker", "Est. Remaining Low", "Est. Remaining High", "Label"} <= set(t.columns)
    j = lc.jawaban(hasil, "TST")
    assert len(j["jawab"]) == 10 and "NET_FLOW_PROXY" in " ".join(a for _, a in j["jawab"]) and len(j["ringkasan"]) == 7
    assert lc.csv_ledger(hasil).startswith(b"trade_date")


def test_net_positif_saja_tidak_cukup():
    import pandas as pd
    tgl = HARI[-25:]
    m = pd.DataFrame({"trade_date": tgl, "close": 1000.0, "volume_shares": 1e7, "open": 1000.0, "high": 1000.0, "low": 1000.0})
    b = pd.DataFrame([dict(ticker="X", trade_date=d, broker_code="ZZ", broker_name="Z", buy_shares=1000.0 + (i % 2) * 1000, sell_shares=0.0, buy_value=1e6,
                           sell_value=0.0, buy_avg_raw=1000.0, sell_avg_raw=None, is_final=True) for i, d in enumerate(tgl)])
    L = lc.bangun(b, m)
    v = lc.analisa_broker(L[L.broker_code == "ZZ"], lc.Cfg(), 1e5)
    assert v["status"] in (lc.NEUTRAL, lc.INSUF) and v["status"] != lc.ACC


def test_data_kurang_insufficient():
    import pandas as pd
    tgl = HARI[-5:]
    m = pd.DataFrame({"trade_date": tgl, "close": 1000.0, "volume_shares": 1e6})
    b = pd.DataFrame([dict(ticker="X", trade_date=d, broker_code="ZZ", broker_name="Z", buy_shares=1e5, sell_shares=0.0, buy_value=1e8, sell_value=0.0,
                           buy_avg_raw=1000.0, sell_avg_raw=None, is_final=True) for d in tgl])
    L = lc.bangun(b, m)
    assert lc.analisa_broker(L, lc.Cfg(), 1e4)["status"] == lc.INSUF


def test_ui_tab_ledger(tmp_path, monkeypatch):
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    import test_app as ta
    url = f"sqlite:///{tmp_path}/ui.db"
    monkeypatch.setenv("LEDGER_DB_URL", url)
    e = lg.engine(url)
    lg.perbarui(Fake(), e, "MCAS", hari_bursa=70, max_calls=100, now=NOW, jeda=0)
    import requests
    from fakes import mcas
    fc = mcas()
    monkeypatch.setattr(requests, "get", lambda url, headers=None, params=None, timeout=None: ta.R(fc.get(url.replace(ad.BASE_URL, ""), params)))
    monkeypatch.setenv("INVEZGO_API_KEY", "dummy")
    at = ta.open_app(monkeypatch)
    [b for b in at.button if b.label == "⚡ Hitung cepat"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    md = " ".join([m.value for m in at.markdown] + [m.value for m in at.subheader])
    assert "Ledger Broker" in md and "Cakupan" in md and "Estimasi posisi" in md and "Bandar Score" in md
    assert any("NET_FLOW_PROXY" in str(c.value) for c in at.caption) or "NET_FLOW_PROXY" in md


def test_ekspor_ai_txt_md_csv(hasil):
    import datetime as dt2
    import ledger_score as ls
    hasil["sinyal"] = ls.hitung(hasil, None)
    L = lc.baris_ai(hasil, "TST")
    t = "\n".join(L)
    assert "Bandar Score ledger" in t and "Bukti berlawanan" in t
    assert "## LEDGER BROKER" in t and "NET_FLOW_PROXY" in t and "Zona distribusi" in t
    import ringkas as rg
    md = "\n".join(rg._md_dari_baris(L))
    assert "| Broker |" in md
    assert lc.csv_ai(hasil).startswith(b"trade_date") and b"NET_FLOW_PROXY" in lc.csv_ai(hasil)
    assert "N/A" in "\n".join(lc.baris_ai(None, "TST"))
    assert lc.csv_ai(None).startswith(b"info")


def test_agregat_summary_dari_ledger(eng):
    import pandas as pd
    c = Fake()
    lg.perbarui(c, eng, "TST", hari_bursa=30, max_calls=100, now=NOW, jeda=0)
    a, z = HARI[-25], HARI[-1]
    out = lg.agregat_summary(eng, "TST", a, z)
    assert out is not None
    # sama dengan jumlah dari respons harian asli
    for br in ("AA", "BB"):
        exp_buy = sum(r["buy_volume"] for d in HARI[-25:] for r in c.daily[d] if r["code"] == br)
        got = [r for r in out if r["code"] == br][0]
        assert got["buy_volume"] == exp_buy and got["net_volume"] == got["buy_volume"] - got["sell_volume"]
    df = pd.DataFrame(out)
    assert {"code", "buy_avg", "net_volume", "net_value"} <= set(df.columns)
    assert lg.agregat_summary(eng, "TST", HARI[-60], z) is None          # sebelum cakupan ledger
    assert lg.agregat_summary(eng, "TST", a, dt.date(2026, 10, 12)) is not None   # tanggal setelah hari bursa terakhir tidak memengaruhi
    assert lg.agregat_summary(eng, "XXX", a, z) is None


def test_summary_hari_payload_tumbuhkan_ledger(eng):
    c = Fake()
    d = HARI[-3]
    p1 = lg.summary_hari_payload(c, eng, "TST", d, NOW)
    n = c.calls
    p2 = lg.summary_hari_payload(c, eng, "TST", d, NOW)
    assert p1 == p2 and c.calls == n                       # kedua kali dari ledger, 0 API
    b, _ = lg.muat(eng, "TST")
    assert set(b.trade_date) == {d} and len(b) == 3


def test_simpan_chart_mengisi_pasar(eng):
    c = Fake()
    raw = c.get("/analysis/chart/stock/TST", {})
    assert lg.simpan_chart(eng, "TST", raw, NOW) == len(HARI)
    _, m = lg.muat(eng, "TST")
    assert len(m) == len(HARI)


def test_rolling_dan_skor(eng):
    import ledger_score as ls
    c = Fake()
    lg.perbarui(c, eng, "TST", hari_bursa=70, max_calls=100, now=NOW, jeda=0)
    b, m = lg.muat(eng, "TST")
    res = lc.jalankan(b, m, lc.Cfg(), lg.log_api(eng), NOW)
    L = res["ledger"]
    M = ls.pasar_harian(L)
    assert len(M) == len(HARI) and (M["volume_lots"] == 10_000).all()              # volume pasar SEKALI per tanggal (1.000.000 lembar = 10.000 lot)
    R = ls.rolling_broker(L)
    aa = R[R.broker_code == "AA"].set_index("trade_date")
    d = HARI[-1]
    exp5 = sum((r["buy_volume"] - r["sell_volume"]) / 100 for dd in HARI[-5:] for r in c.daily[dd] if r["code"] == "AA")
    assert aa.loc[d, "net_lots_5d"] == exp5 and aa.loc[d, "net_lots_1d"] == (c.daily[d][0]["buy_volume"] - c.daily[d][0]["sell_volume"]) / 100
    assert aa.loc[d, "net_sell_streak"] >= 3 and aa.loc[d, "net_lots_20d"] < 0
    s, baru = ls.hitung_dan_simpan(eng, res, lg.muat_meta(eng, "TST"))
    assert baru and s["episode_status"] in (ls.ST_DIST, ls.ST_MARKDOWN, ls.ST_MIXED) and s["formula_version"] == ls.CALC_VERSION
    assert s["supporting_evidence"] is not None and s["contradicting_evidence"] and len(s["score_components"]) == 5
    assert all(0 <= x["poin"] <= x["maks"] for x in s["score_components"]) and 0 <= s["bandar_score"] <= 10
    s2, baru2 = ls.hitung_dan_simpan(eng, res, lg.muat_meta(eng, "TST"))          # input sama -> tidak dihitung ulang
    assert not baru2 and s2["bandar_score"] == s["bandar_score"]
    assert "Dimensi | Poin" in "\n".join(ls.baris(s))


def test_meta_200_sesi_dan_inkremental(eng):
    c = Fake()
    lg.perbarui(c, eng, "TST", hari_bursa=30, max_calls=100, now=NOW, jeda=0)
    m = lg.muat_meta(eng, "TST")
    assert m["sessions_loaded"] == 30 and m["target_sessions"] == 30 and m["history_complete"] and m["last_available_trade_date"] == HARI[-1]
    lg.perbarui(c, eng, "TST", hari_bursa=200, max_calls=0, now=NOW, jeda=0)     # target 200 sesi, hanya 70 sesi tersedia di pasar
    m = lg.muat_meta(eng, "TST")
    assert m["target_sessions"] == 200 and not m["history_complete"] and m["market_sessions"] == len(HARI)
    n0 = c.calls
    r = lg.sinkron_inkremental(c, eng, "TST", max_calls=5, now=NOW)               # tidak ada sesi baru -> hanya panggilan chart
    assert r["diambil"] == 0 and c.calls == n0 + 1
    assert lg.sinkron_inkremental(c, eng, "KOSONG", 5, NOW) is None               # ledger kosong: tidak ada backfill otomatis


def test_sesi_hilang_bukan_nol_buatan(eng):
    c = Fake()
    lg.perbarui(c, eng, "TST", hari_bursa=10, max_calls=100, now=NOW, jeda=0)
    b, m = lg.muat(eng, "TST")
    b = b[b.trade_date != HARI[-4]]
    L = lc.bangun(b, m, lc.Cfg())
    assert HARI[-4] not in set(L.trade_date) and L.trade_date.nunique() == 9
