"""Tes spesifikasi improvement ledger bandarmologi (P0 + P1): tanpa kebocoran masa depan, tabel terpisah, NULL vs 0, anomali, validasi,
fitur rolling, crossing/hari ekstrem, format laporan, dan verifikasi lintas sumber."""
import datetime as dt
import io

import numpy as np
import pandas as pd
import pytest

import ledger as lg
import ledger_calc as lc
import ledger_kualitas as lq

HARI = [d.date() for d in pd.bdate_range(end="2026-10-09", periods=40)]
RET = dt.datetime(2026, 10, 10, 3, 0)
STATUS_OK = {lq.ACC_C, lq.DIST_C, lq.NEUTRAL, lq.INSUF}


def _pasar(close=1000.0, vol_lot=100_000, hari=HARI):
    return pd.DataFrame([dict(ticker="TST", trade_date=d, data_source=lg.SUMBER_PASAR, open=close, high=close * 1.02, low=close * 0.98, close=close,
                              volume_shares=vol_lot * 100.0, retrieved_at=RET) for d in hari])


def _row(d, code, buy_lot, sell_lot, px=1000.0, sumber=lg.SUMBER):
    bs = None if buy_lot is None else buy_lot * 100.0
    ss = None if sell_lot is None else sell_lot * 100.0
    return dict(ticker="TST", trade_date=d, broker_code=code, data_source=sumber, broker_name=code, buy_shares=bs, sell_shares=ss,
                buy_value=None if bs is None else bs * px, sell_value=None if ss is None else ss * px, buy_avg_raw=px if bs else None,
                sell_avg_raw=px if ss else None, net_shares_raw=(bs - ss) if bs is not None and ss is not None else None, is_final=True, retrieved_at=RET)


def _skenario():
    """AA: net beli 30 hari pertama lalu net jual; BB: seimbang; KZ: beli besar hampir tanpa jual (kasus 754.563 vs 1.429);
    SS: jual tanpa beli; DD: hanya muncul di paruh kedua (hari lain = tidak ada baris sumber)."""
    rows = []
    for i, d in enumerate(HARI):
        rows.append(_row(d, "AA", 3000 if i < 30 else 500, 500 if i < 30 else 4000))
        rows.append(_row(d, "BB", 1000 + (i % 3) * 100, 1000 + ((i + 1) % 3) * 100))
        rows.append(_row(d, "KZ", 754_563 / len(HARI), 1429 if i == 5 else 0))
        rows.append(_row(d, "SS", 0, 1500))
        if i >= 25:
            rows.append(_row(d, "DD", 2000, 100))
    return pd.DataFrame(rows), _pasar(vol_lot=100_000)


@pytest.fixture
def res():
    b, m = _skenario()
    return lc.jalankan(b, m, lc.Cfg(min_hari_aktif=5))


# ------------------------------------------------------------------- P0.1 kebocoran
def test_tidak_ada_status_akhir_di_baris_historis(res):
    L = res["ledger"]
    for c in ("flow_status_akhir", "confidence_akhir", "estimated_position_lots_low", "estimated_avg_cost_proxy"):
        assert c not in L.columns
    assert {"daily_flow_status", "rolling_flow_status", "cumulative_net_flow_proxy"} <= set(L.columns)
    assert set(L["daily_flow_status"]) <= STATUS_OK and set(L["rolling_flow_status"]) <= STATUS_OK
    # status harian AA berubah mengikuti data harian (bukan satu label akhir yang disalin)
    aa = L[L.broker_code == "AA"].sort_values("trade_date")
    assert aa["daily_flow_status"].iloc[0] == lq.ACC_C and aa["daily_flow_status"].iloc[-1] == lq.DIST_C
    pe = res["period_end"]
    assert pe.set_index("broker_code").loc["AA", "as_of_date"] == HARI[-1] and pe["broker_code"].is_unique


def test_status_tanggal_t_hanya_memakai_data_sampai_t():
    b, m = _skenario()
    full = lc.jalankan(b, m, lc.Cfg(min_hari_aktif=5))
    T = HARI[27]
    pot = lc.jalankan(b[b.trade_date <= T], m[m.trade_date <= T], lc.Cfg(min_hari_aktif=5))
    kol = ["trade_date", "broker_code", "daily_flow_status", "rolling_flow_status", "cumulative_net_flow_proxy"]
    a = full["ledger"][full["ledger"].trade_date <= T][kol].sort_values(kol[:2]).reset_index(drop=True)
    p = pot["ledger"][kol].sort_values(kol[:2]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, p)
    sa_ = full["status_detail"]
    sp = pot["status_detail"]
    ka = sa_[sa_.trade_date <= T].sort_values(["trade_date", "broker_code", "status_type"]).reset_index(drop=True)
    kp = sp.sort_values(["trade_date", "broker_code", "status_type"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(ka, kp)
    for n in lq.JENDELA_FITUR:            # fitur rolling pada T identik
        pd.testing.assert_frame_equal(full["fitur"][n]["net_flow_pct_market_volume"].loc[:T, pot["fitur"][n]["net_lots"].columns],
                                      pot["fitur"][n]["net_flow_pct_market_volume"], check_freq=False)


def test_status_memuat_semua_field(res):
    sd = res["status_detail"]
    wajib = {"status", "evidence_for", "evidence_against", "data_quality", "confidence", "as_of_date", "lookback_days", "invalidation_condition"}
    assert wajib <= set(sd.columns) and wajib <= set(res["period_end"].columns)
    assert set(sd["status_type"]) == {"daily", "rolling"} and set(sd["status"]) <= STATUS_OK
    r = sd[(sd.status_type == "rolling")]
    assert (r["lookback_days"] == 20).all()
    assert (r[r.trade_date < HARI[19]]["status"] == lq.INSUF).all()           # jendela belum lengkap
    acc = sd[sd.status == lq.ACC_C]
    assert (acc["evidence_for"] != "").all() and (acc["invalidation_condition"].str.len() > 0).all()


# ------------------------------------------------------------------- P0.2 tabel terpisah
def test_volume_pasar_tidak_diulang_per_broker(res):
    bf = lq.broker_flow_daily(res["ledger"])
    assert not {"volume_lots", "close_price", "market_volume_lots"} & set(bf.columns)
    M = res["pasar"]
    assert M["trade_date"].is_unique and len(M) == len(HARI)
    assert (M["market_volume_shares"] == M["market_volume_lots"] * 100).all() and (M["volume_unit_check"] == "OK").all()
    j = bf.merge(M[["trade_date", "market_volume_lots"]], on="trade_date")       # join per tanggal
    assert len(j) == len(bf)
    paket = lc.ekspor_ai(res)
    assert set(paket) == {"broker_flow_daily.csv", "market_daily.csv", "period_end_summary.csv", "broker_flow_features.csv"}
    head = set(paket["broker_flow_daily.csv"].split(b"\n")[0].decode().split(","))
    assert not {"volume_lots", "close_price", "market_volume_lots", "market_volume_shares", "close"} & head
    md = pd.read_csv(io.BytesIO(paket["market_daily.csv"]))
    assert md["trade_date"].is_unique


def test_cek_satuan_volume_mendeteksi_salah_satuan():
    b, m = _skenario()
    m = m.assign(volume_shares=m["volume_shares"] / 100 / 100 + 7)   # seolah sumber melaporkan lot/100 (bukan lembar) dan bukan kelipatan lot
    M = lc.jalankan(b, m, lc.Cfg(min_hari_aktif=5))["pasar"]
    assert (M["volume_unit_check"] != "OK").all()


# ------------------------------------------------------------------- P0.3 NULL vs 0
def test_null_tidak_sama_dengan_nol(res):
    L = res["ledger"]
    dd = L[L.broker_code == "DD"].set_index("trade_date")
    assert dd.loc[HARI[0], "data_completeness"] == lq.NA and pd.isna(dd.loc[HARI[0], "buy_lots"]) and pd.isna(dd.loc[HARI[0], "net_lots"])
    assert dd.loc[HARI[-1], "data_completeness"] == "COMPLETE"
    ss = L[L.broker_code == "SS"]
    assert (ss["buy_lots"] == 0).all() and ss["buy_lots"].notna().all()           # nol dari sumber tetap 0
    assert (L[L.data_completeness == lq.NA]["daily_flow_status"] == lq.INSUF).all()
    assert dd.loc[HARI[0], "cumulative_net_flow_proxy"] == 0 and dd.loc[HARI[-1], "cumulative_net_flow_proxy"] == pytest.approx(1900 * 15)


def test_sisi_hilang_di_sumber_disimpan_null(tmp_path):
    payload = {"data": [dict(code="AA", name="A", buy_volume=1000, buy_value=1e6, buy_avg=1000), dict(code="BB", name="B", buy_volume=0, sell_volume=0)]}
    rows = lg.normalisasi_summary("TST", HARI[-1], payload, RET, True)
    aa, bb = rows
    assert aa["sell_shares"] is None and aa["sell_value"] is None and bb["buy_shares"] == 0.0
    b = pd.DataFrame([dict(r, retrieved_at=RET) for r in rows])
    L = lc.bangun(b, _pasar(hari=[HARI[-1]]))
    a = L[L.broker_code == "AA"].iloc[0]
    assert a["data_completeness"] == "PARTIAL" and pd.isna(a["net_lots"]) and "SIDE_NULL" in a["row_flags"]
    assert not L["crossing"].any()                                                 # sisi NULL tidak pernah dinilai crossing


def test_estimasi_posisi_dan_modal_not_available(res):
    pe = res["period_end"].set_index("broker_code")
    assert (pe["estimated_position_status"] == lq.NA).all() and pe["estimated_position_lots"].isna().all()
    assert pe["estimated_position_reason"].str.contains("persediaan awal").all()
    assert pe.loc["BB", "avg_cost_status"] == lq.NA and pd.isna(pe.loc["BB", "avg_cost_proxy"]) and pe.loc["BB", "avg_cost_reason"]
    t = lc.tabel_posisi(res)
    assert {"Status posisi", "Alasan posisi", "Status estimasi modal", "Cumulative net flow proxy"} <= set(t.columns)


# ------------------------------------------------------------------- P0.4 anomali
def test_anomali_satu_sisi_kasus_kz(res):
    an = res["anomali"]
    assert any(a["flag"] == "ONE_SIDED_BUY" for a in an["KZ"]) and "754,563" in an["KZ"][0]["detail"]
    assert any(a["flag"] == "ONE_SIDED_SELL" for a in an["SS"])
    assert "AA" not in an or not any(a["flag"].startswith("ONE_SIDED") for a in an["AA"])
    pe = res["period_end"].set_index("broker_code")
    assert pe.loc["KZ", "avg_cost_status"] == lq.UNRELIABLE and pe.loc["KZ", "data_quality"].startswith("UNRELIABLE")
    assert pe.loc["KZ", "confidence"] in ("Low", "Insufficient Data") and "ONE_SIDED_BUY" in pe.loc["KZ", "evidence_against"]


def test_anomali_lain_dan_ambang_configurable():
    b, m = _skenario()
    i_ek = HARI[-10]
    b.loc[(b.broker_code == "BB") & (b.trade_date == i_ek), ["buy_shares", "buy_value"]] = [60_000 * 100.0, 60_000 * 100.0 * 1000]
    b.loc[(b.broker_code == "AA") & (b.trade_date == HARI[3]), "buy_value"] = 3000 * 100.0 * 1200     # rata-rata beli 1200 > high 1020
    b = b[~((b.broker_code == "AA") & (b.trade_date >= HARI[-8]))]                                   # AA hilang 8 sesi terakhir
    cfg = lc.Cfg(min_hari_aktif=5)
    cfg.kualitas = lq.CfgQ(muncul_hilang_pct=0.001)
    res = lc.jalankan(b, m, cfg)
    fl = {k: {a["flag"] for a in v} for k, v in res["anomali"].items()}
    assert "EXTREME_NET_VS_VOLUME" in fl["BB"] and "SINGLE_DAY_DOMINANCE" in fl["BB"]
    assert "AVG_PRICE_OUT_OF_RANGE" in fl["AA"] and "DISAPPEARED" in fl["AA"] and "APPEARED" in fl["DD"]
    row = res["ledger"].set_index(["broker_code", "trade_date"]).loc[("BB", i_ek)]
    assert "NET_EXTREME_VS_VOLUME" in row["row_flags"]
    longgar = lc.Cfg(min_hari_aktif=5)
    longgar.kualitas = lq.CfgQ(ekstrem_net_pct=0.9, dominasi_hari=0.99, satu_sisi_rasio=0.0, satu_sisi_min_hari=10_000)
    fl2 = {k: {a["flag"] for a in v} for k, v in lc.jalankan(b, m, longgar)["anomali"].items()}
    assert "EXTREME_NET_VS_VOLUME" not in fl2.get("BB", set()) and "ONE_SIDED_BUY" not in fl2.get("KZ", set())


# ------------------------------------------------------------------- P0.5 validasi & jejak sumber
def test_validasi_net_dan_duplikat():
    b, m = _skenario()
    L = lc.bangun(b, m)
    assert all(c["status"] in ("OK", "PERINGATAN") for c in lq.cek_ledger(L))
    rusak = L.copy()
    rusak.loc[rusak.index[0], "net_lots"] += 5
    ids = {c["id"]: c["status"] for c in lq.cek_ledger(rusak)}
    assert ids["net_lots_valid"] == "GAGAL"
    dup = pd.concat([L, L.iloc[[0]]])
    assert {c["id"]: c["status"] for c in lq.cek_ledger(dup)}["duplikat_kunci_ledger"] == "GAGAL"
    b2 = pd.concat([b, b.iloc[[0]].assign(data_source="sumber_x"), b.iloc[[0]].assign(data_source="sumber_y")])   # dua sumber, segmen sama (UNKNOWN)
    v = lg.validasi(b2, m)
    assert {c["id"]: c["status"] for c in v["cek"]}["duplikat_segmen"] == "PERINGATAN"
    res = lc.jalankan(b, m, lc.Cfg(min_hari_aktif=5))
    assert {"net_lots_valid", "duplikat_kunci_ledger", "null_vs_nol", "nilai_null"} <= {c["id"] for c in res["validasi"]["cek"]}


def test_csv_memuat_sumber_waktu_satuan_versi(res):
    bf = pd.read_csv(io.BytesIO(lc.csv_ledger(res)))
    for c in ("source_id", "ingested_at", "unit_volume", "unit_price", "source_version", "formula_version", "market_segment", "data_completeness", "label"):
        assert c in bf.columns
    ok = bf[bf.data_completeness != lq.NA]
    assert ok["source_id"].str.startswith(lg.SUMBER).all() and ok["ingested_at"].str.endswith("Z").all()
    assert bf[bf.data_completeness == lq.NA]["source_id"].isna().all()
    assert (bf["label"] == "NET_FLOW_PROXY").all()
    md = pd.read_csv(io.BytesIO(lc.csv_market(res)))
    assert {"source_id", "ingested_at", "unit_volume", "adjustment_basis", "external_verification"} <= set(md.columns)


# ------------------------------------------------------------------- P1.6 fitur rolling
def test_fitur_rolling_nilai_dan_denominator():
    tgl = HARI[:6]
    nets = [100, -50, 200, 300, -10, 40]        # lot
    rows = [_row(d, "XX", 1000 + max(n, 0), 1000 + max(-n, 0)) for d, n in zip(tgl, nets)] + [_row(d, "YY", 500, 0) for d in tgl]
    b, m = pd.DataFrame(rows), _pasar(vol_lot=10_000, hari=tgl)
    L = lc.bangun(b, m)
    F = lq.fitur_rolling(L, tgl, (1, 3, 5))
    t = tgl[-1]
    f5 = {k: v.loc[t, "XX"] for k, v in F[5].items()}
    assert f5["net_lots"] == pytest.approx(-50 + 200 + 300 - 10 + 40)
    assert f5["net_flow_pct_market_volume"] == pytest.approx(480 / 50_000)
    assert f5["positive_flow_days_pct"] == pytest.approx(3 / 5)
    assert f5["longest_positive_run"] == 2 and f5["direction_changes"] == 3
    b5, s5 = F[5]["buy_lots"].loc[t, "XX"], F[5]["sell_lots"].loc[t, "XX"]
    assert f5["buy_sell_imbalance"] == pytest.approx((b5 - s5) / (b5 + s5))
    assert f5["share_of_total_net_buy"] == pytest.approx(480 / (480 + 2500))
    assert F[5]["source_coverage"].loc[t, "XX"] == 1 and F[5]["quality_flag"].loc[t, "XX"] == "OK"
    assert F[5]["quality_flag"].loc[tgl[2], "XX"] == "INSUFFICIENT_WINDOW"
    K = lq.konsentrasi(F)
    k = K[(K.as_of_date == t) & (K.lookback_days == 5)].iloc[0]
    assert k["top1_broker"] == "YY" and k["denominator_net_buy_lots"] == pytest.approx(2980) and k["top3_share_net_buy"] == pytest.approx(1.0)
    long = lq.fitur_long(F, ["XX"], [t])
    assert {"as_of_date", "lookback_days", "formula_version", "source_coverage", "quality_flag", "feature_value", "denominator"} <= set(long.columns)
    assert (long["formula_version"] == lq.FORMULA_VERSION).all() and set(long["lookback_days"]) == {1, 3, 5}


def test_fitur_tanda_session_gap_dan_coverage():
    b, m = _skenario()
    b = b[b.trade_date != HARI[-3]]                                                # satu sesi bursa hilang dari ledger
    res = lc.jalankan(b, m, lc.Cfg(min_hari_aktif=5))
    assert res["fitur"][5]["quality_flag"].loc[HARI[-1], "AA"] == "SESSION_GAP"
    assert res["fitur"][20]["source_coverage"].loc[HARI[-1], "DD"] < 1
    assert res["fitur"][20]["quality_flag"].loc[HARI[-1], "DD"] == "PARTIAL_COVERAGE"


# ------------------------------------------------------------------- P1.7 crossing & hari ekstrem
def test_crossing_dipisah_dan_hari_ekstrem_tanpa_menghapus_data():
    b, m = _skenario()
    b = pd.concat([b, pd.DataFrame([_row(d, "CX", 20_000, 20_000) for d in HARI])])
    b.loc[(b.broker_code == "BB") & (b.trade_date == HARI[-10]), ["buy_shares", "buy_value"]] = [60_000 * 100.0, 60_000 * 100.0 * 1000]
    b = pd.concat([b, pd.DataFrame([_row(HARI[-1], "UK", 100, 0, sumber="sumber_lain")])])
    asli = b.copy()
    res = lc.jalankan(b, m, lc.Cfg(min_hari_aktif=5))
    pd.testing.assert_frame_equal(b, asli)
    X = res["crossing"].set_index(["broker_code", "market_segment"])
    assert X.loc[("CX", "REGULAR"), "crossing_days"] == len(HARI) and X.loc[("CX", "REGULAR"), "gross_buy_non_crossing"] == 0
    assert ("UK", "UNKNOWN") in X.index and "tidak diketahui" in X.loc[("UK", "UNKNOWN"), "comparable_note"]
    E = res["ekstrem"].set_index("broker_code")
    assert E.loc["BB", "top_abs_day"] == HARI[-10] and E.loc["BB", "n_extreme_days"] == 1
    assert E.loc["BB", "net_lots_ex_top_day"] == pytest.approx(E.loc["BB", "net_lots_total"] - E.loc["BB", "top_abs_day_net"])
    assert bool(E.loc["BB", "sign_flip_ex_top_day"]) == (np.sign(E.loc["BB", "net_lots_ex_top_day"]) != np.sign(E.loc["BB", "net_lots_total"]))
    assert len(res["ledger"][res["ledger"].broker_code == "BB"]) == len(HARI)       # hari ekstrem tetap ada di data
    assert "net_lots_ex_extreme_days" in res["period_end"].columns


# ------------------------------------------------------------------- P1.8 format laporan
def test_laporan_format_observasi_sampai_data_tambahan(res):
    t = "\n".join(lc.baris_ai(res, "TST"))
    pos = [t.index(f"### {h}") for h in ("OBSERVASI", "INFERENSI", "BUKTI YANG MELAWAN", "KUALITAS DATA", "KESIMPULAN", "INVALIDASI", "DATA TAMBAHAN")]
    assert pos == sorted(pos)
    assert "KZ: ONE_SIDED_BUY" in t and "NOT_AVAILABLE" in t and "belum diverifikasi sumber eksternal" in t
    assert "Estimated remaining position" not in t and "Sisa rendah" not in t
    for kata in ("bandar sedang akumulasi", "posisi aktual broker", "masih mengakumulasi"):
        assert kata not in t.lower()
    import ringkas as rg
    md = "\n".join(rg._md_dari_baris(lc.baris_ai(res, "TST")))
    assert "| Broker |" in md and "### KESIMPULAN" in md


# ------------------------------------------------------------------- P1.9 lintas sumber
def test_verifikasi_sumber_kedua(tmp_path):
    eng = lg.engine(f"sqlite:///{tmp_path}/k.db")
    b, m = _skenario()
    assert lg.periksa_sumber(m)["status"] == "BELUM_DIVERIFIKASI"
    with eng.begin() as c:
        lg.upsert(c, lg.T_MARKET, m.drop(columns=[]).to_dict("records"), ["ticker", "trade_date", "data_source"])
        lg.upsert(c, lg.T_BROKER, b.drop(columns=["net_shares_raw"]).assign(net_shares_raw=b["net_shares_raw"], net_value_raw=None).to_dict("records"),
                  ["ticker", "trade_date", "broker_code", "data_source"])
    ext = pd.DataFrame(dict(date=[str(d) for d in HARI[-3:]], close=[1000.0, 1000.0, 1010.0], volume_lots=[100_000, 100_000, 100_000]))
    assert lg.simpan_pasar_eksternal(eng, "TST", ext, "csv_kedua") == 3
    with pytest.raises(ValueError):
        lg.simpan_pasar_eksternal(eng, "TST", ext, lg.SUMBER_PASAR)
    b2, m2 = lg.muat(eng, "TST")
    assert set(m2["data_source"]) == {lg.SUMBER_PASAR, "csv_kedua"}
    res = lc.jalankan(b2, m2, lc.Cfg(min_hari_aktif=5))
    v = res["sumber"]
    assert v["status"] == "KONFLIK" and v["n_dibanding"] == 3
    K = v["konflik"]
    assert len(K) == 1 and K.iloc[0]["trade_date"] == HARI[-1] and K.iloc[0]["value_a"] == 1000.0 and K.iloc[0]["value_b"] == 1010.0
    M = res["pasar"].set_index("trade_date")
    assert len(M) == len(HARI) and M.loc[HARI[-1], "close"] == 1000.0                 # sumber utama tidak ditimpa
    assert M.loc[HARI[-1], "external_verification"].startswith("KONFLIK") and M.loc[HARI[-2], "external_verification"].startswith("COCOK")
    assert M.loc[HARI[0], "external_verification"] == lq.BELUM_VERIF
    assert lg.simpan_konflik(eng, "TST", K) == 1
    with eng.connect() as c:
        r = c.execute(lg.T_CONFLICT.select()).mappings().all()
    assert len(r) == 1 and r[0]["source_b"] == "csv_kedua" and r[0]["value_b"] == 1010.0
    assert res["ledger"]["trade_date"].nunique() == len(HARI)
