import io

import numpy as np
from openpyxl import load_workbook

import adimology as ad
import fundamental as fd
import laporan as lp
import teknikal as tk
from fakes import setup_case
from test_fundamental import _company
from test_teknikal import ohlc


def _inputs(liab_scale=0.5):
    r = ad.analyze(setup_case(), ad.Params(kode="TEST"))
    x = np.arange(320)
    rk = tk.rekomendasi(ohlc(np.linspace(80, 120, 320) + 6 * np.sin(x * 2 * np.pi / 30)), {"status": "AKUMULASI"})
    f = fd.analisa(harga=r.harga, **_company(liab_scale=liab_scale))
    return r, rk, f


def _nilai(kat):
    return {"kategori": kat, "skor": 4, "skor_maks": 7, "jebakan": ["contoh jebakan"] if kat.startswith("WASPADA") else [],
            "kriteria": [], "info": []}


def test_keputusan_rules():
    r, rk, f = _inputs()
    beli = dict(rk, verdict="BELI DI AREA ENTRY")
    assert lp.keputusan(r, beli, f, nilai=_nilai("AKUMULASI KUAT"))[0] == "BELI"
    kep, alasan = lp.keputusan(r, beli, f, nilai=_nilai("AKUMULASI LEMAH / BELUM SEARAH"))
    assert kep == "TUNGGU" and any("belum searah" in a for a in alasan)
    assert lp.keputusan(r, beli, f, nilai=_nilai("WASPADA: TANDA DISTRIBUSI / JEBAKAN"))[0] == "HINDARI / JUAL"
    assert lp.keputusan(r, dict(rk, verdict="HINDARI"), f, nilai=_nilai("AKUMULASI KUAT"))[0] == "HINDARI / JUAL"
    _, _, f_bad = _inputs(liab_scale=1.3)                         # utang > aset menang atas semuanya
    kep, alasan = lp.keputusan(r, beli, f_bad, nilai=_nilai("AKUMULASI KUAT"))
    assert kep == "HINDARI / JUAL" and any("LEBIH BESAR" in a for a in alasan)
    # tanpa data aliran, penilaian tidak pernah KUAT -> tidak pernah BELI
    assert lp.keputusan(r, beli, f)[0] != "BELI"
    assert any("Skenario Adimology" in a for a in lp.keputusan(r, None, None)[1])


def test_pdf_and_excel_are_built():
    r, rk, f = _inputs()
    pdf = lp.pdf_bytes(r, rk, f, "2026-10-10 15:00 WIB")
    assert pdf[:4] == b"%PDF" and len(pdf) > 20_000             # termasuk grafik
    wb = load_workbook(io.BytesIO(lp.excel_bytes(r, rk, f, "2026-10-10 15:00 WIB")))
    assert {"Ringkasan", "Teknikal", "Broker", "PnL", "Neraca", "Arus kas", "Valuasi"} <= set(wb.sheetnames)
    ring = {row[0].value: row[1].value for row in wb["Ringkasan"].iter_rows(min_row=2)}
    assert ring["Kode"] == "TEST" and ring["Keputusan"] in lp.WARNA_KEP
    pnl = wb["PnL"]
    fills = {pnl.cell(row=i, column=j).fill.fgColor.rgb for i in range(2, pnl.max_row + 1) for j in range(2, pnl.max_column + 1)}
    assert any(str(c).endswith("DCFCE7") for c in fills)          # sel hijau ikut ke Excel


def test_export_with_bpjp_bsjp():
    import intraday as idt
    from test_intraday import C, bars
    from openpyxl import load_workbook
    r, rk, f = _inputs()
    o = idt.statistik(idt.get_bars(C(bars(61)), "TEST", None))
    assert lp.pdf_bytes(r, rk, f, "x", o)[:4] == b"%PDF"
    wb = load_workbook(io.BytesIO(lp.excel_bytes(r, rk, f, "x", o)))
    assert {"BSJP", "BPJP", "BSJP data harian"} <= set(wb.sheetnames)


def test_export_without_technical_or_fundamental():
    r, _, _ = _inputs()
    assert lp.pdf_bytes(r, None, None, "x")[:4] == b"%PDF"
    assert "Ringkasan" in load_workbook(io.BytesIO(lp.excel_bytes(r, None, None, "x"))).sheetnames
