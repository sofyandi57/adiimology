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
    """Vonis kini dari SKOR tiga dimensi (lihat tests/test_skor.py untuk kasus rinci); di sini hanya kontrak antarmuka dan batas keras."""
    r, rk, f = _inputs()
    kep, alasan = lp.keputusan(r, rk, f, nilai=_nilai("AKUMULASI KUAT"))
    assert kep in ("BELI", "TUNGGU", "HINDARI / JUAL") and alasan[0].startswith("RINGKAS: ")
    assert any(a.startswith("Skor Bandarmologi") for a in alasan) and any(a.startswith("Skor Teknikal") for a in alasan) and any(a.startswith("Skor Fundamental") for a in alasan)
    assert any("Skenario Adimology" in a for a in lp.keputusan(r, None, None)[1])
    assert lp.keputusan(r, None, None)[0] == "TUNGGU"                          # data tidak cukup: bukan vonis tegas
    _, _, f_bad = _inputs(liab_scale=1.3)                                      # utang > aset: skor fundamental dibatasi 2,5
    assert any("FATAL" in a or "SOLVABILITAS" in a for a in lp.keputusan(r, rk, f_bad, nilai=_nilai("AKUMULASI KUAT"))[1])


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


def test_beli_needs_hengky_core_rules():
    r, rk, f = _inputs()
    beli = dict(rk, verdict="BELI DI AREA ENTRY")
    gagal = {"checklist": {"lolos_inti": False, "hasil": "TAHAN: 2 dari 4 aturan inti belum terpenuhi"}}
    lolos = {"checklist": {"lolos_inti": True, "hasil": "ATURAN INTI TERPENUHI"}}
    kep, alasan = lp.keputusan(r, beli, f, nilai=_nilai("AKUMULASI KUAT"), hk_=gagal)
    assert kep == "TUNGGU" and any("Hengky" in a for a in alasan)
    assert lp.keputusan(r, beli, f, nilai=_nilai("AKUMULASI KUAT"), hk_=lolos)[0] == "BELI"
