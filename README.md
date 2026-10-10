# Adimology Calculator (Streamlit)

Target harga metode Adimology dari data **broker summary + orderbook Invezgo**. Edukasi, bukan ajakan transaksi.

## Struktur
- `adimology.py`: mesin hitung (tanpa Streamlit; bisa dipakai ulang untuk backtest)
- `app.py`: tampilan (login, Petunjuk, Ringkasan, Rekomendasi, Fundamental, Broker, Orderbook, Sensitivitas, Multi-saham, Histori, Rumus)
- `fundamental.py`: tab Fundamental (PnL, neraca, arus kas per kuartal dari data YTD Invezgo, valuasi, peringatan kuning / sorotan hijau)
- `laporan.py`: tombol Export: PDF dokumen riset (keputusan BELI / TUNGGU / HINDARI-JUAL) dan Excel (semua data per sheet)
- `intraday.py`: statistik BPJP/BSJP dari candle 30 menit (volume pagi/sore vs pergerakan sesi 1 berikutnya, target & jam jual)
- `audit.py`: satu fungsi posisi/risiko (lot, rugi di stop incl. fee), validasi laporan (rekonsiliasi angka, kesegaran data), parser aksi korporasi, daftar batasan data/endpoint
- `ringkas.py`: tombol Unduh TXT: ringkasan padat (±7-12 KB) untuk dianalisa AI, termasuk statistik historis harga (peluang naik, sentuh target sebelum stop)
- `hengky.py`: aturan disiplin trading dari materi Ko Hengky (Remora): checklist uptrend/likuid/level, harga-volume, bid-offer, rencana support/breakout (stop 1 tick), SID, pra-open
- `bandar.py`: penilaian bandarmologi berbasis bukti (7 kriteria + deteksi jebakan); target Adimology diperlakukan sebagai skenario karena backtest tidak menunjukkan edge
- `aliran.py`: konfirmasi aliran: asing, pasar nego, konsistensi bandar 1/3/7/20 hari, afiliasi broker dengan pemegang saham >= 5%, minat ritel (SID + aliran broker ritel XL/XC/PD/CC/YP)
- `teknikal.py`: analisa teknikal tab Rekomendasi (support/resistance, MA 5/20/50/100/200, RSI, MACD, ATR, bandar value)
- `tests/`: 90 tes (audit, fundamental FY/LTM, minat ritel, ringkasan TXT, Hengky, bandar jangka panjang, penilaian bandar, backtest, login, aliran, teknikal, BPJP/BSJP, fundamental, ekspor, rumus, fraksi bertingkat, kasus MCAS sama dengan output Colab, UI smoke test)

## Jalankan lokal
```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # isi INVEZGO_API_KEY
streamlit run app.py
```
App diawali halaman login (email + password dari Secrets). Tanpa `INVEZGO_API_KEY`, app meminta API key lewat sidebar (tidak disimpan).

## Deploy gratis (Streamlit Community Cloud)
1. Push folder ini ke repo GitHub (`.streamlit/secrets.toml` sudah di `.gitignore`).
2. share.streamlit.io, New app, pilih repo, file utama `app.py`.
3. Settings > Secrets, isi `INVEZGO_API_KEY`, `APP_LOGIN_EMAIL`, dan `APP_LOGIN_PASSWORD`.
4. Disarankan app dibuat private (batasi viewer), karena API key Anda dipakai untuk semua pengunjung.

## Tes
```bash
pip install pytest
python -m pytest -q
```

## Catatan
- Cache 5 menit untuk data EOD, 15 detik untuk orderbook.
- Saat pasar tutup orderbook hanya 1 level; pakai "lot per papan manual" atau snapshot historis.
- Fraksi mengikuti aturan BEI standar (tabel `TIERS` di `adimology.py`).
