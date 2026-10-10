# Adimology Calculator (Streamlit)

Target harga metode Adimology dari data **broker summary + orderbook Invezgo**. Edukasi, bukan ajakan transaksi.

## Struktur
- `adimology.py`: mesin hitung (tanpa Streamlit; bisa dipakai ulang untuk backtest)
- `app.py`: tampilan (Ringkasan, Broker, Orderbook, Sensitivitas, Multi-saham, Rumus)
- `tests/`: 13 tes (login, rumus, fraksi bertingkat, kasus MCAS sama dengan output Colab, UI smoke test)

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
