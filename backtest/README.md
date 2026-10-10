# Backtest target Adimology

Menguji apakah sinyal + target Adimology (avg bandar + 5% + P papan, P = net lot bandar / lot per papan;
realistis = 0,5 × P) benar-benar lebih baik daripada pembanding sederhana. Rumus di `adimology.py`
**tidak diubah**; skrip memakai building block-nya, dan tes `test_signal_matches_ad_analyze` memastikan level
serta verdict sama dengan `ad.analyze`. Edukasi, bukan ajakan transaksi.

## Menjalankan
```bash
export INVEZGO_API_KEY='key-anda'      # hanya di environment; jangan tulis ke file yang di-commit
python backtest/run_backtest.py         # 1) cetak estimasi panggilan API, lalu berhenti
python backtest/run_backtest.py --yes   # 2) jalankan
```
- Setiap respons API (termasuk 404/422) disimpan ke `backtest/cache/` (di-gitignore). Run ulang dengan parameter yang
  sama **tidak memakai kuota**; `--offline` membaca cache saja tanpa key.
- `--max-calls N` membatasi panggilan baru per run. Bila tercapai, skrip berhenti (exit 4) dan cache tetap tersimpan,
  jadi bisa dicicil per hari: jalankan ulang dan skrip melanjutkan dari cache.
- Estimasi default (40 saham, 2024-06-01..hari ini, window 5, kedua studi): sekitar **16 ribu panggilan**
  (backtest target ~10.000: broker summary window + hari terakhir per Jumat per saham; studi imbalance ~5.700:
  satu snapshot orderbook per hari bursa per saham sejak 2026-03-25). Mulai kecil dulu, mis.
  `--tickers BBRI,ANTM,MDKA,ESSA --start 2025-06-01 --yes`, atau satu studi saja dengan `--study target` /
  `--study imbalance`.
- Opsi lain: `--windows 5,10,20` (setiap window menambah 1 panggilan per Jumat per saham),
  `--modes top1,top3` (top3 = gabungan 3 akumulator dengan avg beli tertimbang volume, sama dengan
  `mode_bandar="top3"` di app; **tanpa panggilan tambahan**), `--horizons 5,10,20`,
  `--exclude AB,CD` (broker ritel tambahan; default XL, YP, XC, PD), `--ob-time 14:30`, `--imb-time 14:30`,
  `--proxy-scale`, `--seed`. Bila `--ob-time` sama dengan `--imb-time`, snapshot Jumat dipakai bersama lewat cache.
- Mode `ob_multi` app (median 3 snapshot) belum diuji di backtest; backtest memakai satu snapshot.

## Aturan backtest
- **Tanggal evaluasi t**: hari bursa terakhir tiap minggu (Jumat, atau hari sebelumnya bila libur). Hanya data ≤ t.
- **Bandar**: top net buyer pada window broker summary (5 hari default), broker ritel dikecualikan. Distribusi = net
  bandar di hari terakhir negatif dan ≥ 25% net window (sama dengan app).
- **Sinyal Adimology** = verdict "SETUP TERPENUHI" di app: akumulasi, tidak distribusi, harga < target realistis,
  upside ≥ 3%, harga ≤ avg bandar + 5%.
- **Lot per papan**: mulai 2026-03-25 dari orderbook historis Invezgo (snapshot `--ob-time`, default 14:30, pada tanggal t). Sebelumnya
  memakai **proksi** = rata-rata lot transaksi per hari di window (dari broker summary) / median jumlah papan rentang
  harian (low→high, minimal 1). Hasil dilaporkan terpisah per regime (`proxy`, `orderbook`). Diagnostik
  `rasio_lpp_orderbook_per_proxy_median` menunjukkan seberapa jauh proksi dari orderbook; bila jauh dari 1, coba
  `--proxy-scale` dengan rasio itu dan bandingkan (ingat: menyetel parameter dari data yang sama itu overfitting).
- **Entry** = open hari bursa setelah t. **SL** = low terendah di window. **TP** = target realistis.
  Dalam H hari bursa: keluar di TP atau SL mana yang kena dulu; bila TP dan SL kena di hari yang sama dianggap SL
  (konservatif); bila tidak kena keduanya, keluar di close hari ke-H. Gap open di bawah SL / di atas TP keluar di open.
- **Fee**: beli 0,15%, jual 0,25% (sudah termasuk di semua return).
- Sinyal yang rentangnya (window + 20 hari) melintasi stock split / reverse split dibuang dari ringkasan
  (data Invezgo tidak disesuaikan; deteksi pakai `teknikal.sesuaikan_split`).

## Pembanding (baseline)
| strategi | entry | TP |
|---|---|---|
| `adimology` | sinyal Adimology | target realistis |
| `akumulasi` (a) | top net buyer akumulasi & tidak distribusi, tanpa filter target | median upside Adimology |
| `target_tetap` (c) | sinyal Adimology yang sama | median upside Adimology |
| `acak` (b) | saham + Jumat acak dari universe (seed tetap) | median upside Adimology |

Semua memakai SL, horizon dan fee yang sama. `target_tetap` menjawab "apakah *angka* targetnya informatif", `akumulasi`
menjawab "apakah filter/target menambah nilai di atas sekadar ikut bandar", `acak` menjawab "apakah ada edge sama sekali".

## Membaca hasil
File di `backtest/results/` (di-gitignore):
- `sinyal_*.csv`: semua baris (strategi, level, `hasil_H` = tp/sl/waktu, `net_H`, `mfe_H`, `mae_H`, `kalibrasi_H`).
- `ringkasan_*.csv`: per window × regime × H × strategi:
  `hit_tp` (TP kena sebelum SL), `hit_tmax` (target max), `kena_sl`, `expectancy` (rata-rata net return per trade) dan
  CI 95%, `median_net`, `kalibrasi_median` = harga tertinggi yang tercapai / TP (< 1 artinya target rata-rata
  terlalu tinggi, > 1 artinya target terlalu konservatif).
- `perbandingan_*.csv`: selisih Adimology − baseline untuk expectancy dan hit_tp dengan CI 95% bootstrap
  (resample per tanggal evaluasi, karena sinyal di minggu yang sama saling berkorelasi). `kesimpulan`:
  **LEBIH BAIK** hanya bila seluruh CI di atas 0.

Pegangan: lihat dulu `n` dan `n_tanggal` (di bawah ~30 tanggal, CI lebar dan kesimpulan rapuh). Regime `orderbook`
adalah uji rumus yang paling setia, tetapi datanya baru sejak Maret 2026. Hasil `proxy` menguji ide yang sama dengan
lot per papan pengganti, jadi bukan bukti langsung untuk angka orderbook.

## Studi B: imbalance orderbook (klaim slide 12)
Klaim: "total offer lebih tebal dari total bid -> cenderung naik; bid lebih tebal -> sideways/koreksi".
Untuk tiap saham dan tiap hari bursa sejak 2026-03-25: snapshot orderbook `--imb-time` (14:30),
rasio = total lot offer / total lot bid (level yang terlihat di snapshot), lalu return close hari itu ke close
1 dan 5 hari bursa berikutnya (tanpa fee; snapshot sebelum close jadi tidak ada look-ahead).

File: `imbalance_*.csv` (per saham-hari), `imbalance_ringkasan_*.csv` (per bucket rasio: <0,67, 0,67-1, 1-1,5, >=1,5,
plus baseline **semua hari**: rata-rata return dan CI, median, `pct_naik`), `imbalance_perbandingan_*.csv`
(tiap bucket vs semua hari, dan **offer >= bid vs bid > offer**, untuk rata-rata return dan % hari naik).
CI 95% bootstrap klaster per **minggu** (return 5 hari antar hari berurutan saling tumpang tindih, dan semua saham
bergerak bersama IHSG). Klaim didukung bila `offer >= bid` vs `bid > offer` berkesimpulan "kelompok LEBIH TINGGI".
Catatan: data hanya ~6 bulan, sebagian besar dalam satu kondisi pasar; snapshot bisa berisi antrean palsu (spoof).
