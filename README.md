# Simulasi BBM — Comparing Scenario

Repository ini mensimulasikan persediaan BBM kapal, ROB, antrean bunker, dan
operasi tanker untuk skenario yang memprioritaskan checkpoint Surabaya.

## Struktur repository

```text
.
├── fuel_simulation/       # Package dan seluruh logika simulasi
├── data/                  # Workbook input yang diperlukan simulasi
├── tests/                 # Regression test
├── outputs/               # Hasil generated; tidak disimpan di Git
├── requirements.txt       # Dependency Python
└── README.md
```

## Persiapan

Gunakan Python 3.8 atau lebih baru, kemudian pasang dependency:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Aturan checkpoint

- Jika `FULL_SAILING_ROUTE` suatu vesvoy memuat `IDSUB`, checkpoint utama
  vesvoy tersebut adalah `IDSUB`.
- Jika tidak memuat `IDSUB`, checkpoint utama menggunakan `IDJKT`.
- Event menghubungkan checkpoint pilihan vesvoy sekarang dengan checkpoint
  pilihan berikutnya. Karena itu transisinya dapat berupa `IDSUB→IDSUB`,
  `IDSUB→IDJKT`, `IDJKT→IDJKT`, atau `IDJKT→IDSUB`.
- Jakarta tetap menjadi bagian sailing route pada interval `IDSUB→IDSUB`,
  tetapi bukan checkpoint bunker.
- Checkpoint cabang `IDMAK`/`IDAMQ` tetap dapat memotong interval. Khusus TBI,
  setiap kedatangan `IDMAK` tetap menjadi checkpoint cabang.

## Penempatan dan karakteristik tanker

- SIGMA beroperasi di `IDJKT`.
- ZETA beroperasi di `IDSUB`.
- Kapasitas, discharge rate, dan minimum stock mengikuti aset tanker.
- Load rate, pola loading bersamaan/berurutan, dan waktu mobilisasi mengikuti
  daerah operasi.

Konfigurasi efektif:

| Daerah | Tanker | Kapasitas MFO/BIO | Discharge MFO/BIO | Load MFO/BIO | Loading |
|---|---|---:|---:|---:|---|
| IDJKT | SIGMA | 1000/500 KL | 110/90 KL/jam | 100/120 KL/jam | Berurutan |
| IDSUB | ZETA | 750/250 KL | 110/50 KL/jam | 40/70 KL/jam | Bersamaan |

## Aturan lain yang dipertahankan

- Mei 2026 menjadi warm-up; KPI dihitung untuk Juni-Juli 2026.
- Target ROB adalah 150% kebutuhan dasar dan dibatasi kapasitas kapal.
- ROB kapal diteruskan antarevent.
- Antrean tanker memakai EDD berdasarkan ETD, fallback deadline, dengan
  look-ahead wait.
- Penggabungan direct voyage dan seluruh validasi stok tetap aktif.
- Loading hanya boleh dilakukan bila volume menuju penuh mencapai sedikitnya
  250 KL MFO atau 100 KL BIO. Aturan berlaku pada semua kondisi dan kedua
  tangki selalu dipenuhkan.
- Tanker memakai stok secara drain-first. Jika stok tidak cukup, kapal aktif
  dapat menerima bunker bertahap; tanker hanya refill dan kembali bila masih
  dapat memompa kapal yang sama sebelum ATD.
- Perjalanan menuju Pertamina baru boleh dimulai 24 jam setelah loading
  sebelumnya selesai. Tanker tetap boleh bergerak dan bunker selama jeda.
- Look-ahead refill membaca kebutuhan yang tersedia dan 24 jam ke depan, tetapi
  tidak boleh melanggar minimum loading maupun jeda perjalanan refill.
- ATD merupakan hard cutoff. Tanker tidak mendekati kapal bila tidak mungkin
  mulai sebelum ATD, dan pompa yang belum selesai dihentikan tepat saat ATD.
  Hanya volume aktual sebelum ATD yang mengurangi stok tanker; sisanya dicatat
  sebagai shortage.
- ROB event kapal berikutnya tetap memakai ROB rencana. Kekurangan diasumsikan
  dipenuhi sumber lain di luar model dan tidak dicatat pada ledger tanker.

## Menjalankan

Jalankan dari root repository:

```bash
python3 -m fuel_simulation.rob_scenario
```

Hasil ditulis ke:

```text
outputs/rob_scenario/rob_simulation.xlsx
outputs/rob_scenario/summary.md
```

Pemeriksaan regresi skenario dapat dijalankan dengan:

```bash
python3 -m unittest discover -s tests -v
```

Perbandingan tiga skenario dengan flowrate baru dapat dibuat menggunakan:

```bash
python3 -m fuel_simulation.compare_scenarios
```

Hasilnya ditulis ke `outputs/scenario_comparison.xlsx`.

Baseline lama tetap dapat dijalankan sebagai modul untuk kebutuhan audit:

```bash
python3 -m fuel_simulation.core
```

Output baseline ditulis ke `outputs/baseline_simulation.xlsx` dan
`outputs/baseline_simulation.md`.
