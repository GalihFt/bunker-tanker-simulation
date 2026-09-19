#!/usr/bin/env python3
"""Bandingkan tiga kombinasi checkpoint dan penempatan tanker.

Seluruh skenario memakai flowrate loading baru:
- Jakarta: MFO 100 KL/jam, BIO 120 KL/jam, berurutan.
- Surabaya: MFO 40 KL/jam, BIO 70 KL/jam, bersamaan.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import core as sb
from . import rob_scenario as rob


OUTPUT_FILE = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "scenario_comparison.xlsx"
)

# Kolom detail sengaja disamakan dengan workbook simulasi ROB utama. Kolom
# internal DataFrame tidak ikut ditulis agar workbook perbandingan tetap ringkas.
RESULT_COLUMNS = [
    "EVENT_ID", "VESSEL", "VOYAGENO", "VESVOY_GABUNGAN", "TANKER",
    "FULL_SAILING_ROUTE", "CHECKPOINT", "CHECKPOINT_BERIKUT",
    "SAILING_ROUTE_CHECKPOINT_TO_CHECKPOINT", "KEBIJAKAN_BBM", "ATB", "ETD",
    "ATD", "DEADLINE", "MULAI_BUNKER", "SELESAI_BUNKER",
    "KEBUTUHAN_DASAR_MFO_KL", "KAPASITAS_TANGKI_MFO_KL", "ROB_TIBA_MFO_KL",
    "TARGET_ROB_MFO_KL", "VOLUME_DIMINTA_MFO_KL", "KEBUTUHAN_DASAR_BIO_KL",
    "KAPASITAS_TANGKI_BIO_KL", "ROB_TIBA_BIO_KL", "TARGET_ROB_BIO_KL",
    "VOLUME_DIMINTA_BIO_KL", "ROB_AWAL_MFO_TANKER_KL",
    "ROB_AKHIR_MFO_TANKER_KL", "ROB_AWAL_BIO_TANKER_KL",
    "ROB_AKHIR_BIO_TANKER_KL", "TERISI_SBLM_ATD_MFO_KL",
    "TERISI_SBLM_ATD_BIO_KL", "SHORTAGE_MFO_KL", "SHORTAGE_BIO_KL",
    "ROB_PADA_ATD_MFO_KL", "ROB_PADA_ATD_BIO_KL", "SHORTAGE_TARGET_MFO_KL",
    "SHORTAGE_TARGET_BIO_KL", "DEFISIT_RUTE_PADA_ATD_MFO_KL",
    "DEFISIT_RUTE_PADA_ATD_BIO_KL", "STATUS_KAPASITAS", "WAKTU_TUNGGU_JAM",
    "KETERLAMBATAN_JAM", "STATUS", "ALASAN_FLAG",
]

ROB_COLUMNS = [
    "EVENT_ID", "VESSEL", "VOYAGENO", "VESVOY_GABUNGAN", "TANKER",
    "FULL_SAILING_ROUTE", "CHECKPOINT", "CHECKPOINT_BERIKUT",
    "SAILING_ROUTE_CHECKPOINT_TO_CHECKPOINT", "STEP", "ATB", "KEBIJAKAN_BBM",
    "SUMBER_ROB_AWAL", "SERVICE_REQUIRED", "KEBUTUHAN_DASAR_MFO_KL",
    "KAPASITAS_TANGKI_MFO_KL", "ROB_TIBA_MFO_KL", "TARGET_ROB_MFO_KL",
    "VOLUME_DIMINTA_MFO_KL", "ROB_BERANGKAT_RENCANA_MFO_KL",
    "PEMAKAIAN_RUTE_MFO_KL", "DEFISIT_RUTE_RENCANA_MFO_KL",
    "ROB_TIBA_EVENT_BERIKUTNYA_MFO_KL", "KEBUTUHAN_DASAR_BIO_KL",
    "KAPASITAS_TANGKI_BIO_KL", "ROB_TIBA_BIO_KL", "TARGET_ROB_BIO_KL",
    "VOLUME_DIMINTA_BIO_KL", "ROB_BERANGKAT_RENCANA_BIO_KL",
    "PEMAKAIAN_RUTE_BIO_KL", "DEFISIT_RUTE_RENCANA_BIO_KL",
    "ROB_TIBA_EVENT_BERIKUTNYA_BIO_KL", "STATUS_KAPASITAS", "DATA_FLAG",
]

LEDGER_COLUMNS = [
    "OPERATION_ID", "TANKER", "PORT", "JENIS_OPERASI", "EVENT_ID", "VESSEL",
    "MULAI", "SELESAI", "DURASI_JAM", "ROB_AWAL_MFO_TANKER_KL",
    "ROB_AWAL_BIO_TANKER_KL", "MASUK_MFO_KL", "MASUK_BIO_KL",
    "KELUAR_MFO_KL", "KELUAR_BIO_KL", "ROB_AKHIR_MFO_TANKER_KL",
    "ROB_AKHIR_BIO_TANKER_KL", "CATATAN",
]

SCENARIOS = (
    {
        "code": "S1",
        "name": "Prioritas Surabaya - tanker tidak diubah",
        "priority": "IDSUB",
        "jakarta_tanker": "SIGMA",
        "surabaya_tanker": "ZETA",
    },
    {
        "code": "S2",
        "name": "Prioritas Surabaya - tanker ditukar",
        "priority": "IDSUB",
        "jakarta_tanker": "ZETA",
        "surabaya_tanker": "SIGMA",
    },
    {
        "code": "S3",
        "name": "Prioritas Jakarta - tanker tidak diubah",
        "priority": "IDJKT",
        "jakarta_tanker": "SIGMA",
        "surabaya_tanker": "ZETA",
    },
)


def configure_deployment(jakarta_tanker: str, surabaya_tanker: str) -> None:
    sb.TANKERS = {
        "IDJKT": sb.deploy_tanker(jakarta_tanker, "IDJKT"),
        "IDSUB": sb.deploy_tanker(surabaya_tanker, "IDSUB"),
    }


def run_scenarios() -> dict[str, dict[str, pd.DataFrame]]:
    outputs: dict[str, dict[str, pd.DataFrame]] = {}
    for scenario in SCENARIOS:
        configure_deployment(
            scenario["jakarta_tanker"], scenario["surabaya_tanker"]
        )
        outputs[scenario["code"]] = rob.run(scenario["priority"])
    return outputs


def comparison_table(
    outputs: dict[str, dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for scenario in SCENARIOS:
        summary = outputs[scenario["code"]]["summary"].copy()
        summary.insert(0, "SKENARIO", scenario["name"])
        summary.insert(1, "PRIORITAS_CHECKPOINT", scenario["priority"])
        summary.insert(2, "TANKER_JAKARTA", scenario["jakarta_tanker"])
        summary.insert(3, "TANKER_SURABAYA", scenario["surabaya_tanker"])
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def parameter_table() -> pd.DataFrame:
    rows = [
        ("FLOWRATE LOADING JAKARTA MFO", "100 KL/jam"),
        ("FLOWRATE LOADING JAKARTA BIO", "120 KL/jam"),
        ("POLA LOADING JAKARTA", "MFO dan BIO berurutan"),
        ("FLOWRATE LOADING SURABAYA MFO", "40 KL/jam"),
        ("FLOWRATE LOADING SURABAYA BIO", "70 KL/jam"),
        ("POLA LOADING SURABAYA", "MFO dan BIO bersamaan"),
        ("BUNKERING SIGMA MFO/BIO", "110/90 KL/jam"),
        ("BUNKERING ZETA MFO/BIO", "110/50 KL/jam"),
        ("KAPASITAS SIGMA MFO/BIO", "1000/500 KL"),
        ("KAPASITAS ZETA MFO/BIO", "750/250 KL"),
        ("MINIMUM LOADING", "MFO >=250 KL atau BIO >=100 KL pada semua kondisi"),
        ("KEBIJAKAN STOK", "Drain-first; kapal aktif boleh menerima bunker bertahap"),
        ("JEDA REFILL", "Mulai perjalanan ke Pertamina >=24 jam setelah loading sebelumnya selesai"),
        ("LOOK-AHEAD REFILL", "Kebutuhan kapal tersedia dan 24 jam ke depan"),
        ("ATD HARD CUTOFF", "Bunker berhenti saat ATD; kapal yang sudah berangkat tidak dilayani lagi"),
        ("ROB EVENT BERIKUTNYA", "Tetap memakai ROB rencana; pasokan selain tanker berada di luar model"),
        ("PRIORITAS SURABAYA", "Pilih IDSUB bila ada dalam vesvoy; IDJKT fallback"),
        ("PRIORITAS JAKARTA", "Pilih IDJKT bila ada dalam vesvoy; IDSUB fallback"),
        ("ANTREAN", "EDD berdasarkan ETD, fallback deadline, dengan look-ahead"),
        ("OUTPUT", str(OUTPUT_FILE)),
    ]
    return pd.DataFrame(rows, columns=["PARAMETER", "NILAI/ATURAN"])


def write_comparison(outputs: dict[str, dict[str, pd.DataFrame]]) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    comparison = comparison_table(outputs)
    with pd.ExcelWriter(
        OUTPUT_FILE, engine="xlsxwriter", datetime_format="dd/mm/yyyy hh:mm"
    ) as writer:
        rob._write_frame(writer, "PERBANDINGAN", comparison)
        rob._write_frame(writer, "PARAMETER", parameter_table())
        for scenario in SCENARIOS:
            code = scenario["code"]
            out = outputs[code]
            results = out["analysis_results"].reindex(columns=RESULT_COLUMNS)

            rob_events = out["rob_events"].sort_values(["ATB", "EVENT_ID"]).copy()
            rob_events.insert(
                0,
                "PERIODE",
                rob_events["ATB"].map(
                    lambda value: (
                        "WARM-UP MEI"
                        if value < rob.ANALYSIS_START
                        else "ANALISIS JUNI-JULI"
                    )
                ),
            )
            rob_events = rob_events.reindex(columns=["PERIODE", *ROB_COLUMNS])

            tanker = out["ledger"].copy()
            tanker["ROB_AWAL_MFO_TANKER_KL"] = tanker["STOK_AWAL_MFO_KL"]
            tanker["ROB_AWAL_BIO_TANKER_KL"] = tanker["STOK_AWAL_BIO_KL"]
            tanker["ROB_AKHIR_MFO_TANKER_KL"] = tanker["STOK_AKHIR_MFO_KL"]
            tanker["ROB_AKHIR_BIO_TANKER_KL"] = tanker["STOK_AKHIR_BIO_KL"]
            tanker = tanker.reindex(columns=LEDGER_COLUMNS)

            rob._write_frame(writer, f"{code}_HASIL", results)
            rob._write_frame(writer, f"{code}_ROB", rob_events)
            rob._write_frame(writer, f"{code}_TANKER", tanker)
            rob._write_frame(writer, f"{code}_TUNGGU", out["wait_breakdown"])


def main() -> None:
    outputs = run_scenarios()
    write_comparison(outputs)
    comparison = comparison_table(outputs)
    columns = [
        "SKENARIO", "TANKER", "TOTAL_EVENT", "FEASIBLE", "NOT_FEASIBLE",
        "SHORTAGE_MFO_KL", "SHORTAGE_BIO_KL", "MAX_TERLAMBATAN_JAM",
    ]
    print(comparison[columns].to_string(index=False))
    print(f"\nWorkbook: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
