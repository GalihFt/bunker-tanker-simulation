#!/usr/bin/env python3
"""Skenario prioritas checkpoint Surabaya dan pertukaran tanker.

IDSUB dipilih sebagai checkpoint utama vesvoy bila tersedia; IDJKT menjadi
fallback. SIGMA beroperasi di Jakarta dan ZETA di Surabaya. Varian ini sengaja
terpisah dari output baseline. Mei 2026 dipakai sebagai warm-up, sedangkan KPI
utama hanya menghitung Juni-Juli 2026.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import core as sb
from . import queue as common


WARMUP_START = pd.Timestamp("2026-05-01 00:00:00")
ANALYSIS_START = pd.Timestamp("2026-06-01 00:00:00")
ANALYSIS_END = pd.Timestamp("2026-07-31 23:59:59")

SPECIAL_CHECKPOINTS_BY_VESSEL = {
    "TBI": {"IDMAK"},
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rob_scenario"
OUTPUT_FILE = OUTPUT_DIR / "rob_simulation.xlsx"
SUMMARY_FILE = OUTPUT_DIR / "summary.md"

FUELS = ("MFO", "BIO")
EPSILON = 1e-9


def _append_flag(current: Any, flag: str) -> str:
    parts = [] if pd.isna(current) or str(current).strip() in ("", "OK") else str(current).split("; ")
    if flag not in parts:
        parts.append(flag)
    return "; ".join(parts) if parts else "OK"


def apply_rob_policy(events: pd.DataFrame) -> pd.DataFrame:
    """Hitung target, permintaan pengisian, dan ROB berantai per kapal.

    Event direct sudah digabung menjadi satu baris oleh merge_direct_voyages,
    sama seperti penyajian algoritma baseline. Perhitungan ini mengasumsikan
    volume yang diminta akhirnya dapat dikirim. Simulasi antrean kemudian
    menghitung bagian yang sempat terkirim pada ATD serta keterlambatannya.
    """
    planned = events.copy()
    planned["STEP"] = (
        planned["SAILING_ROUTE_CHECKPOINT_TO_CHECKPOINT"]
        .fillna("")
        .astype(str)
        .str.count("-")
    )
    for fuel in FUELS:
        planned[f"KEBUTUHAN_DASAR_{fuel}_KL"] = pd.to_numeric(
            planned[f"KEBUTUHAN_{fuel}_KL"], errors="coerce"
        )
        planned[f"ROB_TIBA_{fuel}_KL"] = np.nan
        planned[f"TARGET_ROB_{fuel}_KL"] = np.nan
        planned[f"VOLUME_DIMINTA_{fuel}_KL"] = np.nan
        planned[f"ROB_BERANGKAT_RENCANA_{fuel}_KL"] = np.nan
        planned[f"PEMAKAIAN_RUTE_{fuel}_KL"] = planned[f"KEBUTUHAN_DASAR_{fuel}_KL"]
        planned[f"ROB_TIBA_EVENT_BERIKUTNYA_{fuel}_KL"] = np.nan
        planned[f"DEFISIT_RUTE_RENCANA_{fuel}_KL"] = np.nan

    planned["KEBIJAKAN_BBM"] = ""
    planned["STATUS_KAPASITAS"] = ""
    planned["SUMBER_ROB_AWAL"] = ""
    planned["SERVICE_REQUIRED"] = False

    # Validasi kapasitas dilakukan terhadap event tunggal atau event direct
    # yang kebutuhan dasarnya sudah digabung.
    for index in planned.index:
        if planned.at[index, "DATA_FLAG"] != "OK":
            continue
        capacity_status: list[str] = []
        for fuel in FUELS:
            base = float(planned.at[index, f"KEBUTUHAN_DASAR_{fuel}_KL"])
            capacity = float(planned.at[index, f"KAPASITAS_TANGKI_{fuel}_KL"])
            if not np.isfinite(base) or not np.isfinite(capacity):
                planned.at[index, "DATA_FLAG"] = _append_flag(
                    planned.at[index, "DATA_FLAG"], f"DATA {fuel} TIDAK VALID"
                )
            elif capacity <= EPSILON and base > EPSILON:
                planned.at[index, "DATA_FLAG"] = _append_flag(
                    planned.at[index, "DATA_FLAG"], f"KAPASITAS {fuel} NOL"
                )
            elif base > capacity + EPSILON:
                capacity_status.append(f"{fuel}_RUTE_MELEBIHI_KAPASITAS")
        planned.at[index, "STATUS_KAPASITAS"] = "; ".join(capacity_status) or "CUKUP"

    for vessel, vessel_events in planned.groupby("VESSEL", sort=False):
        indices = list(
            vessel_events.sort_values(["ATB", "EVENT_ID"], kind="stable").index
        )
        has_warmup = bool((planned.loc[indices, "ATB"] < ANALYSIS_START).any())
        rob = {fuel: np.nan for fuel in FUELS}
        initialized = False

        for index in indices:
            if planned.at[index, "DATA_FLAG"] != "OK":
                continue

            is_direct_pair = bool(planned.at[index, "DIRECT_VOYAGE_PAIR"])
            if is_direct_pair:
                policy = "DIRECT_MERGE_150"
                factor = 1.50
            else:
                policy = "DIRECT_SINGLE_150" if int(planned.at[index, "STEP"]) == 2 else "NON_DIRECT_150"
                factor = 1.50
            cycle_need = {
                fuel: float(planned.at[index, f"KEBUTUHAN_DASAR_{fuel}_KL"])
                for fuel in FUELS
            }

            planned.at[index, "KEBIJAKAN_BBM"] = policy

            # Kapal tanpa warm-up dimulai pada kondisi steady-state dengan
            # asumsi siklus sebelumnya sama dengan siklus pertama yang terlihat.
            if not initialized:
                for fuel in FUELS:
                    capacity = float(planned.at[index, f"KAPASITAS_TANGKI_{fuel}_KL"])
                    initial_target = min(cycle_need[fuel] * factor, capacity)
                    previous_need = cycle_need[fuel]
                    rob[fuel] = (
                        0.0
                        if has_warmup
                        else max(0.0, initial_target - previous_need)
                    )
                planned.at[index, "SUMBER_ROB_AWAL"] = (
                    "WARMUP_MEI_DARI_NOL"
                    if has_warmup
                    else "ESTIMASI_RUTE_SEBELUMNYA_SAMA"
                )
                initialized = True

            for fuel in FUELS:
                capacity = float(planned.at[index, f"KAPASITAS_TANGKI_{fuel}_KL"])
                target = min(cycle_need[fuel] * factor, capacity)
                arrival = float(rob[fuel])
                requested = max(0.0, target - arrival)
                departure = arrival + requested
                planned.at[index, f"ROB_TIBA_{fuel}_KL"] = arrival
                planned.at[index, f"TARGET_ROB_{fuel}_KL"] = target
                planned.at[index, f"VOLUME_DIMINTA_{fuel}_KL"] = requested
                planned.at[index, f"ROB_BERANGKAT_RENCANA_{fuel}_KL"] = departure
                planned.at[index, f"DEFISIT_RUTE_RENCANA_{fuel}_KL"] = max(
                    0.0,
                    float(planned.at[index, f"KEBUTUHAN_DASAR_{fuel}_KL"]) - departure,
                )
                rob[fuel] = max(
                    0.0,
                    departure - float(planned.at[index, f"KEBUTUHAN_DASAR_{fuel}_KL"]),
                )
                planned.at[index, f"ROB_TIBA_EVENT_BERIKUTNYA_{fuel}_KL"] = rob[fuel]

            planned.at[index, "SERVICE_REQUIRED"] = any(
                float(planned.at[index, f"VOLUME_DIMINTA_{fuel}_KL"]) > EPSILON
                for fuel in FUELS
            )
    # Kolom kebutuhan lama sengaja diisi volume permintaan karena kolom inilah
    # yang dikonsumsi state machine tanker dan menentukan durasi pompa.
    for fuel in FUELS:
        planned[f"KEBUTUHAN_{fuel}_KL"] = planned[f"VOLUME_DIMINTA_{fuel}_KL"]

    return planned


def validate_rob(events: pd.DataFrame) -> None:
    valid = events[events["DATA_FLAG"] == "OK"]
    for fuel in FUELS:
        capacity = valid[f"KAPASITAS_TANGKI_{fuel}_KL"]
        target = valid[f"TARGET_ROB_{fuel}_KL"]
        arrival = valid[f"ROB_TIBA_{fuel}_KL"]
        requested = valid[f"VOLUME_DIMINTA_{fuel}_KL"]
        departure = valid[f"ROB_BERANGKAT_RENCANA_{fuel}_KL"]
        next_rob = valid[f"ROB_TIBA_EVENT_BERIKUTNYA_{fuel}_KL"]
        if (target < -EPSILON).any() or (target > capacity + EPSILON).any():
            raise AssertionError(f"Target ROB {fuel} di luar kapasitas")
        if (arrival < -EPSILON).any() or (requested < -EPSILON).any() or (next_rob < -EPSILON).any():
            raise AssertionError(f"ROB/permintaan {fuel} menjadi negatif")
        if (requested > capacity + EPSILON).any():
            raise AssertionError(f"Permintaan {fuel} melebihi kapasitas kapal")
        balance = (arrival + requested - departure).abs().max()
        if balance > 1e-6:
            raise AssertionError(f"Neraca ROB keberangkatan {fuel} tidak seimbang")


def prepare_events(
    main_checkpoint_priority: str = "IDSUB",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    schedule = sb.load_schedule()
    consumption = sb.load_consumption(set(schedule["VESSELID"].dropna()))
    events, issues = sb.build_events(
        schedule,
        consumption,
        simulation_start=WARMUP_START,
        special_checkpoints_by_vessel=SPECIAL_CHECKPOINTS_BY_VESSEL,
        main_checkpoint_priority=main_checkpoint_priority,
    )
    events = sb.attach_tank_capacity(events)
    # Bentuk dan urutan sama dengan algoritma lampiran: dua direct voyage yang
    # lolos aturan 150% dilebur menjadi satu row sebelum kebijakan ROB dihitung.
    events = sb.merge_direct_voyages(events, target_factor=1.50)
    issues = issues[issues["EVENT_ID"].isin(events["EVENT_ID"])].copy()
    events, issues = sb.renumber_events_chronologically(events, issues)
    events = apply_rob_policy(events)
    events = events[events["ATB"] <= ANALYSIS_END].copy()
    issues = issues[issues["EVENT_ID"].isin(events["EVENT_ID"])].copy()
    validate_rob(events)
    return events, issues, consumption


def _event_selection(events: pd.DataFrame, tanker: str) -> pd.DataFrame:
    relevant = events[events["TANKER"] == tanker].copy()
    return relevant[
        relevant["SERVICE_REQUIRED"] | (relevant["DATA_FLAG"] != "OK")
    ].copy()


def run(main_checkpoint_priority: str = "IDSUB") -> dict[str, pd.DataFrame]:
    events, issues, consumption = prepare_events(main_checkpoint_priority)
    result_frames: list[pd.DataFrame] = []
    ledger_frames: list[pd.DataFrame] = []

    for spec in sb.TANKERS.values():
        tanker_events = _event_selection(events, spec.name)
        result, ledger = common.simulate_tanker_priority(
            spec,
            tanker_events,
            mode="EDD",
            lookahead_wait=True,
            simulation_start=WARMUP_START,
        )
        result_frames.append(result)
        ledger_frames.append(ledger)

    branch_events = _event_selection(events, "CABANG")
    if not branch_events.empty:
        result_frames.append(sb.build_branch_results(branch_events))

    all_results = pd.concat(result_frames, ignore_index=True)
    ledger = pd.concat(ledger_frames, ignore_index=True)
    sb.validate_simulation(all_results, ledger)

    # Alias yang lebih mudah dibaca pada HASIL_PENGISIAN. Nilainya adalah stok
    # tanker tepat sebelum pompa pertama dan setelah seluruh pelayanan event.
    for fuel in FUELS:
        all_results[f"ROB_AWAL_{fuel}_TANKER_KL"] = all_results[
            f"STOK_AWAL_{fuel}_TANKER_KL"
        ]
        all_results[f"ROB_AKHIR_{fuel}_TANKER_KL"] = all_results[
            f"STOK_AKHIR_{fuel}_TANKER_KL"
        ]

    # Shortage pelayanan menunjukkan target ROB yang belum tercapai saat ATD.
    # Defisit rute lebih kritis: stok pada ATD belum cukup untuk kebutuhan dasar.
    for fuel in FUELS:
        all_results[f"ROB_PADA_ATD_{fuel}_KL"] = (
            all_results[f"ROB_TIBA_{fuel}_KL"].fillna(0.0)
            + all_results[f"TERISI_SBLM_ATD_{fuel}_KL"].fillna(0.0)
        )
        all_results[f"SHORTAGE_TARGET_{fuel}_KL"] = (
            all_results[f"TARGET_ROB_{fuel}_KL"]
            - all_results[f"ROB_PADA_ATD_{fuel}_KL"]
        ).clip(lower=0.0)
        all_results[f"DEFISIT_RUTE_PADA_ATD_{fuel}_KL"] = (
            all_results[f"KEBUTUHAN_DASAR_{fuel}_KL"]
            - all_results[f"ROB_PADA_ATD_{fuel}_KL"]
        ).clip(lower=0.0)

    analysis_results = all_results[
        all_results["ATB"].between(ANALYSIS_START, ANALYSIS_END)
    ].copy()
    warmup_results = all_results[all_results["ATB"] < ANALYSIS_START].copy()
    analysis_ledger = ledger[
        (ledger["SELESAI"] > ANALYSIS_START) & (ledger["MULAI"] <= ANALYSIS_END)
    ].copy()
    analysis_rob = events[events["ATB"].between(ANALYSIS_START, ANALYSIS_END)].copy()

    summary = sb.build_summary(analysis_results, analysis_ledger)
    record_counts = []
    for tanker in ["SIGMA", "ZETA", "CABANG", "TOTAL"]:
        rows = (
            analysis_rob[analysis_rob["TANKER"].isin(["SIGMA", "ZETA"])]
            if tanker == "TOTAL"
            else analysis_rob[analysis_rob["TANKER"] == tanker]
        )
        record_counts.append(
            {
                "TANKER": tanker,
                "RECORD_ROB": len(rows),
                "EVENT_TANPA_PENGISIAN": int(
                    ((rows["DATA_FLAG"] == "OK") & ~rows["SERVICE_REQUIRED"]).sum()
                ),
            }
        )
    summary = summary.merge(pd.DataFrame(record_counts), on="TANKER", how="left")
    wait_breakdown = sb.build_wait_breakdown(analysis_results, ledger)

    return {
        "summary": summary,
        "analysis_results": analysis_results,
        "warmup_results": warmup_results,
        "rob_events": events,
        "ledger": ledger,
        "wait_breakdown": wait_breakdown,
        "issues": issues,
        "consumption": consumption,
    }


def parameter_table() -> pd.DataFrame:
    rows = [
        ("MODEL", "Checkpoint prioritas Surabaya + EDD look-ahead + ROB kapal berantai"),
        ("WARM-UP", "1 Mei 2026; ROB awal kapal = 0 KL"),
        ("PERIODE KPI", "1 Juni-31 Juli 2026"),
        ("RESET 1 JUNI", "Tidak; ROB kapal, stok, lokasi, waktu, dan antrean tanker diteruskan"),
        ("ROB REALISASI", "Tidak digunakan"),
        ("KAPAL TANPA WARM-UP", "Rute sebelumnya diasumsikan sama dengan rute pertama"),
        ("TARGET ROB", "Semua kategori, termasuk direct pair, memakai 150%; dibatasi kapasitas"),
        ("VOLUME TANKER", "max(0, target ROB - ROB tiba)"),
        ("PEMAKAIAN", "Kebutuhan dasar rute; tidak termasuk safety stock"),
        ("DIRECT PAIR", "Dua voyage dilebur menjadi satu row VESVOY_GABUNGAN sebelum ROB dihitung"),
        ("DIRECT LOOKAHEAD", "Event setelah Juli boleh menjadi pasangan; hasil gabungan memakai ATB event pertama"),
        ("TBI", "Setiap kedatangan IDMAK menjadi checkpoint khusus CABANG"),
        ("CHECKPOINT UTAMA", "Per vesvoy: IDSUB bila full sailing route memuat IDSUB; selain itu IDJKT"),
        ("TRANSISI CHECKPOINT", "Dapat berupa SUB-SUB, SUB-JKT, JKT-JKT, atau JKT-SUB mengikuti vesvoy berikutnya"),
        ("PENEMPATAN TANKER", "SIGMA di IDJKT; ZETA di IDSUB"),
        ("KARAKTERISTIK TANKER", "Kapasitas, discharge rate, dan minimum stock mengikuti aset tanker"),
        ("KARAKTERISTIK DAERAH", "Load rate, loading simultan/berurutan, dan waktu mobilisasi mengikuti daerah"),
        ("MINIMUM LOADING", "Loading hanya jika ullage MFO >=250 KL atau BIO >=100 KL; berlaku pada semua kondisi"),
        ("DRAIN-FIRST", "Stok tanker dipakai ke kapal sampai produk yang masih dibutuhkan habis; kapal aktif kemudian diselesaikan setelah refill"),
        ("JEDA REFILL", "Perjalanan ke Pertamina baru boleh dimulai 24 jam setelah loading sebelumnya selesai"),
        ("LOOK-AHEAD REFILL", "Kebutuhan kapal tersedia dan 24 jam ke depan diperiksa; minimum loading dan jeda tetap wajib"),
        ("ATD HARD CUTOFF", "Bunker berhenti saat ATD; kapal yang sudah berangkat tidak dilayani lagi"),
        ("ROB EVENT BERIKUTNYA", "Tetap memakai ROB rencana; pasokan selain tanker berada di luar model dan tidak dicatat"),
        ("PRIORITAS", "EDD berdasarkan ETD, fallback deadline, dengan look-ahead wait"),
        ("OUTPUT", str(OUTPUT_FILE)),
    ]
    return pd.DataFrame(rows, columns=["PARAMETER", "NILAI/ATURAN"])


def _write_frame(writer: pd.ExcelWriter, name: str, frame: pd.DataFrame) -> None:
    frame.to_excel(writer, sheet_name=name, index=False)
    sheet = writer.sheets[name]
    workbook = writer.book
    header = workbook.add_format(
        {"bold": True, "align": "center", "valign": "vcenter", "border": 1}
    )
    date_format = workbook.add_format({"num_format": "dd/mm/yyyy hh:mm"})
    number_format = workbook.add_format({"num_format": "0.00"})
    sheet.freeze_panes(1, 0)
    if len(frame.columns):
        sheet.autofilter(0, 0, max(len(frame), 1), len(frame.columns) - 1)
    for column_number, column_name in enumerate(frame.columns):
        sheet.write(0, column_number, column_name, header)
        values = frame[column_name].astype(str).replace({"NaT": "", "nan": ""})
        width = min(max([len(str(column_name))] + values.head(500).str.len().tolist()) + 2, 42)
        if pd.api.types.is_datetime64_any_dtype(frame[column_name]):
            cell_format = date_format
        elif pd.api.types.is_numeric_dtype(frame[column_name]):
            cell_format = number_format
        else:
            cell_format = None
        sheet.set_column(column_number, column_number, max(width, 11), cell_format)


def write_excel(out: dict[str, pd.DataFrame]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_columns = [
        "EVENT_ID", "VESSEL", "VOYAGENO", "VESVOY_GABUNGAN", "TANKER",
        "FULL_SAILING_ROUTE",
        "CHECKPOINT", "CHECKPOINT_BERIKUT", "SAILING_ROUTE_CHECKPOINT_TO_CHECKPOINT",
        "KEBIJAKAN_BBM", "ATB", "ETD", "ATD", "DEADLINE", "MULAI_BUNKER",
        "SELESAI_BUNKER", "KEBUTUHAN_DASAR_MFO_KL", "KAPASITAS_TANGKI_MFO_KL",
        "ROB_TIBA_MFO_KL", "TARGET_ROB_MFO_KL", "VOLUME_DIMINTA_MFO_KL",
        "KEBUTUHAN_DASAR_BIO_KL", "KAPASITAS_TANGKI_BIO_KL", "ROB_TIBA_BIO_KL",
        "TARGET_ROB_BIO_KL", "VOLUME_DIMINTA_BIO_KL",
        "ROB_AWAL_MFO_TANKER_KL", "ROB_AKHIR_MFO_TANKER_KL",
        "ROB_AWAL_BIO_TANKER_KL", "ROB_AKHIR_BIO_TANKER_KL",
        "TERISI_SBLM_ATD_MFO_KL", "TERISI_SBLM_ATD_BIO_KL", "SHORTAGE_MFO_KL",
        "SHORTAGE_BIO_KL", "ROB_PADA_ATD_MFO_KL", "ROB_PADA_ATD_BIO_KL",
        "SHORTAGE_TARGET_MFO_KL", "SHORTAGE_TARGET_BIO_KL",
        "DEFISIT_RUTE_PADA_ATD_MFO_KL", "DEFISIT_RUTE_PADA_ATD_BIO_KL",
        "STATUS_KAPASITAS", "WAKTU_TUNGGU_JAM", "KETERLAMBATAN_JAM",
        "STATUS", "ALASAN_FLAG",
    ]
    rob_columns = [
        "EVENT_ID", "VESSEL", "VOYAGENO", "VESVOY_GABUNGAN", "TANKER",
        "FULL_SAILING_ROUTE", "CHECKPOINT",
        "CHECKPOINT_BERIKUT", "SAILING_ROUTE_CHECKPOINT_TO_CHECKPOINT", "STEP", "ATB",
        "KEBIJAKAN_BBM", "SUMBER_ROB_AWAL", "SERVICE_REQUIRED",
        "KEBUTUHAN_DASAR_MFO_KL", "KAPASITAS_TANGKI_MFO_KL", "ROB_TIBA_MFO_KL",
        "TARGET_ROB_MFO_KL", "VOLUME_DIMINTA_MFO_KL", "ROB_BERANGKAT_RENCANA_MFO_KL",
        "PEMAKAIAN_RUTE_MFO_KL", "DEFISIT_RUTE_RENCANA_MFO_KL",
        "ROB_TIBA_EVENT_BERIKUTNYA_MFO_KL",
        "KEBUTUHAN_DASAR_BIO_KL", "KAPASITAS_TANGKI_BIO_KL", "ROB_TIBA_BIO_KL",
        "TARGET_ROB_BIO_KL", "VOLUME_DIMINTA_BIO_KL", "ROB_BERANGKAT_RENCANA_BIO_KL",
        "PEMAKAIAN_RUTE_BIO_KL", "DEFISIT_RUTE_RENCANA_BIO_KL",
        "ROB_TIBA_EVENT_BERIKUTNYA_BIO_KL",
        "STATUS_KAPASITAS", "DATA_FLAG",
    ]
    ledger_columns = [
        "OPERATION_ID", "TANKER", "PORT", "JENIS_OPERASI", "EVENT_ID", "VESSEL",
        "MULAI", "SELESAI", "DURASI_JAM",
        "ROB_AWAL_MFO_TANKER_KL", "ROB_AWAL_BIO_TANKER_KL",
        "MASUK_MFO_KL", "MASUK_BIO_KL", "KELUAR_MFO_KL", "KELUAR_BIO_KL",
        "ROB_AKHIR_MFO_TANKER_KL", "ROB_AKHIR_BIO_TANKER_KL", "CATATAN",
    ]
    tanker_monitoring = out["ledger"].copy()
    tanker_monitoring["ROB_AWAL_MFO_TANKER_KL"] = tanker_monitoring["STOK_AWAL_MFO_KL"]
    tanker_monitoring["ROB_AWAL_BIO_TANKER_KL"] = tanker_monitoring["STOK_AWAL_BIO_KL"]
    tanker_monitoring["ROB_AKHIR_MFO_TANKER_KL"] = tanker_monitoring["STOK_AKHIR_MFO_KL"]
    tanker_monitoring["ROB_AKHIR_BIO_TANKER_KL"] = tanker_monitoring["STOK_AKHIR_BIO_KL"]
    rob_events = out["rob_events"].sort_values(["ATB", "EVENT_ID"]).copy()
    rob_events.insert(
        0,
        "PERIODE",
        np.where(rob_events["ATB"] < ANALYSIS_START, "WARM-UP MEI", "ANALISIS JUNI-JULI"),
    )

    with pd.ExcelWriter(OUTPUT_FILE, engine="xlsxwriter", datetime_format="dd/mm/yyyy hh:mm") as writer:
        _write_frame(writer, "RINGKASAN", out["summary"])
        _write_frame(
            writer,
            "HASIL_PENGISIAN",
            out["analysis_results"].reindex(columns=result_columns).sort_values("EVENT_ID"),
        )
        _write_frame(
            writer,
            "HASIL_WARMUP",
            out["warmup_results"].reindex(columns=result_columns).sort_values("EVENT_ID"),
        )
        _write_frame(
            writer,
            "MONITORING_KAPAL",
            rob_events.reindex(columns=["PERIODE", *rob_columns]),
        )
        _write_frame(
            writer,
            "MONITORING_TANKER",
            tanker_monitoring.reindex(columns=ledger_columns).sort_values(["MULAI", "TANKER"]),
        )
        _write_frame(writer, "BREAKDOWN_TUNGGU", out["wait_breakdown"])
        _write_frame(writer, "DATA_ISSUE", out["issues"])
        _write_frame(writer, "PARAMETER", parameter_table())


def write_summary(out: dict[str, pd.DataFrame]) -> None:
    summary = out["summary"]
    lines = [
        "# Ringkasan Simulasi ROB",
        "",
        "Simulasi memakai Mei 2026 sebagai warm-up dan Juni-Juli sebagai periode KPI.",
        "ROB realisasi tidak digunakan. Khusus TBI, setiap IDMAK menjadi checkpoint cabang.",
        "",
        "| Tanker | Record ROB | Tanpa pengisian | Event pengisian | Feasible | Not feasible | Shortage MFO | Shortage BIO |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.TANKER} | {int(row.RECORD_ROB)} | {int(row.EVENT_TANPA_PENGISIAN)} | "
            f"{int(row.TOTAL_EVENT)} | {int(row.FEASIBLE)} | {int(row.NOT_FEASIBLE)} | "
            f"{row.SHORTAGE_MFO_KL:.2f} KL | {row.SHORTAGE_BIO_KL:.2f} KL |"
        )
    lines.extend(["", f"Workbook: `{OUTPUT_FILE.name}`", ""])
    SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    out = run()
    write_excel(out)
    write_summary(out)
    print(out["summary"].to_string(index=False))
    print(f"\nWorkbook: {OUTPUT_FILE}")
    print(f"Ringkasan: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
