#!/usr/bin/env python3
"""Modul bersama untuk simulasi antrean bunker dengan aturan prioritas berbeda.

Aturan prioritas yang didukung:
  - FIFO : first-in-first-out berdasarkan ATB (sama seperti simulasi utama).
  - EDD  : earliest due date menggunakan ETD (kapal ber-ETD paling dekat lebih dulu),
           dengan opsi look-ahead wait dalam rentang waktu mobilisasi ke SANDAR.

Keterlambatan (keterlambatan/shortage) TETAP dihitung terhadap DEADLINE
(ATD, fallback ETD) sesuai model utama. ETD dipakai hanya untuk menentukan
prioritas antrean; bila ETD kosong, fallback ke DEADLINE.

Fungsi pembacaan data dan validasi dipakai ulang dari `fuel_simulation.core`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import core as sb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DOC_DIR = OUTPUT_DIR

PRIORITY_DESCRIPTIONS = {
    "FIFO": "First-in-first-out berdasarkan urutan ATB (kapal yang berthing lebih dulu dilayani lebih dulu).",
    "EDD": "Earliest Due Date berdasarkan ETD: di antara kapal yang sudah berthing, kapal dengan ETD paling dekat dilayani lebih dulu. Bila ETD kosong, fallback ke DEADLINE.",
}

LOOKAHEAD_DESCRIPTION = (
    "Look-ahead wait: sebelum tanker berangkat LABUH→SANDAR atau PERTAMINA→SANDAR, "
    "tanker melihat ke depan sejauh rentang waktu mobilisasi (lama perpindahan ke SANDAR). "
    "Bila ada kapal yang akan berthing dalam rentang itu dengan ETD lebih dekat daripada "
    "kapal terbaik yang sudah berthing, tanker menunggu kapal itu berthing dulu."
)


def _priority_time(event: dict[str, Any]) -> pd.Timestamp:
    """Waktu yang dipakai sebagai dasar prioritas: ETD, fallback DEADLINE."""
    etd = event.get("ETD")
    if pd.isna(etd):
        return event["DEADLINE"]
    return etd


def simulate_tanker_priority(
    spec: sb.TankerSpec,
    events: pd.DataFrame,
    mode: str,
    lookahead_wait: bool = False,
    simulation_start: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulasi satu tanker dengan prioritas antrean tertentu (mode).

    Struktur ledger dan hasil identik dengan `sb.simulate_tanker`; yang berbeda
    hanya pemilihan kapal berikutnya di antara kapal yang sudah berthing.

    Bila `lookahead_wait` aktif, sebelum komit ke kapal terbaik yang sudah
    berthing, tanker mengecek kapal yang akan berthing dalam rentang waktu
    mobilisasi (LABUH→SANDAR atau PERTAMINA→SANDAR) dan menunggu kapal yang
    lebih urgen (ETD lebih dekat) berthing dulu.
    """
    valid = events[events["DATA_FLAG"] == "OK"].copy()
    valid = valid.sort_values(["ATB", "EVENT_ID"]).reset_index(drop=True)
    invalid = events[events["DATA_FLAG"] != "OK"].copy()

    now = sb.SIMULATION_START if simulation_start is None else pd.Timestamp(simulation_start)
    next_refill_trip_allowed = now
    stock_mfo = spec.capacity_mfo
    stock_bio = spec.capacity_bio
    location = sb.LOC_LABUH
    unscheduled = valid.to_dict("records")
    ledger: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    operation_no = 0

    def add_operation(
        operation_type: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        before_mfo: float,
        before_bio: float,
        after_mfo: float,
        after_bio: float,
        event: dict[str, Any] | None = None,
        in_mfo: float = 0.0,
        in_bio: float = 0.0,
        out_mfo: float = 0.0,
        out_bio: float = 0.0,
        note: str = "",
    ) -> None:
        nonlocal operation_no
        operation_no += 1
        ledger.append(
            {
                "OPERATION_ID": f"{spec.name}-OP-{operation_no:04d}",
                "TANKER": spec.name,
                "PORT": spec.port,
                "JENIS_OPERASI": operation_type,
                "EVENT_ID": event["EVENT_ID"] if event else "",
                "VESSEL": event["VESSEL"] if event else "",
                "MULAI": start,
                "SELESAI": end,
                "DURASI_JAM": sb.hours_between(end, start),
                "STOK_AWAL_MFO_KL": before_mfo,
                "STOK_AWAL_BIO_KL": before_bio,
                "MASUK_MFO_KL": in_mfo,
                "MASUK_BIO_KL": in_bio,
                "KELUAR_MFO_KL": out_mfo,
                "KELUAR_BIO_KL": out_bio,
                "STOK_AKHIR_MFO_KL": after_mfo,
                "STOK_AKHIR_BIO_KL": after_bio,
                "CATATAN": note,
            }
        )

    def do_move(destination: str, event: dict[str, Any] | None = None, note: str = "") -> None:
        nonlocal now, location
        duration = spec.move_time(location, destination)
        if destination == sb.LOC_SANDAR and location == sb.LOC_SANDAR:
            op_type = "PINDAH ANTAR KAPAL"
        elif destination == sb.LOC_SANDAR:
            op_type = "MANUVER & PERSIAPAN KE KAPAL"
        elif destination == sb.LOC_LABUH:
            op_type = "KEMBALI KE BASE"
        else:
            op_type = "PERJALANAN KE PERTAMINA"
        if note:
            note = f"{note} ({location} → {destination})"
        else:
            note = f"{location} → {destination}"
        if duration > 1e-9:
            start = now
            end = start + pd.Timedelta(hours=duration)
            add_operation(
                op_type, start, end, stock_mfo, stock_bio, stock_mfo, stock_bio, event, note=note
            )
            now = end
        location = destination

    def do_refill_pump(note: str, event: dict[str, Any] | None = None) -> None:
        nonlocal now, stock_mfo, stock_bio, next_refill_trip_allowed
        if not sb.refill_meets_minimum(spec, stock_mfo, stock_bio):
            in_mfo, in_bio = sb.refill_amounts(spec, stock_mfo, stock_bio)
            raise RuntimeError(
                f"Loading {spec.name} di bawah minimum: "
                f"MFO {in_mfo:.2f} KL, BIO {in_bio:.2f} KL"
            )
        duration = sb.refill_duration(spec, stock_mfo, stock_bio)
        if duration <= 1e-9:
            return
        start = now
        end = start + pd.Timedelta(hours=duration)
        before_mfo, before_bio = stock_mfo, stock_bio
        in_mfo = spec.capacity_mfo - stock_mfo
        in_bio = spec.capacity_bio - stock_bio
        stock_mfo, stock_bio = spec.capacity_mfo, spec.capacity_bio
        add_operation(
            "REFILL PERTAMINA", start, end, before_mfo, before_bio,
            stock_mfo, stock_bio, event, in_mfo=in_mfo, in_bio=in_bio, note=note,
        )
        now = end
        next_refill_trip_allowed = end + pd.Timedelta(
            hours=sb.REFILL_TRIP_COOLDOWN_HOURS
        )

    def do_refill(note: str, event: dict[str, Any] | None = None) -> bool:
        nonlocal now, location
        if not sb.refill_meets_minimum(spec, stock_mfo, stock_bio):
            return False
        if now < next_refill_trip_allowed:
            start = now
            end = next_refill_trip_allowed
            add_operation(
                "MENUNGGU IZIN PERJALANAN REFILL",
                start,
                end,
                stock_mfo,
                stock_bio,
                stock_mfo,
                stock_bio,
                event,
                note="Menunggu 24 jam sejak loading sebelumnya selesai",
            )
            now = end
        do_move(sb.LOC_PERTAMINA, event, note=f"{note} (menuju Pertamina)")
        do_refill_pump(note, event)
        return True

    def priority_key(event: dict[str, Any]) -> tuple[Any, ...]:
        if mode == "EDD":
            return (_priority_time(event), event["ATB"], event["EVENT_ID"])
        return (event["ATB"], event["EVENT_ID"])

    while unscheduled:
        available = [event for event in unscheduled if event["ATB"] <= now]
        if not available:
            next_release = min(event["ATB"] for event in unscheduled)
            next_event = min(unscheduled, key=lambda item: (item["ATB"], item["EVENT_ID"]))
            forecast_limit = max(
                next_release,
                now + pd.Timedelta(hours=sb.REFILL_LOOKAHEAD_HOURS),
            )
            forecast = [
                event for event in unscheduled if event["ATB"] <= forecast_limit
            ]
            forecast_mfo = sum(float(event["KEBUTUHAN_MFO_KL"]) for event in forecast)
            forecast_bio = sum(float(event["KEBUTUHAN_BIO_KL"]) for event in forecast)
            need_refill = forecast_mfo > stock_mfo + 1e-9 or forecast_bio > stock_bio + 1e-9
            if need_refill and sb.refill_meets_minimum(spec, stock_mfo, stock_bio):
                earliest_departure = max(now, next_refill_trip_allowed)
                refill_complete = (
                    earliest_departure
                    + pd.Timedelta(hours=spec.move_time(location, sb.LOC_PERTAMINA))
                    + pd.Timedelta(hours=sb.refill_duration(spec, stock_mfo, stock_bio))
                )
                if refill_complete <= next_release:
                    do_refill("REFILL LOOK-AHEAD 24 JAM SAAT IDLE", next_event)
            now = max(now, next_release)
            continue

        event = min(available, key=priority_key)

        # LOOK-AHEAD WAIT (opsional): bila tanker akan berangkat LABUH→SANDAR
        # atau PERTAMINA→SANDAR, rentang mobilisasi = lama pindah ke SANDAR.
        # Kapal yang berthing dalam rentang itu dicek; bila ada yang ETD-nya
        # lebih dekat daripada kapal terbaik yang sudah berthing, tunggu kapal
        # itu berthing dulu sebelum memilih.
        if lookahead_wait and location in (sb.LOC_LABUH, sb.LOC_PERTAMINA):
            horizon = spec.move_time(location, sb.LOC_SANDAR)
            imminent = [
                candidate
                for candidate in unscheduled
                if now < candidate["ATB"] <= now + pd.Timedelta(hours=horizon)
            ]
            if imminent:
                best_imminent = min(imminent, key=priority_key)
                if priority_key(best_imminent) < priority_key(event):
                    now = best_imminent["ATB"]
                    continue

        unscheduled.remove(event)
        event["PRIORITAS_ETD"] = _priority_time(event)
        remaining_mfo = float(event["KEBUTUHAN_MFO_KL"])
        remaining_bio = float(event["KEBUTUHAN_BIO_KL"])
        delivery_segments: list[dict[str, Any]] = []
        first_delivery_start: pd.Timestamp | None = None
        stock_before_bunker_mfo: float | None = None
        stock_before_bunker_bio: float | None = None

        need_bunker = remaining_mfo > 1e-9 or remaining_bio > 1e-9
        deadline = event["DEADLINE"]
        can_service = not need_bunker

        if need_bunker:
            # Drain-first sudah terpenuhi ketika produk yang dibutuhkan nol.
            # Refill sebelum approach agar tanker tidak lebih dulu bermanuver
            # ke kapal lalu berbalik menuju Pertamina dengan stok kosong.
            if (remaining_mfo > 1e-6 and stock_mfo <= 1e-6) or (
                remaining_bio > 1e-6 and stock_bio <= 1e-6
            ):
                do_refill("REFILL SETELAH STOK HABIS SEBELUM APPROACH", event)
            arrival_at_ship = now + pd.Timedelta(
                hours=spec.move_time(location, sb.LOC_SANDAR)
            )
            # ATD adalah hard cutoff. Jangan melakukan approach apabila tanker
            # sudah tidak mungkin mulai memompa sebelum kapal berangkat.
            if arrival_at_ship < deadline:
                do_move(sb.LOC_SANDAR, event, note="Menuju kapal sebelum memompa")
                can_service = True

        segment_count = 0
        service_epsilon = 1e-6
        while can_service and (
            remaining_mfo > service_epsilon or remaining_bio > service_epsilon
        ):
            segment_count += 1
            if segment_count > 20:
                raise RuntimeError(
                    f"Terlalu banyak segmen bunker {event['EVENT_ID']}: "
                    f"sisa MFO {remaining_mfo}, BIO {remaining_bio}, "
                    f"stok MFO {stock_mfo}, BIO {stock_bio}, waktu {now}"
                )
            if stock_mfo <= service_epsilon:
                stock_mfo = 0.0
            if stock_bio <= service_epsilon:
                stock_bio = 0.0
            if remaining_mfo <= service_epsilon:
                remaining_mfo = 0.0
            if remaining_bio <= service_epsilon:
                remaining_bio = 0.0

            if now >= deadline:
                break

            if (remaining_mfo > service_epsilon and stock_mfo <= service_epsilon) or (
                remaining_bio > service_epsilon and stock_bio <= service_epsilon
            ):
                if not do_refill("REFILL LANJUTAN UNTUK KAPAL YANG SAMA", event):
                    raise RuntimeError(
                        f"Event {event['EVENT_ID']} membutuhkan refill tetapi "
                        "volume loading belum memenuhi minimum"
                    )
                arrival_after_refill = now + pd.Timedelta(
                    hours=spec.move_time(location, sb.LOC_SANDAR)
                )
                # Loading yang sudah dilakukan tetap menjadi persediaan tanker
                # untuk antrean berikutnya, tetapi tanker tidak kembali ke kapal
                # aktif bila kapal akan/telah berangkat.
                if arrival_after_refill >= deadline:
                    break
                do_move(sb.LOC_SANDAR, event, note="Refill lanjutan (kembali ke kapal yang sama)")

            chunk_mfo = min(remaining_mfo, stock_mfo)
            chunk_bio = min(remaining_bio, stock_bio)
            mfo_hours = chunk_mfo / spec.discharge_rate_mfo if chunk_mfo > 0 else 0.0
            bio_hours = chunk_bio / spec.discharge_rate_bio if chunk_bio > 0 else 0.0
            duration = max(mfo_hours, bio_hours)
            if duration <= 1e-12:
                raise RuntimeError(f"Tidak dapat melanjutkan event {event['EVENT_ID']}")
            start = now
            natural_end = start + pd.Timedelta(hours=duration)
            end = min(natural_end, deadline)
            pumping_hours = max(0.0, (end - start).total_seconds() / 3600.0)
            actual_mfo = min(chunk_mfo, pumping_hours * spec.discharge_rate_mfo)
            actual_bio = min(chunk_bio, pumping_hours * spec.discharge_rate_bio)
            if end <= start or (actual_mfo <= 1e-9 and actual_bio <= 1e-9):
                break
            if first_delivery_start is None:
                first_delivery_start = start
                stock_before_bunker_mfo = stock_mfo
                stock_before_bunker_bio = stock_bio
            before_mfo, before_bio = stock_mfo, stock_bio
            stock_mfo = max(0.0, stock_mfo - actual_mfo)
            stock_bio = max(0.0, stock_bio - actual_bio)
            remaining_mfo -= actual_mfo
            remaining_bio -= actual_bio
            if stock_mfo <= service_epsilon:
                stock_mfo = 0.0
            if stock_bio <= service_epsilon:
                stock_bio = 0.0
            if remaining_mfo <= service_epsilon:
                remaining_mfo = 0.0
            if remaining_bio <= service_epsilon:
                remaining_bio = 0.0
            delivery_segments.append(
                {
                    "start": start,
                    "end": end,
                    "mfo": actual_mfo,
                    "bio": actual_bio,
                }
            )
            add_operation(
                "BUNKER KE KAPAL", start, end, before_mfo, before_bio,
                stock_mfo, stock_bio, event,
                out_mfo=actual_mfo, out_bio=actual_bio,
                note=(
                    "MFO dan BIO dipompa bersamaan; dihentikan saat ATD"
                    if end < natural_end
                    else "MFO dan BIO dipompa bersamaan"
                ),
            )
            now = end

        delivered_mfo = sum(segment["mfo"] for segment in delivery_segments)
        delivered_bio = sum(segment["bio"] for segment in delivery_segments)

        req_mfo = float(event["KEBUTUHAN_MFO_KL"])
        req_bio = float(event["KEBUTUHAN_BIO_KL"])
        shortage_mfo = max(0.0, req_mfo - delivered_mfo)
        shortage_bio = max(0.0, req_bio - delivered_bio)
        completion = delivery_segments[-1]["end"] if delivery_segments else pd.NaT
        # Tidak ada lagi pelayanan terlambat: kapal ditutup tepat pada ATD.
        # Kegagalan pelayanan direpresentasikan oleh shortage, bukan durasi
        # bunker fiktif setelah kapal berangkat.
        delay = 0.0
        if first_delivery_start is None:
            wait = (
                max(0.0, (deadline - event["ATB"]).total_seconds() / 3600.0)
                if need_bunker
                else 0.0
            )
        else:
            wait = max(
                0.0,
                (first_delivery_start - event["ATB"]).total_seconds() / 3600.0,
            )
        if first_delivery_start is None and need_bunker:
            wait_reason = "TANKER TIDAK SEMPAT MULAI SEBELUM ATD"
        elif wait <= 1e-9:
            wait_reason = "TIDAK MENUNGGU"
        else:
            wait_operations = [
                operation
                for operation in ledger
                if operation["SELESAI"] > event["ATB"]
                and operation["MULAI"] < first_delivery_start
            ]
            reasons: list[str] = []
            if any(
                operation["JENIS_OPERASI"] in ("BUNKER KE KAPAL", "PINDAH ANTAR KAPAL")
                for operation in wait_operations
            ):
                reasons.append("ANTREAN BUNKER KAPAL LAIN")
            if any(
                operation["JENIS_OPERASI"] in ("REFILL PERTAMINA", "PERJALANAN KE PERTAMINA")
                for operation in wait_operations
            ):
                reasons.append("REFILL PERTAMINA")
            if any(
                operation["JENIS_OPERASI"] == "MENUNGGU IZIN PERJALANAN REFILL"
                for operation in wait_operations
            ):
                reasons.append("JEDA 24 JAM MENUJU PERTAMINA")
            if any(
                operation["JENIS_OPERASI"] == "MANUVER & PERSIAPAN KE KAPAL"
                for operation in wait_operations
            ):
                reasons.append("PROSES APPROACH & PERSIAPAN")
            if any(operation["JENIS_OPERASI"] == "KEMBALI KE BASE" for operation in wait_operations):
                reasons.append("KEMBALI KE BASE / SIKLUS KELUAR-MASUK")
            wait_reason = " + ".join(reasons) if reasons else "MENUNGGU KETERSEDIAAN TANKER"
        feasible = shortage_mfo <= 1e-6 and shortage_bio <= 1e-6

        result = dict(event)
        result.update(
            {
                "MULAI_BUNKER": first_delivery_start,
                "SELESAI_BUNKER": completion,
                "WAKTU_TUNGGU_JAM": wait,
                "REASON_WAKTU_TUNGGU": wait_reason,
                "STOK_AWAL_MFO_TANKER_KL": stock_before_bunker_mfo,
                "STOK_AWAL_BIO_TANKER_KL": stock_before_bunker_bio,
                "STOK_AKHIR_MFO_TANKER_KL": stock_mfo,
                "STOK_AKHIR_BIO_TANKER_KL": stock_bio,
                "DURASI_PROSES_JAM": (
                    (completion - first_delivery_start).total_seconds() / 3600.0
                    if first_delivery_start is not None
                    else 0.0
                ),
                "TERISI_SBLM_ATD_MFO_KL": delivered_mfo,
                "TERISI_SBLM_ATD_BIO_KL": delivered_bio,
                "SHORTAGE_MFO_KL": shortage_mfo,
                "SHORTAGE_BIO_KL": shortage_bio,
                "KETERLAMBATAN_JAM": delay,
                "STATUS": "FEASIBLE" if feasible else "NOT FEASIBLE",
                "ALASAN_FLAG": "OK"
                if feasible
                else "Kapal berangkat saat ATD; permintaan tidak terisi seluruhnya",
            }
        )
        results.append(result)

        if need_bunker:
            next_waiting = any(event_queued["ATB"] <= now for event_queued in unscheduled)
            if next_waiting:
                pass
            else:
                do_move(sb.LOC_LABUH, event, note="Kembali ke base setelah bunker (idle)")

    for _, event in invalid.iterrows():
        result = event.to_dict()
        result.update(
            {
                "MULAI_BUNKER": pd.NaT,
                "SELESAI_BUNKER": pd.NaT,
                "WAKTU_TUNGGU_JAM": np.nan,
                "REASON_WAKTU_TUNGGU": "DATA ISSUE",
                "STOK_AWAL_MFO_TANKER_KL": np.nan,
                "STOK_AWAL_BIO_TANKER_KL": np.nan,
                "STOK_AKHIR_MFO_TANKER_KL": np.nan,
                "STOK_AKHIR_BIO_TANKER_KL": np.nan,
                "DURASI_PROSES_JAM": np.nan,
                "TERISI_SBLM_ATD_MFO_KL": 0.0,
                "TERISI_SBLM_ATD_BIO_KL": 0.0,
                "SHORTAGE_MFO_KL": result.get("KEBUTUHAN_MFO_KL", np.nan),
                "SHORTAGE_BIO_KL": result.get("KEBUTUHAN_BIO_KL", np.nan),
                "KETERLAMBATAN_JAM": np.nan,
                "STATUS": "DATA ISSUE",
                "ALASAN_FLAG": result["DATA_FLAG"],
            }
        )
        results.append(result)

    return pd.DataFrame(results), pd.DataFrame(ledger)


def run(mode: str, lookahead_wait: bool = False) -> dict[str, pd.DataFrame]:
    """Jalankan simulasi penuh (kedua tanker + cabang) dengan prioritas `mode`."""
    schedule = sb.load_schedule()
    consumption = sb.load_consumption(set(schedule["VESSELID"].dropna()))
    events, issues = sb.build_events(schedule, consumption)
    events = sb.attach_tank_capacity(events)
    events = sb.merge_direct_voyages(events)
    events = events[events["ATB"] <= sb.SIMULATION_END].copy()
    issues = issues[issues["EVENT_ID"].isin(events["EVENT_ID"])].copy()
    events = sb.apply_vessel_fuel_policy(events)
    events, issues = sb.renumber_events_chronologically(events, issues)

    all_results: list[pd.DataFrame] = []
    all_ledgers: list[pd.DataFrame] = []
    for port, spec in sb.TANKERS.items():
        tanker_events = events[events["TANKER"] == spec.name].copy()
        result, ledger = simulate_tanker_priority(spec, tanker_events, mode, lookahead_wait)
        all_results.append(result)
        all_ledgers.append(ledger)

    branch_events = events[events["TANKER"] == "CABANG"].copy()
    if not branch_events.empty:
        all_results.append(sb.build_branch_results(branch_events))

    results = pd.concat(all_results, ignore_index=True)
    ledger = pd.concat(all_ledgers, ignore_index=True)
    sb.validate_simulation(results, ledger)
    summary = sb.build_summary(results, ledger)
    wait_breakdown = sb.build_wait_breakdown(results, ledger)
    return {
        "summary": summary,
        "results": results,
        "ledger": ledger,
        "wait_breakdown": wait_breakdown,
        "issues": issues,
        "consumption": consumption,
    }


def _parameter_table(mode: str, lookahead_wait: bool = False) -> pd.DataFrame:
    params = sb.parameter_table()
    params = params[params["PARAMETER"] != "PRIORITAS ANTREAN"]
    extra_rows = [
        ("PRIORITAS ANTREAN", PRIORITY_DESCRIPTIONS[mode]),
        ("DASAR PRIORITAS", "ETD (fallback DEADLINE jika ETD kosong)"),
        ("PERHITUNGAN KETERLAMBATAN", "Terhadap DEADLINE (ATD, fallback ETD)"),
    ]
    if lookahead_wait:
        extra_rows.append(("LOOK-AHEAD WAIT", LOOKAHEAD_DESCRIPTION))
    extra = pd.DataFrame(extra_rows, columns=["PARAMETER", "NILAI/ATURAN"])
    return pd.concat([params, extra], ignore_index=True)


def write_priority_excel(
    out: dict[str, pd.DataFrame], mode: str, lookahead_wait: bool, xlsx_path: Path
) -> None:
    summary = out["summary"]
    results = out["results"]
    ledger = out["ledger"]
    wait_breakdown = out["wait_breakdown"]

    result_columns = [
        "EVENT_ID", "VESSEL", "VOYAGENO", "VESVOY_GABUNGAN", "FULL_SAILING_ROUTE",
        "SAILING_ROUTE_CHECKPOINT_TO_CHECKPOINT", "STEP", "TANKER",
        "CHECKPOINT", "CHECKPOINT_BERIKUT",
        "JARAK_KE_CHECKPOINT_NM",
        "KEBUTUHAN_DASAR_MFO_KL", "KEBUTUHAN_MFO_KL", "KAPASITAS_TANGKI_MFO_KL",
        "KEBUTUHAN_DASAR_BIO_KL", "KEBUTUHAN_BIO_KL", "KAPASITAS_TANGKI_BIO_KL",
        "KEBIJAKAN_BBM", "STATUS_KAPASITAS",
        "ATB", "ETD", "ATD", "DEADLINE", "PRIORITAS_ETD",
        "MULAI_BUNKER", "SELESAI_BUNKER",
        "WAKTU_TUNGGU_JAM", "REASON_WAKTU_TUNGGU",
        "STOK_AWAL_MFO_TANKER_KL", "STOK_AWAL_BIO_TANKER_KL",
        "DURASI_PROSES_JAM", "TERISI_SBLM_ATD_MFO_KL", "TERISI_SBLM_ATD_BIO_KL",
        "SHORTAGE_MFO_KL", "SHORTAGE_BIO_KL",
        "STOK_AKHIR_MFO_TANKER_KL", "STOK_AKHIR_BIO_TANKER_KL",
        "KETERLAMBATAN_JAM", "STATUS", "ALASAN_FLAG",
    ]
    results_out = results.reindex(columns=result_columns).sort_values("EVENT_ID")

    ledger_columns = [
        "OPERATION_ID", "TANKER", "PORT", "JENIS_OPERASI", "EVENT_ID", "VESSEL",
        "MULAI", "SELESAI", "DURASI_JAM", "STOK_AWAL_MFO_KL", "STOK_AWAL_BIO_KL",
        "MASUK_MFO_KL", "MASUK_BIO_KL", "KELUAR_MFO_KL", "KELUAR_BIO_KL",
        "STOK_AKHIR_MFO_KL", "STOK_AKHIR_BIO_KL",
    ]
    ledger_out = ledger.reindex(columns=ledger_columns).sort_values(["MULAI", "TANKER", "OPERATION_ID"])

    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter", datetime_format="dd/mm/yyyy hh:mm") as writer:
        summary.to_excel(writer, sheet_name="RINGKASAN", index=False)
        results_out.to_excel(writer, sheet_name="HASIL_KAPAL", index=False)
        ledger_out.to_excel(writer, sheet_name="MONITORING_TANKER", index=False)
        wait_breakdown.to_excel(writer, sheet_name="BREAKDOWN_TUNGGU", index=False)
        _parameter_table(mode, lookahead_wait).to_excel(writer, sheet_name="PARAMETER", index=False)

        workbook = writer.book
        header_fmt = workbook.add_format({"bold": True, "align": "center", "valign": "vcenter", "border": 1})
        number_fmt = workbook.add_format({"num_format": "0.00"})
        date_fmt = workbook.add_format({"num_format": "dd/mm/yyyy hh:mm"})
        red_fmt = workbook.add_format({"bg_color": "#FFC7CE", "font_color": "#9C0006"})
        green_fmt = workbook.add_format({"bg_color": "#C6EFCE", "font_color": "#006100"})
        amber_fmt = workbook.add_format({"bg_color": "#FFEB9C", "font_color": "#9C6500"})

        frames = {
            "RINGKASAN": summary,
            "HASIL_KAPAL": results_out,
            "MONITORING_TANKER": ledger_out,
            "BREAKDOWN_TUNGGU": wait_breakdown,
            "PARAMETER": _parameter_table(mode, lookahead_wait),
        }
        for name, frame in frames.items():
            sheet = writer.sheets[name]
            sheet.freeze_panes(1, 0)
            if len(frame.columns):
                sheet.autofilter(0, 0, max(len(frame), 1), len(frame.columns) - 1)
            for col_num, col_name in enumerate(frame.columns):
                sheet.write(0, col_num, col_name, header_fmt)
                sample = frame[col_name].astype(str).replace("NaT", "").replace("nan", "")
                max_len = max([len(str(col_name))] + [len(value) for value in sample.head(500)])
                width = min(max(max_len + 2, 11), 42)
                fmt = date_fmt if pd.api.types.is_datetime64_any_dtype(frame[col_name]) else (
                    number_fmt if pd.api.types.is_numeric_dtype(frame[col_name]) else None
                )
                sheet.set_column(col_num, col_num, width, fmt)

        result_sheet = writer.sheets["HASIL_KAPAL"]
        if not results_out.empty:
            status_col = results_out.columns.get_loc("STATUS")
            result_sheet.conditional_format(1, status_col, len(results_out), status_col, {
                "type": "cell", "criteria": "==", "value": '"NOT FEASIBLE"', "format": red_fmt
            })
            result_sheet.conditional_format(1, status_col, len(results_out), status_col, {
                "type": "cell", "criteria": "==", "value": '"FEASIBLE"', "format": green_fmt
            })
            result_sheet.conditional_format(1, status_col, len(results_out), status_col, {
                "type": "cell", "criteria": "==", "value": '"DATA ISSUE"', "format": amber_fmt
            })


def write_priority_doc(
    out: dict[str, pd.DataFrame], mode: str, lookahead_wait: bool, doc_path: Path, xlsx_name: str
) -> None:
    summary = out["summary"]
    results = out["results"]
    total = summary[summary["TANKER"] == "TOTAL"].iloc[0]
    combined_count = int(results["VESVOY_GABUNGAN"].fillna("").ne("").sum())
    unpaired_direct = results[
        (results["STEP"] == 2) & results["VESVOY_GABUNGAN"].fillna("").eq("")
    ]
    unpaired_text = ", ".join(
        f"{row.VESSEL} {row.VOYAGENO}" for row in unpaired_direct.itertuples()
    ) or "Tidak ada"

    flagged = results[results["STATUS"] == "NOT FEASIBLE"].sort_values(
        "KETERLAMBATAN_JAM", ascending=False
    )
    top_flags = "\n".join(
        f"- {row.VESSEL} di {row.CHECKPOINT}, ATA {row.ATA:%d/%m/%Y %H:%M}: "
        f"shortage MFO {row.SHORTAGE_MFO_KL:.2f} KL, BIO {row.SHORTAGE_BIO_KL:.2f} KL, "
        f"terlambat {row.KETERLAMBATAN_JAM:.2f} jam."
        for row in flagged.head(10).itertuples()
    ) or "- Tidak ada event yang melewati deadline."

    if int(total.NOT_FEASIBLE) == 0:
        conclusion = (
            f"SIGMA dan ZETA cukup untuk seluruh {int(total.EVENT_VALID)} event valid. "
            f"Tidak ada shortage terhadap deadline."
        )
    else:
        conclusion = (
            f"Terdapat {int(total.NOT_FEASIBLE)} dari {int(total.EVENT_VALID)} event valid yang "
            f"tidak selesai sebelum deadline. Total shortage MFO {total.SHORTAGE_MFO_KL:.2f} KL, "
            f"BIO {total.SHORTAGE_BIO_KL:.2f} KL; keterlambatan maksimum "
            f"{total.MAX_TERLAMBATAN_JAM:.2f} jam."
        )

    lookahead_line = f"\n## Look-ahead wait\n\n{LOOKAHEAD_DESCRIPTION}\n" if lookahead_wait else ""

    text = f"""# Simulasi Antrean Bunker — Prioritas {mode}{' + Look-ahead Wait' if lookahead_wait else ''}

## Aturan prioritas

{PRIORITY_DESCRIPTIONS[mode]}
{lookahead_line}
- Prioritas ditentukan dengan **ETD**; bila ETD kosong, fallback ke DEADLINE.
- **Keterlambatan dan shortage tetap dihitung terhadap DEADLINE (ATD, fallback ETD).**
- SIGMA dan ZETA memiliki antrean terpisah; kapal yang sudah mulai dibunker tidak diinterupsi.
- Seluruh aturan lain (refill look-ahead, waktu perpindahan, flow rate, stok minimum) sama dengan simulasi utama.

File hasil: `{xlsx_name}`.

## Pembentukan event dan penggabungan voyage

- Rute direct adalah interval dengan `STEP=2` sebelum penggabungan.
- Penggabungan dibatasi pada dua event direct `STEP=2` yang berurutan dan tersambung untuk vessel yang sama. Merge hanya dilakukan jika target 130% MFO dan BIO sama-sama muat. Jika gagal, event pertama diproses sendiri dan event kedua tetap dapat diuji dengan event berikutnya.
- `VESVOY_GABUNGAN` mencatat pasangan sumber, misalnya `PFA 12/2026 + PFA 13/2026`. `VOYAGENO` tetap menunjukkan voyage pertama.
- Jika event direct berikutnya bukan `STEP=2` atau tidak tersedia, event tidak dipaksa bergabung dan tetap `STEP=2`.
- Event setelah akhir Juli hanya dibaca sebagai look-ahead pasangan; event tersebut tidak masuk hasil sebagai event awal baru.
- Jumlah event gabungan pada run ini: **{combined_count}**.
- Direct tanpa pasangan pada run ini: **{unpaired_text}**.

## Kebijakan volume BBM kapal

- Konsumsi dasar dihitung dari `total NMILE × liter/NM ÷ 1.000`.
- Direct gabungan: pengisian dan penggunaan MFO maupun BIO adalah total kebutuhan dasar dua voyage × 130%.
- Event non-direct dan direct tanpa pasangan: pengisian dan penggunaan MFO maupun BIO adalah minimum 150% kebutuhan dasar atau kapasitas fisik.
- ROB tidak dimodelkan karena ROB awal 0 KL dan pengisian selalu sama dengan penggunaan.
- `KEBUTUHAN_*_KL` adalah volume aktual yang diminta sekaligus digunakan; pembatasan kapasitas dicatat pada `STATUS_KAPASITAS`.
- Kapal yang tidak memiliki data langsung MFO/MILES dan BIO/MILES dikecualikan sesuai `EXCLUDED_VESSELS`.

## Hasil

| Tanker | Event | Feasible | Not feasible | Data issue | Kebutuhan MFO | Kebutuhan BIO | Shortage MFO | Shortage BIO | Max terlambat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{_summary_md_rows(summary)}

## Kesimpulan

{conclusion}

## Event dengan keterlambatan terbesar

{top_flags}
"""
    doc_path.write_text(text, encoding="utf-8")


def _summary_md_rows(summary: pd.DataFrame) -> str:
    lines = []
    for _, row in summary.iterrows():
        lines.append(
            f"| {row.TANKER} | {int(row.TOTAL_EVENT)} | {int(row.FEASIBLE)} | "
            f"{int(row.NOT_FEASIBLE)} | {int(row.DATA_ISSUE)} | "
            f"{row.KEBUTUHAN_MFO_KL:.2f} KL | {row.KEBUTUHAN_BIO_KL:.2f} KL | "
            f"{row.SHORTAGE_MFO_KL:.2f} KL | {row.SHORTAGE_BIO_KL:.2f} KL | "
            f"{row.MAX_TERLAMBATAN_JAM:.2f} jam |"
        )
    return "\n".join(lines)


def produce(mode: str, lookahead_wait: bool = False) -> dict[str, pd.DataFrame]:
    """Jalankan, simpan Excel + dokumentasi, dan kembalikan hasil."""
    out = run(mode, lookahead_wait)
    tag = mode.lower() + ("_lookahead" if lookahead_wait else "")
    xlsx_name = f"hasil_antrean_{tag}.xlsx"
    doc_name = f"SIMULASI_BBM_{mode}{'_LOOKAHEAD' if lookahead_wait else ''}.md"
    write_priority_excel(out, mode, lookahead_wait, OUTPUT_DIR / xlsx_name)
    write_priority_doc(out, mode, lookahead_wait, DOC_DIR / doc_name, xlsx_name)
    return out
