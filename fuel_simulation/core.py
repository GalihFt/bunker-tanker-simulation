#!/usr/bin/env python3
"""Simulasi pengisian BBM kapal operasional oleh tanker ZETA dan SIGMA.

Input:
  - data/schedule.xlsx
  - data/fuel_consumption.xlsx
  - data/vessel_tank_capacity.xlsx

Output:
  - outputs/baseline_simulation.xlsx
  - outputs/baseline_simulation.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SCHEDULE_FILE = DATA_DIR / "schedule.xlsx"
CONSUMPTION_FILE = DATA_DIR / "fuel_consumption.xlsx"
TANK_CAPACITY_FILE = DATA_DIR / "vessel_tank_capacity.xlsx"
OUTPUT_FILE = OUTPUT_DIR / "baseline_simulation.xlsx"
DOC_FILE = OUTPUT_DIR / "baseline_simulation.md"

SIMULATION_START = pd.Timestamp("2026-06-01 00:00:00")
SIMULATION_END = pd.Timestamp("2026-07-31 23:59:59")

FLEETS = {
    "A": "HSJ ODI OJA ORU OSA VEI".split(),
    "B": "HSA OGO MBU SBR TIT VER".split(),
    "C": "HAN HAP HAS HAY HJE PSM".split(),
    "D": "AKA DER FOR MAG KAA RAT RUM".split(),
    "E": "OEM OSI ASG ASN ASR MAN".split(),
    "G": "APE LUZ OGA PBE PFA PRI".split(),
    "H": "HSG PHK PLA PWE TBE TBI TFL".split(),
    "K": "BAU BGI BKU BSA PAH PST".split(),
    "J1": "KLE SLE RAH RET REN".split(),
    "J2": "OPA PRA PNN".split(),
}
FLEET_BY_VESSEL = {vessel: fleet for fleet, vessels in FLEETS.items() for vessel in vessels}
BIO_ONLY = {"KXP", "LBR", "MAN", "PBE", "PSM", "THT"}
VIRTUAL_PORTS = {"IDRDE", "IDDOK"}
BRANCH_CHECKPOINTS = {"IDMAK", "IDAMQ"}
# Kedatangan checkpoint utama yang hanya berjarak satu leg dari checkpoint
# terpilih sebelumnya dilewati sebagai titik bunker. Port tersebut tetap masuk
# ke sailing route, lalu endpoint digeser ke checkpoint utama berikutnya.
MIN_LEGS_BETWEEN_MAIN_CHECKPOINTS = 2
MIN_LOADING_MFO_KL = 250.0
MIN_LOADING_BIO_KL = 100.0
REFILL_TRIP_COOLDOWN_HOURS = 24.0
REFILL_LOOKAHEAD_HOURS = 24.0
# Kapal yang dikecualikan manual dari simulasi (tidak dibuatkan event dan tidak
# dihitung kebutuhannya). Kapal tanpa data konsumsi MFO/MILES dan BIO/MILES
# dikeluarkan agar analisis tidak memakai fallback rata-rata global.
EXCLUDED_VESSELS = {
    "KAP",
    "BET", "KME", "MAA", "MAR", "MDA", "MEN", "MHI", "MIA",
    "MPR", "MSM", "MTD", "MTI", "MVI", "MWA", "TCI", "YYU",
}
# Voyage bernomor non-SPIL (kapal Meratus): pola STxxxN/2026 atau xxxJ/P/2026.
# Vessel yang memakai format ini dianggap bukan kapal SPIL dan otomatis dikecualikan.
MERATUS_VOYAGE_RE = re.compile(r"^(ST\d+N|\d+[JP])/\d{4}$")


LOC_LABUH = "LABUH"
LOC_PERTAMINA = "PERTAMINA"
LOC_SANDAR = "SANDAR"


@dataclass(frozen=True)
class TankerAssetSpec:
    name: str
    capacity_mfo: float
    capacity_bio: float
    discharge_rate_mfo: float
    discharge_rate_bio: float
    min_stock_mfo: float
    min_stock_bio: float


@dataclass(frozen=True)
class PortOperationSpec:
    port: str
    load_rate_mfo: float
    load_rate_bio: float
    load_simultaneous: bool
    move_times: dict[str, float]


@dataclass(frozen=True)
class TankerSpec:
    """Tanker yang sudah ditempatkan di suatu daerah operasi.

    Kapasitas, discharge rate, dan minimum stock berasal dari aset tanker.
    Load rate, pola loading, dan waktu mobilisasi berasal dari daerah operasi.
    """

    name: str
    port: str
    capacity_mfo: float
    capacity_bio: float
    load_rate_mfo: float
    load_rate_bio: float
    load_simultaneous: bool
    discharge_rate_mfo: float
    discharge_rate_bio: float
    move_times: dict[str, float]
    min_stock_mfo: float
    min_stock_bio: float

    def move_time(self, from_loc: str, to_loc: str) -> float:
        """Waktu pindah tanker dari from_loc ke to_loc (jam).

        Shift antar kapal (SANDAR → SANDAR) memakai nilai dari matriks; pasangan
        lokasi yang tidak terdefinisi (mis. LABUH → LABUH) dianggap 0 jam.
        """
        return float(self.move_times.get(f"{from_loc}->{to_loc}", 0.0))


TANKER_ASSETS = {
    "SIGMA": TankerAssetSpec("SIGMA", 1000.0, 500.0, 110.0, 90.0, 250.0, 50.0),
    "ZETA": TankerAssetSpec("ZETA", 750.0, 250.0, 110.0, 50.0, 250.0, 50.0),
}

PORT_OPERATIONS = {
    "IDJKT": PortOperationSpec(
        "IDJKT", 100.0, 120.0, False,
        {
            "LABUH->PERTAMINA": 6.0,
            "LABUH->SANDAR": 6.0,
            "PERTAMINA->LABUH": 4.0,
            "PERTAMINA->SANDAR": 4.0,
            "SANDAR->LABUH": 4.0,
            "SANDAR->PERTAMINA": 4.0,
            "SANDAR->SANDAR": 2.0,
        },
    ),
    "IDSUB": PortOperationSpec(
        "IDSUB", 40.0, 70.0, True,
        {
            "LABUH->PERTAMINA": 1.5,
            "LABUH->SANDAR": 1.5,
            "PERTAMINA->LABUH": 3.0,
            "PERTAMINA->SANDAR": 2.0,
            "SANDAR->LABUH": 3.0,
            "SANDAR->PERTAMINA": 2.0,
            "SANDAR->SANDAR": 1.0,
        },
    ),
}


def deploy_tanker(asset_name: str, port: str) -> TankerSpec:
    asset = TANKER_ASSETS[asset_name]
    operation = PORT_OPERATIONS[port]
    return TankerSpec(
        name=asset.name,
        port=operation.port,
        capacity_mfo=asset.capacity_mfo,
        capacity_bio=asset.capacity_bio,
        load_rate_mfo=operation.load_rate_mfo,
        load_rate_bio=operation.load_rate_bio,
        load_simultaneous=operation.load_simultaneous,
        discharge_rate_mfo=asset.discharge_rate_mfo,
        discharge_rate_bio=asset.discharge_rate_bio,
        move_times=operation.move_times,
        min_stock_mfo=asset.min_stock_mfo,
        min_stock_bio=asset.min_stock_bio,
    )


# Skenario terisolasi: checkpoint memprioritaskan Surabaya, sedangkan
# penempatan tanker kembali ke SIGMA di Jakarta dan ZETA di Surabaya.
TANKERS = {
    "IDJKT": deploy_tanker("SIGMA", "IDJKT"),
    "IDSUB": deploy_tanker("ZETA", "IDSUB"),
}


def parse_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%d/%m/%Y %H:%M", errors="coerce")


def weighted_rate(df: pd.DataFrame, fuel_col: str) -> float:
    usable = df.dropna(subset=["MILES", fuel_col])
    miles = usable["MILES"].sum()
    return float(usable[fuel_col].sum() / miles) if miles else np.nan


def load_consumption(schedule_vessels: set[str]) -> pd.DataFrame:
    mfo = pd.read_excel(CONSUMPTION_FILE, sheet_name="MFO")
    bio = pd.read_excel(CONSUMPTION_FILE, sheet_name="BIO")
    mfo["VESSEL"] = mfo["VESSEL"].astype(str).str.strip().str.upper()
    bio["VESSEL"] = bio["VESSEL"].astype(str).str.strip().str.upper()

    mfo_direct = mfo.set_index("VESSEL")["MFO/MILES"].astype(float).to_dict()
    bio_direct = bio.set_index("VESSEL")["BIO/MILES"].astype(float).to_dict()
    global_mfo = weighted_rate(mfo, "MFO")
    global_bio = weighted_rate(bio, "BIO")

    fleet_mfo: dict[str, float] = {}
    fleet_bio: dict[str, float] = {}
    for fleet, members in FLEETS.items():
        fleet_mfo[fleet] = weighted_rate(mfo[mfo["VESSEL"].isin(members)], "MFO")
        fleet_bio[fleet] = weighted_rate(bio[bio["VESSEL"].isin(members)], "BIO")

    rows: list[dict[str, Any]] = []
    all_vessels = (
        schedule_vessels | set(FLEET_BY_VESSEL) | set(mfo_direct) | set(bio_direct)
    ) - EXCLUDED_VESSELS
    for vessel in sorted(all_vessels):
        fleet = FLEET_BY_VESSEL.get(vessel)

        if vessel in BIO_ONLY:
            mfo_rate, mfo_source = 0.0, "BIO_ONLY (MFO=0)"
        elif vessel in mfo_direct:
            mfo_rate, mfo_source = float(mfo_direct[vessel]), "DATA YTD JAN-JUL 2026"
        elif fleet and np.isfinite(fleet_mfo.get(fleet, np.nan)):
            mfo_rate, mfo_source = fleet_mfo[fleet], f"RATA-RATA SISTERSHIP {fleet}"
        else:
            mfo_rate, mfo_source = global_mfo, "RATA-RATA GLOBAL"

        if vessel in bio_direct:
            bio_rate, bio_source = float(bio_direct[vessel]), "DATA YTD JAN-JUL 2026"
        elif fleet and np.isfinite(fleet_bio.get(fleet, np.nan)):
            bio_rate, bio_source = fleet_bio[fleet], f"RATA-RATA SISTERSHIP {fleet}"
        else:
            bio_rate, bio_source = global_bio, "RATA-RATA GLOBAL"

        rows.append(
            {
                "VESSEL": vessel,
                "FLEET": fleet or "TIDAK TERDAFTAR",
                "BIO_ONLY": "YA" if vessel in BIO_ONLY else "TIDAK",
                "MFO_L_PER_NM": mfo_rate,
                "SUMBER_MFO": mfo_source,
                "BIO_L_PER_NM": bio_rate,
                "SUMBER_BIO": bio_source,
            }
        )
    return pd.DataFrame(rows)


def load_tank_capacity() -> pd.DataFrame:
    """Load kapasitas tangki kapal dari sheet terbaru/lengkap (Sheet2)."""
    capacity = pd.read_excel(TANK_CAPACITY_FILE, sheet_name="Sheet2")
    capacity.columns = capacity.columns.str.strip()
    capacity["VESSEL"] = capacity["Vessel"].astype(str).str.strip().str.upper()
    capacity["KAPASITAS_TANGKI_MFO_KL"] = pd.to_numeric(
        capacity["MFO"], errors="coerce"
    )
    capacity["KAPASITAS_TANGKI_BIO_KL"] = pd.to_numeric(
        capacity["BIO"], errors="coerce"
    )
    # Sel kosong berarti kapal tidak memakai produk tersebut, sehingga
    # kapasitas operasionalnya diperlakukan sebagai 0 KL.
    capacity[["KAPASITAS_TANGKI_MFO_KL", "KAPASITAS_TANGKI_BIO_KL"]] = (
        capacity[["KAPASITAS_TANGKI_MFO_KL", "KAPASITAS_TANGKI_BIO_KL"]]
        .fillna(0.0)
        .clip(lower=0.0)
    )
    if capacity["VESSEL"].duplicated().any():
        duplicates = sorted(capacity.loc[capacity["VESSEL"].duplicated(False), "VESSEL"].unique())
        raise ValueError(
            "VESSEL duplikat pada vessel_tank_capacity.xlsx: "
            + ", ".join(duplicates)
        )
    return capacity[
        ["VESSEL", "KAPASITAS_TANGKI_MFO_KL", "KAPASITAS_TANGKI_BIO_KL"]
    ]


def attach_tank_capacity(events: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan kapasitas tangki kapal ke setiap event berdasarkan VESSEL."""
    return events.merge(
        load_tank_capacity(),
        on="VESSEL",
        how="left",
        validate="many_to_one",
        sort=False,
    )


def merge_direct_voyages(
    events: pd.DataFrame, target_factor: float = 1.30
) -> pd.DataFrame:
    """Gabungkan dua direct berurutan jika target berdasarkan faktor muat.

    Jika pasangan tidak muat untuk salah satu produk, event pertama diproses
    sendiri dan event kedua tetap tersedia untuk diuji dengan event berikutnya.
    Default 130% mempertahankan perilaku simulasi baseline.
    """
    source = events.copy()
    source["STEP"] = (
        source["SAILING_ROUTE_CHECKPOINT_TO_CHECKPOINT"]
        .fillna("")
        .astype(str)
        .str.count("-")
    )
    def combine(
        vessel: str, first: dict[str, Any], second: dict[str, Any]
    ) -> dict[str, Any]:
        first_route = str(first["SAILING_ROUTE_CHECKPOINT_TO_CHECKPOINT"])
        second_route = str(second["SAILING_ROUTE_CHECKPOINT_TO_CHECKPOINT"])
        first_ports = first_route.split("-")
        second_ports = second_route.split("-")
        combined_ports = (
            first_ports + second_ports[1:]
            if first_ports[-1] == second_ports[0]
            else first_ports + second_ports
        )
        combined = dict(first)
        combined["VESVOY_GABUNGAN"] = (
            f"{vessel} {first['VOYAGENO']} + {vessel} {second['VOYAGENO']}"
        )
        combined["FULL_SAILING_ROUTE"] = (
            f"{first['FULL_SAILING_ROUTE']} | {second['FULL_SAILING_ROUTE']}"
        )
        combined["SAILING_ROUTE_CHECKPOINT_TO_CHECKPOINT"] = "-".join(combined_ports)
        combined["STEP"] = len(combined_ports) - 1
        combined["CHECKPOINT_BERIKUT"] = second["CHECKPOINT_BERIKUT"]
        combined["NEXT_CHECKPOINT"] = second["NEXT_CHECKPOINT"]
        combined["JARAK_KE_CHECKPOINT_NM"] = (
            float(first["JARAK_KE_CHECKPOINT_NM"])
            + float(second["JARAK_KE_CHECKPOINT_NM"])
        )
        combined["KEBUTUHAN_MFO_KL"] = (
            float(first["KEBUTUHAN_MFO_KL"])
            + float(second["KEBUTUHAN_MFO_KL"])
        )
        combined["KEBUTUHAN_BIO_KL"] = (
            float(first["KEBUTUHAN_BIO_KL"])
            + float(second["KEBUTUHAN_BIO_KL"])
        )
        combined["DIRECT_FIRST_VOYAGE_MFO_KL"] = float(first["KEBUTUHAN_MFO_KL"])
        combined["DIRECT_FIRST_VOYAGE_BIO_KL"] = float(first["KEBUTUHAN_BIO_KL"])
        combined["DIRECT_VOYAGE_PAIR"] = True
        return combined

    def pair_fits(first: dict[str, Any], second: dict[str, Any]) -> bool:
        first_route = str(first["SAILING_ROUTE_CHECKPOINT_TO_CHECKPOINT"]).split("-")
        second_route = str(second["SAILING_ROUTE_CHECKPOINT_TO_CHECKPOINT"]).split("-")
        if not first_route or not second_route or first_route[-1] != second_route[0]:
            return False
        if str(first["VOYAGENO"]) == str(second["VOYAGENO"]):
            return False
        for fuel in ("MFO", "BIO"):
            base_pair = float(first[f"KEBUTUHAN_{fuel}_KL"]) + float(
                second[f"KEBUTUHAN_{fuel}_KL"]
            )
            capacity = float(first[f"KAPASITAS_TANGKI_{fuel}_KL"])
            if not np.isfinite(base_pair) or not np.isfinite(capacity):
                return False
            if base_pair * target_factor > capacity + 1e-9:
                return False
        return True

    merged_rows: list[dict[str, Any]] = []
    for vessel, vessel_events in source.groupby("VESSEL", sort=False):
        rows = vessel_events.sort_values(
            ["ATB", "EVENT_ID"], na_position="last", kind="stable"
        ).to_dict("records")
        vessel_merged: list[dict[str, Any]] = []
        index = 0
        while index < len(rows):
            current = rows[index]
            can_merge = (
                current["DATA_FLAG"] == "OK"
                and int(current["STEP"]) == 2
                and index + 1 < len(rows)
                and rows[index + 1]["DATA_FLAG"] == "OK"
                and int(rows[index + 1]["STEP"]) == 2
                and pair_fits(current, rows[index + 1])
            )
            if not can_merge:
                current["DIRECT_VOYAGE_PAIR"] = False
                current["VESVOY_GABUNGAN"] = ""
                vessel_merged.append(current)
                index += 1
                continue

            following = rows[index + 1]
            vessel_merged.append(combine(vessel, current, following))
            index += 2

        merged_rows.extend(vessel_merged)

    return pd.DataFrame(merged_rows)


def apply_vessel_fuel_policy(events: pd.DataFrame) -> pd.DataFrame:
    """Hitung volume aktual dengan penggunaan selalu sama dengan pengisian.

    KEBUTUHAN_DASAR_* menyimpan konsumsi sebelum buffer. KEBUTUHAN_* setelah
    fungsi ini adalah volume aktual yang diminta sekaligus digunakan kapal.
    """
    planned = events.copy()
    planned["STEP"] = (
        planned["SAILING_ROUTE_CHECKPOINT_TO_CHECKPOINT"]
        .fillna("")
        .astype(str)
        .str.count("-")
    )
    planned["KEBUTUHAN_DASAR_MFO_KL"] = planned["KEBUTUHAN_MFO_KL"]
    planned["KEBUTUHAN_DASAR_BIO_KL"] = planned["KEBUTUHAN_BIO_KL"]
    planned["KEBIJAKAN_BBM"] = ""
    planned["STATUS_KAPASITAS"] = ""

    for index in planned.index:
        if planned.at[index, "DATA_FLAG"] != "OK":
            continue

        direct_pair = bool(planned.at[index, "DIRECT_VOYAGE_PAIR"])
        if direct_pair:
            policy = "DIRECT_MERGE_130"
            factor = 1.30
        elif int(planned.at[index, "STEP"]) == 2:
            policy = "DIRECT_SINGLE_150"
            factor = 1.50
        else:
            policy = "NON_DIRECT_150"
            factor = 1.50

        limited: list[str] = []
        inconsistent = False
        for fuel in ("MFO", "BIO"):
            base_need = float(planned.at[index, f"KEBUTUHAN_DASAR_{fuel}_KL"])
            capacity = float(planned.at[index, f"KAPASITAS_TANGKI_{fuel}_KL"])
            if not np.isfinite(base_need) or not np.isfinite(capacity):
                inconsistent = True
                continue
            if capacity <= 1e-9 and base_need > 1e-9:
                inconsistent = True
            target = base_need * factor
            actual = min(target, capacity)
            planned.at[index, f"KEBUTUHAN_{fuel}_KL"] = actual
            if target > capacity + 1e-9:
                limited.append(fuel)

        planned.at[index, "KEBIJAKAN_BBM"] = policy
        if inconsistent:
            planned.at[index, "STATUS_KAPASITAS"] = "DATA_TIDAK_KONSISTEN"
            planned.at[index, "DATA_FLAG"] = (
                "KAPASITAS 0/TIDAK VALID DENGAN KONSUMSI POSITIF"
            )
        elif limited == ["MFO", "BIO"]:
            planned.at[index, "STATUS_KAPASITAS"] = "MFO_DAN_BIO_TERBATAS"
        elif limited:
            planned.at[index, "STATUS_KAPASITAS"] = f"{limited[0]}_TERBATAS"
        else:
            planned.at[index, "STATUS_KAPASITAS"] = "CUKUP"

    return planned


def load_schedule() -> pd.DataFrame:
    schedule = pd.read_excel(SCHEDULE_FILE, dtype=str)
    schedule.columns = schedule.columns.str.strip()
    schedule["VESSELID"] = schedule["VESSELID"].str.strip().str.upper()
    # Kapal yang voyage-nya berformat non-SPIL (Meratus) otomatis dikecualikan,
    # selain daftar manual EXCLUDED_VESSELS.
    meratus_vessels = set(
        schedule.loc[
            schedule["VOYAGENO"].astype(str).str.match(MERATUS_VOYAGE_RE, na=False),
            "VESSELID",
        ]
    )
    excluded = EXCLUDED_VESSELS | meratus_vessels
    schedule = schedule[~schedule["VESSELID"].isin(excluded)].reset_index(drop=True)
    schedule["POL"] = schedule["POL"].str.strip().str.upper()
    schedule["POD"] = schedule["POD"].str.strip().str.upper()
    schedule["NMILE_NUM"] = pd.to_numeric(schedule["NMILE"], errors="coerce")
    for col in ["ETA_POD", "ATA_POD", "ETB_POD", "ATB_POD", "ETD_POD", "ATD_POD"]:
        schedule[f"{col}_DT"] = parse_datetime(schedule[col])
    schedule["ARRIVAL_ORDER"] = schedule["ATA_POD_DT"].fillna(schedule["ETA_POD_DT"])
    schedule["EXCEL_ROW"] = np.arange(2, len(schedule) + 2)
    return schedule


def normalize_route(raw_route: Any) -> list[str]:
    """Return physical route ports; local/virtual calls do not create movement."""
    ports: list[str] = []
    if not isinstance(raw_route, str):
        return ports
    for raw_port in raw_route.split("-"):
        port = raw_port.strip().upper()
        if not port or port in VIRTUAL_PORTS:
            continue
        if not ports or ports[-1] != port:
            ports.append(port)
    return ports


def voyage_sort_key(voyage: Any) -> tuple[int, int, str]:
    """Sort 05/2026, 111A/2026, etc. by year, numeric voyage, suffix."""
    text = str(voyage).strip().upper()
    match = re.match(r"^(\d+)([^/]*)/(\d{4})$", text)
    if match:
        return int(match.group(3)), int(match.group(1)), match.group(2)
    return 9999, 999999, text


def build_events(
    schedule: pd.DataFrame,
    consumption: pd.DataFrame,
    simulation_start: pd.Timestamp | None = None,
    special_checkpoints_by_vessel: dict[str, set[str]] | None = None,
    main_checkpoint_priority: str = "IDSUB",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one checkpoint-to-checkpoint demand interval per vessel-voyage.

    Checkpoint utama dipilih per vesvoy berdasarkan `main_checkpoint_priority`;
    port utama lain menjadi fallback. Default-nya IDSUB bila full sailing route
    memuat Surabaya, selain itu IDJKT. Karena marker seluruh vesvoy kemudian
    disusun berantai, interval dapat berupa SUB→SUB, SUB→JKT, JKT→JKT, atau
    JKT→SUB sesuai checkpoint pilihan vesvoy berikutnya.
    Checkpoint cabang tetap menjadi batas operasional. Route origin adalah
    checkpoint sebelumnya dan tidak dihitung sebagai kedatangan baru. Kebutuhan
    tiap event dihitung dari SEGMEN
    BERIKUTNYA (checkpoint saat ini -> checkpoint berikutnya): kapal mengisi BBM
    saat berthing di checkpoint ini untuk berlayar ke checkpoint berikutnya,
    bukan untuk segmen yang sudah ditempuh. FULL_SAILING_ROUTE tetap berisi rute
    asli voyage tempat checkpoint saat ini berada; SAILING_ROUTE_CHECKPOINT_TO_
    CHECKPOINT berisi rute segmen berikutnya. Voyage terakhir tanpa checkpoint
    lanjutan diberi status DATA ISSUE.
    """
    event_start = SIMULATION_START if simulation_start is None else pd.Timestamp(simulation_start)
    special_checkpoints_by_vessel = special_checkpoints_by_vessel or {}
    if main_checkpoint_priority not in TANKERS:
        raise ValueError(
            f"Prioritas checkpoint tidak dikenal: {main_checkpoint_priority}"
        )
    fallback_main_checkpoint = (
        "IDJKT" if main_checkpoint_priority == "IDSUB" else "IDSUB"
    )
    consumption_lookup = consumption.set_index("VESSEL").to_dict("index")
    event_rows: list[dict[str, Any]] = []
    issue_rows: list[dict[str, Any]] = []
    event_no = 0

    schedule = schedule.copy()
    schedule["VOYAGE_KEY"] = schedule["VOYAGENO"].map(voyage_sort_key)

    for vessel, vessel_data in schedule.groupby("VESSELID", sort=False):
        voyage_groups: list[dict[str, Any]] = []
        grouped = vessel_data.groupby(["VOYAGENO", "FULL_SAILING_ROUTE"], dropna=False, sort=False)
        for (voyage, raw_route), raw_group in grouped:
            route = normalize_route(raw_route)
            if not route:
                continue
            special_checkpoints = special_checkpoints_by_vessel.get(vessel, set())
            # Prioritas checkpoint berlaku per vesvoy, bukan berdasarkan urutan
            # kemunculan JKT/SUB. Port prioritas dipakai bila terdapat di route;
            # port utama lainnya menjadi fallback.
            preferred_main_checkpoint = (
                main_checkpoint_priority
                if main_checkpoint_priority in route
                else fallback_main_checkpoint
            )
            checkpoint_candidates = [
                (index, port)
                for index, port in enumerate(route[1:], start=1)
                if port == preferred_main_checkpoint or port in special_checkpoints
            ]

            unused = list(raw_group.sort_values("EXCEL_ROW", kind="stable").to_dict("records"))
            legs: list[dict[str, Any]] = []
            for pol, pod in zip(route[:-1], route[1:]):
                match_index = next(
                    (index for index, row in enumerate(unused) if row["POL"] == pol and row["POD"] == pod),
                    None,
                )
                if match_index is None:
                    legs.append({"POL": pol, "POD": pod, "NMILE_NUM": np.nan, "MISSING": True})
                else:
                    row = unused.pop(match_index)
                    row["MISSING"] = False
                    legs.append(row)

            voyage_groups.append(
                {
                    "voyage": voyage,
                    "voyage_key": voyage_sort_key(voyage),
                    "raw_route": raw_route,
                    "route": route,
                    "checkpoint_candidates": checkpoint_candidates,
                    "legs": legs,
                    "extra_rows": unused,
                }
            )

        voyage_groups.sort(key=lambda item: item["voyage_key"])

        # A branch checkpoint closes the prior voyage only when the next voyage
        # demonstrably starts at the same branch port.
        for group_index, voyage_group in enumerate(voyage_groups[:-1]):
            next_group = voyage_groups[group_index + 1]
            route = voyage_group["route"]
            next_route = next_group["route"]
            if (
                not voyage_group["checkpoint_candidates"]
                and route
                and next_route
                and route[-1] == next_route[0]
                and route[-1] in BRANCH_CHECKPOINTS
            ):
                voyage_group["checkpoint_candidates"].append(
                    (len(route) - 1, route[-1])
                )

        all_legs: list[dict[str, Any]] = []
        markers: list[dict[str, Any]] = []

        for voyage_group in voyage_groups:
            offset = len(all_legs)
            all_legs.extend(voyage_group["legs"])
            for checkpoint_index, checkpoint in voyage_group["checkpoint_candidates"]:
                position = offset + int(checkpoint_index)
                inbound = all_legs[position - 1] if position > 0 else None
                if inbound is not None and inbound.get("POD") != checkpoint:
                    inbound = None
                marker = {
                    **voyage_group,
                    "checkpoint": checkpoint,
                    "checkpoint_index": checkpoint_index,
                    "position": position,
                    "inbound": inbound,
                }
                # The end of one voyage and start of the next can describe the
                # same physical port call. Keep the first marker (the voyage that
                # arrives at the checkpoint) so it does not create a zero-NM event.
                if (
                    markers
                    and markers[-1]["position"] == position
                    and markers[-1]["checkpoint"] == checkpoint
                ):
                    if markers[-1]["inbound"] is None and marker["inbound"] is not None:
                        markers[-1]["inbound"] = marker["inbound"]
                else:
                    markers.append(marker)

        # Kandidat utama sudah dipilih per vesvoy. Tidak ada lagi eliminasi
        # berdasarkan minimum dua leg antara JKT dan SUB; checkpoint cabang
        # tetap berada dalam urutan dan dapat memotong interval utama.

        for marker_index in range(len(markers)):
            marker = markers[marker_index]
            previous_marker = markers[marker_index - 1] if marker_index > 0 else None
            next_marker = markers[marker_index + 1] if marker_index + 1 < len(markers) else None
            previous_inbound = previous_marker["inbound"] if previous_marker else None
            inbound = marker["inbound"]

            ata = inbound.get("ATA_POD_DT", pd.NaT) if inbound else pd.NaT
            atb = inbound.get("ATB_POD_DT", pd.NaT) if inbound else pd.NaT
            if pd.isna(atb):
                atb = inbound.get("ETB_POD_DT", pd.NaT) if inbound else pd.NaT
            if pd.isna(atb):
                atb = ata
            # Pengisian hanya saat kapal berthing: jendela periode mengikuti ATB.
            # Event setelah akhir periode tetap dibentuk sebagai look-ahead agar
            # voyage direct terakhir dalam periode dapat dipasangkan. Event
            # look-ahead dibuang lagi setelah proses penggabungan.
            if pd.isna(atb) or atb < event_start:
                continue

            event_no += 1
            next_inbound = next_marker["inbound"] if next_marker else None
            next_checkpoint_time = next_inbound.get("ARRIVAL_ORDER", pd.NaT) if next_inbound else pd.NaT

            # Kebutuhan = segmen BERIKUTNYA (checkpoint saat ini -> checkpoint
            # berikutnya). Kapal mengisi BBM saat berthing di checkpoint ini untuk
            # berlayar ke checkpoint berikutnya, bukan untuk segmen yang sudah
            # ditempuh. Voyage terakhir tanpa checkpoint lanjutan -> DATA ISSUE.
            if next_marker is not None:
                segment = all_legs[marker["position"] : next_marker["position"]]
            else:
                segment = []

            sailing_ports = [marker["checkpoint"]]
            discontinuities: list[str] = []
            for leg in segment:
                if sailing_ports[-1] != leg["POL"]:
                    discontinuities.append(f"{sailing_ports[-1]}→{leg['POL']}")
                    sailing_ports.append(leg["POL"])
                sailing_ports.append(leg["POD"])
            sailing_route = "-".join(sailing_ports)

            missing_legs = [leg for leg in segment if leg.get("MISSING") or pd.isna(leg.get("NMILE_NUM"))]
            valid_distance = bool(segment) and not missing_legs
            distance_nm = sum(float(leg["NMILE_NUM"]) for leg in segment) if valid_distance else np.nan

            deadline = inbound.get("ATD_POD_DT", pd.NaT) if inbound else pd.NaT
            deadline_source = "ATD"
            if pd.isna(deadline):
                deadline = inbound.get("ETD_POD_DT", pd.NaT) if inbound else pd.NaT
                deadline_source = "ETD (FALLBACK)" if pd.notna(deadline) else "TIDAK ADA"

            cons = consumption_lookup[vessel]
            req_mfo = distance_nm * cons["MFO_L_PER_NM"] / 1000.0 if valid_distance else np.nan
            req_bio = distance_nm * cons["BIO_L_PER_NM"] / 1000.0 if valid_distance else np.nan

            data_flags: list[str] = []
            notes: list[str] = []
            if next_marker is None:
                data_flags.append("TIDAK ADA CHECKPOINT BERIKUTNYA")
                notes.append(
                    "Voyage terakhir tanpa checkpoint lanjutan; kebutuhan segmen berikutnya tidak dapat dihitung"
                )
            if missing_legs:
                data_flags.append("DATA SEGMEN VOYAGE TIDAK LENGKAP")
                notes.append("Missing: " + ", ".join(f"{leg['POL']}-{leg['POD']}" for leg in missing_legs))
            if discontinuities:
                data_flags.append("SEQUENCE ANTAR VOYAGE TIDAK TERSAMBUNG")
                notes.append("Gap: " + ", ".join(discontinuities))
            if not valid_distance or distance_nm <= 0:
                data_flags.append("DATA JARAK TIDAK CUKUP")
            if pd.isna(deadline):
                data_flags.append("DEADLINE TIDAK TERSEDIA")
            if pd.notna(deadline) and deadline < ata:
                data_flags.append("ATD SEBELUM ATA")
            # Preserve order while removing duplicate flag text.
            data_flags = list(dict.fromkeys(data_flags))

            event_id = f"EVT-{event_no:04d}"
            source_row = int(inbound["EXCEL_ROW"]) if inbound and "EXCEL_ROW" in inbound else 0
            supply_checkpoint = marker["checkpoint"]
            event_rows.append(
                {
                    "EVENT_ID": event_id,
                    "EXCEL_ROW": source_row,
                    "VESSEL": vessel,
                    "FLEET": cons["FLEET"],
                    "TANKER": (
                        TANKERS[supply_checkpoint].name
                        if supply_checkpoint in TANKERS
                        else "CABANG"
                    ),
                    "CHECKPOINT": marker["checkpoint"],
                    "CHECKPOINT_BERIKUT": next_marker["checkpoint"] if next_marker else "",
                    "VOYAGENO": marker["voyage"],
                    "CHECKPOINT_SEBELUM": previous_marker["checkpoint"] if previous_marker else "",
                    "ATD_CHECKPOINT_SEBELUM": (
                        previous_inbound.get("ATD_POD_DT", pd.NaT) if previous_inbound else pd.NaT
                    ),
                    "ATA": ata,
                    "ATB": atb,
                    "ATD": inbound.get("ATD_POD_DT", pd.NaT) if inbound else pd.NaT,
                    "ETD": inbound.get("ETD_POD_DT", pd.NaT) if inbound else pd.NaT,
                    "DEADLINE": deadline,
                    "SUMBER_DEADLINE": deadline_source,
                    "NEXT_CHECKPOINT": next_checkpoint_time,
                    "FULL_SAILING_ROUTE": marker["raw_route"],
                    "SAILING_ROUTE_CHECKPOINT_TO_CHECKPOINT": sailing_route,
                    "JARAK_KE_CHECKPOINT_NM": float(distance_nm) if valid_distance else np.nan,
                    "SUMBER_JARAK": "AKUMULASI NMILE CHECKPOINT-TO-CHECKPOINT ANTAR VOYAGE",
                    "MFO_L_PER_NM": cons["MFO_L_PER_NM"],
                    "SUMBER_MFO": cons["SUMBER_MFO"],
                    "BIO_L_PER_NM": cons["BIO_L_PER_NM"],
                    "SUMBER_BIO": cons["SUMBER_BIO"],
                    "KEBUTUHAN_MFO_KL": req_mfo,
                    "KEBUTUHAN_BIO_KL": req_bio,
                    "DATA_FLAG": "; ".join(data_flags) if data_flags else "OK",
                }
            )
            if data_flags:
                issue_rows.append(
                    {
                        "EVENT_ID": event_id,
                        "VESSEL": vessel,
                        "CHECKPOINT": marker["checkpoint"],
                        "ATA": ata,
                        "MASALAH": "; ".join(data_flags),
                        "KETERANGAN": "; ".join(notes) or sailing_route,
                        "SOURCE_EXCEL_ROW": source_row,
                    }
                )

    return pd.DataFrame(event_rows), pd.DataFrame(issue_rows)


def renumber_events_chronologically(
    events: pd.DataFrame, issues: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Make EVENT_ID follow global ATB (berthing) → ATD/deadline → vessel order."""
    ordered = events.sort_values(
        ["ATB", "DEADLINE", "VESSEL", "EXCEL_ROW"], na_position="last", kind="stable"
    ).reset_index(drop=True)
    old_ids = ordered["EVENT_ID"].copy()
    ordered["EVENT_ID"] = [f"EVT-{number:04d}" for number in range(1, len(ordered) + 1)]
    id_map = dict(zip(old_ids, ordered["EVENT_ID"]))
    if not issues.empty:
        issues = issues.copy()
        issues["EVENT_ID"] = issues["EVENT_ID"].map(id_map)
        issues = issues.sort_values("EVENT_ID").reset_index(drop=True)
    return ordered, issues


def hours_between(end: pd.Timestamp, start: pd.Timestamp) -> float:
    return max(0.0, (end - start).total_seconds() / 3600.0)


def refill_duration(spec: TankerSpec, stock_mfo: float, stock_bio: float) -> float:
    """Lama pompa mengisi tanker sampai penuh (tanpa waktu perpindahan)."""
    mfo_hours = max(0.0, spec.capacity_mfo - stock_mfo) / spec.load_rate_mfo
    bio_hours = max(0.0, spec.capacity_bio - stock_bio) / spec.load_rate_bio
    pumping_hours = max(mfo_hours, bio_hours) if spec.load_simultaneous else mfo_hours + bio_hours
    return max(0.0, pumping_hours)


def refill_amounts(
    spec: TankerSpec, stock_mfo: float, stock_bio: float
) -> tuple[float, float]:
    """Volume yang masuk bila kedua tangki tanker dipenuhkan."""
    return (
        max(0.0, spec.capacity_mfo - stock_mfo),
        max(0.0, spec.capacity_bio - stock_bio),
    )


def refill_meets_minimum(
    spec: TankerSpec, stock_mfo: float, stock_bio: float
) -> bool:
    """Loading sah bila MFO >=250 KL atau BIO >=100 KL."""
    in_mfo, in_bio = refill_amounts(spec, stock_mfo, stock_bio)
    return (
        in_mfo >= MIN_LOADING_MFO_KL - 1e-9
        or in_bio >= MIN_LOADING_BIO_KL - 1e-9
    )


def simulate_tanker(spec: TankerSpec, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Gunakan satu state machine bersama agar FIFO dan EDD tunduk pada aturan
    # minimum loading, drain-first, serta jeda perjalanan refill yang sama.
    from . import queue as common

    return common.simulate_tanker_priority(
        spec,
        events,
        mode="FIFO",
        lookahead_wait=False,
        simulation_start=SIMULATION_START,
    )

    # Implementasi lama dipertahankan di bawah sebagai referensi historis,
    # tetapi tidak dieksekusi pada skenario terisolasi ini.
    valid = events[(events["DATA_FLAG"] == "OK")].copy()
    valid = valid.sort_values(["ATB", "EVENT_ID"]).reset_index(drop=True)
    invalid = events[events["DATA_FLAG"] != "OK"].copy()

    now = SIMULATION_START
    stock_mfo = spec.capacity_mfo
    stock_bio = spec.capacity_bio
    # Lokasi tanker saat ini: LABUH (anchorage), PERTAMINA, atau SANDAR (di
    # samping kapal). Waktu pindah antar lokasi mengikuti matriks move_times.
    location = LOC_LABUH
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
                "DURASI_JAM": hours_between(end, start),
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
        """Pindahkan tanker dari lokasi sekarang ke destination (pakai matriks waktu)."""
        nonlocal now, location
        duration = spec.move_time(location, destination)
        if destination == LOC_SANDAR and location == LOC_SANDAR:
            op_type = "PINDAH ANTAR KAPAL"
        elif destination == LOC_SANDAR:
            op_type = "MANUVER & PERSIAPAN KE KAPAL"
        elif destination == LOC_LABUH:
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
                op_type,
                start,
                end,
                stock_mfo,
                stock_bio,
                stock_mfo,
                stock_bio,
                event,
                note=note,
            )
            now = end
        location = destination

    def do_refill_pump(note: str, event: dict[str, Any] | None = None) -> None:
        """Pompa mengisi tanker sampai penuh (diasumsikan sudah di Pertamina)."""
        nonlocal now, stock_mfo, stock_bio
        duration = refill_duration(spec, stock_mfo, stock_bio)
        if duration <= 1e-9:
            return
        start = now
        end = start + pd.Timedelta(hours=duration)
        before_mfo, before_bio = stock_mfo, stock_bio
        in_mfo = spec.capacity_mfo - stock_mfo
        in_bio = spec.capacity_bio - stock_bio
        stock_mfo, stock_bio = spec.capacity_mfo, spec.capacity_bio
        add_operation(
            "REFILL PERTAMINA",
            start,
            end,
            before_mfo,
            before_bio,
            stock_mfo,
            stock_bio,
            event,
            in_mfo=in_mfo,
            in_bio=in_bio,
            note=note,
        )
        now = end

    def do_refill(note: str, event: dict[str, Any] | None = None) -> None:
        """Pindah ke Pertamina lalu pompa sampai penuh; tanker berakhir di Pertamina."""
        nonlocal location
        if refill_duration(spec, stock_mfo, stock_bio) <= 1e-9:
            return
        do_move(LOC_PERTAMINA, event, note=f"{note} (menuju Pertamina)")
        do_refill_pump(note, event)

    while unscheduled:
        available = [event for event in unscheduled if event["ATB"] <= now]
        if not available:
            next_release = min(event["ATB"] for event in unscheduled)
            next_event = min(unscheduled, key=lambda item: (item["ATB"], item["EVENT_ID"]))
            need_refill = (
                stock_mfo < spec.min_stock_mfo - 1e-9
                or stock_bio < spec.min_stock_bio - 1e-9
                or next_event["KEBUTUHAN_MFO_KL"] > stock_mfo + 1e-9
                or next_event["KEBUTUHAN_BIO_KL"] > stock_bio + 1e-9
            )
            # Refill look-ahead saat idle, bila selesai sebelum kapal berikutnya tiba.
            # Setelah refill tanker tetap di Pertamina agar siap melayani kapal besar.
            if need_refill:
                total_refill = (
                    spec.move_time(location, LOC_PERTAMINA)
                    + refill_duration(spec, stock_mfo, stock_bio)
                )
                if total_refill > 1e-9 and now + pd.Timedelta(hours=total_refill) <= next_release:
                    do_refill("REFILL LOOK-AHEAD SAAT IDLE", next_event)
            now = max(now, next_release)
            continue

        event = min(available, key=lambda item: (item["ATB"], item["EVENT_ID"]))
        unscheduled.remove(event)
        remaining_mfo = float(event["KEBUTUHAN_MFO_KL"])
        remaining_bio = float(event["KEBUTUHAN_BIO_KL"])
        delivery_segments: list[dict[str, Any]] = []
        first_delivery_start: pd.Timestamp | None = None
        stock_before_bunker_mfo: float | None = None
        stock_before_bunker_bio: float | None = None

        need_bunker = remaining_mfo > 1e-9 or remaining_bio > 1e-9

        # Refill look-ahead sebelum melayani: bila stok menyentuh batas minimum
        # ATAU kebutuhan kapal ini melebihi stok (pre-fill kapal besar agar tidak
        # terjadi refill di tengah pengisian). Setelah refill tanker berada di
        # Pertamina dan langsung bergerak ke kapal (PERTAMINA → SANDAR) tanpa
        # kembali ke base.
        if (
            stock_mfo < spec.min_stock_mfo - 1e-9
            or stock_bio < spec.min_stock_bio - 1e-9
            or remaining_mfo > stock_mfo + 1e-9
            or remaining_bio > stock_bio + 1e-9
        ):
            do_refill("REFILL LOOK-AHEAD SEBELUM MELAYANI", event)

        # MANUVER & PERSIAPAN KE KAPAL (approach + izin + pandu + tambat/persiapan)
        # dihitung dari lokasi tanker saat ini ke SANDAR. Jika tanker sudah di
        # area sandar (pindah langsung antar kapal), waktu shift yang dipakai.
        if need_bunker:
            do_move(LOC_SANDAR, event, note="Menuju kapal sebelum memompa")

        while remaining_mfo > 1e-9 or remaining_bio > 1e-9:
            if (remaining_mfo > 1e-9 and stock_mfo <= 1e-9) or (
                remaining_bio > 1e-9 and stock_bio <= 1e-9
            ):
                # Refill lanjutan: tanker di area kapal, pergi ke Pertamina, isi
                # penuh, lalu kembali ke kapal yang sama untuk melanjutkan.
                do_move(LOC_PERTAMINA, event, note="Refill lanjutan (menuju Pertamina)")
                do_refill_pump("REFILL LANJUTAN UNTUK KAPAL YANG SAMA", event)
                do_move(LOC_SANDAR, event, note="Refill lanjutan (kembali ke kapal yang sama)")

            chunk_mfo = min(remaining_mfo, stock_mfo)
            chunk_bio = min(remaining_bio, stock_bio)
            mfo_hours = chunk_mfo / spec.discharge_rate_mfo if chunk_mfo > 0 else 0.0
            bio_hours = chunk_bio / spec.discharge_rate_bio if chunk_bio > 0 else 0.0
            duration = max(mfo_hours, bio_hours)
            if duration <= 1e-12:
                raise RuntimeError(f"Tidak dapat melanjutkan event {event['EVENT_ID']}")

            start = now
            end = start + pd.Timedelta(hours=duration)
            if first_delivery_start is None:
                first_delivery_start = start
                stock_before_bunker_mfo = stock_mfo
                stock_before_bunker_bio = stock_bio
            before_mfo, before_bio = stock_mfo, stock_bio
            stock_mfo -= chunk_mfo
            stock_bio -= chunk_bio
            stock_mfo = max(0.0, stock_mfo)
            stock_bio = max(0.0, stock_bio)
            remaining_mfo -= chunk_mfo
            remaining_bio -= chunk_bio
            delivery_segments.append(
                {
                    "start": start,
                    "end": end,
                    "mfo": chunk_mfo,
                    "bio": chunk_bio,
                    "mfo_hours": mfo_hours,
                    "bio_hours": bio_hours,
                }
            )
            add_operation(
                "BUNKER KE KAPAL",
                start,
                end,
                before_mfo,
                before_bio,
                stock_mfo,
                stock_bio,
                event,
                out_mfo=chunk_mfo,
                out_bio=chunk_bio,
                note="MFO dan BIO dipompa bersamaan",
            )
            now = end

        deadline = event["DEADLINE"]
        delivered_mfo = 0.0
        delivered_bio = 0.0
        for segment in delivery_segments:
            available_hours = max(0.0, (deadline - segment["start"]).total_seconds() / 3600.0)
            delivered_mfo += min(segment["mfo"], available_hours * spec.discharge_rate_mfo)
            delivered_bio += min(segment["bio"], available_hours * spec.discharge_rate_bio)

        req_mfo = float(event["KEBUTUHAN_MFO_KL"])
        req_bio = float(event["KEBUTUHAN_BIO_KL"])
        shortage_mfo = max(0.0, req_mfo - delivered_mfo)
        shortage_bio = max(0.0, req_bio - delivered_bio)
        completion = now
        delay = max(0.0, (completion - deadline).total_seconds() / 3600.0)
        wait = max(0.0, (first_delivery_start - event["ATB"]).total_seconds() / 3600.0)
        if wait <= 1e-9:
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
                operation["JENIS_OPERASI"] == "MANUVER & PERSIAPAN KE KAPAL"
                for operation in wait_operations
            ):
                reasons.append("PROSES APPROACH & PERSIAPAN")
            if any(
                operation["JENIS_OPERASI"] == "KEMBALI KE BASE"
                for operation in wait_operations
            ):
                reasons.append("KEMBALI KE BASE / SIKLUS KELUAR-MASUK")
            wait_reason = " + ".join(reasons) if reasons else "MENUNGGU KETERSEDIAAN TANKER"
        feasible = shortage_mfo <= 1e-6 and shortage_bio <= 1e-6 and completion <= deadline

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
                "DURASI_PROSES_JAM": (completion - first_delivery_start).total_seconds() / 3600.0,
                "TERISI_SBLM_ATD_MFO_KL": delivered_mfo,
                "TERISI_SBLM_ATD_BIO_KL": delivered_bio,
                "SHORTAGE_MFO_KL": shortage_mfo,
                "SHORTAGE_BIO_KL": shortage_bio,
                "KETERLAMBATAN_JAM": delay,
                "STATUS": "FEASIBLE" if feasible else "NOT FEASIBLE",
                "ALASAN_FLAG": "OK"
                if feasible
                else "Pengisian tidak selesai seluruhnya sebelum deadline",
            }
        )
        results.append(result)

        # Setelah bunker, tentukan posisi tanker berikutnya berdasarkan antrean
        # dan sisa stok:
        # - Perlu refill: pindah SANDAR → PERTAMINA, isi penuh; bila ada kapal
        #   menunggu tanker tetap di Pertamina, bila tidak kembali ke base.
        # - Ada kapal menunggu & stok cukup: tanker TETAP di area sandar; kapal
        #   berikutnya membayar shift (SANDAR → SANDAR).
        # - Tidak ada antrean: kembali ke base (LABUH).
        if need_bunker:
            need_refill_now = (
                stock_mfo < spec.min_stock_mfo - 1e-9
                or stock_bio < spec.min_stock_bio - 1e-9
            )
            next_waiting = any(event_queued["ATB"] <= now for event_queued in unscheduled)
            if need_refill_now:
                do_refill("REFILL MINIMUM STOK SETELAH MELAYANI", event)
                if not any(event_queued["ATB"] <= now for event_queued in unscheduled):
                    do_move(LOC_LABUH, event, note="Pulang ke base setelah refill (idle)")
            elif next_waiting:
                pass  # tetap di SANDAR; kapal berikutnya membayar shift
            else:
                do_move(LOC_LABUH, event, note="Kembali ke base setelah bunker (idle)")

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


def build_branch_results(events: pd.DataFrame) -> pd.DataFrame:
    """Report branch supply without consuming SIGMA/ZETA time or stock."""
    results: list[dict[str, Any]] = []
    for _, event in events.iterrows():
        result = event.to_dict()
        valid = result["DATA_FLAG"] == "OK"
        result.update(
            {
                "MULAI_BUNKER": pd.NaT,
                "SELESAI_BUNKER": pd.NaT,
                "WAKTU_TUNGGU_JAM": 0.0 if valid else np.nan,
                "REASON_WAKTU_TUNGGU": "PENGISIAN CABANG - DI LUAR SIMULASI TANKER" if valid else "DATA ISSUE",
                "STOK_AWAL_MFO_TANKER_KL": np.nan,
                "STOK_AWAL_BIO_TANKER_KL": np.nan,
                "DURASI_PROSES_JAM": np.nan,
                "TERISI_SBLM_ATD_MFO_KL": result["KEBUTUHAN_MFO_KL"] if valid else 0.0,
                "TERISI_SBLM_ATD_BIO_KL": result["KEBUTUHAN_BIO_KL"] if valid else 0.0,
                "SHORTAGE_MFO_KL": 0.0 if valid else result.get("KEBUTUHAN_MFO_KL", np.nan),
                "SHORTAGE_BIO_KL": 0.0 if valid else result.get("KEBUTUHAN_BIO_KL", np.nan),
                "KETERLAMBATAN_JAM": 0.0 if valid else np.nan,
                "STATUS": "FEASIBLE" if valid else "DATA ISSUE",
                "ALASAN_FLAG": "PENGISIAN CABANG; TIDAK MEMPENGARUHI SIGMA/ZETA" if valid else result["DATA_FLAG"],
            }
        )
        results.append(result)
    return pd.DataFrame(results)


def build_summary(results: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for tanker in ["SIGMA", "ZETA", "CABANG", "TOTAL"]:
        r = (
            results[results["TANKER"].isin(["SIGMA", "ZETA"])]
            if tanker == "TOTAL"
            else results[results["TANKER"] == tanker]
        )
        l = ledger if tanker == "TOTAL" else ledger[ledger["TANKER"] == tanker]
        valid = r[r["STATUS"].isin(["FEASIBLE", "NOT FEASIBLE"])]
        rows.append(
            {
                "TANKER": tanker,
                "TOTAL_EVENT": len(r),
                "EVENT_VALID": len(valid),
                "FEASIBLE": int((r["STATUS"] == "FEASIBLE").sum()),
                "NOT_FEASIBLE": int((r["STATUS"] == "NOT FEASIBLE").sum()),
                "DATA_ISSUE": int((r["STATUS"] == "DATA ISSUE").sum()),
                "KEBUTUHAN_MFO_KL": valid["KEBUTUHAN_MFO_KL"].sum(),
                "KEBUTUHAN_BIO_KL": valid["KEBUTUHAN_BIO_KL"].sum(),
                "SHORTAGE_MFO_KL": valid["SHORTAGE_MFO_KL"].sum(),
                "SHORTAGE_BIO_KL": valid["SHORTAGE_BIO_KL"].sum(),
                "MAX_TUNGGU_JAM": valid["WAKTU_TUNGGU_JAM"].max(),
                "MAX_TERLAMBATAN_JAM": valid["KETERLAMBATAN_JAM"].max(),
                "JUMLAH_REFILL": int((l["JENIS_OPERASI"] == "REFILL PERTAMINA").sum()) if not l.empty else 0,
                "TOTAL_REFILL_MFO_KL": l["MASUK_MFO_KL"].sum() if not l.empty else 0.0,
                "TOTAL_REFILL_BIO_KL": l["MASUK_BIO_KL"].sum() if not l.empty else 0.0,
                "STOK_MIN_MFO_KL": l["STOK_AKHIR_MFO_KL"].min() if not l.empty else np.nan,
                "STOK_MIN_BIO_KL": l["STOK_AKHIR_BIO_KL"].min() if not l.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _idle_wait_row(start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    return {
        "OPERATION_ID": "",
        "JENIS_OPERASI": "IDLE (TIDAK ADA OPERASI)",
        "OP_EVENT_ID": "",
        "OP_VESSEL": "",
        "OP_MULAI": start,
        "OP_SELESAI": end,
        "DURASI_OPERASI_JAM": hours_between(end, start),
        "DURASI_DLM_TUNGGU_JAM": hours_between(end, start),
    }


def build_wait_breakdown(results: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    """Rincian per event: operasi tanker apa saja yang mengisi waktu tunggu.

    Untuk setiap event valid dengan WAKTU_TUNGGU > 0, daftarkan seluruh operasi
    tanker yang terjadi di jendela [ATB, MULAI_BUNKER) beserta durasinya dalam
    jendela tersebut. Sisa waktu yang tidak tertutup operasi dicatat sebagai IDLE.
    Jumlah DURASI_DLM_TUNGGU_JAM = WAKTU_TUNGGU_JAM sehingga dapat diaudit.
    """
    columns = [
        "EVENT_ID", "VESSEL", "TANKER", "CHECKPOINT", "ATB", "MULAI_BUNKER",
        "WAKTU_TUNGGU_JAM", "OPERATION_ID", "JENIS_OPERASI", "OP_EVENT_ID",
        "OP_VESSEL", "OP_MULAI", "OP_SELESAI", "DURASI_OPERASI_JAM",
        "DURASI_DLM_TUNGGU_JAM", "KUMULATIF_JAM",
    ]
    rows: list[dict[str, Any]] = []
    valid = results[results["STATUS"] != "DATA ISSUE"].copy()
    for _, event in valid.iterrows():
        wait = float(event.get("WAKTU_TUNGGU_JAM", np.nan))
        if not np.isfinite(wait) or wait <= 1e-9:
            continue
        atb = event["ATB"]
        mulai_bunker = event["MULAI_BUNKER"]
        if pd.isna(atb) or pd.isna(mulai_bunker):
            continue
        tanker = event["TANKER"]
        ops = ledger[ledger["TANKER"] == tanker]

        window_ops: list[dict[str, Any]] = []
        for _, op in ops.iterrows():
            op_start, op_end = op["MULAI"], op["SELESAI"]
            if op_end <= atb or op_start >= mulai_bunker:
                continue
            overlap = hours_between(min(op_end, mulai_bunker), max(op_start, atb))
            if overlap <= 1e-9:
                continue
            window_ops.append(
                {
                    "OPERATION_ID": op["OPERATION_ID"],
                    "JENIS_OPERASI": op["JENIS_OPERASI"],
                    "OP_EVENT_ID": op["EVENT_ID"],
                    "OP_VESSEL": op["VESSEL"],
                    "OP_MULAI": op_start,
                    "OP_SELESAI": op_end,
                    "DURASI_OPERASI_JAM": float(op["DURASI_JAM"]),
                    "DURASI_DLM_TUNGGU_JAM": overlap,
                }
            )
        window_ops.sort(key=lambda item: item["OP_MULAI"])

        # Isi celah waktu yang tidak tertutup operasi sebagai IDLE.
        cursor = atb
        idle_rows: list[dict[str, Any]] = []
        for op in window_ops:
            if op["OP_MULAI"] > cursor + pd.Timedelta(microseconds=1):
                idle_rows.append(_idle_wait_row(cursor, op["OP_MULAI"]))
            cursor = max(cursor, op["OP_SELESAI"])
        if mulai_bunker > cursor + pd.Timedelta(microseconds=1):
            idle_rows.append(_idle_wait_row(cursor, mulai_bunker))

        combined = idle_rows + window_ops
        combined.sort(key=lambda item: item["OP_MULAI"])

        cumulative = 0.0
        for item in combined:
            cumulative += item["DURASI_DLM_TUNGGU_JAM"]
            rows.append(
                {
                    "EVENT_ID": event["EVENT_ID"],
                    "VESSEL": event["VESSEL"],
                    "TANKER": tanker,
                    "CHECKPOINT": event["CHECKPOINT"],
                    "ATB": atb,
                    "MULAI_BUNKER": mulai_bunker,
                    "WAKTU_TUNGGU_JAM": round(wait, 4),
                    **item,
                    "KUMULATIF_JAM": round(cumulative, 4),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def validate_simulation(results: pd.DataFrame, ledger: pd.DataFrame) -> None:
    """Fail fast if the generated ledger violates core stock/scheduling rules."""
    if results["EVENT_ID"].duplicated().any():
        raise AssertionError("EVENT_ID duplikat pada hasil simulasi")
    valid_routes = results[results["STATUS"] != "DATA ISSUE"]
    for event in valid_routes.itertuples(index=False):
        route = str(event.SAILING_ROUTE_CHECKPOINT_TO_CHECKPOINT).split("-")
        if route[0] != event.CHECKPOINT or route[-1] != event.CHECKPOINT_BERIKUT:
            raise AssertionError(
                f"Endpoint sailing route {event.EVENT_ID} tidak sesuai checkpoint"
            )
    for event in valid_routes.itertuples(index=False):
        for fuel in ("MFO", "BIO"):
            actual = float(getattr(event, f"KEBUTUHAN_{fuel}_KL"))
            capacity = float(getattr(event, f"KAPASITAS_TANGKI_{fuel}_KL"))
            if actual > capacity + 1e-6:
                raise AssertionError(
                    f"Pengisian {fuel} kapal {event.VESSEL} melebihi kapasitas"
                )
    deadlines = valid_routes.set_index("EVENT_ID")["DEADLINE"]
    bunker_operations = ledger[ledger["JENIS_OPERASI"] == "BUNKER KE KAPAL"]
    for operation in bunker_operations.itertuples(index=False):
        deadline = deadlines.get(operation.EVENT_ID, pd.NaT)
        if pd.notna(deadline) and operation.SELESAI > deadline + pd.Timedelta(microseconds=1):
            raise AssertionError(
                f"Bunker {operation.EVENT_ID} berlanjut setelah ATD/deadline"
            )
    for spec in TANKERS.values():
        operations = ledger[ledger["TANKER"] == spec.name].sort_values("MULAI").reset_index(drop=True)
        if operations.empty:
            continue
        if (operations["MULAI"].iloc[1:].reset_index(drop=True) < operations["SELESAI"].iloc[:-1].reset_index(drop=True)).any():
            raise AssertionError(f"Aktivitas {spec.name} bertabrakan")
        if (operations["STOK_AKHIR_MFO_KL"] < -1e-6).any() or (operations["STOK_AKHIR_BIO_KL"] < -1e-6).any():
            raise AssertionError(f"Stok {spec.name} menjadi negatif")
        if (operations["STOK_AKHIR_MFO_KL"] > spec.capacity_mfo + 1e-6).any():
            raise AssertionError(f"Stok MFO {spec.name} melebihi kapasitas")
        if (operations["STOK_AKHIR_BIO_KL"] > spec.capacity_bio + 1e-6).any():
            raise AssertionError(f"Stok BIO {spec.name} melebihi kapasitas")
        mfo_balance = (
            operations["STOK_AWAL_MFO_KL"] + operations["MASUK_MFO_KL"]
            - operations["KELUAR_MFO_KL"] - operations["STOK_AKHIR_MFO_KL"]
        ).abs().max()
        bio_balance = (
            operations["STOK_AWAL_BIO_KL"] + operations["MASUK_BIO_KL"]
            - operations["KELUAR_BIO_KL"] - operations["STOK_AKHIR_BIO_KL"]
        ).abs().max()
        if mfo_balance > 1e-6 or bio_balance > 1e-6:
            raise AssertionError(f"Neraca stok {spec.name} tidak seimbang")
        refills = operations[operations["JENIS_OPERASI"] == "REFILL PERTAMINA"]
        if not refills.empty:
            below_minimum = (
                (refills["MASUK_MFO_KL"] < MIN_LOADING_MFO_KL - 1e-6)
                & (refills["MASUK_BIO_KL"] < MIN_LOADING_BIO_KL - 1e-6)
            )
            if below_minimum.any():
                raise AssertionError(f"Loading {spec.name} di bawah minimum")

            refill_rows = list(refills.itertuples(index=False))
            for previous, current in zip(refill_rows, refill_rows[1:]):
                travel = operations[
                    (operations["JENIS_OPERASI"] == "PERJALANAN KE PERTAMINA")
                    & (operations["MULAI"] >= previous.SELESAI)
                    & (operations["SELESAI"] <= current.MULAI)
                ]
                departure = travel["MULAI"].iloc[-1] if not travel.empty else current.MULAI
                earliest = previous.SELESAI + pd.Timedelta(
                    hours=REFILL_TRIP_COOLDOWN_HOURS
                )
                if departure < earliest - pd.Timedelta(microseconds=1):
                    raise AssertionError(
                        f"Perjalanan refill {spec.name} dimulai sebelum jeda 24 jam"
                    )


def parameter_table() -> pd.DataFrame:
    rows = [
        ("PERIODE", "Waktu berthing (ATB) 1 Juni 2026 s.d. 31 Juli 2026"),
        ("DEADLINE", "ATD; ETD hanya menjadi fallback jika ATD kosong"),
        ("ROB KAPAL", "Tidak dimodelkan: ROB awal 0 KL dan pengisian selalu sama dengan penggunaan"),
        ("WAKTU PENGISIAN", "Pengisian hanya dilakukan saat kapal berthing; waktu berthing = ATB, fallback ETB, fallback ATA"),
        ("CHECKPOINT", "Semua kedatangan IDJKT/IDSUB menjadi kandidat. Kandidat utama yang hanya satu leg dari checkpoint terpilih sebelumnya dilewati sebagai event bunker dan pencarian diteruskan ke kandidat berikutnya; port yang dilewati tetap masuk sailing route"),
        ("MINIMUM STEP CHECKPOINT UTAMA", f"{MIN_LEGS_BETWEEN_MAIN_CHECKPOINTS} leg antara checkpoint bunker utama terpilih; checkpoint cabang tetap menjadi batas operasional"),
        ("JARAK KEBUTUHAN", "Akumulasi NMILE dari checkpoint voyage saat ini sampai checkpoint tujuan voyage berikutnya untuk vessel yang sama (kebutuhan segmen yang akan datang, bukan segmen yang sudah ditempuh)"),
        ("PENGGABUNGAN RUTE DIRECT", "Dua event STEP=2 berurutan dan tersambung digabung hanya jika target 130% MFO dan BIO sama-sama muat. Jika gagal, event pertama diproses sendiri dan event kedua tetap dapat diuji dengan event sesudahnya"),
        ("RUTE DIRECT (GABUNGAN 2 VOYAGE)", "Target/pengisian/penggunaan tiap produk = total kebutuhan dasar dua voyage × 130%; merge dibatalkan jika salah satu produk tidak muat"),
        ("RUTE DIRECT TUNGGAL", "Target tiap produk = 150% kebutuhan dasar; aktual = minimum(target, kapasitas fisik)"),
        ("RUTE NON-DIRECT", "MFO dan BIO: target = 150% kebutuhan dasar; aktual = minimum(target, kapasitas fisik)"),
        ("KEBUTUHAN PADA OUTPUT", "Volume aktual yang diminta dari tanker sekaligus digunakan kapal; pengisian = penggunaan"),
        ("FULL_SAILING_ROUTE", "Rute asli vessel-voyage tempat checkpoint tujuan berada (voyage yang sedang dihitung, dari data schedule)"),
        ("SAILING_ROUTE_CHECKPOINT_TO_CHECKPOINT", "Rute dari checkpoint saat ini sampai checkpoint berikutnya, termasuk lintas voyage"),
        ("CHECKPOINT BATAS VOYAGE", "Checkpoint yang sama pada akhir voyage dan awal voyage berikutnya dianggap satu port call; event memakai voyage yang tiba lebih dahulu"),
        ("CHECKPOINT CABANG", "IDMAK atau IDAMQ menjadi checkpoint pada akhir voyage jika voyage berikutnya dimulai dari cabang yang sama"),
        ("PENGISIAN CABANG", "Ditampilkan sebagai TANKER=CABANG dan dianggap tersedia penuh; tidak memakai waktu, stok, kapasitas, atau antrean SIGMA/ZETA"),
        ("NEXT_CHECKPOINT", "Kedatangan checkpoint terpilih berikutnya dan menjadi batas akhir rute kebutuhan; dapat berada pada voyage yang sama atau voyage berikutnya"),
        ("KONSUMSI", "Rata-rata YTD Januari-Juli 2026, satuan liter/NM"),
        ("MISSING RATE", "Rata-rata tertimbang sistership; jika tidak ada memakai rata-rata global"),
        ("BIO ONLY", ", ".join(sorted(BIO_ONLY))),
        ("KAPAL DIKECUALIKAN", ", ".join(sorted(EXCLUDED_VESSELS))),
        ("PORT VIRTUAL", "IDRDE dan IDDOK diabaikan sebagai perpindahan; kapal dianggap tetap di port yang sama"),
        ("PRIORITAS ANTREAN", "First-in-first-out berdasarkan urutan ATB (kapal yang berthing lebih dulu dilayani lebih dulu)"),
        ("PERTAMINA", "Selalu tersedia; refill mengisi kedua produk sampai penuh"),
        ("KONFLIK", "Satu tanker hanya menjalankan satu aktivitas dalam satu waktu"),
        ("FLOW RATE", "Menggunakan nilai tetap sesuai konfigurasi (Catatan_waktu)"),
        ("MINIMUM STOK", "MFO 250 KL; BIO 50 KL; refill dipicu saat salah satu produk menyentuh batas minimum"),
        ("REFILL LOOK-AHEAD", "Refill dilakukan bila stok di bawah minimum ATAU kebutuhan kapal berikutnya > stok (pre-fill kapal besar, hindari refill tengah); saat idle bila refill selesai sebelum kapal berikutnya tiba; selalu isi sampai penuh; saat sedang mengisi kapal, stok dihabiskan dulu"),
        ("WAKTU PERPINDAHAN", "Waktu pindah tanker antar lokasi (Labuh, Pertamina, Sandar) mengikuti matriks perpindahan per tanker (move_times). Tidak ada perpindahan bila tanker sudah berada di lokasi tujuan (0 jam)."),
        ("SHIFT ANTAR KAPAL", "Jika kapal berikutnya sudah berthing (ATB ≤ waktu sekarang) dan stok cukup, tanker TETAP di area sandar dan pindah langsung antar kapal dengan waktu shift (SANDAR → SANDAR). Kembali ke base hanya dilakukan saat tanker idle (tidak ada kapal menunggu) atau harus refill."),
    ]
    for spec in TANKERS.values():
        move_text = "; ".join(
            f"{key.replace('->', '→')} {value:g} jam" for key, value in spec.move_times.items()
        )
        rows.extend(
            [
                (f"{spec.name} PORT", spec.port),
                (f"{spec.name} KAPASITAS", f"MFO {spec.capacity_mfo:g} KL; BIO {spec.capacity_bio:g} KL"),
                (
                    f"{spec.name} PERTAMINA→TANKER",
                    f"MFO {spec.load_rate_mfo:g} KL/jam; BIO {spec.load_rate_bio:g} KL/jam; "
                    + ("bersamaan" if spec.load_simultaneous else "bergantian"),
                ),
                (
                    f"{spec.name} TANKER→KAPAL",
                    f"MFO {spec.discharge_rate_mfo:g} KL/jam; BIO {spec.discharge_rate_bio:g} KL/jam; bersamaan",
                ),
                (f"{spec.name} MATRIKS PERPINDAHAN", move_text),
                (
                    f"{spec.name} MINIMUM STOK",
                    f"MFO {spec.min_stock_mfo:g} KL; BIO {spec.min_stock_bio:g} KL (salah satu menyentuh → refill)",
                ),
            ]
        )
    return pd.DataFrame(rows, columns=["PARAMETER", "NILAI/ATURAN"])


def movement_matrix_table() -> pd.DataFrame:
    """Tabel matriks waktu perpindahan tanker (DARI x KE) dalam jam."""
    destinations = (LOC_LABUH, LOC_PERTAMINA, LOC_SANDAR)
    frames: list[pd.DataFrame] = []
    for spec in TANKERS.values():
        rows = [
            {
                "TANKER": spec.name,
                "DARI \\ KE": origin,
                **{f"KE_{dest}": spec.move_time(origin, dest) for dest in destinations},
            }
            for origin in destinations
        ]
        frames.append(pd.DataFrame(rows))
    matrix = pd.concat(frames, ignore_index=True)
    matrix.columns = ["TANKER", "DARI \\ KE", "LABUH", "PERTAMINA", "SANDAR"]
    return matrix


def write_excel(
    summary: pd.DataFrame,
    results: pd.DataFrame,
    ledger: pd.DataFrame,
    consumption: pd.DataFrame,
    issues: pd.DataFrame,
    wait_breakdown: pd.DataFrame,
) -> None:
    result_columns = [
        "EVENT_ID", "VESSEL", "VOYAGENO", "VESVOY_GABUNGAN", "FULL_SAILING_ROUTE",
        "SAILING_ROUTE_CHECKPOINT_TO_CHECKPOINT", "TANKER",
        "CHECKPOINT", "CHECKPOINT_BERIKUT",
        "JARAK_KE_CHECKPOINT_NM",
        "KEBUTUHAN_DASAR_MFO_KL", "KEBUTUHAN_MFO_KL", "KAPASITAS_TANGKI_MFO_KL",
        "KEBUTUHAN_DASAR_BIO_KL", "KEBUTUHAN_BIO_KL", "KAPASITAS_TANGKI_BIO_KL",
        "KEBIJAKAN_BBM", "STATUS_KAPASITAS",
        "ATB", "MULAI_BUNKER", "SELESAI_BUNKER", "ATD",
        "WAKTU_TUNGGU_JAM", "REASON_WAKTU_TUNGGU", "STOK_AWAL_MFO_TANKER_KL",
        "STOK_AWAL_BIO_TANKER_KL", "DURASI_PROSES_JAM", "TERISI_SBLM_ATD_MFO_KL",
        "TERISI_SBLM_ATD_BIO_KL", "SHORTAGE_MFO_KL", "SHORTAGE_BIO_KL",
        "STOK_AKHIR_MFO_TANKER_KL", "STOK_AKHIR_BIO_TANKER_KL",
        "KETERLAMBATAN_JAM", "STATUS", "ALASAN_FLAG",
    ]
    results = results.reindex(columns=result_columns).sort_values("EVENT_ID")
    results = results.rename(
        columns={
            "ATB": "ATB (WAKTU BERTHING)",
            "ATD": "ATD_NEXT_VESVOY",
        }
    )
    ledger_columns = [
        "OPERATION_ID", "TANKER", "PORT", "JENIS_OPERASI", "EVENT_ID", "VESSEL",
        "MULAI", "SELESAI", "DURASI_JAM", "STOK_AWAL_MFO_KL", "STOK_AWAL_BIO_KL",
        "MASUK_MFO_KL", "MASUK_BIO_KL", "KELUAR_MFO_KL", "KELUAR_BIO_KL",
        "STOK_AKHIR_MFO_KL", "STOK_AKHIR_BIO_KL",
    ]
    ledger = ledger.reindex(columns=ledger_columns).sort_values(["MULAI", "TANKER", "OPERATION_ID"])
    if issues.empty:
        issues = pd.DataFrame(columns=["EVENT_ID", "VESSEL", "CHECKPOINT", "ATA", "MASALAH", "KETERANGAN", "SOURCE_EXCEL_ROW"])

    with pd.ExcelWriter(OUTPUT_FILE, engine="xlsxwriter", datetime_format="dd/mm/yyyy hh:mm") as writer:
        summary.to_excel(writer, sheet_name="RINGKASAN", index=False)
        results.to_excel(writer, sheet_name="HASIL_KAPAL", index=False)
        ledger.to_excel(writer, sheet_name="MONITORING_TANKER", index=False)
        wait_breakdown.to_excel(writer, sheet_name="BREAKDOWN_TUNGGU", index=False)
        parameter_table().to_excel(writer, sheet_name="PARAMETER", index=False)

        workbook = writer.book
        header_fmt = workbook.add_format({"bold": True, "align": "center", "valign": "vcenter", "border": 1})
        number_fmt = workbook.add_format({"num_format": "0.00"})
        date_fmt = workbook.add_format({"num_format": "dd/mm/yyyy hh:mm"})
        red_fmt = workbook.add_format({"bg_color": "#FFC7CE", "font_color": "#9C0006"})
        green_fmt = workbook.add_format({"bg_color": "#C6EFCE", "font_color": "#006100"})
        amber_fmt = workbook.add_format({"bg_color": "#FFEB9C", "font_color": "#9C6500"})

        frames = {
            "RINGKASAN": summary,
            "HASIL_KAPAL": results,
            "MONITORING_TANKER": ledger,
            "BREAKDOWN_TUNGGU": wait_breakdown,
            "PARAMETER": parameter_table(),
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
        if not results.empty:
            status_col = results.columns.get_loc("STATUS")
            result_sheet.conditional_format(1, status_col, len(results), status_col, {
                "type": "cell", "criteria": "==", "value": '"NOT FEASIBLE"', "format": red_fmt
            })
            result_sheet.conditional_format(1, status_col, len(results), status_col, {
                "type": "cell", "criteria": "==", "value": '"FEASIBLE"', "format": green_fmt
            })
            result_sheet.conditional_format(1, status_col, len(results), status_col, {
                "type": "cell", "criteria": "==", "value": '"DATA ISSUE"', "format": amber_fmt
            })


def write_documentation(summary: pd.DataFrame, results: pd.DataFrame, consumption: pd.DataFrame) -> None:
    total = summary[summary["TANKER"] == "TOTAL"].iloc[0]
    sigma = summary[summary["TANKER"] == "SIGMA"].iloc[0]
    zeta = summary[summary["TANKER"] == "ZETA"].iloc[0]
    branch = summary[summary["TANKER"] == "CABANG"].iloc[0]
    global_mfo = consumption.loc[consumption["SUMBER_MFO"] == "RATA-RATA GLOBAL", "MFO_L_PER_NM"]
    global_bio = consumption.loc[consumption["SUMBER_BIO"] == "RATA-RATA GLOBAL", "BIO_L_PER_NM"]
    global_mfo_value = global_mfo.iloc[0] if len(global_mfo) else np.nan
    global_bio_value = global_bio.iloc[0] if len(global_bio) else np.nan

    flagged = results[results["STATUS"] == "NOT FEASIBLE"].sort_values("KETERLAMBATAN_JAM", ascending=False)
    top_flags = "\n".join(
        f"- {row.VESSEL} di {row.CHECKPOINT}, ATA {row.ATA:%d/%m/%Y %H:%M}: "
        f"shortage MFO {row.SHORTAGE_MFO_KL:.2f} KL, BIO {row.SHORTAGE_BIO_KL:.2f} KL, "
        f"terlambat {row.KETERLAMBATAN_JAM:.2f} jam."
        for row in flagged.head(10).itertuples()
    ) or "- Tidak ada event yang melewati deadline."
    if int(total.NOT_FEASIBLE) == 0:
        capacity_conclusion = (
            f"SIGMA dan ZETA cukup untuk seluruh {int(total.EVENT_VALID)} event yang datanya lengkap. "
            f"Tidak ada shortage terhadap deadline dan waktu tunggu terpanjang adalah "
            f"{total.MAX_TUNGGU_JAM:.2f} jam."
        )
    else:
        capacity_conclusion = (
            f"Terdapat {int(total.NOT_FEASIBLE)} dari {int(total.EVENT_VALID)} event valid yang tidak selesai "
            f"sebelum deadline. Total shortage saat deadline adalah MFO {total.SHORTAGE_MFO_KL:.2f} KL "
            f"dan BIO {total.SHORTAGE_BIO_KL:.2f} KL; keterlambatan maksimum "
            f"{total.MAX_TERLAMBATAN_JAM:.2f} jam."
        )

    move_lines = []
    for spec in TANKERS.values():
        move_text = "; ".join(
            f"{key.replace('->', '→')} {value:g} jam" for key, value in spec.move_times.items()
        )
        move_lines.append(f"- {spec.name}: {move_text}.")
    overhead_text = "\n".join(move_lines)

    text = f"""# Simulasi Kapasitas Tanker ZETA dan SIGMA

## Tujuan

Simulasi ini menguji apakah kapasitas, stok, flow rate, dan ketersediaan waktu tanker ZETA dan SIGMA cukup untuk memenuhi kebutuhan BBM kapal operasional selama Juni–Juli 2026.

File hasil utama adalah `baseline_simulation.xlsx`. Simulasi dapat dijalankan ulang dengan:

```bash
python3 -m fuel_simulation.core
```

## Aturan yang digunakan

- Event dipilih berdasarkan waktu berthing (ATB) antara 1 Juni 2026 pukul 00:00 sampai 31 Juli 2026 pukul 23:59. Waktu berthing = `ATB_POD`, fallback `ETB_POD`, fallback `ATA_POD`.
- Pengisian hanya dilakukan saat kapal berthing, bukan saat arrival (sesuai praktik tim bunker).
- ATD adalah deadline pengisian. ETD digunakan hanya ketika ATD kosong dan tersedia.
- ROB kapal tidak dimodelkan: ROB awal dianggap 0 KL dan pengisian selalu sama dengan penggunaan.
- Semua kedatangan Jakarta dan Surabaya menjadi kandidat checkpoint bunker. Kandidat utama yang hanya satu leg dari checkpoint terpilih sebelumnya dilewati sebagai event bunker; pencarian dilanjutkan ke kandidat berikutnya sampai interval memiliki minimum {MIN_LEGS_BETWEEN_MAIN_CHECKPOINTS} leg. Port yang dilewati tetap berada di dalam `SAILING_ROUTE_CHECKPOINT_TO_CHECKPOINT`, dan event berikutnya dimulai dari checkpoint akhir yang benar-benar terpilih.
- Port origin rute tidak dihitung sebagai kedatangan baru karena merupakan checkpoint sebelumnya.
- Voyage masing-masing vessel diurutkan berdasarkan tahun, nomor voyage, dan suffix voyage. Baris leg di dalam voyage mengikuti urutan port pada `FULL_SAILING_ROUTE`.
- Jarak kebutuhan adalah akumulasi `NMILE` dari checkpoint voyage saat ini sampai checkpoint tujuan voyage berikutnya (kebutuhan untuk segmen yang akan datang, bukan segmen yang sudah ditempuh). Interval dapat melintasi pergantian voyage dan diatribusikan ke voyage tempat checkpoint saat ini berada (voyage yang sedang dihitung).
- Kolom `FULL_SAILING_ROUTE` berisi rute asli voyage tempat checkpoint tujuan berada (voyage yang sedang dihitung; berasal dari data schedule).
- Kolom `SAILING_ROUTE_CHECKPOINT_TO_CHECKPOINT` berisi rute hasil susunan dari checkpoint saat ini sampai checkpoint berikutnya (segmen yang akan datang, termasuk lintas voyage).
- Jika Jakarta atau Surabaya muncul beberapa kali sebagai port kedatangan, setiap kemunculan menjadi kandidat dan dipilih secara berantai memakai aturan minimum leg tersebut.
- Checkpoint yang sama pada batas akhir-awal dua voyage dianggap satu port call, bukan dua event berjarak 0 NM. Event tersebut menggunakan nomor voyage yang tiba di checkpoint lebih dahulu.
- Makassar (`IDMAK`) dan Ambon (`IDAMQ`) menjadi checkpoint khusus pada akhir voyage jika voyage berikutnya dimulai dari cabang yang sama.
- Pengisian pada checkpoint cabang ditampilkan sebagai `TANKER=CABANG` dan dianggap tersedia langsung. Event tersebut memutus sailing route dan kebutuhan, tetapi tidak masuk antrean serta tidak mengurangi stok SIGMA/ZETA.
- Konsumsi dasar rute dihitung dengan rumus `total NMILE segmen × liter/NM ÷ 1.000`.
- Dua event direct `STEP=2` yang berurutan dan tersambung hanya digabung jika target 130% MFO dan BIO sama-sama muat pada kapasitas fisiknya. Jika salah satu tidak muat, event pertama diproses sendiri dan event kedua tetap dapat diuji dengan event sesudahnya.
- Untuk event direct gabungan, pengisian dan penggunaan tiap produk adalah total kebutuhan dasar dua voyage × 130%.
- Untuk direct tunggal dan rute non-direct, pengisian dan penggunaan MFO maupun BIO adalah minimum 150% kebutuhan dasar atau kapasitas fisiknya. Kapal tetap berangkat ketika dibatasi kapasitas dan kondisi tersebut dicatat pada `STATUS_KAPASITAS`.
- Kapasitas kosong dianggap 0 KL. Kapasitas 0 dengan konsumsi positif diberi `DATA ISSUE`; kapasitas 0 dengan konsumsi 0 tetap valid.
- Audit volume menggunakan empat kolom inti: `KEBUTUHAN_DASAR_MFO_KL`, `KEBUTUHAN_DASAR_BIO_KL`, `KEBIJAKAN_BBM`, dan `STATUS_KAPASITAS`. `KEBUTUHAN_*_KL` adalah volume aktual yang diminta sekaligus digunakan.
- Konsumsi menggunakan YTD Januari–Juli 2026 dari `data/fuel_consumption.xlsx`.
- Data konsumsi yang kosong menggunakan rata-rata tertimbang sistership. Jika sistership tidak tersedia, digunakan rata-rata global tertimbang.
- Kapal yang dikecualikan manual tidak dibuatkan event dan tidak dihitung kebutuhannya: {', '.join(sorted(EXCLUDED_VESSELS))}.
- Kapal BIO-only adalah: {', '.join(sorted(BIO_ONLY))}. Kebutuhan MFO kapal tersebut ditetapkan 0 KL.
- `IDRDE` dan `IDDOK` dianggap sebagai lokasi lokal tanpa perpindahan pelabuhan. Keduanya dihapus dari urutan rute sebelum jarak dihitung.
- ZETA dan SIGMA mulai dalam kondisi penuh pada 1 Juni 2026 pukul 00:00.
- Refill tanker selalu mengisi kedua produk sampai penuh. Pertamina dianggap selalu tersedia.
- Satu tanker hanya dapat melakukan satu aktivitas pada satu waktu.
- Antrean menggunakan first-in-first-out berdasarkan urutan ATB (kapal yang berthing lebih dulu dilayani lebih dulu).
- Refill menggunakan kebijakan look-ahead: tanker mengisi sampai penuh bila stok menyentuh minimum (MFO 250 KL, BIO 50 KL) ATAU kebutuhan kapal berikutnya melebihi stok (pre-fill kapal besar agar tidak terjadi refill di tengah pengisian). Saat sedang mengisi kapal, stok dihabiskan lebih dulu (refill lanjutan hanya bila stok benar-benar habis dan kebutuhan kapal masih tersisa).
- Refill saat idle dilakukan bila stok di bawah minimum atau kebutuhan kapal berikutnya melebihi stok, asalkan refill dapat selesai sebelum kapal berikutnya tiba.
- Setiap perpindahan tanker antar lokasi (Labuh, Pertamina, Sandar) memakai waktu perpindahan sesuai matriks per tanker. Perpindahan ke kapal (approach) dihitung dari lokasi tanker saat itu: dari Labuh, dari Pertamina, atau shift antar kapal (SANDAR → SANDAR) saat kapal berikutnya sudah berthing.
- Setelah bunker: bila kapal berikutnya sudah berthing dan stok cukup, tanker tetap di area sandar dan pindah langsung antar kapal (shift). Kembali ke base hanya dilakukan saat tanker idle (tidak ada kapal menunggu) atau harus refill.
- Setelah refill di Pertamina: bila kapal sudah menunggu, tanker langsung bergerak PERTAMINA → SANDAR tanpa kembali ke base; bila idle, tanker kembali ke Labuh.
- Refill lanjutan di tengah pengisian (stok habis): tanker bergerak SANDAR → PERTAMINA, mengisi penuh, lalu kembali SANDAR untuk melanjutkan kapal yang sama.
- Waktu perpindahan SIGMA menyertakan SPOG (izin masuk) sesuai jadwal Syahbandar; ZETA tidak menyertakan SPOG karena ditangani langsung pihak SBRA.
- Semua flow rate memakai nilai tetap sesuai konfigurasi (Catatan_waktu).

## Parameter tanker

| Tanker | Port | Kapasitas MFO | Kapasitas BIO | Pertamina → tanker | Tanker → kapal |
|---|---|---:|---:|---|---|
| SIGMA | IDJKT | 1.000 KL | 500 KL | MFO 90 dan BIO 90 KL/jam, bergantian | MFO 110 dan BIO 50 KL/jam, bersamaan |
| ZETA | IDSUB | 750 KL | 250 KL | MFO 30 dan BIO 30 KL/jam, bersamaan | MFO 110 dan BIO 90 KL/jam, bersamaan |

Matriks waktu perpindahan tanker (jam):

{overhead_text}

## Hasil run saat ini

| Tanker | Event | Feasible | Not feasible | Data issue | Refill | Kebutuhan MFO | Kebutuhan BIO |
|---|---:|---:|---:|---:|---:|---:|---:|
| SIGMA | {int(sigma.TOTAL_EVENT)} | {int(sigma.FEASIBLE)} | {int(sigma.NOT_FEASIBLE)} | {int(sigma.DATA_ISSUE)} | {int(sigma.JUMLAH_REFILL)} | {sigma.KEBUTUHAN_MFO_KL:.2f} KL | {sigma.KEBUTUHAN_BIO_KL:.2f} KL |
| ZETA | {int(zeta.TOTAL_EVENT)} | {int(zeta.FEASIBLE)} | {int(zeta.NOT_FEASIBLE)} | {int(zeta.DATA_ISSUE)} | {int(zeta.JUMLAH_REFILL)} | {zeta.KEBUTUHAN_MFO_KL:.2f} KL | {zeta.KEBUTUHAN_BIO_KL:.2f} KL |
| CABANG | {int(branch.TOTAL_EVENT)} | {int(branch.FEASIBLE)} | {int(branch.NOT_FEASIBLE)} | {int(branch.DATA_ISSUE)} | 0 | {branch.KEBUTUHAN_MFO_KL:.2f} KL | {branch.KEBUTUHAN_BIO_KL:.2f} KL |
| Total tanker (SIGMA+ZETA) | {int(total.TOTAL_EVENT)} | {int(total.FEASIBLE)} | {int(total.NOT_FEASIBLE)} | {int(total.DATA_ISSUE)} | {int(total.JUMLAH_REFILL)} | {total.KEBUTUHAN_MFO_KL:.2f} KL | {total.KEBUTUHAN_BIO_KL:.2f} KL |

Rata-rata global fallback yang dihasilkan adalah MFO {global_mfo_value:.4f} liter/NM dan BIO {global_bio_value:.4f} liter/NM.

### Kesimpulan kapasitas

{capacity_conclusion}

Kesimpulan ini belum mencakup {int(total.DATA_ISSUE)} event berstatus `DATA ISSUE`. Detailnya tetap tampil pada `HASIL_KAPAL`; event tersebut tidak dipaksakan memakai angka asumsi agar hasil kapasitas tetap dapat diaudit.

### Event dengan masalah jadwal terbesar

{top_flags}

## Isi workbook

- `RINGKASAN`: KPI per tanker dan total.
- `HASIL_KAPAL`: kebutuhan, jadwal bunker, stok tanker sebelum bunker, alasan tunggu, jumlah yang sempat terisi sebelum ATD, shortage, dan flag setiap event.
- `MONITORING_TANKER`: ledger kronologis stok masuk/keluar dan sisa MFO/BIO setelah setiap aktivitas.
- `BREAKDOWN_TUNGGU`: rincian per event — operasi tanker apa saja yang mengisi waktu tunggu (mulai ATB sampai pompa dimulai), termasuk bagian IDLE, agar waktu tunggu dapat diaudit.
- `PARAMETER`: seluruh parameter dan aturan simulasi.

## Arti status

- `FEASIBLE`: seluruh kebutuhan MFO dan BIO selesai diisi sebelum atau tepat pada deadline.
- `NOT FEASIBLE`: pengisian selesai setelah deadline; kolom shortage menunjukkan volume yang belum masuk saat ATD/ETD.
- `DATA ISSUE`: event tidak disimulasikan karena jarak rute atau deadline tidak tersedia.

## Batasan model

- Antrean eksternal Pertamina (LO blocked, kendala siklus uang), prioritas pengisian (TNI-AL/PLN/PELNI), cuaca, dan variabilitas waktu pandu (normal vs crowded) belum dimodelkan penuh; waktu overhead yang dipakai adalah nilai tetap/tengah.
- Pemakaian BBM kapal dianggap linear terhadap NMILE dan memakai rata-rata historis YTD.
- `NEXT_CHECKPOINT (NEXT_VOYAGE)` adalah kedatangan checkpoint terpilih pada voyage berikutnya dan menjadi batas akhir rute kebutuhan.
- Voyage terakhir yang belum memiliki checkpoint lanjutan serta interval dengan segmen schedule yang tidak lengkap diberi status `DATA ISSUE`; jaraknya tidak ditutup menggunakan rute asumsi.
"""
    DOC_FILE.parent.mkdir(parents=True, exist_ok=True)
    DOC_FILE.write_text(text, encoding="utf-8")


def main() -> None:
    schedule = load_schedule()
    consumption = load_consumption(set(schedule["VESSELID"].dropna()))
    events, issues = build_events(schedule, consumption)
    events = attach_tank_capacity(events)
    events = merge_direct_voyages(events)
    events = events[events["ATB"] <= SIMULATION_END].copy()
    issues = issues[issues["EVENT_ID"].isin(events["EVENT_ID"])].copy()
    events = apply_vessel_fuel_policy(events)
    events, issues = renumber_events_chronologically(events, issues)

    all_results: list[pd.DataFrame] = []
    all_ledgers: list[pd.DataFrame] = []
    for port, spec in TANKERS.items():
        tanker_events = events[events["TANKER"] == spec.name].copy()
        result, ledger = simulate_tanker(spec, tanker_events)
        all_results.append(result)
        all_ledgers.append(ledger)

    branch_events = events[events["TANKER"] == "CABANG"].copy()
    if not branch_events.empty:
        all_results.append(build_branch_results(branch_events))

    results = pd.concat(all_results, ignore_index=True)
    ledger = pd.concat(all_ledgers, ignore_index=True)
    validate_simulation(results, ledger)
    summary = build_summary(results, ledger)
    wait_breakdown = build_wait_breakdown(results, ledger)
    write_excel(summary, results, ledger, consumption, issues, wait_breakdown)
    write_documentation(summary, results, consumption)

    print(summary.to_string(index=False))
    print(f"\nWorkbook: {OUTPUT_FILE}")
    print(f"Dokumentasi: {DOC_FILE}")


if __name__ == "__main__":
    main()
