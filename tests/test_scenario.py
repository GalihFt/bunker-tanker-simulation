from __future__ import annotations

import unittest

import pandas as pd

from fuel_simulation import core as sb
from fuel_simulation import rob_scenario as rob


class ScenarioConfigurationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.events, cls.issues, _ = rob.prepare_events()

    def test_tanker_deployment_and_port_loading(self) -> None:
        jakarta = sb.TANKERS["IDJKT"]
        surabaya = sb.TANKERS["IDSUB"]

        self.assertEqual(jakarta.name, "SIGMA")
        self.assertEqual((jakarta.capacity_mfo, jakarta.capacity_bio), (1000.0, 500.0))
        self.assertEqual((jakarta.load_rate_mfo, jakarta.load_rate_bio), (100.0, 120.0))
        self.assertEqual((jakarta.discharge_rate_mfo, jakarta.discharge_rate_bio), (110.0, 90.0))
        self.assertFalse(jakarta.load_simultaneous)

        self.assertEqual(surabaya.name, "ZETA")
        self.assertEqual((surabaya.capacity_mfo, surabaya.capacity_bio), (750.0, 250.0))
        self.assertEqual((surabaya.load_rate_mfo, surabaya.load_rate_bio), (40.0, 70.0))
        self.assertEqual((surabaya.discharge_rate_mfo, surabaya.discharge_rate_bio), (110.0, 50.0))
        self.assertTrue(surabaya.load_simultaneous)

    def test_minimum_loading_uses_or_rule(self) -> None:
        sigma = sb.TANKERS["IDJKT"]
        self.assertTrue(sb.refill_meets_minimum(sigma, 750.0, 450.0))
        self.assertTrue(sb.refill_meets_minimum(sigma, 920.0, 400.0))
        self.assertFalse(sb.refill_meets_minimum(sigma, 800.01, 400.01))

    def test_main_checkpoint_follows_route_preference(self) -> None:
        main = self.events[self.events["CHECKPOINT"].isin(["IDJKT", "IDSUB"])]
        jakarta = main[main["CHECKPOINT"] == "IDJKT"]
        surabaya = main[main["CHECKPOINT"] == "IDSUB"]

        jakarta_source_route = jakarta["FULL_SAILING_ROUTE"].fillna("").str.split(" | ", regex=False).str[0]
        surabaya_source_route = surabaya["FULL_SAILING_ROUTE"].fillna("").str.split(" | ", regex=False).str[0]

        self.assertFalse(jakarta_source_route.str.contains("IDSUB").any())
        self.assertTrue(surabaya_source_route.str.contains("IDSUB").all())

    def test_all_four_main_checkpoint_transitions_exist(self) -> None:
        transitions = set(
            self.events.loc[
                self.events["CHECKPOINT"].isin(["IDJKT", "IDSUB"])
                & self.events["CHECKPOINT_BERIKUT"].isin(["IDJKT", "IDSUB"]),
                ["CHECKPOINT", "CHECKPOINT_BERIKUT"],
            ].itertuples(index=False, name=None)
        )
        self.assertEqual(
            transitions,
            {
                ("IDSUB", "IDSUB"),
                ("IDSUB", "IDJKT"),
                ("IDJKT", "IDJKT"),
                ("IDJKT", "IDSUB"),
            },
        )

    def test_oru_is_surabaya_to_surabaya(self) -> None:
        oru = self.events[
            (self.events["VESSEL"] == "ORU")
            & self.events["ATB"].between(rob.ANALYSIS_START, rob.ANALYSIS_END)
        ]
        self.assertFalse(oru.empty)
        self.assertTrue((oru["CHECKPOINT"] == "IDSUB").all())
        self.assertTrue((oru["CHECKPOINT_BERIKUT"] == "IDSUB").all())
        self.assertTrue(
            oru["SAILING_ROUTE_CHECKPOINT_TO_CHECKPOINT"]
            .str.startswith("IDSUB-")
            .all()
        )
        self.assertTrue(
            oru["SAILING_ROUTE_CHECKPOINT_TO_CHECKPOINT"]
            .str.endswith("-IDSUB")
            .all()
        )

    def test_only_known_branch_checkpoints_are_added(self) -> None:
        other = set(self.events["CHECKPOINT"]) - {"IDJKT", "IDSUB"}
        self.assertEqual(other, {"IDMAK", "IDAMQ"})
        self.assertTrue(self.issues.empty)

    def test_jakarta_priority_is_symmetric(self) -> None:
        events, issues, _ = rob.prepare_events("IDJKT")
        main = events[events["CHECKPOINT"].isin(["IDJKT", "IDSUB"])]
        jakarta = main[main["CHECKPOINT"] == "IDJKT"]
        surabaya = main[main["CHECKPOINT"] == "IDSUB"]

        jakarta_source_route = jakarta["FULL_SAILING_ROUTE"].fillna("").str.split(" | ", regex=False).str[0]
        surabaya_source_route = surabaya["FULL_SAILING_ROUTE"].fillna("").str.split(" | ", regex=False).str[0]

        self.assertTrue(jakarta_source_route.str.contains("IDJKT").all())
        self.assertFalse(surabaya_source_route.str.contains("IDJKT").any())
        self.assertTrue(issues.empty)

    def test_bunker_stops_at_deadline(self) -> None:
        out = rob.run("IDSUB")
        all_results = pd.concat(
            [out["warmup_results"], out["analysis_results"]], ignore_index=True
        )
        deadlines = all_results.set_index("EVENT_ID")["DEADLINE"]
        bunker = out["ledger"][
            out["ledger"]["JENIS_OPERASI"] == "BUNKER KE KAPAL"
        ]
        for operation in bunker.itertuples(index=False):
            self.assertLessEqual(operation.SELESAI, deadlines[operation.EVENT_ID])

        delivered = bunker.groupby("EVENT_ID")[["KELUAR_MFO_KL", "KELUAR_BIO_KL"]].sum()
        results = all_results.set_index("EVENT_ID")
        for event_id, volume in delivered.iterrows():
            self.assertAlmostEqual(
                volume["KELUAR_MFO_KL"],
                results.at[event_id, "TERISI_SBLM_ATD_MFO_KL"],
                places=6,
            )
            self.assertAlmostEqual(
                volume["KELUAR_BIO_KL"],
                results.at[event_id, "TERISI_SBLM_ATD_BIO_KL"],
                places=6,
            )


if __name__ == "__main__":
    unittest.main()
