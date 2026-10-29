"""Focused checks for the frozen 0/A scheduling helper."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from plan_matched_0a import digest, plan


def block(index: int, stratum: str) -> dict:
    return {"pair_id": f"p{index}", "problem_id": f"q{index // 2}",
            "stratum": stratum, "packet_sha256": digest({"packet": index}),
            "trigger": "construct_problem", "seed": index}


class ScheduleTests(unittest.TestCase):
    def test_balanced_within_stratum_and_reproducible(self):
        blocks = [block(i, "cto" if i < 5 else "tim") for i in range(10)]
        one = plan(blocks, "registered-random-seed")
        self.assertEqual(one, plan(blocks, "registered-random-seed"))
        self.assertEqual(one["blocks_sha256"], digest(blocks))
        self.assertEqual(one["episodes_sha256"], digest(one["episodes"]))
        self.assertEqual(len(one["episodes"]), 20)
        for i in range(0, 20, 2):
            pair = one["episodes"][i:i + 2]
            self.assertEqual({row["condition"] for row in pair}, {"0", "A"})
            self.assertEqual({row["pair_id"] for row in pair}, {pair[0]["pair_id"]})
            self.assertEqual([row["within_pair_order"] for row in pair], [1, 2])
        for stratum in ("cto", "tim"):
            firsts = [row["condition"] for row in one["episodes"] if row["stratum"] == stratum and row["within_pair_order"] == 1]
            self.assertLessEqual(abs(firsts.count("0") - firsts.count("A")), 1)

    def test_refuses_bad_registry(self):
        good = block(1, "cto")
        for blocks in ([], [good, good], [{**good, "packet_sha256": "bad"}],
                       [{**good, "packet_sha256": None}], [{**good, "seed": -1}],
                       [{**good, "seed": 2_147_483_648}], [{**good, "stratum": []}],
                       [{**good, "problem_id": "private research question text"}],
                       [{**good, "private_text": "do not store"}]):
            with self.subTest(blocks=blocks), self.assertRaises(ValueError):
                plan(blocks, "seed")
        with self.assertRaises(ValueError):
            plan([good], "private seed with spaces")

    def test_cli_refuses_to_replace_frozen_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, schedule = root / "blocks.json", root / "schedule.json"
            registry.write_text(json.dumps([block(1, "cto"), block(2, "tim")]))
            command = [sys.executable, str(Path(__file__).with_name("plan_matched_0a.py")),
                       "--blocks", str(registry), "--assignment-seed", "frozen-seed",
                       "--out", str(schedule)]
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
            original = schedule.read_bytes()
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(schedule.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
