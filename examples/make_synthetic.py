"""Generate clearly fictional phase-0 inputs for a local smoke test."""
import csv
from pathlib import Path

from openpyxl import Workbook


def write_csv(path, headers, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)


out = Path("examples/synthetic")
out.mkdir(parents=True, exist_ok=True)
write_csv(out / "sources.csv", ["source_id", "journal_name", "inclusion_rationale", "active", "version", "validated_on", "validated_by"],
          [["999999999", "TEST ONLY Journal", "synthetic fixture only", "true", "demo-v1", "2026-01-01", "Fixture Human"]])
headers = ["seed_id", "community", "version", "doi", "eid", "title", "authors", "publication_year", "journal", "top_rank", "expert_rationale", "status", "retrieval_note", "validated_by", "validated_on"]
write_csv(out / "benchmark_cto.csv", headers, [["", "CTO", "demo-v1", "10.9999/example", "", "Synthetic example", "A Tester", "2020", "TEST ONLY Journal", "fixture", "synthetic test", "eligible", "fixture source", "Fixture Human", "2026-01-01"]])
write_csv(out / "benchmark_tim.csv", headers, [["", "TIM", "demo-v1", "", "", "Out of range", "B Tester", "2020", "TEST ONLY Journal", "fixture", "synthetic test", "out_of_scope", "fixture excluded", "Fixture Human", "2026-01-01"]])
book = Workbook()
sheet = book.active
sheet.append(["DOI", "EID", "Title", "Authors", "Year", "Source title", "Source ID", "Document Type", "Source Type", "Extra"])
sheet.append(["10.9999/example", "2-s2.0-TEST", "Synthetic example", "A Tester", 2020, "TEST ONLY Journal", "999999999", "Article", "Journal", "retained"])
book.save(out / "scopus_synthetic.xlsx")
print(out)
