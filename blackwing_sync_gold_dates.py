import csv
import sys
import shutil
from pathlib import Path
from datetime import datetime, date
from collections import Counter

try:
    from openpyxl import load_workbook
except ImportError:
    print("Missing openpyxl. Run: python -m pip install openpyxl")
    sys.exit(1)

base = Path.cwd().parent

workbooks = list(base.rglob("*CORRECTED*.xlsx"))
csv_files = list(base.rglob("gold_question_results.csv"))

if not workbooks:
    sys.exit("ERROR: Corrected Excel workbook not found.")

if not csv_files:
    sys.exit("ERROR: Evaluation CSV not found.")

if len(workbooks) != 1 or len(csv_files) != 1:
    print("Multiple candidate files found. No changes made.")
    print("Workbooks:", workbooks)
    print("CSVs:", csv_files)
    sys.exit(1)

workbook = workbooks[0]
csv_path = csv_files[0]

print("Workbook:", workbook)
print("CSV:", csv_path)

wb = load_workbook(workbook, read_only=True, data_only=True)
if "80_QUESTIONS" not in wb.sheetnames:
    sys.exit("ERROR: 80_QUESTIONS sheet missing.")

ws = wb["80_QUESTIONS"]
values = ws.values
headers = [str(x).strip() if x is not None else "" for x in next(values)]
excel_rows = [
    dict(zip(headers, row))
    for row in values
    if any(x is not None for x in row)
]
wb.close()

with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    csv_rows = list(reader)

required = {"case_label", "query", "review_date", "human_validation_status"}

if not required.issubset(headers):
    sys.exit("ERROR: Required Excel columns missing.")

if not {"case_label", "query", "review_date",
        "review_date_iso", "review_metadata_integrity"}.issubset(fields):
    sys.exit("ERROR: Required CSV columns missing.")

if len(excel_rows) != 80 or len(csv_rows) != 80:
    sys.exit(f"ERROR: Excel={len(excel_rows)}, CSV={len(csv_rows)}. Expected 80 each.")

def key(row):
    return (
        str(row["case_label"]).strip().upper(),
        " ".join(str(row["query"]).split()).casefold()
    )

excel_keys = [key(r) for r in excel_rows]
csv_keys = [key(r) for r in csv_rows]

if len(set(excel_keys)) != 80 or len(set(csv_keys)) != 80:
    sys.exit("ERROR: Duplicate question keys detected.")

if set(excel_keys) != set(csv_keys):
    sys.exit("ERROR: Excel and CSV questions do not match.")

if any(str(r["human_validation_status"]).strip().upper() != "APPROVED"
       for r in excel_rows):
    sys.exit("ERROR: Not all gold answers are approved.")

review_map = {key(r): r for r in excel_rows}

updates = []
as_of = date(2026, 11, 19)

for row in csv_rows:
    source = review_map[key(row)]
    original_date = source["review_date"]

    if isinstance(original_date, (datetime, date)):
        parsed = original_date.date() if isinstance(original_date, datetime) else original_date
    else:
        parsed = datetime.strptime(str(original_date).strip(), "%d.%m.%Y").date()

    if parsed > as_of:
        sys.exit("ERROR: Future review date detected.")

    updates.append((
        row,
        parsed.strftime("%d.%m.%Y"),
        parsed.isoformat()
    ))

print("\nVERIFICATION SUCCESSFUL")
print("Excel questions:", len(excel_rows))
print("CSV questions:", len(csv_rows))
print("Matched questions:", len(updates))
print("Review dates:", dict(Counter(x[1] for x in updates)))

print("\nExisting flags:",
      dict(Counter(r["review_metadata_integrity"] for r in csv_rows)))

backup = csv_path.with_name(
    csv_path.stem + ".BEFORE_DATE_SYNC_" +
    datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
)

shutil.copy2(csv_path, backup)

extra = ["review_metadata_integrity_original", "review_metadata_rechecked_on"]
new_fields = list(fields) + [x for x in extra if x not in fields]

for row, display_date, iso_date in updates:
    if not row.get("review_metadata_integrity_original"):
        row["review_metadata_integrity_original"] = row["review_metadata_integrity"]

    row["review_date"] = display_date
    row["review_date_iso"] = iso_date
    row["review_metadata_integrity"] = "DATE_NOT_IN_FUTURE"
    row["review_metadata_rechecked_on"] = as_of.isoformat()

temporary = csv_path.with_name(csv_path.name + ".tmp")

with temporary.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=new_fields)
    writer.writeheader()
    writer.writerows(csv_rows)

with temporary.open("r", encoding="utf-8-sig", newline="") as f:
    verified = list(csv.DictReader(f))

if len(verified) != 80:
    temporary.unlink()
    sys.exit("ERROR: Post-write verification failed. Original CSV preserved.")

temporary.replace(csv_path)

print("\nSYNCHRONIZATION COMPLETED")
print("Records updated:", len(updates))
print("Current future-date flags: 0")
print("Backup:", backup)
print("CSV:", csv_path)
print("Original date-status flags preserved.")
print("AI responses and scoring columns were not modified.")
