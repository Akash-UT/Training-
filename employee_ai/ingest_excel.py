import pandas as pd

from app.config import settings
from app.database import (
    clear_employee_data,
    initialize_database,
    insert_employee_rows,
)

REQUIRED_COLUMNS = [
    "Employee Name",
    "Department",
    "Position",
    "Leave Type",
    "Start Date",
    "End Date",
    "Days Taken",
    "Total Leave Entitlement",
    "Leave Taken So Far",
    "Remaining Leaves",
    "month",
]


def clean_value(value):
    if pd.isna(value):
        return None
    return value


def load_excel() -> list[dict]:
    df = pd.read_excel(settings.excel_path)

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    rows = []

    for _, row in df.iterrows():
        rows.append(
            {
                "employee_name": clean_value(row["Employee Name"]),
                "department": clean_value(row["Department"]),
                "position": clean_value(row["Position"]),
                "leave_type": clean_value(row["Leave Type"]),
                "start_date": (
                    pd.to_datetime(row["Start Date"]).date()
                    if not pd.isna(row["Start Date"])
                    else None
                ),
                "end_date": (
                    pd.to_datetime(row["End Date"]).date()
                    if not pd.isna(row["End Date"])
                    else None
                ),
                "days_taken": clean_value(row["Days Taken"]),
                "total_leave_entitlement": clean_value(
                    row["Total Leave Entitlement"]
                ),
                "leave_taken_so_far": clean_value(
                    row["Leave Taken So Far"]
                ),
                "remaining_leaves": clean_value(
                    row["Remaining Leaves"]
                ),
                "month": clean_value(row["month"]),
            }
        )

    return rows


def main():
    initialize_database()

    rows = load_excel()

    clear_employee_data()
    insert_employee_rows(rows)

    print(f"Inserted {len(rows)} employee leave records.")


if __name__ == "__main__":
    main()