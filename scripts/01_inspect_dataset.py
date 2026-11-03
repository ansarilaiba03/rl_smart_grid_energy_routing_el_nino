import pandas as pd
from pathlib import Path


# ============================================================
# 1. FILE LOCATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "maharashtra_electricity_2015_2026.csv"


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 70)
print("EL NIÑO SMART GRID - DATASET INSPECTION")
print("=" * 70)

print("\nLoading dataset...")
print(f"File: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")


# ============================================================
# 3. BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("BASIC INFORMATION")
print("=" * 70)

print(f"Number of rows    : {len(df):,}")
print(f"Number of columns : {len(df.columns)}")


# ============================================================
# 4. COLUMN NAMES
# ============================================================

print("\n" + "=" * 70)
print("COLUMNS")
print("=" * 70)

for i, column in enumerate(df.columns, start=1):
    print(f"{i:2}. {column}")


# ============================================================
# 5. FIRST 5 ROWS
# ============================================================

print("\n" + "=" * 70)
print("FIRST 5 ROWS")
print("=" * 70)

print(df.head().to_string())


# ============================================================
# 6. DATA TYPES
# ============================================================

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(df.dtypes.to_string())


# ============================================================
# 7. MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = df.isnull().sum()

missing = missing[missing > 0]

if len(missing) == 0:
    print("No missing values found.")
else:
    print(missing.to_string())


# ============================================================
# 8. DUPLICATE ROWS
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATES")
print("=" * 70)

duplicates = df.duplicated().sum()

print(f"Duplicate rows: {duplicates:,}")


# ============================================================
# 9. DATE INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATE INFORMATION")
print("=" * 70)

if "date" in df.columns:

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    print(f"Minimum date: {df['date'].min()}")
    print(f"Maximum date: {df['date'].max()}")

    print("\nNumber of records by year:")

    yearly_records = df["date"].dt.year.value_counts().sort_index()

    print(yearly_records.to_string())

else:
    print("WARNING: 'date' column not found.")


# ============================================================
# 10. CITIES
# ============================================================

print("\n" + "=" * 70)
print("CITIES")
print("=" * 70)

if "city" in df.columns:

    cities = sorted(df["city"].dropna().unique())

    print(f"Number of cities: {len(cities)}")

    for city in cities:
        print(f" - {city}")

else:
    print("WARNING: 'city' column not found.")


# ============================================================
# 11. ELECTRICITY CATEGORIES
# ============================================================

print("\n" + "=" * 70)
print("ELECTRICITY CATEGORIES")
print("=" * 70)

if "category" in df.columns:

    categories = sorted(df["category"].dropna().unique())

    print(f"Number of categories: {len(categories)}")

    for category in categories:
        print(f" - {category}")

else:
    print("WARNING: 'category' column not found.")


# ============================================================
# 12. PRIORITY TIERS
# ============================================================

print("\n" + "=" * 70)
print("PRIORITY TIERS")
print("=" * 70)

if "priority_tier" in df.columns:

    print(df["priority_tier"].value_counts(dropna=False).to_string())

else:
    print("WARNING: 'priority_tier' column not found.")


# ============================================================
# 13. ENSO PHASE
# ============================================================

print("\n" + "=" * 70)
print("ENSO PHASE")
print("=" * 70)

if "enso_phase" in df.columns:

    print(df["enso_phase"].value_counts(dropna=False).to_string())

else:
    print("WARNING: 'enso_phase' column not found.")


# ============================================================
# 14. NUMERICAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("NUMERICAL SUMMARY")
print("=" * 70)

numeric_columns = df.select_dtypes(include="number").columns

print(
    df[numeric_columns]
    .describe()
    .transpose()
    .to_string()
)


# ============================================================
# 15. IMPORTANT ELECTRICITY VARIABLES
# ============================================================

print("\n" + "=" * 70)
print("IMPORTANT ELECTRICITY VARIABLES")
print("=" * 70)

important_columns = [
    "demand_mwh",
    "allocated_mwh",
    "shortage_mwh",
    "state_demand_mwh",
    "total_available_mwh",
    "state_shortage_mwh",
    "thermal_mwh",
    "hydro_mwh",
    "solar_mwh",
    "wind_mwh",
    "oni",
    "avg_temp_c",
]


for column in important_columns:

    if column in df.columns:

        print(f"\n{column}")

        print(f"  Minimum : {df[column].min():,.2f}")
        print(f"  Maximum : {df[column].max():,.2f}")
        print(f"  Mean    : {df[column].mean():,.2f}")
        print(f"  Median  : {df[column].median():,.2f}")


# ============================================================
# 16. CONSISTENCY CHECKS
# ============================================================

print("\n" + "=" * 70)
print("CONSISTENCY CHECKS")
print("=" * 70)


if all(
    column in df.columns
    for column in [
        "demand_mwh",
        "allocated_mwh",
        "shortage_mwh",
    ]
):

    calculated_shortage = (
        df["demand_mwh"] - df["allocated_mwh"]
    ).clip(lower=0)

    difference = (
        calculated_shortage - df["shortage_mwh"]
    ).abs()

    print(
        "Demand = Allocated + Shortage check:"
    )

    print(
        f"Maximum difference: {difference.max():.6f} MWh"
    )

    if difference.max() < 0.001:
        print("PASS: shortage values are consistent.")
    else:
        print("WARNING: shortage values may not exactly match.")


# ============================================================
# 17. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("DATASET INSPECTION COMPLETE")
print("=" * 70)

print("\nNext step:")
print("We will use these results to design the RL environment.")
print("=" * 70)