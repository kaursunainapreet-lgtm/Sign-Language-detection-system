import pandas as pd

INPUT_FILE = "two_hand_words_raw.csv"
OUTPUT_FILE = "two_hand_words_clean.csv"

EXPECTED_FEATURES = 84  # two hands
EXPECTED_COLUMNS = EXPECTED_FEATURES + 1  # + label

print("Loading raw data...")
df = pd.read_csv(INPUT_FILE)

print("Original rows:", len(df))
print("Original columns:", len(df.columns))

# Drop rows with missing values
df = df.dropna()

# Keep only rows with correct number of columns
df = df[df.apply(lambda row: len(row) == EXPECTED_COLUMNS, axis=1)]

# Ensure all feature values are numeric
feature_cols = df.columns[:-1]

df[feature_cols] = df[feature_cols].apply(
    pd.to_numeric, errors="coerce"
)

# Drop rows that became NaN after conversion
df = df.dropna()

# Reset index
df = df.reset_index(drop=True)

print("Cleaned rows:", len(df))
print("Final columns:", len(df.columns))

# Save clean CSV
df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Clean CSV saved as {OUTPUT_FILE}")

# Show label distribution
print("\nLabel distribution:")
print(df["label"].value_counts())
