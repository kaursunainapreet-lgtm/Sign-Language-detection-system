import pandas as pd

# Load original data
df = pd.read_csv("alphabet_data.csv")

# Keep only x and y columns + label
cols_to_keep = [col for col in df.columns if col.startswith("x") or col.startswith("y") or col == "label"]

df_fixed = df[cols_to_keep]

# Save new CSV
df_fixed.to_csv("alphabet_data_xy.csv", index=False)

print("✅ Fixed CSV created: alphabet_data_xy.csv")
print("Total columns:", df_fixed.shape[1])
