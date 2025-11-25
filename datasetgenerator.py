import pandas as pd
import re
import os

# -------------------------------------------
# Paths
# -------------------------------------------

wb_path = r"C:\Users\jcsch\Downloads\STAT451Project\World_Bank1.csv"
pwt_path = r"C:\Users\jcsch\Downloads\STAT451Project\Penn_World_Table.xlsx"

# -------------------------------------------
# Load World Bank + reshape to long
# -------------------------------------------

df = pd.read_csv(wb_path, encoding="latin-1")

clean_cols = []
for col in df.columns:
    m = re.search(r"\d{4}", col)
    if m:
        clean_cols.append(m.group(0))
    else:
        clean_cols.append(col)
df.columns = clean_cols

year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}", c)]
id_cols = [c for c in df.columns if c not in year_cols]

df_long = df.melt(
    id_vars=id_cols,
    value_vars=year_cols,
    var_name="year",
    value_name="wb_value"
)

df_long["year"] = df_long["year"].astype(int)

# -------------------------------------------
# Load Penn World Table
# -------------------------------------------

pwt = pd.read_excel(pwt_path, sheet_name="Data")

# normalize headers
pwt.columns = [c.lower().strip() for c in pwt.columns]

# ensure numeric year
pwt["year"] = pwt["year"].astype(int)
# --- Standardize column names ---

# lowercase both
df_long.columns = [c.lower().strip() for c in df_long.columns]
pwt.columns     = [c.lower().strip() for c in pwt.columns]

# rename World Bank "country name" → "country"
if "country name" in df_long.columns:
    df_long = df_long.rename(columns={"country name": "country"})

# ensure PWT already has "country"
if "country" not in pwt.columns:
    raise ValueError("PWT dataset missing 'country' column")

# ensure YEAR is clean
df_long["year"] = df_long["year"].astype(int)
pwt["year"]     = pwt["year"].astype(int)

# --- MERGE ---
merged = df_long.merge(pwt, on=["country", "year"], how="inner")




# -------------------------------------------
# Load Demographics
# -------------------------------------------

demo_path = r"C:\Users\jcsch\Downloads\STAT451Project\Demographics.csv"
demo = pd.read_csv(demo_path, encoding="utf-8-sig")

demo.columns = [c.lower().strip() for c in demo.columns]

# drop first bogus header row
demo = demo[ demo['country'] != 'Country' ]

# clean naming
demo = demo.rename(columns={'t03': 'demographic_indicator'})

# ensure types
demo['year'] = demo['year'].astype(int)

merged = merged.merge(
    demo,
    on=['country','year'],
    how='left'
)


# -------------------------------------------
# SAVE UPDATED PANEL
# -------------------------------------------

out_2 = r"C:\Users\jcsch\Downloads\STAT451Project\Merged_WB_PWT_DEMO.csv"
merged.to_csv(out_2, index=False)

print("\nSaved final merged dataset to:")
print(out_2)
