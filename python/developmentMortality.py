# --------------------------------------------------------------------------------
# Data Management with World Bank Data
# BEE2041 example: merging, reshaping, and visualizing
# --------------------------------------------------------------------------------
# Steps I want to look at:
#   1. Downloading data from the World Bank API
#   2. Filtering to actual countries (no aggregates)
#   3. Wide-format merge with diagnostic checks
#   4. Scatter plot: GDP vs Infant Mortality Rate (single year)
#   5. Reshaping to long format
#   6. Long-format merge
#   7. Line plot: IMR over time for a country
#
#
# Notes: This script uses tools we have looked at previously, with the exception
# of the World Bank API for data download (pip install wbgapi)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import wbgapi as wb        

# --------------------------------------------------------------------------------
# 1. DOWNLOAD DATA FROM THE WORLD BANK
# --------------------------------------------------------------------------------
# Indicators:
#   NY.GDP.PCAP.PP.KD  – GDP per capita, PPP (constant 2017 intl $)
#   SH.DYN.MORT        – Infant mortality rate (per 1,000 live births)

YEARS = range(2000, 2023)  
GDP_IND  = "NY.GDP.PCAP.PP.KD"
IMR_IND  = "SH.DYN.MORT"

print("Downloading GDP data...")
gdp_raw = wb.data.DataFrame(GDP_IND, time=YEARS, labels=True)

print("Downloading Infant Mortality Rate data...")
imr_raw = wb.data.DataFrame(IMR_IND, time=YEARS, labels=True)

# wbgapi returns a DataFrame indexed by economy, with columns
# named "YR2000", "YR2001", ... Let's inspect the shape:
print("GDP shape:", gdp_raw.shape)
print("IMR shape:", imr_raw.shape)
print(gdp_raw.head())

# --------------------------------------------------------------------------------
# 2. KEEP ONLY ACTUAL COUNTRIES (drop regional aggregates)
# --------------------------------------------------------------------------------
# wbgapi's income-level list lets us identify true countries.
# Aggregates have an 'aggregate' flag in the economy metadata.

economy_meta = wb.economy.DataFrame()           
true_countries = economy_meta[economy_meta["aggregate"] == False].index.tolist()

print(f"Total economies in WB: {len(economy_meta)}")
print(f"True countries:        {len(true_countries)}")

def keep_countries(df):
    return df[df.index.isin(true_countries)]

gdp = keep_countries(gdp_raw).copy()
imr = keep_countries(imr_raw).copy()

print(f"GDP after filter: {gdp.shape}")
print(f"IMR after filter: {imr.shape}")

# --------------------------------------------------------------------------------
# 3. WIDE-FORMAT JOIN WITH DIAGNOSTIC CHECKS
# --------------------------------------------------------------------------------
# The data has a simple economy index and year columns (YR2000, YR2001, ...).
# We add a suffix to each dataset's year columns before merging so we can
# tell them apart: YR2019_gdp vs YR2019_imr.

gdp_year_cols = [c for c in gdp.columns if c.startswith("YR")]
imr_year_cols = [c for c in imr.columns if c.startswith("YR")]

gdp = gdp.rename(columns={c: c + "_gdp" for c in gdp_year_cols})
imr = imr.rename(columns={c: c + "_imr" for c in imr_year_cols})

# --- Diagnostic checks BEFORE merging ---
print("---------- Merge diagnostics (wide format) ----------")
print("Unique countries in GDP:", gdp.index.nunique())
print("Unique countries in IMR:", imr.index.nunique())

# Countries in GDP but not IMR (and vice versa)
gdp_codes = set(gdp.index)
imr_codes = set(imr.index)
print("In GDP but not IMR:", gdp_codes - imr_codes)
print("In IMR but not GDP:", imr_codes - gdp_codes)

# Check for duplicate index entries
assert not gdp.index.duplicated().any(), "Duplicate rows in GDP!"
assert not imr.index.duplicated().any(), "Duplicate rows in IMR!"
print("No duplicates found")

# --- Merge (outer join so we keep all countries) --- NOTE: We can use either below, my pref is merge as more generalisable
# Option 1:
#wide = gdp.join(imr, how="outer", lsuffix="", rsuffix="_imr_dup")
# Option 2 (using this):
wide = pd.merge(gdp, imr, left_index=True, right_index=True, how="outer", validate="1:1")
wide = wide.drop(columns="Country_y")
wide = wide.rename(columns={"Country_x": "Country"})

# Option 3 (join with key):
#wide = pd.merge(
#    gdp,
#    imr.drop(columns="Country"),
#    on="economy",
#    how="outer",
#    validate="1:1"
#)

print(f"Merged wide shape: {wide.shape}")
print(wide.head())

# --------------------------------------------------------------------------------
# 4. SCATTER PLOT: GDP vs IMR FOR A SINGLE YEAR
# --------------------------------------------------------------------------------
PLOT_YEAR = 2019

gdp_col = f"YR{PLOT_YEAR}_gdp"
imr_col = f"YR{PLOT_YEAR}_imr"

## Commands like this should be inspected very carefully!
snapshot = wide[["Country", gdp_col, imr_col]].dropna()

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(snapshot[gdp_col], snapshot[imr_col],
           alpha=0.6, edgecolors="steelblue", facecolors="steelblue", s=50)

# Annotate a handful of countries for context
highlight = ["CHN", "IND", "NGA", "BRA", "DEU", "JPN", "ZAF"]
for code in highlight:
    if code in snapshot.index:
        row = snapshot.loc[code]
        ax.annotate(code, (row[gdp_col], row[imr_col]),
                    textcoords="offset points", xytext=(5, 3), fontsize=8)

ax.set_xscale("log")
ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
ax.set_xlabel("GDP per capita, PPP (log scale)", fontsize=12)
ax.set_ylabel("Infant Mortality Rate (per 1,000 births)", fontsize=12)
ax.set_title(f"GDP per Capita vs Infant Mortality Rate ({PLOT_YEAR})", fontsize=14)
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("gdp_vs_imr_scatter.png", dpi=150)
plt.show()
print("Scatter plot saved: gdp_vs_imr_scatter.png")

# --------------------------------------------------------------------------------
# 5. RESHAPE EACH DATASET TO LONG FORMAT
# --------------------------------------------------------------------------------
# Currently each dataset is wide: one row per country, one column per year.
# We use pd.melt() to reshape to long: one row per country-year.
#NOTE: The below is somewhat convoluted because we want to do it for two dfs, so make a function
#      In practice, we would likley do this "by hand", and then build up!

def wide_to_long(df, value_col, suffix):
    """Melt wide DataFrame: tidy long DataFrame."""
    year_cols = [c for c in df.columns if c.startswith("YR")]
    long = df.reset_index()[["economy", "Country"] + year_cols].copy()
    long = pd.melt(long,
                   id_vars=["economy", "Country"],    # columns to keep as-is
                   value_vars=year_cols,              # columns to unpivot
                   var_name="year",                   # new column for year label
                   value_name=value_col)              # new column for the value
    # Convert "YR2019_gdp": integer 2019
    long["year"] = long["year"].str.replace("YR", "").str.replace(f"_{suffix}", "").astype(int)
    long.rename(columns={"economy": "code"}, inplace=True)
    long.sort_values(["code", "year"], inplace=True)
    long.reset_index(drop=True, inplace=True)
    return long

gdp_long = wide_to_long(gdp, "gdp_pc", "gdp")
imr_long = wide_to_long(imr, "imr",    "imr")

print("------ Long format ------")
print("GDP long shape:", gdp_long.shape)
print(gdp_long.head())
print("IMR long shape:", imr_long.shape)
print(imr_long.head())

# --------------------------------------------------------------------------------
# 6. MERGE THE LONG DATASETS
# --------------------------------------------------------------------------------
# Key: (code, year) — both datasets must have the same granularity.

print("------ Long-format merge diagnostics ------")
print("GDP long - unique (code, year):", gdp_long[["code","year"]].drop_duplicates().shape[0])
print("IMR long - unique (code, year):", imr_long[["code","year"]].drop_duplicates().shape[0])

# Check for duplicates
assert not gdp_long.duplicated(["code","year"]).any(), "Duplicates in gdp_long!"
assert not imr_long.duplicated(["code","year"]).any(), "Duplicates in imr_long!"
print("No duplicate (code, year) pairs")

long = pd.merge(
    gdp_long,
    imr_long[["code", "year", "imr"]],
    on=["code", "year"],
    how="outer",
    indicator=True           # adds _merge column for diagnostics
)

print("Merge indicator summary:")
print(long["_merge"].value_counts())
long.drop(columns="_merge", inplace=True)

print(f"Final long shape: {long.shape}")
print(long.head())

# --------------------------------------------------------------------------------
# 7. LINE PLOT: IMR OVER TIME FOR SELECTED COUNTRIES
# --------------------------------------------------------------------------------
COUNTRIES = ["USA", "IND", "CHN", "NGA", "BRA"]

fig, ax = plt.subplots(figsize=(10, 6))

for code in COUNTRIES:
    subset = long[long["code"] == code].dropna(subset=["imr"]).sort_values("year")
    label = subset["Country"].iloc[0]          # Grab the country name
    ax.plot(subset["year"], subset["imr"], marker="o", markersize=4, label=label)

ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Infant Mortality Rate (per 1,000 births)", fontsize=12)
ax.set_title("Infant Mortality Rate Over Time", fontsize=14)
ax.legend(title="Country", fontsize=10)
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("imr_over_time.png", dpi=150)
plt.show()
print("Line plot saved as imr_over_time.png")

