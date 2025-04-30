import os
import pandas as pd
from collections import defaultdict

# Root directory of  HRRR data
data_root = '../data/data_lake/HRRR/data'
output_root = '../data/hrrr_combined'

# Dictionary to hold dataframes grouped by FIPS
fips_data = defaultdict(list)

# Walk through all year/state folders and collect CSV files
for year in os.listdir(data_root):
    year_path = os.path.join(data_root, year)
    if not os.path.isdir(year_path):
        continue
    for state in os.listdir(year_path):
        state_path = os.path.join(year_path, state)
        if not os.path.isdir(state_path):
            continue
        for file in os.listdir(state_path):
            if file.endswith('.csv'):
                file_path = os.path.join(state_path, file)
                print(f"Reading {file_path}")
                df = pd.read_csv(file_path)
                for fips, group in df.groupby("FIPS Code"):
                    fips_data[fips].append(group)

# Create the combined folder structure and save CSVs
os.makedirs(output_root, exist_ok=True)

for fips_code, dfs in fips_data.items():
    combined_df = pd.concat(dfs, ignore_index=True)
    fips_folder = os.path.join(output_root, str(fips_code))
    os.makedirs(fips_folder, exist_ok=True)
    output_path = os.path.join(fips_folder, f"FIPS_{fips_code}.csv")
    combined_df.to_csv(output_path, index=False)
    print(f"Saved combined data for FIPS {fips_code} to {output_path}")
