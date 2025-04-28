import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm

COLUMNS = [
    'Year', 'Month', 'Day', 'Daily/Monthly',
    'State', 'County', 'FIPS Code', 'Grid Index',
    'Lat (llcrnr)', 'Lon (llcrnr)', 'Lat (urcrnr)', 'Lon (urcrnr)',
    'Avg Temperature (K)', 'Max Temperature (K)', 'Min Temperature (K)',
    'Precipitation (kg m**-2)', 'Relative Humidity (%)',
    'Wind Gust (m s**-1)', 'Wind Speed (m s**-1)',
    'U Component of Wind (m s**-1)', 'V Component of Wind (m s**-1)',
    'Downward Shortwave Radiation Flux (W m**-2)',
    'Vapor Pressure Deficit (kPa)'
]

def create_fips_folders_from_csv(csv_path: str, output_base_dir: str = "data_lake_organized/"):
    # Correct column names based on your dataset
    columns = [
        'Year', 'Month', 'Day', 'Daily/Monthly',
        'State', 'County', 'FIPS Code', 'Grid Index',
        'Lat (llcrnr)', 'Lon (llcrnr)', 'Lat (urcrnr)', 'Lon (urcrnr)',
        'Avg Temperature (K)', 'Max Temperature (K)', 'Min Temperature (K)',
        'Precipitation (kg m**-2)', 'Relative Humidity (%)',
        'Wind Gust (m s**-1)', 'Wind Speed (m s**-1)',
        'U Component of Wind (m s**-1)', 'V Component of Wind (m s**-1)',
        'Downward Shortwave Radiation Flux (W m**-2)',
        'Vapor Pressure Deficit (kPa)'
    ]
    
    # Load CSV with correct headers
    df = pd.read_csv(csv_path, header=None, names=columns)
    
    # Get unique FIPS codes
    unique_fips = df['FIPS Code'].unique()
    """ print(f"Found {len(unique_fips)} unique FIPS codes.") """
    unique_fips = unique_fips[1:]
    """ print("FIPS Codes:", unique_fips) """
    
    for fips_code in unique_fips:
        # Some FIPS codes might be floats due to csv parsing, convert to int
        fips_code_int = int(fips_code)
        fips_folder = os.path.join(output_base_dir, str(fips_code_int))
        os.makedirs(fips_folder, exist_ok=True)
        """ print(f"📁 Folder ready for FIPS {fips_code_int}") """
    
    """ print("✅ All FIPS folders created successfully.") """

 
# Base directory
def list_all_files(base_dir):
    base_dir = Path(base_dir)
    directories_final = [
        str(file)
        for year_dir in base_dir.iterdir() if year_dir.is_dir()
        for state_dir in year_dir.iterdir() if state_dir.is_dir()
        for file in state_dir.iterdir() if file.is_file()
    ]
    return directories_final

def list_unique_years(base_dir):
    base_dir = Path(base_dir)
    years = {year_dir.name for year_dir in base_dir.iterdir() if year_dir.is_dir()}
    return sorted(years)

## TODO: UNCOMMENT THIS BLOCK TO CREATE FIPS FOLDERS AND YEARS FOLDERS
base_dir = 'data_lake_organized/'

all_files = list_all_files('data_lake/HRRR/data')
## Create all FIPS folders from all files in the data lake
for f in tqdm(all_files, desc="Processing files"):
    create_fips_folders_from_csv(f)
    
## Add in all the years folders to every FIPS folder
new_base_dir = 'data_lake_organized/'
years = list_unique_years('data_lake/HRRR/data')
sub_dirs = [entry for entry in os.listdir(new_base_dir) if os.path.isdir(os.path.join(new_base_dir, entry))]
for sub_dir in tqdm(sub_dirs, desc="Processing subdirectories"):
    for year in years:
        year_folder = os.path.join(base_dir, sub_dir, year)
        if not os.path.exists(year_folder):
            os.makedirs(year_folder)
            

def process_fips_year_timeseries(data_lake_path="data_lake/HRRR/data/", output_base="data_lake_organized/"):
    """
    Processes the HRRR data to create full year weather time series per FIPS code,
    keeping ONLY daily weather data (removing monthly summaries),
    skipping first row, dropping useless columns, and aggregating by (Year, Month, Day) to get 365/366 rows.
    """
    base_path = Path(data_lake_path)
    years = [p.name for p in base_path.iterdir() if p.is_dir()]
    
    for year in tqdm(years, desc="Processing years"):
        year_path = base_path / year
        
        monthly_files = []
        for state_dir in year_path.iterdir():
            if state_dir.is_dir():
                for file in state_dir.iterdir():
                    if file.is_file() and file.suffix == '.csv':
                        monthly_files.append(file)
        
        if not monthly_files:
            continue  # No files for this year
        
        # 🧹 SKIP first row (bad header inside the CSV)
        dfs = [pd.read_csv(f, header=None, names=COLUMNS, skiprows=1) for f in monthly_files]
        full_year_df = pd.concat(dfs, ignore_index=True)
        
        # Drop NaN FIPS just in case
        full_year_df = full_year_df.dropna(subset=['FIPS Code'])
        
        # Keep only 'Daily' entries
        full_year_df = full_year_df[full_year_df['Daily/Monthly'] == 'Daily']
        
        # Get unique FIPS codes
        unique_fips = full_year_df['FIPS Code'].unique()

        for fips_code in tqdm(unique_fips, desc=f"Processing FIPS codes for {year}", leave=False):
            fips_code_int = int(fips_code)

            # Filter rows for this FIPS
            fips_df = full_year_df[full_year_df['FIPS Code'] == fips_code]

            # Drop irrelevant columns
            fips_df = fips_df.drop(columns=[
                'Lat (llcrnr)', 'Lon (llcrnr)', 'Lat (urcrnr)', 'Lon (urcrnr)',
                'Grid Index', 'Daily/Monthly', 'State', 'County', 'FIPS Code'
            ])

            # 🛠️ Force all weather columns to numeric
            for col in [
                'Avg Temperature (K)', 'Max Temperature (K)', 'Min Temperature (K)',
                'Precipitation (kg m**-2)', 'Relative Humidity (%)',
                'Wind Gust (m s**-1)', 'Wind Speed (m s**-1)',
                'U Component of Wind (m s**-1)', 'V Component of Wind (m s**-1)',
                'Downward Shortwave Radiation Flux (W m**-2)',
                'Vapor Pressure Deficit (kPa)'
            ]:
                if col in fips_df.columns:
                    fips_df[col] = pd.to_numeric(fips_df[col], errors='coerce')

            # 🛠️ Group by (Year, Month, Day) and take mean
            fips_df = fips_df.groupby(['Year', 'Month', 'Day']).mean().reset_index()

            # Save the final clean file
            output_folder = Path(output_base) / str(fips_code_int) / year
            output_folder.mkdir(parents=True, exist_ok=True)
            output_file = output_folder / f"WeatherTimeSeries{year}.csv"
            
            fips_df.to_csv(output_file, index=False)

    print("✅ Completed extracting full-resolution daily FIPS timeseries (365/366 rows each).")

process_fips_year_timeseries()