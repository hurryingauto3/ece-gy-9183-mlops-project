import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm

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

""" all_files = list_all_files('data_lake/HRRR/data')

## TODO: Create all FIPS folders from all files in the data lake
for f in tqdm(all_files, desc="Processing files"):
    create_fips_folders_from_csv(f)
    
## TODO: Add in all the years folders to every FIPS folder
years = list_unique_years('data_lake/HRRR/data')
 """

base_dir = 'data_lake_organized/'
sub_dirs = [entry for entry in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, entry))]
print(sub_dirs)