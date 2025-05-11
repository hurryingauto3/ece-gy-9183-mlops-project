import torch
from torch.utils.data import Dataset
import pandas as pd
import structlog
from typing import Dict, List, Tuple

logger = structlog.get_logger(__name__)

# Match these to your actual columns after preprocessing
WEATHER_FEATURE_COLUMNS = [
    'Avg Temperature (K)', 'Max Temperature (K)', 'Min Temperature (K)',
    'Precipitation (kg m**-2)', 'Relative Humidity (%)', 'Wind Gust (m s**-1)',
    'Wind Speed (m s**-1)', 'U Component of Wind (m s**-1)', 'V Component of Wind (m s**-1)',
    'Downward Shortwave Radiation Flux (W m**-2)', 'Vapor Pressure Deficit (kPa)'
]

class LocalCropYieldDataset(Dataset):
    """
    Loads crop yield and weather feature data from a local CSV.
    Assumes 'Yield' column exists and one row per day (or timestep).
    """

    def __init__(self, csv_path: str, transform=None):
        self.csv_path = csv_path
        self.transform = transform
        self._samples_with_year: List[Tuple[torch.Tensor, torch.Tensor, int, str]] = []
        self._samples_by_year_indices: Dict[str, List[int]] = {}
        self.fips_to_id: Dict[str, int] = {}
        self.id_to_fips: Dict[int, str] = {}
        self._next_fips_id = 0

        logger.info(f"Loading local dataset from: {csv_path}")
        self._load_csv()

    def _load_csv(self):
        df = pd.read_csv(self.csv_path)

        required_cols = set(WEATHER_FEATURE_COLUMNS + ["Year", "Yield", "FIPS Code"])
        if not required_cols.issubset(set(df.columns)):
            missing = required_cols - set(df.columns)
            raise ValueError(f"CSV missing required columns: {missing}")

        grouped = df.groupby(["Year", "FIPS Code"])
        for (year, fips), group_df in grouped:
            year = str(year)
            fips = str(fips)

            if fips not in self.fips_to_id:
                self.fips_to_id[fips] = self._next_fips_id
                self.id_to_fips[self._next_fips_id] = fips
                self._next_fips_id += 1

            fips_id = self.fips_to_id[fips]

            try:
                features = group_df[WEATHER_FEATURE_COLUMNS].dropna().astype(float).values
                yield_val = float(group_df["Yield"].iloc[0])
                if features.shape[0] == 0:
                    continue

                weather_tensor = torch.tensor(features, dtype=torch.float32)
                yield_tensor = torch.tensor(yield_val, dtype=torch.float32)
                self._samples_with_year.append((weather_tensor, yield_tensor, fips_id, year))

                self._samples_by_year_indices.setdefault(year, []).append(len(self._samples_with_year) - 1)
            except Exception as e:
                logger.warning(f"Failed to process sample for FIPS {fips}, Year {year}: {e}")

        logger.info(f"Loaded {len(self._samples_with_year)} samples from {self.csv_path}")

    def __len__(self):
        return len(self._samples_with_year)

    def __getitem__(self, idx):
        weather_tensor, yield_tensor, fips_id, _ = self._samples_with_year[idx]
        if self.transform:
            weather_tensor = self.transform(weather_tensor)
        return weather_tensor, yield_tensor, fips_id

    def get_fips_mapping(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        return self.fips_to_id, self.id_to_fips

    def get_num_fips(self) -> int:
        return len(self.fips_to_id)

    def get_years(self) -> List[str]:
        return sorted(self._samples_by_year_indices.keys())

    def get_sample_indices_by_year(self) -> Dict[str, List[int]]:
        return self._samples_by_year_indices

    def get_sample_metadata(self, idx) -> Tuple[int, str]:
        _, _, fips_id, year_str = self._samples_with_year[idx]
        return fips_id, year_str
