import torch
from torch.utils.data import Dataset
import pandas as pd
import structlog
from typing import Dict, List, Tuple

logger = structlog.get_logger(__name__)

# These columns must exactly match the ones in your output CSVs
WEATHER_FEATURE_COLUMNS = [
    'Avg Temperature (K)', 'Max Temperature (K)', 'Min Temperature (K)',
    'Precipitation (kg m**-2)', 'Relative Humidity (%)',
    'Wind Gust (m s**-1)', 'Wind Speed (m s**-1)',
    'U Component of Wind (m s**-1)', 'V Component of Wind (m s**-1)',
    'Downward Shortwave Radiation Flux (W m**-2)',
    'Vapor Pressure Deficit (kPa)'
]

class LocalCropYieldDataset(Dataset):
    """
    Loads a single CSV (train/eval/test) with rows for a single year + FIPS.
    Converts weather time series into a torch tensor and yield into a scalar.
    """

    def __init__(self, csv_path: str, transform=None):
        self.transform = transform
        self._samples: List[Tuple[torch.Tensor, torch.Tensor, int, str]] = []
        self._by_year: Dict[str, List[int]] = {}
        self._fips_to_id: Dict[str, int] = {}
        self._id_to_fips: Dict[int, str] = {}
        self._next_id = 0

        logger.info(f"Loading CSV: {csv_path}")
        self._load(csv_path)
        logger.info(f"Loaded {len(self._samples)} samples")

    def _load(self, path: str):
        df = pd.read_csv(path)
        required = set(WEATHER_FEATURE_COLUMNS + ["Year", "Yield", "FIPS Code"])
        if not required.issubset(df.columns):
            raise ValueError(f"CSV missing columns: {required - set(df.columns)}")

        for (year, fips), group in df.groupby(["Year", "FIPS Code"]):
            ystr = str(year)
            fstr = str(fips)
            if fstr not in self._fips_to_id:
                self._fips_to_id[fstr] = self._next_id
                self._id_to_fips[self._next_id] = fstr
                self._next_id += 1
            fid = self._fips_to_id[fstr]

            arr = group[WEATHER_FEATURE_COLUMNS].dropna().values
            if arr.shape[0] == 0:
                continue
            wt = torch.tensor(arr, dtype=torch.float32)
            yt = torch.tensor(float(group["Yield"].iloc[0]), dtype=torch.float32)

            idx = len(self._samples)
            self._samples.append((wt, yt, fid, ystr))
            self._by_year.setdefault(ystr, []).append(idx)

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, i):
        wt, yt, fid, _ = self._samples[i]
        if self.transform:
            wt = self.transform(wt)
        return wt, yt, fid

    def get_fips_mapping(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        return self._fips_to_id, self._id_to_fips

    def get_num_fips(self) -> int:
        return len(self._fips_to_id)

    def get_years(self) -> List[str]:
        return sorted(self._by_year.keys())

    def get_sample_indices_by_year(self) -> Dict[str, List[int]]:
        return self._by_year

class MultiCropYieldDataset(Dataset):
    def __init__(self, csv_path: str, transform=None):
        self.transform = transform
        self._samples = []
        self._by_year = {}
        self._fips_to_id = {}
        self._id_to_fips = {}
        self._crop_to_id = {}
        self._id_to_crop = {}
        self._fips_next_id = 0
        self._crop_next_id = 0

        logger.info(f"Loading CSV: {csv_path}")
        self._load(csv_path)
        logger.info(f"Loaded {len(self._samples)} samples")

    def _load(self, path: str):
        df = pd.read_csv(path)
        crop_columns = [
            "Yield_corn", "Yield_soybeans", "Yield_cotton", "Yield_winterwheat"
        ]
        required = set(WEATHER_FEATURE_COLUMNS + ["Year", "FIPS Code"] + crop_columns)
        if not required.issubset(df.columns):
            raise ValueError(f"CSV missing columns: {required - set(df.columns)}")

        crops = {
            "corn": "Yield_corn",
            "soybeans": "Yield_soybeans",
            "cotton": "Yield_cotton",
            "winterwheat": "Yield_winterwheat"
        }

        for (year, fips), group in df.groupby(["Year", "FIPS Code"]):
            for crop_name, target_col in crops.items():
                if pd.isna(group[target_col].iloc[0]):
                    continue

                ystr = str(year)
                fstr = str(fips)

                if fstr not in self._fips_to_id:
                    self._fips_to_id[fstr] = self._fips_next_id
                    self._id_to_fips[self._fips_next_id] = fstr
                    self._fips_next_id += 1
                fid = self._fips_to_id[fstr]

                if crop_name not in self._crop_to_id:
                    self._crop_to_id[crop_name] = self._crop_next_id
                    self._id_to_crop[self._crop_next_id] = crop_name
                    self._crop_next_id += 1
                cid = self._crop_to_id[crop_name]

                arr = group[WEATHER_FEATURE_COLUMNS].dropna().values
                if arr.shape[0] == 0:
                    continue
                wt = torch.tensor(arr, dtype=torch.float32)
                yt = torch.tensor(float(group[target_col].iloc[0]), dtype=torch.float32)

                idx = len(self._samples)
                self._samples.append((wt, yt, fid, cid, ystr))
                self._by_year.setdefault(ystr, []).append(idx)

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, i):
        wt, yt, fid, cid, _ = self._samples[i]
        if self.transform:
            wt = self.transform(wt)
        return wt, yt, fid, cid

    def get_num_fips(self):
        return len(self._fips_to_id)

    def get_num_crops(self):
        return len(self._crop_to_id)

    def get_crop_mapping(self):
        return self._crop_to_id, self._id_to_crop