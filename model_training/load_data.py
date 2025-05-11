# data_loader.py
import torch
from torch.utils.data import Dataset
import pandas as pd
import structlog
from typing import Dict, List, Tuple

logger = structlog.get_logger(__name__)

# Must match exactly the columns in your CSVs after fetch_data.py adds them:
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
    One CSV with columns [Year, FIPS Code, ...weather..., Yield].
    Groups by (Year, FIPS) to produce one sample per county-year.
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
        needed = set(WEATHER_FEATURE_COLUMNS + ["Year","Yield","FIPS Code"])
        if not needed.issubset(df.columns):
            missing = needed - set(df.columns)
            raise ValueError(f"CSV missing: {missing}")

        for (year, fips), group in df.groupby(["Year","FIPS Code"]):
            ystr = str(year)
            fstr = str(fips)
            if fstr not in self._fips_to_id:
                self._fips_to_id[fstr] = self._next_id
                self._id_to_fips[self._next_id] = fstr
                self._next_id += 1
            fid = self._fips_to_id[fstr]

            arr = group[WEATHER_FEATURE_COLUMNS].dropna().values
            if arr.shape[0]==0:
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

    def get_fips_mapping(self) -> Tuple[Dict[str,int], Dict[int,str]]:
        return self._fips_to_id, self._id_to_fips

    def get_num_fips(self) -> int:
        return len(self._fips_to_id)

    def get_years(self) -> List[str]:
        return sorted(self._by_year.keys())

    def get_sample_indices_by_year(self) -> Dict[str,List[int]]:
        return self._by_year
