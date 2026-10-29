#!/usr/bin/env python3
"""
Plot annual global screen temperature, precipitation, MOC from vmo,
and September NH sea-ice area
from converted RTD NetCDF files.

Includes all ensemble members for:
- c7hist
- c7scen7h (H)
- c7scen7m (M)
- c7scen7vl (VL)

Default input directory:
  /home/rng001/site7/rtd_nc
"""

import argparse
import csv
import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset, num2date
import xarray as xr

time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)


TEMP_SCENARIO_PATTERNS = {
    "historical": "sc_v51p2-c7hist-r*_185001_202112_rtd074.nc",
    "H": "sc_v51p2-c7scen7h-r*_202201_210012_rtd074.nc",
    "M": "sc_v51p2-c7scen7m-r*_202201_210012_rtd074.nc",
    "VL": "sc_v51p2-c7scen7vl-r*_202201_210012_rtd074.nc",
}

ICE_SCENARIO_PATTERNS = {
    "historical": "sc_v51p2-c7hist-r*_185001_202112_nemo_ice_rtd.nc",
    "H": "sc_v51p2-c7scen7h-r*_202201_210012_nemo_ice_rtd.nc",
    "M": "sc_v51p2-c7scen7m-r*_202201_210012_nemo_ice_rtd.nc",
    "VL": "sc_v51p2-c7scen7vl-r*_202201_210012_nemo_ice_rtd.nc",
}

PHYSICAL_SCENARIO_PATTERNS = {
    "historical": "sc_v51p2-c7hist-r*_185001_202112_nemo_physical_rtd.nc",
    "H": "sc_v51p2-c7scen7h-r*_202201_210012_nemo_physical_rtd.nc",
    "M": "sc_v51p2-c7scen7m-r*_202201_210012_nemo_physical_rtd.nc",
    "VL": "sc_v51p2-c7scen7vl-r*_202201_210012_nemo_physical_rtd.nc",
}

CMIP6_ICE_SCENARIO_PATTERNS = {
    "CMIP6 historical": [
        "sc_v5.1-p2-histens-*_nemo_ice_rtd.nc",
    ],
    "SSP1-1.9": [
        "sc_v5.1.p2-s119-*_nemo_ice_rtd.nc",
        "sc_v51p2-c6s119-r*_201501_210012_nemo_ice_rtd.nc",
    ],
    "SSP1-2.6": [
        "sc_v5.1.p2-s126-*_nemo_ice_rtd.nc",
        "sc_v51p2-c6s126-r*_201501_210012_nemo_ice_rtd.nc",
    ],
    "SSP2-4.5": [
        "sc_v5.1.p2-s245-*_nemo_ice_rtd.nc",
    ],
   "SSP4-6.0": [
        "sc_v5.1.p2-s460-*_nemo_ice_rtd.nc",
        "sc_v51p2-c6s460-r*_201501_210012_nemo_ice_rtd.nc",
    ],
    "SSP3-7.0": [
        "sc_v5.1.p2-s370-*_nemo_ice_rtd.nc",
    ],
    "SSP5-8.5": [
        "sc_v5.1-p2-ssp585-*_nemo_ice_rtd.nc",
    ],
}

CMIP6_PHYSICAL_SCENARIO_PATTERNS = {
    "CMIP6 historical": [
        "sc_v5.1-p2-histens-*_nemo_physical_rtd.nc",
    ],
    "SSP1-1.9": [
        "sc_v5.1.p2-s119-*_nemo_physical_rtd.nc",
        "sc_v51p2-c6s119-r*_201501_210012_nemo_physical_rtd.nc",
    ],
    "SSP1-2.6": [
        "sc_v5.1.p2-s126-*_nemo_physical_rtd.nc",
        "sc_v51p2-c6s126-r*_201501_210012_nemo_physical_rtd.nc",
    ],
    "SSP2-4.5": [
        "sc_v5.1.p2-s245-*_nemo_physical_rtd.nc",
    ],
    "SSP4-6.0": [
        "sc_v5.1.p2-s460-*_nemo_physical_rtd.nc",
        "sc_v51p2-c6s460-r*_201501_210012_nemo_physical_rtd.nc",
    ],
    "SSP3-7.0": [
        "sc_v5.1.p2-s370-*_nemo_physical_rtd.nc",
    ],
    "SSP5-8.5": [
        "sc_v5.1-p2-ssp585-*_nemo_physical_rtd.nc",
    ],
}

CMIP6_LABEL_TO_SCENARIO = {
    "CMIP6 historical": "historical",
    "SSP1-1.9": "ssp119",
    "SSP1-2.6": "ssp126",
    "SSP2-4.5": "ssp245",
    "SSP4-6.0": "ssp460",
    "SSP3-7.0": "ssp370",
    "SSP5-8.5": "ssp585",
}

CMIP7_LABEL_TO_SCENARIO = {
    "historical": "historical",
    "H": "scen7-h",
    "M": "scen7-m",
    "VL": "scen7-vl",
}

DEFAULT_CMIP6_HIST_ROOT = "/fs/site7/eccc/crd/cccma/model_output/CMIP6/final/CMIP6/CMIP/CCCma/CanESM5-1/historical"
DEFAULT_CMIP6_SCEN_ROOT = "/fs/site7/eccc/crd/cccma/model_output/CMIP6/final/CMIP6/ScenarioMIP/CCCma/CanESM5-1"
DEFAULT_CMIP7_HIST_MEMBER_DIR_GLOB = (
    "/space/hall8/sitestore/eccc/crd/cccma/users/scrd103/canesm_runs/"
    "v51p2-c7hist-r*/data/nc_output/MIP-DRS7/CMIP7/CMIP/CCCma/CanESM5-1/"
    "historical/r*i1p2f1"
)
DEFAULT_CMIP7_SCEN7H_MEMBER_DIR_GLOB = (
    "/space/hall8/sitestore/eccc/crd/cccma/users/scrd103/canesm_runs/"
    "v51p2-c7scen7h-r*/data/nc_output/MIP-DRS7/CMIP7/ScenarioMIP/CCCma/CanESM5-1/"
    "scen7-h/r*i1p2f1"
)
DEFAULT_CMIP7_SCEN7M_MEMBER_DIR_GLOB = (
    "/space/hall8/sitestore/eccc/crd/cccma/users/scrd103/canesm_runs/"
    "v51p2-c7scen7m-r*/data/nc_output/MIP-DRS7/CMIP7/ScenarioMIP/CCCma/CanESM5-1/"
    "scen7-m/r*i1p2f1"
)
DEFAULT_CMIP7_SCEN7VL_MEMBER_DIR_GLOB = (
    "/space/hall8/sitestore/eccc/crd/cccma/users/scrd103/canesm_runs/"
    "v51p2-c7scen7vl-r*/data/nc_output/MIP-DRS7/CMIP7/ScenarioMIP/CCCma/CanESM5-1/"
    "scen7-vl/r*i1p2f1"
)
CMIP6_EXTRA_SCEN_DIR_GLOBS = {
    scenario: (
        f"/home/scrd103/site8/canesm_runs/v51p2-c6s{tag}-r*/data/nc_output/"
        f"CMIP6/ScenarioMIP/CCCma/CanESM5-1/{scenario}"
    )
    for scenario, tag in (("ssp119", "119"), ("ssp126", "126"), ("ssp460", "460"))
}
CMIP6_EXTRA_HIST_DIR_GLOBS = "/home/scrd106/model_output/canesm-5.1/reference/CCCma/CCCma/CanESM5-v5.1-p2-histens-*/historical/"

CMIP6_TEMP_SCENARIO_PATTERNS = {
    "CMIP6 historical": [
        "sc_v5.1-p2-histens-*_rtd074.nc",
    ],
    "SSP1-1.9": [
        "sc_v5.1.p2-s119-*_rtd074.nc",
        "sc_v51p2-c6s119-r*_201501_210012_rtd074.nc",
    ],
    "SSP1-2.6": [
        "sc_v5.1.p2-s126-*_rtd074.nc",
        "sc_v51p2-c6s126-r*_201501_210012_rtd074.nc",
    ],
    "SSP2-4.5": [
        "sc_v5.1.p2-s245-*_rtd074.nc",
    ],
    "SSP4-6.0": [
        "sc_v5.1.p2-s460-*_rtd074.nc",
        "sc_v51p2-c6s460-r*_201501_210012_rtd074.nc",
    ],
    "SSP3-7.0": [
        "sc_v5.1.p2-s370-*_rtd074.nc",
    ],
    "SSP5-8.5": [
        "sc_v5.1-p2-ssp585-*_rtd074.nc",
    ],
}

CMIP6_ICE_SCENARIO_GREYS = {
    "CMIP6 historical": "0.45",
    "SSP1-1.9": "0.85",
    "SSP1-2.6": "0.78",
    "SSP2-4.5": "0.70",
    "SSP4-6.0": "0.54",
    "SSP3-7.0": "0.62",
    "SSP5-8.5": "0.36",
}

SCENARIO_COLORS = {
    "historical": "tab:gray",
    "H": "tab:red",
    "M": "tab:orange",
    "VL": "tab:blue",
}


def extract_member_number(filename: str) -> int:
    match = re.search(r"-r(\d+)_", filename)
    return int(match.group(1)) if match else 9999


def pick_temperature_variable(var_names: List[str]) -> str:
    # Prefer exact-style names from RTD files, fallback to closest match.
    exact_candidates = [
        v for v in var_names
        if v.startswith("SCREEN_TEMPERATURE_GLOBAL__K__________________ANN_00001")
    ]
    if exact_candidates:
        return exact_candidates[0]

    raise KeyError("Could not find a global screen temperature variable in the NetCDF file")


def pick_precipitation_variable(var_names: List[str]) -> str:
    exact_candidates = [
        v for v in var_names
        if v.startswith("PRECIPITATION_GLOBAL__MMpDAY__________________ANN_00001")
    ]
    if exact_candidates:
        return exact_candidates[0]

    raise KeyError("Could not find a global precipitation variable in the NetCDF file")


def pick_year_variable(var_names: List[str]) -> str:
    candidates = [v for v in var_names if v.startswith("YEAR")]
    if candidates:
        return candidates[0]
    raise KeyError("Could not find YEAR variable in the NetCDF file")


def to_annual_series(years: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    years = np.asarray(years).astype(int)
    values = np.asarray(values).astype(float)

    valid = np.isfinite(years) & np.isfinite(values)
    years = years[valid]
    values = values[valid]

    if years.size == 0:
        return years, values

    # If repeated years exist (e.g., monthly), reduce to annual mean.
    unique_years, inverse = np.unique(years, return_inverse=True)
    annual_vals = np.full(unique_years.shape, np.nan, dtype=float)

    for i in range(unique_years.size):
        member_vals = values[inverse == i]
        annual_vals[i] = np.nanmean(member_vals)

    return unique_years, annual_vals


def read_member_series(nc_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with Dataset(nc_path, mode="r") as ds:
        var_names = list(ds.variables.keys())
        temp_var = pick_temperature_variable(var_names)
        year_var = pick_year_variable(var_names)

        years = np.array(ds.variables[year_var][:]).squeeze()
        values = np.array(ds.variables[temp_var][:]).squeeze()

    years_annual, values_annual = to_annual_series(years, values)
    # Convert from Kelvin to degrees Celsius for plotting.
    values_annual = values_annual - 273.15
    return years_annual, values_annual


def read_precip_member_series(nc_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with Dataset(nc_path, mode="r") as ds:
        var_names = list(ds.variables.keys())
        precip_var = pick_precipitation_variable(var_names)
        year_var = pick_year_variable(var_names)

        years = np.array(ds.variables[year_var][:]).squeeze()
        values = np.array(ds.variables[precip_var][:]).squeeze()

    years_annual, values_annual = to_annual_series(years, values)
    return years_annual, values_annual


def read_september_area_nh_series(nc_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with Dataset(nc_path, mode="r") as ds:
        if "Area_NH" not in ds.variables:
            raise KeyError(f"Area_NH variable not found in {nc_path}")
        if "time" not in ds.variables:
            raise KeyError(f"time variable not found in {nc_path}")

        time_var = ds.variables["time"]
        area_nh = np.array(ds.variables["Area_NH"][:]).squeeze().astype(float)

        fill_value = getattr(ds.variables["Area_NH"], "_FillValue", None)
        if fill_value is not None:
            area_nh = np.where(area_nh == fill_value, np.nan, area_nh)

        units = getattr(time_var, "units", None)
        calendar = getattr(time_var, "calendar", "standard")
        if units is None:
            raise KeyError(f"time units attribute not found in {nc_path}")

        time_vals = np.array(time_var[:]).squeeze()
        dates = num2date(time_vals, units=units, calendar=calendar)

    years = np.array([d.year for d in dates], dtype=int)
    months = np.array([d.month for d in dates], dtype=int)

    sep_mask = (months == 9) & np.isfinite(area_nh)
    sep_years = years[sep_mask]
    sep_vals = area_nh[sep_mask]

    if sep_years.size == 0:
        return np.array([]), np.array([])

    unique_years, inverse = np.unique(sep_years, return_inverse=True)
    annual_sep_vals = np.full(unique_years.shape, np.nan, dtype=float)
    for i in range(unique_years.size):
        annual_sep_vals[i] = np.nanmean(sep_vals[inverse == i])

    # Convert from m^2 to million km^2
    annual_sep_vals = annual_sep_vals / 1e12
    return unique_years, annual_sep_vals


def scenario_mean(members: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    if not members:
        return np.array([]), np.array([])

    all_years = sorted({int(y) for years, _ in members.values() for y in years})
    year_to_idx = {y: i for i, y in enumerate(all_years)}
    stack = np.full((len(members), len(all_years)), np.nan, dtype=float)

    for row, (_, (years, vals)) in enumerate(members.items()):
        for y, v in zip(years, vals):
            stack[row, year_to_idx[int(y)]] = v

    mean_vals = np.nanmean(stack, axis=0)
    return np.array(all_years, dtype=int), mean_vals


def extract_member_token(member_id: str) -> str:
    match = re.search(r"r(\d+)", member_id)
    if match:
        return str(int(match.group(1)))

    match = re.search(r"-(\d+)", member_id)
    if match:
        return str(int(match.group(1)))

    return ""


def baseline_over_period(years: np.ndarray, values: np.ndarray, start_year: int = 1850, end_year: int = 1900) -> float:
    years = np.asarray(years, dtype=int)
    values = np.asarray(values, dtype=float)
    mask = (years >= start_year) & (years <= end_year) & np.isfinite(values)
    if not np.any(mask):
        return np.nan
    return float(np.nanmean(values[mask]))


def apply_member_anomalies(
    members: Dict[str, Tuple[np.ndarray, np.ndarray]],
    historical_members: Dict[str, Tuple[np.ndarray, np.ndarray]],
    start_year: int = 1850,
    end_year: int = 1900,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    baselines_by_key: Dict[str, float] = {}
    baselines_by_token_lists: Dict[str, List[float]] = {}
    historical_baselines: List[float] = []

    for hist_key, (hist_years, hist_vals) in historical_members.items():
        b = baseline_over_period(hist_years, hist_vals, start_year=start_year, end_year=end_year)
        if not np.isfinite(b):
            continue

        baselines_by_key[hist_key] = b
        historical_baselines.append(b)

        token = extract_member_token(hist_key)
        if token:
            baselines_by_token_lists.setdefault(token, []).append(b)

    baselines_by_token = {
        token: float(np.nanmean(vals))
        for token, vals in baselines_by_token_lists.items()
        if len(vals) > 0
    }
    overall_baseline = float(np.nanmean(historical_baselines)) if historical_baselines else np.nan

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for member_key, (years, vals) in members.items():
        baseline = baselines_by_key.get(member_key, np.nan)

        if not np.isfinite(baseline):
            token = extract_member_token(member_key)
            if token in baselines_by_token:
                baseline = baselines_by_token[token]

        if not np.isfinite(baseline):
            baseline = overall_baseline

        vals = np.asarray(vals, dtype=float)
        if np.isfinite(baseline):
            anomaly_vals = vals - baseline
        else:
            anomaly_vals = np.full(vals.shape, np.nan, dtype=float)

        out[member_key] = (np.asarray(years, dtype=int), anomaly_vals)

    return out


def align_series_on_years(
    years_a: np.ndarray,
    values_a: np.ndarray,
    years_b: np.ndarray,
    values_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    years_a = np.asarray(years_a, dtype=int)
    values_a = np.asarray(values_a, dtype=float)
    years_b = np.asarray(years_b, dtype=int)
    values_b = np.asarray(values_b, dtype=float)

    common_years, idx_a, idx_b = np.intersect1d(years_a, years_b, return_indices=True)
    return common_years.astype(int), values_a[idx_a], values_b[idx_b]


def ratio_to_reference_series(
    years: np.ndarray,
    values: np.ndarray,
    ref_years: np.ndarray,
    ref_values: np.ndarray,
    min_abs_ref: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    common_years, aligned_values, aligned_ref = align_series_on_years(years, values, ref_years, ref_values)
    ratio_vals = np.full(aligned_values.shape, np.nan, dtype=float)
    valid = np.isfinite(aligned_values) & np.isfinite(aligned_ref)# & (np.abs(aligned_ref) >= min_abs_ref)
    ratio_vals[valid] = aligned_values[valid] / aligned_ref[valid]
    return common_years, ratio_vals


def series_against_reference(
    years: np.ndarray,
    values: np.ndarray,
    ref_years: np.ndarray,
    ref_values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    _, aligned_values, aligned_ref = align_series_on_years(years, values, ref_years, ref_values)
    valid = np.isfinite(aligned_values) & np.isfinite(aligned_ref)
    return aligned_ref[valid], aligned_values[valid]


def years_and_series_against_reference(
    years: np.ndarray,
    values: np.ndarray,
    ref_years: np.ndarray,
    ref_values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    common_years, aligned_values, aligned_ref = align_series_on_years(years, values, ref_years, ref_values)
    valid = np.isfinite(aligned_values) & np.isfinite(aligned_ref)
    return common_years[valid], aligned_ref[valid], aligned_values[valid]


def change_at_latest_warming_crossing(
    years: np.ndarray,
    gsat: np.ndarray,
    values: np.ndarray,
    level: float,
    direction: str = "up",
) -> Tuple[float, float]:
    """Return (crossing year, value) at the latest crossing of `level`, linearly interpolated.

    `direction` is "up" for rising through the level or "down" for falling through it.
    """
    order = np.argsort(years)
    years, gsat, values = years[order], gsat[order], values[order]
    if direction == "up":
        crossings = np.where((gsat[:-1] < level) & (gsat[1:] >= level))[0]
    else:
        crossings = np.where((gsat[:-1] >= level) & (gsat[1:] < level))[0]
    if crossings.size == 0:
        return np.nan, np.nan
    i = crossings[-1]
    frac = (level - gsat[i]) / (gsat[i + 1] - gsat[i])
    return (
        years[i] + frac * (years[i + 1] - years[i]),
        values[i] + frac * (values[i + 1] - values[i]),
    )


def set_robust_ylim(
    ax: plt.Axes,
    series_collection: List[Tuple[np.ndarray, np.ndarray]],
    lower_quantile: float = 0.02,
    upper_quantile: float = 0.98,
    pad_fraction: float = 0.08,
) -> None:
    finite_chunks: List[np.ndarray] = []
    for _, values in series_collection:
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            finite_chunks.append(vals)

    if not finite_chunks:
        return

    merged = np.concatenate(finite_chunks)
    lower = float(np.nanquantile(merged, lower_quantile))
    upper = float(np.nanquantile(merged, upper_quantile))

    if not np.isfinite(lower) or not np.isfinite(upper):
        return

    if np.isclose(lower, upper):
        pad = max(0.1, abs(lower) * pad_fraction)
        ax.set_ylim(lower - pad, upper + pad)
        return

    pad = (upper - lower) * pad_fraction
    ax.set_ylim(lower - pad, upper + pad)




MAX_ENSEMBLE_SIZE = 10


def natural_member_key(path: str) -> Tuple[int, str]:
    match = re.search(r"-r?(\d+)_", os.path.basename(path))
    return (int(match.group(1)) if match else 9999, path)


def collect_files(input_dir: str, scenario_patterns: Dict[str, str]) -> Dict[str, List[str]]:
    files = {}
    for scenario, pattern in scenario_patterns.items():
        paths = sorted(glob.glob(os.path.join(input_dir, pattern)), key=lambda p: extract_member_number(os.path.basename(p)))
        files[scenario] = paths[:MAX_ENSEMBLE_SIZE]
    return files


def collect_cmip6_ice_files_by_scenario(input_dir: str) -> Dict[str, List[str]]:
    files_by_scenario: Dict[str, List[str]] = {}
    for scenario, patterns in CMIP6_ICE_SCENARIO_PATTERNS.items():
        matched = []
        for pattern in patterns:
            matched.extend(glob.glob(os.path.join(input_dir, pattern)))
        files_by_scenario[scenario] = sorted(set(matched), key=natural_member_key)[:MAX_ENSEMBLE_SIZE]
    return files_by_scenario


def collect_cmip6_temp_files_by_scenario(input_dir: str) -> Dict[str, List[str]]:
    files_by_scenario: Dict[str, List[str]] = {}
    for scenario, patterns in CMIP6_TEMP_SCENARIO_PATTERNS.items():
        matched = []
        for pattern in patterns:
            matched.extend(glob.glob(os.path.join(input_dir, pattern)))
        files_by_scenario[scenario] = sorted(set(matched), key=natural_member_key)[:MAX_ENSEMBLE_SIZE]
    return files_by_scenario


def collect_cmip6_physical_files_by_scenario(input_dir: str) -> Dict[str, List[str]]:
    files_by_scenario: Dict[str, List[str]] = {}
    for scenario, patterns in CMIP6_PHYSICAL_SCENARIO_PATTERNS.items():
        matched = []
        for pattern in patterns:
            matched.extend(glob.glob(os.path.join(input_dir, pattern)))
        files_by_scenario[scenario] = sorted(set(matched), key=natural_member_key)[:MAX_ENSEMBLE_SIZE]
    return files_by_scenario


def discover_cmip6_cmor_files(
    cmip6_hist_root: str,
    cmip6_scen_root: str,
    scenario: str,
    grid: str,
    variable: str,
) -> Dict[str, List[str]]:
    base_dirs = [cmip6_hist_root] if scenario == "historical" else [
        os.path.join(cmip6_scen_root, scenario)
    ]
    if scenario == "historical":
        base_dirs.extend(sorted(glob.glob(CMIP6_EXTRA_HIST_DIR_GLOBS)))
    else:
        extra_glob = CMIP6_EXTRA_SCEN_DIR_GLOBS.get(scenario)
        if extra_glob:
            base_dirs.extend(sorted(glob.glob(extra_glob)))

    member_dirs: List[str] = []
    for base_dir in base_dirs:
        member_dirs.extend(sorted(glob.glob(os.path.join(base_dir, "r*i1p2f*"))))

    files_by_member: Dict[str, List[str]] = {}
    for member_dir in member_dirs:
        member = os.path.basename(member_dir)
        version_dirs = sorted(glob.glob(os.path.join(member_dir, "Omon", variable, grid, "v*")))
        if not version_dirs:
            continue
        latest_version = version_dirs[-1]
        pattern = f"{variable}_Omon_*_{scenario}_{member}_{grid}_*.nc"
        files = sorted(glob.glob(os.path.join(latest_version, pattern)))
        if files:
            files_by_member[member] = files
    return files_by_member


def discover_cmip7_cmor_files(
    member_dir_glob: str,
    scenario: str,
    model: str,
    variable: str,
) -> Dict[str, List[str]]:
    files_by_member: Dict[str, List[str]] = {}
    for member_dir in sorted(glob.glob(member_dir_glob)):
        if not os.path.isdir(member_dir):
            continue
        matches = []
        for nc_file in sorted(glob.glob(os.path.join(member_dir, "**", "*.nc"), recursive=True)):
            name = os.path.basename(nc_file)
            if not name.startswith(variable + "_") or "_mon_" not in name:
                continue
            if model not in name or f"_{scenario}_" not in name:
                continue
            if re.search(rf"_{re.escape(scenario)}_(r\d+i\d+p\d+f\d+)_", name):
                matches.append(nc_file)
        if matches:
            member_match = re.search(
                rf"_{re.escape(scenario)}_(r\d+i\d+p\d+f\d+)_",
                os.path.basename(matches[0]),
            )
            if member_match:
                files_by_member[member_match.group(1)] = matches
    return files_by_member


def member_id_from_path(path: str) -> Optional[str]:
    """Take a physical RTD path and return the member ID in the form `r#i1p2f1`,
    which is a key in the CMOR vmo file dictionaries.
    """
    filename = os.path.basename(path)
    match = re.search(r"^sc_[a-zA-Z0-9-.]+?(\d{1,2})_\d{6}_\d{6}_nemo_physical_rtd.nc", filename)
    if match:
        member_number = int(match.group(1))  # int() to strip leading 0's
        return f"r{member_number}i1p2f1"
    return None


def matching_cmor_files(files_by_member: Dict[str, List[str]], physical_path: str) -> List[str]:
    physical_member = member_id_from_path(physical_path)
    if physical_member is None:
        raise ValueError(f"Could not identify ensemble member in {physical_path}")
    if physical_member in files_by_member:
        return files_by_member[physical_member]
    raise FileNotFoundError(
        f"No cmor files found for ensemble member {physical_member} from {physical_path}"
    )


def read_moc_from_vmo_files(paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    vmo = xr.open_mfdataset(
        paths,
        decode_times=time_coder,
        concat_dim="time",
        combine="nested",
        data_vars=["vmo"],
        compat="override",
        coords="minimal",
        combine_attrs="override",
        drop_variables=["time_bnds", "lev_bnds", "vertices_latitude", "vertices_longitude"],
    )["vmo"]
    return moc_from_vmo(vmo)


def moc_from_vmo(vmo: xr.DataArray) -> Tuple[np.ndarray, np.ndarray]:
    """Annual maximum Atlantic overturning streamfunction (Sv) from vmo (time, lev, j, i) [kg/s]."""
    j = 188  # In Atlantic, mean latitude is 26.495161056518555
    ii = slice(208, 274)  # Atlantic boundaries (this slice will get all ocean cells, no extra land at lev=0)
    k = np.argmax(vmo.lev.data >= 500)  # index to first level that is 500m or deeper. k=19 for orca1
    assert k == 19, "Expected k = 19 in moc_from_vmo. Is the grid not ORCA1?"
    rho0 = 1035.0  # Boussinesq reference density [kg m-3]

    # Cumulatively sum y-ward transport from the bottom up to level k, integrate zonally,
    # take the minimum over depth, then convert to Sv with the sign convention applied.
    moc = vmo.isel(j=j, i=ii, lev=slice(None, k - 1, -1)).cumsum(dim="lev").sum(dim="i").min(dim="lev") / (
        -rho0 * 1e6
    )
    years = vmo.time.dt.year
    return to_annual_series(years.values, moc.values)


# --- Global warming vs cumulative diagnosed CO2 emissions (panel a of scaling figure) ---

CC_CMIP7_SCENARIO_PATTERNS = {
    "historical": "sc_v51p2-c7hist-r*_185001_202112_rtd074.nc",
    "H": "sc_v51p2-c7scen7h-r*_202201_210012_rtd074.nc",
    "M": "sc_v51p2-c7scen7m-r*_202201_210012_rtd074.nc",
    "VL": "sc_v51p2-c7scen7vl-r*_202201_210012_rtd074.nc",
}


CC_CMIP6_SCENARIO_PATTERNS = {
    "CMIP6 historical": [
        "sc_v5.1-p2-histens-*_rtd074.nc",
    ],
    "SSP1-1.9": [
        "sc_v5.1.p2-s119-*_rtd074.nc",
        "sc_v51p2-c6s119-r*_201501_210012_rtd074.nc",
    ],
    "SSP1-2.6": [
        "sc_v5.1.p2-s126-*_rtd074.nc",
        "sc_v51p2-c6s126-r*_201501_210012_rtd074.nc",
    ],
    "SSP2-4.5": [
        "sc_v5.1.p2-s245-*_rtd074.nc",
    ],
    "SSP4-6.0": [
        "sc_v5.1.p2-s460-*_rtd074.nc",
        "sc_v51p2-c6s460-r*_201501_210012_rtd074.nc",
    ],
    "SSP3-7.0": [
        "sc_v5.1.p2-s370-*_rtd074.nc",
    ],
    "SSP5-8.5": [
        "sc_v5.1-p2-ssp585-*_rtd074.nc",
    ],
}


CC_CMIP6_SCENARIO_GREYS = {
    "CMIP6 historical": "0.45",
    "SSP1-1.9": "0.85",
    "SSP1-2.6": "0.78",
    "SSP2-4.5": "0.70",
    "SSP4-6.0": "0.62",
    "SSP3-7.0": "0.54",
    "SSP5-8.5": "0.36",
}


CC_CMIP7_COLORS = {
    "historical": "black",
    "H": "tab:red",
    "M": "tab:orange",
    "VL": "tab:blue",
}


CC_CMIP7_ORDER = ["historical", "H", "M", "VL"]


CC_CMIP6_ORDER = [
    "CMIP6 historical",
    "SSP1-1.9",
    "SSP1-2.6",
    "SSP2-4.5",
    "SSP4-6.0",
    "SSP3-7.0",
    "SSP5-8.5",
]


CC_HISTORICAL_START_YEAR = 1850


CC_ONEPCT_PATTERN = "sc_v51p2-c71ppy-r1_*_rtd074.nc"


CC_ONEPCT_BRANCH_YEAR = 6600


CC_CMIP6_SCENARIO_START_YEAR = 2015


CC_CMIP7_SCENARIO_START_YEAR = 2022


CC_DEFAULT_BRANCH_TIMES_CSV = "/home/rng001/plots/scenarios/canesm5-1_tas_precip_regional_anoms_2x2_branch_times.csv"


CC_DEFAULT_CMIP6_PICONTROL = "/home/rng001/site7/rtd_nc/sc_v510b-p2-pict-joined_600001_901712_rtd074.nc"


CC_DEFAULT_CMIP7_PICONTROL = "/home/rng001/site7/rtd_nc/sc_v51p2-c7pic-r1_650001_749912_rtd074.nc"


CC_TEMP_VAR = "SCREEN_TEMPERATURE_GLOBAL__K__________________ANN_00001"


def cc_extract_member_number(filename: str) -> int:
    match = re.search(r"-r(\d+)_", filename)
    return int(match.group(1)) if match else 9999


def cc_member_key_from_filename(filename: str) -> str:
    match = re.search(r"-r(\d+)_", filename)
    if match:
        return f"r{int(match.group(1))}"

    match = re.search(r"-(\d+)_", filename)
    if match:
        return f"r{int(match.group(1))}"

    return filename


def cc_pick_year_variable(var_names: List[str]) -> str:
    candidates = [v for v in var_names if v.startswith("YEAR")]
    if candidates:
        return candidates[0]
    raise KeyError("Could not find YEAR variable in the NetCDF file")


def cc_to_annual_series(years: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    years = np.asarray(years).astype(int)
    values = np.asarray(values).astype(float)

    valid = np.isfinite(years) & np.isfinite(values)
    years = years[valid]
    values = values[valid]

    if years.size == 0:
        return years, values

    unique_years, inverse = np.unique(years, return_inverse=True)
    annual_vals = np.full(unique_years.shape, np.nan, dtype=float)

    for i in range(unique_years.size):
        annual_vals[i] = np.nanmean(values[inverse == i])

    return unique_years, annual_vals


def cc_read_member_variable_series(nc_path: str, variable_name: str) -> Tuple[np.ndarray, np.ndarray]:
    with Dataset(nc_path, mode="r") as ds:
        var_names = list(ds.variables.keys())
        year_var = cc_pick_year_variable(var_names)

        if variable_name not in ds.variables:
            raise KeyError(f"Variable '{variable_name}' not found in {nc_path}")

        years = np.array(ds.variables[year_var][:]).squeeze()
        values = np.array(ds.variables[variable_name][:]).squeeze()

    return cc_to_annual_series(years, values)


def cc_read_temperature_series_celsius(nc_path: str) -> Tuple[np.ndarray, np.ndarray]:
    years, vals_k = cc_read_member_variable_series(nc_path, CC_TEMP_VAR)
    return years, vals_k - 273.15


def cc_merge_series(
    years_a: np.ndarray,
    vals_a: np.ndarray,
    years_b: np.ndarray,
    vals_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    merged = {}

    for y, v in zip(np.asarray(years_a, dtype=int), np.asarray(vals_a, dtype=float)):
        merged[int(y)] = float(v)
    for y, v in zip(np.asarray(years_b, dtype=int), np.asarray(vals_b, dtype=float)):
        merged[int(y)] = float(v)

    years = np.array(sorted(merged.keys()), dtype=int)
    vals = np.array([merged[y] for y in years], dtype=float)
    return years, vals


def cc_cumulative_sum(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    out = np.full(vals.shape, np.nan, dtype=float)

    running = 0.0
    for i, v in enumerate(vals):
        if np.isfinite(v):
            running += v
        out[i] = running

    return out


def cc_align_on_common_years(
    years_a: np.ndarray,
    vals_a: np.ndarray,
    years_b: np.ndarray,
    vals_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    set_a = {int(y) for y in years_a}
    set_b = {int(y) for y in years_b}
    common_years = np.array(sorted(set_a & set_b), dtype=int)

    if common_years.size == 0:
        return common_years, np.array([], dtype=float), np.array([], dtype=float)

    idx_a = {int(y): i for i, y in enumerate(np.asarray(years_a, dtype=int))}
    idx_b = {int(y): i for i, y in enumerate(np.asarray(years_b, dtype=int))}

    aligned_a = np.array([vals_a[idx_a[int(y)]] for y in common_years], dtype=float)
    aligned_b = np.array([vals_b[idx_b[int(y)]] for y in common_years], dtype=float)
    return common_years, aligned_a, aligned_b


def cc_scenario_mean(members: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    if not members:
        return np.array([], dtype=int), np.array([], dtype=float)

    all_years = sorted({int(y) for years, _ in members.values() for y in years})
    year_to_idx = {y: i for i, y in enumerate(all_years)}
    stack = np.full((len(members), len(all_years)), np.nan, dtype=float)

    for row, (years, vals) in enumerate(members.values()):
        for y, v in zip(years, vals):
            stack[row, year_to_idx[int(y)]] = float(v)

    valid_counts = np.sum(np.isfinite(stack), axis=0)
    summed_vals = np.nansum(stack, axis=0)
    mean_vals = np.full(len(all_years), np.nan, dtype=float)
    good = valid_counts > 0
    mean_vals[good] = summed_vals[good] / valid_counts[good]
    return np.array(all_years, dtype=int), mean_vals


def cc_collect_cmip7_files(input_dir: str) -> Dict[str, Dict[str, str]]:
    files: Dict[str, Dict[str, str]] = {}
    for scenario, pattern in CC_CMIP7_SCENARIO_PATTERNS.items():
        paths = sorted(
            glob.glob(os.path.join(input_dir, pattern)),
            key=lambda p: cc_extract_member_number(os.path.basename(p)),
        )[:MAX_ENSEMBLE_SIZE]
        scenario_map = {}
        for path in paths:
            member_key = cc_member_key_from_filename(os.path.basename(path))
            scenario_map[member_key] = path
        files[scenario] = scenario_map
    return files


def cc_collect_cmip6_files(input_dir: str) -> Dict[str, Dict[str, str]]:
    files: Dict[str, Dict[str, str]] = {}
    for scenario, patterns in CC_CMIP6_SCENARIO_PATTERNS.items():
        matched = []
        for pattern in patterns:
            matched.extend(glob.glob(os.path.join(input_dir, pattern)))

        scenario_map = {}
        for path in sorted(set(matched), key=natural_member_key)[:MAX_ENSEMBLE_SIZE]:
            member_key = cc_member_key_from_filename(os.path.basename(path))
            scenario_map[member_key] = path
        files[scenario] = scenario_map
    return files


def cc_append_member_series(
    historical_path: str,
    scenario_path: str,
    reader,
) -> Tuple[np.ndarray, np.ndarray]:
    hist_years, hist_vals = reader(historical_path)
    scen_years, scen_vals = reader(scenario_path)
    return cc_merge_series(hist_years, hist_vals, scen_years, scen_vals)


def cc_subtract_control(
    years: np.ndarray,
    vals: np.ndarray,
    control: Dict[int, float],
    family: str,
    member: str,
    branch_times: Dict[Tuple[str, str], int],
) -> np.ndarray:
    member_id = f"r{int(member[1:])}i1p2f1"
    if (family, member_id) not in branch_times:
        raise KeyError(f"No piControl branch time for {family} member {member_id}")
    control_years = branch_times[(family, member_id)] + (years - CC_HISTORICAL_START_YEAR)
    reference = np.array([control.get(int(y), np.nan) for y in control_years], dtype=float)
    return vals - reference


def cc_build_family_means(
    files_by_scenario: Dict[str, Dict[str, str]],
    order: List[str],
    historical_name: str,
    diagnosed: Dict[str, Tuple[np.ndarray, np.ndarray]],
    temp_control: Dict[int, float],
    family: str,
    branch_times: Dict[Tuple[str, str], int],
    scenario_start_year: int,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, int]]:
    historical_members = files_by_scenario.get(historical_name, {})

    temp_members_by_scenario: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {s: {} for s in order}

    for scenario in order:
        scenario_members = files_by_scenario.get(scenario, {})

        if scenario == historical_name:
            for member, path in scenario_members.items():
                t_years, t_vals = cc_read_temperature_series_celsius(path)
                t_vals = cc_subtract_control(t_years, t_vals, temp_control, family, member, branch_times)
                temp_members_by_scenario[scenario][member] = (t_years, t_vals)
            continue

        for member, scen_path in scenario_members.items():
            hist_path = historical_members.get(member)
            if hist_path is None:
                continue

            t_years, t_vals = cc_append_member_series(hist_path, scen_path, cc_read_temperature_series_celsius)
            t_vals = cc_subtract_control(t_years, t_vals, temp_control, family, member, branch_times)
            temp_members_by_scenario[scenario][member] = (t_years, t_vals)

    temp_mean_by_scenario = {
        s: cc_scenario_mean(temp_members_by_scenario[s])
        for s in order
    }

    member_counts = {s: len(temp_members_by_scenario[s]) for s in order}

    scenario_xy: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for scenario in order:
        t_years, t_vals = temp_mean_by_scenario[scenario]
        e_years, e_vals = diagnosed.get(scenario, (np.array([], dtype=int), np.array([], dtype=float)))
        years, t_aligned, e_aligned = cc_align_on_common_years(t_years, t_vals, e_years, e_vals)
        if scenario != historical_name:
            keep = years >= scenario_start_year
            t_aligned, e_aligned = t_aligned[keep], e_aligned[keep]
        scenario_xy[scenario] = (e_aligned, t_aligned)

    return scenario_xy, temp_mean_by_scenario, member_counts


CC_BURDEN_VAR = "ATMOSPHERIC_CO2_BURDEN_JAN_1__PG_C__SAMPLED___ANN_00001"


CC_OCEAN_FLUX_VAR = "OCEANmATMOS_CO2_FLUX__PG_CpYR_________________ANN_00001"


CC_NBP_VAR = "NET_BIOME_PRODUCTIVITY__PG_CpYR_______________ANN_00001"


CC_LUC_VARS = [
    "LUC_VEGd_COMBUSTd_EMISSION__PG_CpYR___________ANN_00001",
    "LUC_LITTER_INPUTS__PG_CpYR____________________ANN_00001",
    "LUC_SOIL_CARBON_INPUTS__PG_CpYR_______________ANN_00001",
]


def cc_read_luc_emissions_series(nc_path: str) -> Tuple[np.ndarray, np.ndarray]:
    years = None
    total = None
    for var in CC_LUC_VARS:
        y, v = cc_read_member_variable_series(nc_path, var)
        if years is None:
            years, total = y, v
            continue
        years, total, v_aligned = cc_align_on_common_years(years, total, y, v)
        total = total + v_aligned
    return years, total


def cc_build_component_members(
    files_by_scenario: Dict[str, Dict[str, str]],
    order: List[str],
    historical_name: str,
    reader,
    cumulative: bool,
    control: Optional[Dict[int, float]] = None,
    family: str = "",
    branch_times: Optional[Dict[Tuple[str, str], int]] = None,
) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    historical_members = files_by_scenario.get(historical_name, {})
    result: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}

    for scenario in order:
        members: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for member, path in files_by_scenario.get(scenario, {}).items():
            if scenario == historical_name:
                years, vals = reader(path)
            else:
                hist_path = historical_members.get(member)
                if hist_path is None:
                    continue
                years, vals = cc_append_member_series(hist_path, path, reader)
            if control is not None:
                vals = cc_subtract_control(years, vals, control, family, member, branch_times)
            if cumulative:
                vals = cc_cumulative_sum(vals)
            members[member] = (years, vals)
        result[scenario] = members

    return result


def cc_load_branch_times(csv_path: str) -> Dict[Tuple[str, str], int]:
    branch_times: Dict[Tuple[str, str], int] = {}
    with open(csv_path, newline="", encoding="utf-8") as branch_file:
        for row in csv.DictReader(branch_file):
            branch_year = int(row["YMDH_branch_time_in_parent"].split(":", 1)[0])
            branch_times[(row["family"], row["member"])] = branch_year
    return branch_times


def cc_read_control_by_year(path: str, reader) -> Dict[int, float]:
    years, vals = reader(path)
    return {int(y): float(v) for y, v in zip(years, vals) if np.isfinite(v)}


CC_COMPONENTS = [
    ("a) Atmosphere carbon burden change", CC_BURDEN_VAR, False),
    ("b) Ocean carbon burden change", CC_OCEAN_FLUX_VAR, True),
    ("c) Land carbon burden change", CC_NBP_VAR, True),
    ("d) Cumulative land use change emissions", None, True),
]


def cc_make_reader(var: Optional[str]):
    if var is None:
        return cc_read_luc_emissions_series
    return lambda p: cc_read_member_variable_series(p, var)


def cc_compute_component_means(
    files: Dict[str, Dict[str, str]],
    order: List[str],
    hist_name: str,
    family: str,
    branch_times: Dict[Tuple[str, str], int],
    pic_path: str,
) -> Tuple[
    List[Dict[str, Tuple[np.ndarray, np.ndarray]]],
    Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
]:
    component_members = []
    for _, var, cumulative in CC_COMPONENTS:
        reader = cc_make_reader(var)
        control = cc_read_control_by_year(pic_path, reader)
        component_members.append(
            cc_build_component_members(
                files, order, hist_name, reader, cumulative, control, family, branch_times
            )
        )

    component_means = [
        {scenario: cc_scenario_mean(members[scenario]) for scenario in order}
        for members in component_members
    ]
    cumulative_members: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    for scenario in order:
        scenario_totals = {}
        common_members = set.intersection(
            *(set(component[scenario]) for component in component_members)
        )
        for member in common_members:
            years, total = component_members[0][scenario][member]
            for component in component_members[1:]:
                component_years, component_values = component[scenario][member]
                years, total, aligned_values = cc_align_on_common_years(
                    years, total, component_years, component_values
                )
                total = total + aligned_values
            scenario_totals[member] = (years, total)
        cumulative_members[scenario] = scenario_totals

    return component_means, cumulative_members


def cc_sum_components(
    component_means: List[Dict[str, Tuple[np.ndarray, np.ndarray]]],
    order: List[str],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    total = {}
    for scenario in order:
        years, vals = component_means[0][scenario]
        for comp in component_means[1:]:
            c_years, c_vals = comp[scenario]
            years, vals, c_aligned = cc_align_on_common_years(years, vals, c_years, c_vals)
            vals = vals + c_aligned
        total[scenario] = (years, vals)
    return total


def cc_anomaly_vs_branch(
    years: np.ndarray, vals: np.ndarray, control: Dict[int, float], branch_year: int
) -> np.ndarray:
    control_years = branch_year + (years - years[0])
    reference = np.array([control.get(int(y), np.nan) for y in control_years], dtype=float)
    return vals - reference


def cc_compute_single_run_xy(
    path: str, pic_path: str, branch_year: int
) -> Tuple[np.ndarray, np.ndarray]:
    total_years = total = None
    for _, var, cumulative in CC_COMPONENTS:
        reader = cc_make_reader(var)
        years, vals = reader(path)
        vals = cc_anomaly_vs_branch(years, vals, cc_read_control_by_year(pic_path, reader), branch_year)
        if cumulative:
            vals = cc_cumulative_sum(vals)
        if total is None:
            total_years, total = years, vals
        else:
            total_years, total, vals_aligned = cc_align_on_common_years(total_years, total, years, vals)
            total = total + vals_aligned

    t_years, t_vals = cc_read_temperature_series_celsius(path)
    t_vals = cc_anomaly_vs_branch(
        t_years, t_vals, cc_read_control_by_year(pic_path, cc_read_temperature_series_celsius), branch_year
    )
    _, x_vals, y_vals = cc_align_on_common_years(total_years, total, t_years, t_vals)
    return x_vals, y_vals


def cc_compute_temp_vs_emissions(
    input_dir: str, branch_times_csv: str, cmip6_picontrol: str, cmip7_picontrol: str
) -> dict:
    cmip7_files = cc_collect_cmip7_files(input_dir)
    cmip6_files = cc_collect_cmip6_files(input_dir)

    if sum(len(v) for v in cmip7_files.values()) == 0 and sum(len(v) for v in cmip6_files.values()) == 0:
        raise FileNotFoundError(f"No matching RTD files found in {input_dir}")

    branch_times = cc_load_branch_times(branch_times_csv)
    cmip7_components, cmip7_cumulative_members = cc_compute_component_means(
        cmip7_files, CC_CMIP7_ORDER, "historical", "CMIP7", branch_times, cmip7_picontrol
    )
    cmip6_components, cmip6_cumulative_members = cc_compute_component_means(
        cmip6_files, CC_CMIP6_ORDER, "CMIP6 historical", "CMIP6", branch_times, cmip6_picontrol
    )
    cmip7_total = cc_sum_components(cmip7_components, CC_CMIP7_ORDER)
    cmip6_total = cc_sum_components(cmip6_components, CC_CMIP6_ORDER)

    cmip7_xy, _, cmip7_counts = cc_build_family_means(
        files_by_scenario=cmip7_files,
        order=CC_CMIP7_ORDER,
        historical_name="historical",
        diagnosed=cmip7_total,
        temp_control=cc_read_control_by_year(cmip7_picontrol, cc_read_temperature_series_celsius),
        family="CMIP7",
        branch_times=branch_times,
        scenario_start_year=CC_CMIP7_SCENARIO_START_YEAR,
    )
    cmip6_xy, _, cmip6_counts = cc_build_family_means(
        files_by_scenario=cmip6_files,
        order=CC_CMIP6_ORDER,
        historical_name="CMIP6 historical",
        diagnosed=cmip6_total,
        temp_control=cc_read_control_by_year(cmip6_picontrol, cc_read_temperature_series_celsius),
        family="CMIP6",
        branch_times=branch_times,
        scenario_start_year=CC_CMIP6_SCENARIO_START_YEAR,
    )

    onepct_xy = None
    onepct_files = sorted(glob.glob(os.path.join(input_dir, CC_ONEPCT_PATTERN)))
    if onepct_files:
        onepct_xy = cc_compute_single_run_xy(onepct_files[0], cmip7_picontrol, CC_ONEPCT_BRANCH_YEAR)
    else:
        print(f"WARNING: no file matching {CC_ONEPCT_PATTERN} in {input_dir}")

    return {
        "cmip7_components": cmip7_components,
        "cmip6_components": cmip6_components,
        "cmip7_cumulative_members": cmip7_cumulative_members,
        "cmip6_cumulative_members": cmip6_cumulative_members,
        "cmip7_total": cmip7_total,
        "cmip6_total": cmip6_total,
        "cmip7_xy": cmip7_xy,
        "cmip6_xy": cmip6_xy,
        "cmip7_counts": cmip7_counts,
        "cmip6_counts": cmip6_counts,
        "onepct_xy": onepct_xy,
    }


def cc_plot_temp_vs_emissions(ax, data: dict, short_labels: bool = False) -> None:
    cmip7_xy, cmip7_counts = data["cmip7_xy"], data["cmip7_counts"]
    cmip6_xy, cmip6_counts = data["cmip6_xy"], data["cmip6_counts"]

    if data["onepct_xy"] is not None:
        x_vals, y_vals = data["onepct_xy"]
        ax.plot(x_vals, y_vals, color="black", linestyle="--", linewidth=1.2, zorder=1, label="1%/yr CO2")

    for scenario in CC_CMIP7_ORDER:
        x_vals, y_vals = cmip7_xy.get(scenario, (np.array([]), np.array([])))
        if x_vals.size == 0 or y_vals.size == 0:
            continue

        label = (
            f"CMIP7 historical ensemble mean ({cmip7_counts.get(scenario, 0)} members)"
            if scenario == "historical"
            else f"{scenario} ensemble mean ({cmip7_counts.get(scenario, 0)} members)"
        )
        if short_labels:
            label = "CMIP7 historical" if scenario == "historical" else scenario

        ax.plot(
            x_vals,
            y_vals,
            color=CC_CMIP7_COLORS[scenario],
            linestyle="-",
            linewidth=3.0,
            zorder=3,
            label=label,
        )

    for scenario in CC_CMIP6_ORDER:
        x_vals, y_vals = cmip6_xy.get(scenario, (np.array([]), np.array([])))
        if x_vals.size == 0 or y_vals.size == 0:
            continue

        label = f"{scenario} mean ({cmip6_counts.get(scenario, 0)} members)"
        if short_labels:
            label = scenario

        ax.plot(
            x_vals,
            y_vals,
            color=CC_CMIP6_SCENARIO_GREYS[scenario],
            linestyle="-",
            linewidth=2.6,
            zorder=1.5,
            label=label,
        )

    ax.set_xlabel("Diagnosed cumulative CO2 emissions (Pg C)")
    ax.set_ylabel("Global mean screen temperature anomaly relative to piControl (degC)")
    ax.grid(True, alpha=0.3)


CC_XLIM = (2000, 2100)
CC_SUMMARY_START_YEAR = 2081
CC_SUMMARY_END_YEAR = 2100
CC_DEFAULT_CUMULATIVE_SUMMARY = "/home/rng001/plots/scenarios/cmip7_cmip6_cumulative_co2_emissions_2081-2100.csv"


def cc_write_cumulative_emissions_summary(
    output_path: str,
    cmip6_members: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    cmip7_members: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    cmip6_warming: Dict[str, Tuple[np.ndarray, np.ndarray]],
    cmip7_warming: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    warming_by_family = {"CMIP6": cmip6_warming, "CMIP7": cmip7_warming}

    with open(output_path, "w", newline="", encoding="utf-8") as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow(
            (
                "family",
                "scenario",
                "requested_period_start_year",
                "requested_period_end_year",
                "included_period_start_year",
                "included_period_end_year",
                "years_included",
                "mean_diagnosed_cumulative_co2_emissions_anomaly_pg_c",
                "members_included",
                "mean_global_warming_relative_to_1850_1900_degc",
            )
        )
        for family, scenarios in (("CMIP6", cmip6_members), ("CMIP7", cmip7_members)):
            for scenario, members in scenarios.items():
                member_periods = []
                for years, values in members.values():
                    period_mask = (years >= CC_SUMMARY_START_YEAR) & (years <= CC_SUMMARY_END_YEAR)
                    finite = np.isfinite(values[period_mask])
                    member_periods.append(
                        {
                            int(year): float(value)
                            for year, value in zip(years[period_mask][finite], values[period_mask][finite])
                        }
                    )

                common_years = sorted(
                    set.intersection(*(set(period) for period in member_periods))
                ) if member_periods else []
                member_means = [
                    float(np.mean([period[year] for year in common_years]))
                    for period in member_periods
                ] if common_years else []
                mean_cumulative_emissions = (
                    float(np.mean(member_means)) if member_means else ""
                )
                warming_years, warming_vals = warming_by_family[family].get(
                    scenario, (np.array([], dtype=int), np.array([], dtype=float))
                )
                warming_mask = (
                    (np.asarray(warming_years) >= CC_SUMMARY_START_YEAR)
                    & (np.asarray(warming_years) <= CC_SUMMARY_END_YEAR)
                    & np.isfinite(warming_vals)
                )
                mean_warming = float(np.mean(np.asarray(warming_vals)[warming_mask])) if np.any(warming_mask) else ""
                writer.writerow(
                    (
                        family,
                        scenario,
                        CC_SUMMARY_START_YEAR,
                        CC_SUMMARY_END_YEAR,
                        common_years[0] if common_years else "",
                        common_years[-1] if common_years else "",
                        len(common_years),
                        mean_cumulative_emissions,
                        len(member_means),
                        mean_warming,
                    )
                )


def cc_plot_scenario_lines(ax, cmip6_means, cmip7_means, labelled: bool) -> None:
    families = [
        (cmip6_means, CC_CMIP6_ORDER, "CMIP6 historical", CC_CMIP6_SCENARIO_GREYS, 2.6, 1.5, CC_CMIP6_SCENARIO_START_YEAR),
        (cmip7_means, CC_CMIP7_ORDER, "historical", CC_CMIP7_COLORS, 3.0, 3.0, CC_CMIP7_SCENARIO_START_YEAR),
    ]
    for means, order, hist_name, colors, lw, z, start_year in families:
        for scenario in order:
            years, vals = means[scenario]
            min_year = CC_XLIM[0] if scenario == hist_name else max(start_year, CC_XLIM[0])
            keep = years >= min_year
            years, vals = years[keep], vals[keep]
            if years.size == 0:
                continue
            label = "CMIP7 historical" if scenario == "historical" else scenario
            if not labelled:
                label = "_nolegend_"
            ax.plot(years, vals, color=colors[scenario], linewidth=lw, zorder=z, label=label)
    ax.set_xlim(*CC_XLIM)
    ax.grid(True, alpha=0.3)


def cc_save_figure(fig, output: str) -> None:
    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Saved plot to {output}")


def cc_plot_component_panels(
    cmip6_components: List[Dict[str, Tuple[np.ndarray, np.ndarray]]],
    cmip7_components: List[Dict[str, Tuple[np.ndarray, np.ndarray]]],
    output: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)

    for i, (ax, (title, _, _)) in enumerate(zip(axes.ravel(), CC_COMPONENTS)):
        cc_plot_scenario_lines(ax, cmip6_components[i], cmip7_components[i], labelled=(i == 0))
        ax.set_title(title)
        ax.set_ylabel("Pg C")

    for ax in axes[1]:
        ax.set_xlabel("Year")
    axes[0, 0].legend(loc="upper left", ncol=1, frameon=False, fontsize=10)
    fig.tight_layout(pad=1.0)
    cc_save_figure(fig, output)


def cc_plot_diagnosed_emissions(cmip6_total, cmip7_total, output: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    cc_plot_scenario_lines(ax, cmip6_total, cmip7_total, labelled=True)
    ax.set_title("Diagnosed cumulative CO2 emissions")
    ax.set_xlabel("Year")
    ax.set_ylabel("Pg C")
    ax.legend(loc="upper left", ncol=1, frameon=False, fontsize=10)
    fig.tight_layout(pad=1.0)
    cc_save_figure(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot annual global screen temperature, precipitation, MOC from vmo, and September NH sea-ice area from RTD NetCDF ensembles")
    parser.add_argument("--input-dir", default="/home/rng001/site7/rtd_nc", help="Directory containing converted atmospheric RTD NetCDF files")
    parser.add_argument("--ice-input-dir", default="/home/scrd104/rtd_browser_files/cannemo", help="Directory containing NEMO ice RTD NetCDF files")
    parser.add_argument("--output", default="/home/rng001/plots/scenarios/cmip7_scenarios_climate_response.png", help="Output PNG path")
    parser.add_argument("--scaling-output", default="/home/rng001/plots/scenarios/cmip7_scenarios_climate_response_scaling.png", help="Output PNG path for the GSAT-scaled figure")
    parser.add_argument("--gsat-output", default="/home/rng001/plots/scenarios/cmip7_scenarios_variable_changes_vs_gsat.png", help="Output PNG path for variable changes against GSAT change")
    parser.add_argument("--cmip6-hist-root", default=DEFAULT_CMIP6_HIST_ROOT, help="CMIP6 historical data root for vmo files")
    parser.add_argument("--cmip6-scen-root", default=DEFAULT_CMIP6_SCEN_ROOT, help="CMIP6 scenario data root for vmo files")
    parser.add_argument("--cmip7-hist-member-dir-glob", default=DEFAULT_CMIP7_HIST_MEMBER_DIR_GLOB, help="CMIP7 historical member directories for CMOR files")
    parser.add_argument("--cmip7-scen7h-member-dir-glob", default=DEFAULT_CMIP7_SCEN7H_MEMBER_DIR_GLOB, help="CMIP7 scen7-h member directories for CMOR files")
    parser.add_argument("--cmip7-scen7m-member-dir-glob", default=DEFAULT_CMIP7_SCEN7M_MEMBER_DIR_GLOB, help="CMIP7 scen7-m member directories for CMOR files")
    parser.add_argument("--cmip7-scen7vl-member-dir-glob", default=DEFAULT_CMIP7_SCEN7VL_MEMBER_DIR_GLOB, help="CMIP7 scen7-vl member directories for CMOR files")
    parser.add_argument("--model", default="CanESM5-1", help="CMIP model name used when locating CMOR files")
    parser.add_argument("--grid", default="gn", help="CMIP6 grid label used when locating CMOR files")
    parser.add_argument("--branch-times-csv", default=CC_DEFAULT_BRANCH_TIMES_CSV, help="CSV of historical member piControl branch times")
    parser.add_argument("--cmip6-picontrol", default=CC_DEFAULT_CMIP6_PICONTROL, help="CMIP6 piControl RTD NetCDF file")
    parser.add_argument("--cmip7-picontrol", default=CC_DEFAULT_CMIP7_PICONTROL, help="CMIP7 piControl RTD NetCDF file")
    parser.add_argument("--components-output", default="/home/rng001/plots/scenarios/cmip7_cmip6_co2_budget_components.png", help="Output PNG path for the four-panel carbon budget component plot")
    parser.add_argument("--diagnosed-output", default="/home/rng001/plots/scenarios/cmip7_cmip6_diagnosed_cumulative_co2_emissions.png", help="Output PNG path for the diagnosed cumulative CO2 emissions plot")
    parser.add_argument("--cumulative-summary", default=CC_DEFAULT_CUMULATIVE_SUMMARY, help="CSV output for mean cumulative CO2 emissions in 2081-2100")
    args = parser.parse_args()

    temp_files_by_scenario = collect_files(args.input_dir, TEMP_SCENARIO_PATTERNS)
    ice_files_by_scenario = collect_files(args.ice_input_dir, ICE_SCENARIO_PATTERNS)
    physical_files_by_scenario = collect_files(args.ice_input_dir, PHYSICAL_SCENARIO_PATTERNS)
    cmip7_cmor_member_dir_globs = {
        "historical": args.cmip7_hist_member_dir_glob,
        "H": args.cmip7_scen7h_member_dir_glob,
        "M": args.cmip7_scen7m_member_dir_glob,
        "VL": args.cmip7_scen7vl_member_dir_glob,
    }
    cmip7_vmo_files_by_scenario = {
        scenario: discover_cmip7_cmor_files(
            cmip7_cmor_member_dir_globs[scenario],
            CMIP7_LABEL_TO_SCENARIO[scenario],
            args.model,
            "vmo",
        )
        for scenario in CMIP7_LABEL_TO_SCENARIO
    }

    n_found = sum(len(v) for v in temp_files_by_scenario.values())
    if n_found == 0:
        raise FileNotFoundError(f"No matching RTD NetCDF files found in {args.input_dir}")

    scenario_members: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {
        "historical": {},
        "H": {},
        "M": {},
        "VL": {},
    }

    for scenario, paths in temp_files_by_scenario.items():
        for path in paths:
            fname = os.path.basename(path)
            member_id_match = re.search(r"(r\d+)", fname)
            member_id = member_id_match.group(1) if member_id_match else fname
            years, vals = read_member_series(path)
            scenario_members[scenario][member_id] = (years, vals)

    scenario_precip_members: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {
        "historical": {},
        "H": {},
        "M": {},
        "VL": {},
    }

    for scenario, paths in temp_files_by_scenario.items():
        for path in paths:
            fname = os.path.basename(path)
            member_id_match = re.search(r"(r\d+)", fname)
            member_id = member_id_match.group(1) if member_id_match else fname
            years, vals = read_precip_member_series(path)
            scenario_precip_members[scenario][member_id] = (years, vals)

    scenario_ice_members: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {
        "historical": {},
        "H": {},
        "M": {},
        "VL": {},
    }

    scenario_moc_members: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {
        "historical": {},
        "H": {},
        "M": {},
        "VL": {},
    }

    for scenario, paths in ice_files_by_scenario.items():
        for path in paths:
            fname = os.path.basename(path)
            member_id_match = re.search(r"(r\d+)", fname)
            member_id = member_id_match.group(1) if member_id_match else fname
            years, vals = read_september_area_nh_series(path)
            scenario_ice_members[scenario][member_id] = (years, vals)

    for scenario, paths in physical_files_by_scenario.items():
        for path in paths:
            fname = os.path.basename(path)
            member_id_match = re.search(r"(r\d+)", fname)
            member_id = member_id_match.group(1) if member_id_match else fname
            vmo_paths = matching_cmor_files(cmip7_vmo_files_by_scenario[scenario], path)
            print(f"Using vmo for AMOC in {scenario}, {path}")
            years, vals = read_moc_from_vmo_files(vmo_paths)
            scenario_moc_members[scenario][member_id] = (years, vals)

    scenario_temp_anom_members: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    scenario_precip_anom_members: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    scenario_moc_anom_members: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    scenario_ice_anom_members: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}

    scenario_order = ["historical", "H", "M", "VL"]
    for scenario in scenario_order:
        scenario_temp_anom_members[scenario] = apply_member_anomalies(
            scenario_members[scenario],
            scenario_members["historical"],
        )
        scenario_precip_anom_members[scenario] = apply_member_anomalies(
            scenario_precip_members[scenario],
            scenario_precip_members["historical"],
        )
        scenario_moc_anom_members[scenario] = apply_member_anomalies(
            scenario_moc_members[scenario],
            scenario_moc_members["historical"],
        )
        scenario_ice_anom_members[scenario] = apply_member_anomalies(
            scenario_ice_members[scenario],
            scenario_ice_members["historical"],
        )

    cmip6_ice_files_by_scenario = collect_cmip6_ice_files_by_scenario(args.ice_input_dir)
    cmip6_temp_files_by_scenario = collect_cmip6_temp_files_by_scenario(args.input_dir)
    cmip6_physical_files_by_scenario = collect_cmip6_physical_files_by_scenario(args.ice_input_dir)
    cmip6_vmo_files_by_scenario = {
        label: discover_cmip6_cmor_files(
            args.cmip6_hist_root,
            args.cmip6_scen_root,
            cmip_scenario,
            args.grid,
            "vmo",
        )
        for label, cmip_scenario in CMIP6_LABEL_TO_SCENARIO.items()
    }

    cmip6_temp_members: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {
        scenario: {} for scenario in CMIP6_TEMP_SCENARIO_PATTERNS
    }
    cmip6_precip_members: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {
        scenario: {} for scenario in CMIP6_TEMP_SCENARIO_PATTERNS
    }
    for scenario, paths in cmip6_temp_files_by_scenario.items():
        for path in paths:
            member_key = os.path.basename(path).replace("_rtd074.nc", "")
            years, vals = read_member_series(path)
            cmip6_temp_members[scenario][member_key] = (years, vals)
            p_years, p_vals = read_precip_member_series(path)
            cmip6_precip_members[scenario][member_key] = (p_years, p_vals)

    cmip6_ice_members: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {
        scenario: {} for scenario in CMIP6_ICE_SCENARIO_PATTERNS
    }
    cmip6_moc_members: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {
        scenario: {} for scenario in CMIP6_PHYSICAL_SCENARIO_PATTERNS
    }
    for scenario, paths in cmip6_ice_files_by_scenario.items():
        for path in paths:
            member_key = os.path.basename(path).replace("_nemo_ice_rtd.nc", "")
            years, vals = read_september_area_nh_series(path)
            cmip6_ice_members[scenario][member_key] = (years, vals)

    for scenario, paths in cmip6_physical_files_by_scenario.items():
        if scenario in ("SSP2-4.5", "SSP3-7.0", "SSP5-8.5"):
            # Skip scenarios without a full set of vmo ensemble members, as in plot_scenarios_vmo.py
            continue
        for path in paths:
            member_key = os.path.basename(path).replace("_nemo_physical_rtd.nc", "")
            vmo_paths = matching_cmor_files(cmip6_vmo_files_by_scenario[scenario], path)
            print(f"Using vmo for AMOC in {scenario}, {path}")
            years, vals = read_moc_from_vmo_files(vmo_paths)
            cmip6_moc_members[scenario][member_key] = (years, vals)

    cmip6_temp_anom_members: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    cmip6_precip_anom_members: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    cmip6_moc_anom_members: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    cmip6_ice_anom_members: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 10,
    })
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.2), sharex=True)
    ax_temp, ax_precip, ax_moc, ax_ice = axes.ravel()

    for scenario in scenario_order:
        color = SCENARIO_COLORS[scenario]
        members = scenario_temp_anom_members[scenario]

        if not members:
            print(f"No files found for {scenario}")
            continue

#        for member_id in sorted(members.keys(), key=lambda x: extract_member_number(f"-{x}_")):
#            years, vals = members[member_id]
#            ax_temp.plot(years, vals, color=color, alpha=0.22, linewidth=0.9)

        mean_years, mean_vals = scenario_mean(members)
        mean_color = "black" if scenario == "historical" else color
        label = "CMIP7 historical" if scenario == "historical" else scenario
        ax_temp.plot(mean_years, mean_vals, color=mean_color, linewidth=3.0, label=label, zorder=3)

    cmip6_order = [
        "CMIP6 historical",
        "SSP1-1.9",
        "SSP1-2.6",
        "SSP2-4.5",
        "SSP3-7.0",
        "SSP4-6.0",
        "SSP5-8.5",
    ]

    for scenario in cmip6_order:
        cmip6_temp_anom_members[scenario] = apply_member_anomalies(
            cmip6_temp_members.get(scenario, {}),
            cmip6_temp_members.get("CMIP6 historical", {}),
        )
        cmip6_precip_anom_members[scenario] = apply_member_anomalies(
            cmip6_precip_members.get(scenario, {}),
            cmip6_precip_members.get("CMIP6 historical", {}),
        )
        cmip6_moc_anom_members[scenario] = apply_member_anomalies(
            cmip6_moc_members.get(scenario, {}),
            cmip6_moc_members.get("CMIP6 historical", {}),
        )
        cmip6_ice_anom_members[scenario] = apply_member_anomalies(
            cmip6_ice_members.get(scenario, {}),
            cmip6_ice_members.get("CMIP6 historical", {}),
        )

    for scenario in cmip6_order:
        members = cmip6_temp_anom_members.get(scenario, {})
        grey = CMIP6_ICE_SCENARIO_GREYS[scenario]
        if not members:
            continue

        mean_years, mean_vals = scenario_mean(members)
        mean_color = "0.55" if scenario == "CMIP6 historical" else grey
        ax_temp.plot(
            mean_years,
            mean_vals,
            color=mean_color,
            linewidth=2.6,
            zorder=1.5,
            label=scenario,
        )

    for scenario in scenario_order:
        color = SCENARIO_COLORS[scenario]
        members = scenario_precip_anom_members[scenario]

        if not members:
            print(f"No precipitation files found for {scenario}")
            continue

        mean_years, mean_vals = scenario_mean(members)
        mean_color = "black" if scenario == "historical" else color
        label = "_nolegend_"
        ax_precip.plot(mean_years, mean_vals * 365.0, color=mean_color, linewidth=3.0, label=label, zorder=3)

    for scenario in cmip6_order:
        members = cmip6_precip_anom_members.get(scenario, {})
        grey = CMIP6_ICE_SCENARIO_GREYS[scenario]
        if not members:
            continue

        mean_years, mean_vals = scenario_mean(members)
        mean_color = "0.55" if scenario == "CMIP6 historical" else grey
        ax_precip.plot(
            mean_years,
            mean_vals * 365.0,
            color=mean_color,
            linewidth=2.6,
            zorder=1.5,
            label=f"{scenario} mean ({len(members)} members)",
        )

    for scenario in scenario_order:
        color = SCENARIO_COLORS[scenario]
        members = scenario_moc_anom_members[scenario]

        if not members:
            print(f"No NEMO physical files found for {scenario}")
            continue

        mean_years, mean_vals = scenario_mean(members)
        mean_color = "black" if scenario == "historical" else color
        label = f"CMIP7 historical ensemble mean ({len(members)} members)" if scenario == "historical" else f"{scenario} ensemble mean ({len(members)} members)"
        ax_moc.plot(mean_years, mean_vals, color=mean_color, linewidth=3.0, label=label, zorder=3)

    for scenario in cmip6_order:
        members = cmip6_moc_anom_members.get(scenario, {})
        grey = CMIP6_ICE_SCENARIO_GREYS[scenario]
        if not members:
            continue

        mean_years, mean_vals = scenario_mean(members)
        mean_color = "0.55" if scenario == "CMIP6 historical" else grey
        ax_moc.plot(
            mean_years,
            mean_vals,
            color=mean_color,
            linewidth=2.6,
            zorder=1.5,
            label=f"{scenario} mean ({len(members)} members)",
        )

    for scenario in cmip6_order:
        members = cmip6_ice_members.get(scenario, {})
        grey = CMIP6_ICE_SCENARIO_GREYS[scenario]
        if not members:
            continue

#        for member_key in sorted(members.keys()):
#            years, vals = members[member_key]
#            ax_ice.plot(years, vals, color=grey, alpha=0.28, linewidth=0.8, zorder=1)

        mean_years, mean_vals = scenario_mean(members)
        mean_color = "0.55" if scenario == "CMIP6 historical" else grey
        ax_ice.plot(
            mean_years,
            mean_vals,
            color=mean_color,
            linewidth=2.6,
            zorder=1.5,
            label=f"{scenario} mean ({len(members)} members)",
        )

    for scenario in scenario_order:
        color = SCENARIO_COLORS[scenario]
        members = scenario_ice_members[scenario]

        if not members:
            print(f"No NEMO sea-ice files found for {scenario}")
            continue

#        for member_id in sorted(members.keys(), key=lambda x: extract_member_number(f"-{x}_")):
#            years, vals = members[member_id]
#            ax_ice.plot(years, vals, color=color, alpha=0.22, linewidth=0.9, zorder=2)

        mean_years, mean_vals = scenario_mean(members)
        mean_color = "black" if scenario == "historical" else color
        ax_ice.plot(mean_years, mean_vals, color=mean_color, linewidth=3.0, label=f"{scenario} ensemble mean ({len(members)} members)", zorder=3)

    ax_temp.set_title("a) Global mean temperature anomaly")
    ax_temp.set_xlabel("Year")
    ax_temp.set_ylabel("°C")
    ax_temp.set_xlim(2000, 2100)
    ax_temp.set_ylim(bottom=0)
    ax_temp.grid(True, alpha=0.3)
    ax_temp.legend(loc="upper left", ncol=1, frameon=False)

    ax_precip.set_title("b) Global mean precipitation anomaly")
    ax_precip.set_xlabel("Year")
    ax_precip.set_ylabel("mm/year")
    ax_precip.set_xlim(2000, 2100)
    ax_precip.set_ylim(bottom=0)
    ax_precip.grid(True, alpha=0.3)

    ax_moc.set_title("c) Atlantic Meridional Overturning Circulation anomaly")
    ax_moc.set_xlabel("Year")
    ax_moc.set_ylabel("Sv")
    ax_moc.set_xlim(2000, 2100)
    ax_moc.grid(True, alpha=0.3)

    ax_ice.set_title("d) September Arctic sea ice area")
    ax_ice.set_xlabel("Year")
    ax_ice.set_ylabel("million km²")
    ax_ice.set_xlim(2000, 2100)
    ax_ice.set_ylim(bottom=0)
    ax_ice.grid(True, alpha=0.3)
    fig.tight_layout(pad=1.0)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.savefig(args.output, dpi=180)
    plt.close(fig)

    print(f"Saved plot to {args.output}")
    temp_emissions_data = cc_compute_temp_vs_emissions(
        args.input_dir, args.branch_times_csv, args.cmip6_picontrol, args.cmip7_picontrol
    )

    scenario_temp_display: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    scenario_precip_display: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    scenario_moc_display: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    scenario_ice_display: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for scenario in scenario_order:
        if scenario_temp_anom_members[scenario]:
            scenario_temp_display[scenario] = scenario_mean(scenario_temp_anom_members[scenario])
        if scenario_precip_anom_members[scenario]:
            scenario_precip_display[scenario] = scenario_mean(scenario_precip_anom_members[scenario])
        if scenario_moc_anom_members[scenario]:
            scenario_moc_display[scenario] = scenario_mean(scenario_moc_anom_members[scenario])
        if scenario_ice_anom_members[scenario]:
            scenario_ice_display[scenario] = scenario_mean(scenario_ice_anom_members[scenario])

    cmip6_temp_display: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    cmip6_precip_display: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    cmip6_moc_display: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    cmip6_ice_display: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for scenario in cmip6_order:
        if cmip6_temp_anom_members.get(scenario):
            mean_years, mean_vals = scenario_mean(cmip6_temp_anom_members[scenario])
            cmip6_temp_display[scenario] = (mean_years, mean_vals)
        if cmip6_precip_anom_members.get(scenario):
            mean_years, mean_vals = scenario_mean(cmip6_precip_anom_members[scenario])
            cmip6_precip_display[scenario] = (mean_years, mean_vals)
        if cmip6_moc_anom_members.get(scenario):
            mean_years, mean_vals = scenario_mean(cmip6_moc_anom_members[scenario])
            cmip6_moc_display[scenario] = (mean_years, mean_vals)
        if cmip6_ice_anom_members.get(scenario):
            mean_years, mean_vals = scenario_mean(cmip6_ice_anom_members[scenario])
            cmip6_ice_display[scenario] = (mean_years, mean_vals)

    scenario_temp_ratio_display: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    scenario_precip_ratio_display: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    scenario_moc_ratio_display: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    scenario_ice_ratio_display: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for scenario in scenario_order:
        if scenario in scenario_temp_display:
            temp_years, temp_vals = scenario_temp_display[scenario]
            scenario_temp_ratio_display[scenario] = ratio_to_reference_series(temp_years, temp_vals, temp_years, temp_vals)
        if scenario in scenario_precip_display and scenario in scenario_temp_display:
            val_years, val_vals = scenario_precip_display[scenario]
            temp_years, temp_vals = scenario_temp_display[scenario]
            scenario_precip_ratio_display[scenario] = ratio_to_reference_series(val_years, val_vals, temp_years, temp_vals)
        if scenario in scenario_moc_display and scenario in scenario_temp_display:
            val_years, val_vals = scenario_moc_display[scenario]
            temp_years, temp_vals = scenario_temp_display[scenario]
            scenario_moc_ratio_display[scenario] = ratio_to_reference_series(val_years, val_vals, temp_years, temp_vals)
        if scenario in scenario_ice_display and scenario in scenario_temp_display:
            val_years, val_vals = scenario_ice_display[scenario]
            temp_years, temp_vals = scenario_temp_display[scenario]
            scenario_ice_ratio_display[scenario] = ratio_to_reference_series(val_years, val_vals, temp_years, temp_vals)

    cmip6_temp_ratio_display: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    cmip6_precip_ratio_display: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    cmip6_moc_ratio_display: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    cmip6_ice_ratio_display: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for scenario in cmip6_order:
        if scenario in cmip6_temp_display:
            temp_years, temp_vals = cmip6_temp_display[scenario]
            cmip6_temp_ratio_display[scenario] = ratio_to_reference_series(temp_years, temp_vals, temp_years, temp_vals)
        if scenario in cmip6_precip_display and scenario in cmip6_temp_display:
            val_years, val_vals = cmip6_precip_display[scenario]
            temp_years, temp_vals = cmip6_temp_display[scenario]
            cmip6_precip_ratio_display[scenario] = ratio_to_reference_series(val_years, val_vals, temp_years, temp_vals)
        if scenario in cmip6_moc_display and scenario in cmip6_temp_display:
            val_years, val_vals = cmip6_moc_display[scenario]
            temp_years, temp_vals = cmip6_temp_display[scenario]
            cmip6_moc_ratio_display[scenario] = ratio_to_reference_series(val_years, val_vals, temp_years, temp_vals)
        if scenario in cmip6_ice_display and scenario in cmip6_temp_display:
            val_years, val_vals = cmip6_ice_display[scenario]
            temp_years, temp_vals = cmip6_temp_display[scenario]
            cmip6_ice_ratio_display[scenario] = ratio_to_reference_series(val_years, val_vals, temp_years, temp_vals)

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
    })
    ratio_output = args.scaling_output
    fig_ratio, axes_ratio = plt.subplots(2, 2, figsize=(10.8, 8.2), sharex=False)
    ax_temp_ratio, ax_precip_ratio, ax_moc_ratio, ax_ice_ratio = axes_ratio.ravel()

    for scenario in scenario_order:
        color = SCENARIO_COLORS[scenario]
        if scenario in scenario_precip_ratio_display:
            years, vals = scenario_precip_ratio_display[scenario]
            mean_color = "black" if scenario == "historical" else color
            ax_precip_ratio.plot(years, vals * 365.0, color=mean_color, linewidth=3.0, label="_nolegend_", zorder=3)
        if scenario in scenario_moc_ratio_display:
            years, vals = scenario_moc_ratio_display[scenario]
            mean_color = "black" if scenario == "historical" else color
            ax_moc_ratio.plot(years, vals, color=mean_color, linewidth=3.0, label="_nolegend_", zorder=3)
        if scenario in scenario_ice_ratio_display:
            years, vals = scenario_ice_ratio_display[scenario]
            mean_color = "black" if scenario == "historical" else color
            ax_ice_ratio.plot(years, vals, color=mean_color, linewidth=3.0, label="_nolegend_", zorder=3)

    for scenario in cmip6_order:
        grey = CMIP6_ICE_SCENARIO_GREYS[scenario]
        mean_color = "0.55" if scenario == "CMIP6 historical" else grey
        if scenario in cmip6_precip_ratio_display:
            years, vals = cmip6_precip_ratio_display[scenario]
            ax_precip_ratio.plot(years, vals * 365.0, color=mean_color, linewidth=2.6, zorder=1.5, label=f"{scenario} mean ({len(cmip6_precip_anom_members[scenario])} members)")
        if scenario in cmip6_moc_ratio_display:
            years, vals = cmip6_moc_ratio_display[scenario]
            ax_moc_ratio.plot(years, vals, color=mean_color, linewidth=2.6, zorder=1.5, label=f"{scenario} mean ({len(cmip6_moc_anom_members[scenario])} members)")
        if scenario in cmip6_ice_ratio_display:
            years, vals = cmip6_ice_ratio_display[scenario]
            ax_ice_ratio.plot(years, vals, color=mean_color, linewidth=2.6, zorder=1.5, label=f"{scenario} mean ({len(cmip6_ice_anom_members[scenario])} members)")

    cc_plot_temp_vs_emissions(ax_temp_ratio, temp_emissions_data, short_labels=True)
    ax_temp_ratio.set_title("a) Global warming vs cumulative CO2 emissions")
    ax_temp_ratio.set_xlabel("Cumulative CO2 emissions (PgC)")
    ax_temp_ratio.set_ylabel("°C")
    ax_temp_ratio.legend(loc="lower right", ncol=1, frameon=False)

    ax_precip_ratio.set_title("b) Precipitation change per degree warming")
    ax_precip_ratio.set_xlabel("Year")
    ax_precip_ratio.set_ylabel("mm/year/°C")
    ax_precip_ratio.set_xlim(2000, 2100)
    ax_precip_ratio.set_ylim(0, 25)
    ax_precip_ratio.grid(True, alpha=0.3)

    ax_moc_ratio.set_title("c) Overturning circulation change per degree warming")
    ax_moc_ratio.set_xlabel("Year")
    ax_moc_ratio.set_ylabel("Sv/°C")
    ax_moc_ratio.set_xlim(2000, 2100)
    ax_moc_ratio.set_ylim(-1, 1)
    ax_moc_ratio.grid(True, alpha=0.3)

    ax_ice_ratio.set_title("d) Sea ice area change per degree warming")
    ax_ice_ratio.set_xlabel("Year")
    ax_ice_ratio.set_ylabel("million km²/°C")
    ax_ice_ratio.set_xlim(2000, 2100)
    ax_ice_ratio.set_ylim(-3, 0)
    ax_ice_ratio.grid(True, alpha=0.3)
    fig_ratio.tight_layout(pad=1.0)
    output_dir = os.path.dirname(ratio_output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig_ratio.savefig(ratio_output, dpi=180)
    plt.close(fig_ratio)

    print(f"Saved GSAT-scaled plot to {ratio_output}")

    gsat_output = args.gsat_output
    fig_gsat, axes_gsat = plt.subplots(3, 1, figsize=(6.5, 12.0), sharex=True)
    variable_specs = [
        ("a) Precipitation change", "mm/year", scenario_precip_display, cmip6_precip_display, 365.0),
        ("b) Overturning circulation change", "Sv", scenario_moc_display, cmip6_moc_display, 1.0),
        ("c) September NH sea-ice area change", "million km²", scenario_ice_display, cmip6_ice_display, 1.0),
    ]

    for ax, (title, ylabel, cmip7_values, cmip6_values, scale) in zip(axes_gsat, variable_specs):
        for scenario in scenario_order:
            if scenario not in cmip7_values or scenario not in scenario_temp_display:
                continue
            temp_years, temp_vals = scenario_temp_display[scenario]
            value_years, values = cmip7_values[scenario]
            _, gsat_change, variable_change = years_and_series_against_reference(
                value_years, values, temp_years, temp_vals
            )
            ax.plot(
                gsat_change,
                variable_change * scale,
                color="black" if scenario == "historical" else SCENARIO_COLORS[scenario],
                linewidth=3.0,
                label="CMIP7 historical" if scenario == "historical" else scenario,
                zorder=3,
            )

        for scenario in cmip6_order:
            if scenario not in cmip6_values or scenario not in cmip6_temp_display:
                continue
            temp_years, temp_vals = cmip6_temp_display[scenario]
            value_years, values = cmip6_values[scenario]
            years, gsat_change, variable_change = years_and_series_against_reference(
                value_years, values, temp_years, temp_vals
            )
            period = years < 2015 if scenario == "CMIP6 historical" else years > 2015
            if not np.any(period):
                continue
            ax.plot(
                gsat_change[period],
                variable_change[period] * scale,
                color=CMIP6_ICE_SCENARIO_GREYS[scenario],
                linewidth=2.6,
                label=scenario,
                zorder=1.5,
            )

        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    axes_gsat[-1].set_xlabel("Global mean warming (°C)")
    handles, labels = axes_gsat[0].get_legend_handles_labels()
    fig_gsat.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig_gsat.tight_layout(rect=(0, 0.08, 1, 1), pad=1.0)
    gsat_dir = os.path.dirname(gsat_output)
    if gsat_dir:
        os.makedirs(gsat_dir, exist_ok=True)
    fig_gsat.savefig(gsat_output, dpi=180)
    plt.close(fig_gsat)

    print(f"Saved variable changes against GSAT change plot to {gsat_output}")

    warming_level = 2.0
    print(f"\nChange in global mean precipitation at {warming_level:g}°C global warming (latest rising crossing, plus latest descending crossing where one exists):")
    crossing_sets = [
        (scenario_order, scenario_temp_display, scenario_precip_display, lambda s: "CMIP7 historical" if s == "historical" else f"CMIP7 {s}"),
        (cmip6_order, cmip6_temp_display, cmip6_precip_display, lambda s: s),
    ]
    for order, temp_display, precip_display, name in crossing_sets:
        for scenario in order:
            if scenario not in temp_display or scenario not in precip_display:
                continue
            temp_years, temp_vals = temp_display[scenario]
            precip_years, precip_vals = precip_display[scenario]
            years, gsat_change, precip_change = years_and_series_against_reference(
                precip_years, precip_vals, temp_years, temp_vals
            )
            cross_year, cross_precip = change_at_latest_warming_crossing(
                years, gsat_change, precip_change * 365.0, warming_level
            )
            if np.isnan(cross_year):
                print(f"  {name(scenario)}: {warming_level:g}°C not reached")
            else:
                print(f"  {name(scenario)}: {cross_precip:+.1f} mm/year (crossing at {cross_year:.1f})")
            down_year, down_precip = change_at_latest_warming_crossing(
                years, gsat_change, precip_change * 365.0, warming_level, direction="down"
            )
            if not np.isnan(down_year):
                print(f"  {name(scenario)} (descending): {down_precip:+.1f} mm/year (crossing at {down_year:.1f})")

    cc_write_cumulative_emissions_summary(
        args.cumulative_summary,
        temp_emissions_data["cmip6_cumulative_members"],
        temp_emissions_data["cmip7_cumulative_members"],
        cmip6_temp_display,
        scenario_temp_display,
    )
    print(f"Saved 2081-2100 cumulative CO2 emissions summary to {args.cumulative_summary}")
    cc_plot_component_panels(
        temp_emissions_data["cmip6_components"], temp_emissions_data["cmip7_components"], args.components_output
    )
    cc_plot_diagnosed_emissions(
        temp_emissions_data["cmip6_total"], temp_emissions_data["cmip7_total"], args.diagnosed_output
    )
    for scenario in scenario_order:
        print(f"{scenario}: {len(scenario_members[scenario])} members")
    for scenario in scenario_order:
        print(f"{scenario} sea-ice: {len(scenario_ice_members[scenario])} members")
    cmip6_total = 0
    for scenario in cmip6_order:
        n = len(cmip6_ice_members.get(scenario, {}))
        cmip6_total += n
        print(f"{scenario} sea-ice: {n} members")
    print(f"CMIP6-forced sea-ice members total: {cmip6_total}")


if __name__ == "__main__":
    main()
