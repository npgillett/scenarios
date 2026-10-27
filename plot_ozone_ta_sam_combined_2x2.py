#!/usr/bin/env python3
"""Plot combined ozone, temperature, and SAM scenario diagnostics.

Two figures are written: a four-panel figure (below) and a two-panel figure of
global-mean stratospheric and tropospheric column ozone.

This script assembles a four-panel figure using the existing scenario-processing
logic from the ozone, temperature, and SAM plotting scripts:

1. Global-mean total column ozone (DU)
2. September 60S-90S total column ozone (DU)
3. SON 60S-90S mean temperature at 100 hPa (degC)
4. DJF SAM index anomaly (hPa, vs 1850-1900)
"""

import argparse
import glob
import os
from collections import OrderedDict
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import xarray as xr
from netCDF4 import Dataset, num2date


DEFAULT_OUTPUT = "/home/rng001/plots/scenarios/ozone_ta_sam_combined_2x2.png"
DEFAULT_OZONE_OUTPUT = "/home/rng001/plots/scenarios/ozone_strat_trop_1x2.png"

DEFAULT_CMIP6_HIST_ROOT = "/fs/site7/eccc/crd/cccma/model_output/CMIP6/final/CMIP6/CMIP/CCCma/CanESM5-1/historical"
DEFAULT_CMIP6_SCEN_ROOT = "/fs/site7/eccc/crd/cccma/model_output/CMIP6/final/CMIP6/ScenarioMIP/CCCma/CanESM5-1"
DEFAULT_SAM_CMIP6_HIST_ROOT = DEFAULT_CMIP6_HIST_ROOT
DEFAULT_SAM_CMIP6_SCEN_ROOT = DEFAULT_CMIP6_SCEN_ROOT
CMIP6_EXTRA_SCEN_DIR_GLOBS = {
    scenario: (
        f"/home/scrd103/site8/canesm_runs/v51p2-c6s{tag}-r*/data/nc_output/"
        f"CMIP6/ScenarioMIP/CCCma/CanESM5-1/{scenario}"
    )
    for scenario, tag in (("ssp119", "119"), ("ssp126", "126"), ("ssp460", "460"))
}
MAX_ENSEMBLE_MEMBERS = 10
DEFAULT_CMIP7_SCEN7VL_ROOT_CANDIDATES = [
    "/space/hall8/sitestore/eccc/crd/cccma/users/scrd103/canesm_runs/"
    "v51p2-c7scen7vl-r1/data/nc_output/MIP-DRS7/CMIP7/ScenarioMIP/CCCma/CanESM5-1/"
    "scen7-vl/r1i1p2f1",
    "/space/hall8/sitestore/eccc/crd/cccma/users/scrd103/canesm_runs/"
    "v51p2-c7scen7vl-r1.cd/data/nc_output/MIP-DRS7/CMIP7/ScenarioMIP/CCCma/CanESM5-1/"
    "scen7-vl/r1i1p2f1",
]
DEFAULT_CMIP7_HIST_MEMBER_DIR_GLOB = (
    "/space/hall8/sitestore/eccc/crd/cccma/users/scrd103/canesm_runs/"
    "v51p2-c7hist-r*/data/nc_output/MIP-DRS7/CMIP7/CMIP/CCCma/CanESM5-1/"
    "historical/r*i1p2f1/glb/mon/ta/tavg-p19-hxy-air/g120/v20190429"
)
DEFAULT_CMIP7_SCEN7H_MEMBER_DIR_GLOB = (
    "/space/hall8/sitestore/eccc/crd/cccma/users/scrd103/canesm_runs/"
    "v51p2-c7scen7h-r*/data/nc_output/MIP-DRS7/CMIP7/ScenarioMIP/CCCma/CanESM5-1/"
    "scen7-h/r*i1p2f1/glb/mon/ta/tavg-p19-hxy-air/g120/v20190429"
)
DEFAULT_CMIP7_SCEN7M_MEMBER_DIR_GLOB = (
    "/space/hall8/sitestore/eccc/crd/cccma/users/scrd103/canesm_runs/"
    "v51p2-c7scen7m-r*/data/nc_output/MIP-DRS7/CMIP7/ScenarioMIP/CCCma/CanESM5-1/"
    "scen7-m/r*i1p2f1/glb/mon/ta/tavg-p19-hxy-air/g120/v20190429"
)
DEFAULT_CMIP7_SCEN7VL_MEMBER_DIR_GLOB = (
    "/space/hall8/sitestore/eccc/crd/cccma/users/scrd103/canesm_runs/"
    "v51p2-c7scen7vl-r*/data/nc_output/MIP-DRS7/CMIP7/ScenarioMIP/CCCma/CanESM5-1/"
    "scen7-vl/r*i1p2f1/glb/mon/ta/tavg-p19-hxy-air/g120/v20190429"
)
DEFAULT_SAM_CMIP7_HIST_MEMBER_DIR_GLOB = (
    "/space/hall8/sitestore/eccc/crd/cccma/users/scrd103/canesm_runs/"
    "v51p2-c7hist-r*/data/nc_output/MIP-DRS7/CMIP7/CMIP/CCCma/CanESM5-1/"
    "historical/r*i1p2f1/glb/mon/psl/tavg-u-hxy-u/g120/v20190429"
)
DEFAULT_SAM_CMIP7_SCEN7H_MEMBER_DIR_GLOB = (
    "/space/hall8/sitestore/eccc/crd/cccma/users/scrd103/canesm_runs/"
    "v51p2-c7scen7h-r*/data/nc_output/MIP-DRS7/CMIP7/ScenarioMIP/CCCma/CanESM5-1/"
    "scen7-h/r*i1p2f1/glb/mon/psl/tavg-u-hxy-u/g120/v20190429"
)
DEFAULT_SAM_CMIP7_SCEN7M_MEMBER_DIR_GLOB = (
    "/space/hall8/sitestore/eccc/crd/cccma/users/scrd103/canesm_runs/"
    "v51p2-c7scen7m-r*/data/nc_output/MIP-DRS7/CMIP7/ScenarioMIP/CCCma/CanESM5-1/"
    "scen7-m/r*i1p2f1/glb/mon/psl/tavg-u-hxy-u/g120/v20190429"
)
DEFAULT_SAM_CMIP7_SCEN7VL_MEMBER_DIR_GLOB = (
    "/space/hall8/sitestore/eccc/crd/cccma/users/scrd103/canesm_runs/"
    "v51p2-c7scen7vl-r*/data/nc_output/MIP-DRS7/CMIP7/ScenarioMIP/CCCma/CanESM5-1/"
    "scen7-vl/r*i1p2f1/glb/mon/psl/tavg-u-hxy-u/g120/v20190429"
)
CMIP6_SCENARIOS = ["historical", "ssp119", "ssp126", "ssp245", "ssp370", "ssp460", "ssp585"]
CMIP6_LABELS = {
    "historical": "CMIP6 historical",
    "ssp119": "SSP1-1.9",
    "ssp126": "SSP1-2.6",
    "ssp245": "SSP2-4.5",
    "ssp460": "SSP4-6.0",
    "ssp370": "SSP3-7.0",
    "ssp585": "SSP5-8.5",
}
CMIP6_ORDER = list(CMIP6_LABELS.values())
CMIP7_ORDER = ["CMIP7 historical", "CMIP7 H", "CMIP7 M", "CMIP7 VL"]
CMIP6_COLORS = {
    "CMIP6 historical": "0.45",
    "SSP1-1.9": "0.85",
    "SSP1-2.6": "0.78",
    "SSP2-4.5": "0.70",
    "SSP4-6.0": "0.62",
    "SSP3-7.0": "0.54",
    "SSP5-8.5": "0.36",
}
CMIP7_COLORS = {
    "CMIP7 historical": "black",
    "CMIP7 H": "tab:red",
    "CMIP7 M": "tab:orange",
    "CMIP7 VL": "tab:blue",
}
CMIP6_MEMBER_GLOB = "r*i1p2f*"
PSL_VAR = "psl"
PSL_UNIT_SCALE = 0.01
CMIP7_PSL_STREAM = "tavg-u-hxy-u"
SAM_SMOOTH_WINDOW = 11

SCENARIO_ORDER = [
    "CMIP6 historical",
    "SSP1-1.9",
    "SSP1-2.6",
    "SSP2-4.5",
    "SSP4-6.0",
    "SSP3-7.0",
    "SSP5-8.5",
    "CMIP7 historical",
    "CMIP7 H",
    "CMIP7 M",
    "CMIP7 VL",
]

SCENARIO_COLORS = {
    **CMIP6_COLORS,
    **CMIP7_COLORS,
}


def decode_pressure_levels(pp_values: np.ndarray) -> np.ndarray:
    pp_values = np.asarray(pp_values, dtype=float)
    pressure = np.empty_like(pp_values, dtype=float)
    positive = pp_values >= 0
    pressure[positive] = pp_values[positive]
    coded = np.abs(pp_values[~positive]).astype(int)
    mantissa = (coded % 1000) / 100.0
    exponent = -(coded // 1000)
    pressure[~positive] = mantissa * np.power(10.0, exponent)
    return pressure


def scenario_label_from_path(path: str) -> str:
    name = os.path.splitext(os.path.basename(path))[0]
    for token, label in (
        ("cmip6_radozone_histo", "CMIP6 historical"),
        ("ssp119", "SSP1-1.9"),
        ("ssp126", "SSP1-2.6"),
        ("ssp245", "SSP2-4.5"),
        ("ssp460", "SSP4-6.0"),
        ("ssp370", "SSP3-7.0"),
        ("ssp585", "SSP5-8.5"),
        ("ozone_cmip7_hist", "CMIP7 historical"),
        ("scenh", "CMIP7 H"),
        ("scenm", "CMIP7 M"),
        ("scenvl", "CMIP7 VL"),
    ):
        if token in name:
            return label
    raise ValueError(f"Could not infer scenario label from file name: {os.path.basename(path)}")


def read_scenario_data(input_dir: str) -> OrderedDict:
    files = [
        os.path.join(input_dir, entry)
        for entry in sorted(os.listdir(input_dir))
        if entry.endswith(".nc") and ("ozone" in entry.lower() or "radozone" in entry.lower())
    ]
    if not files:
        raise FileNotFoundError(f"No ozone NetCDF files found in {input_dir}")
    datasets: OrderedDict = OrderedDict()
    for path in files:
        datasets.setdefault(scenario_label_from_path(path), []).append(path)
    return datasets


def reference_scenario_path(data_by_scenario: Dict[str, List[str]]) -> str:
    if data_by_scenario.get("CMIP7 historical"):
        return data_by_scenario["CMIP7 historical"][0]
    for scenario in CMIP6_ORDER + CMIP7_ORDER:
        if data_by_scenario.get(scenario):
            return data_by_scenario[scenario][0]
    raise RuntimeError("No ozone file was available to build the regime mask")


def build_stratosphere_mask(reference_path: str) -> Tuple[xr.DataArray, xr.DataArray]:
    with xr.open_dataset(reference_path) as ds:
        try:
            ozone_data = ds["OZ"]
        except KeyError as exc:
            raise KeyError(f"Required ozone variable 'OZ' not found in {reference_path}") from exc
        lev = decode_pressure_levels(ds["lev"].values)
        lat = ds["lat"].values
        mean_zonal_ppb = (ozone_data.mean(dim=("time", "lon")) * 1e9).assign_coords(lev=lev, lat=lat)
        mask = xr.DataArray(
            np.asarray(mean_zonal_ppb.values > 150.0, dtype=bool),
            dims=("lev", "lat"),
            coords={"lev": lev, "lat": lat},
        )
    return mean_zonal_ppb, mask


def time_to_years(time_values: np.ndarray) -> np.ndarray:
    values = np.asarray(time_values, dtype=float)
    if np.nanmax(values) > 3000.0:
        return np.floor(values / 100.0).astype(int)
    return np.floor(values).astype(int)


def pressure_layer_thickness(levels_hpa: np.ndarray) -> np.ndarray:
    levels_hpa = np.asarray(levels_hpa, dtype=float)
    interfaces = np.empty(levels_hpa.size + 1, dtype=float)
    interfaces[0] = 0.0
    interfaces[1:-1] = 0.5 * (levels_hpa[:-1] + levels_hpa[1:])
    interfaces[-1] = 2.0 * levels_hpa[-1] - interfaces[-2]
    return np.diff(interfaces)


def process_scenario_file(
    path: str,
    troposphere: bool,
    stratosphere_mask: xr.DataArray,
) -> Tuple[np.ndarray, np.ndarray]:
    with xr.open_dataset(path) as ds:
        try:
            ozone_data = ds["OZ"]
        except KeyError as exc:
            raise KeyError(f"Required ozone variable 'OZ' not found in {path}") from exc
        lev = decode_pressure_levels(ds["lev"].values)
        lat = ds["lat"].values
        lon = ds["lon"].values
        ozone_subset = ozone_data.transpose("time", "lev", "lat", "lon").assign_coords(lev=lev, lat=lat)
        region_mask = stratosphere_mask if not troposphere else ~stratosphere_mask
        region_mask = region_mask.reindex(lev=lev, lat=lat, fill_value=False)
        pressure_weights = xr.DataArray(
            pressure_layer_thickness(lev), dims=("lev",), coords={"lev": lev}
        )
        lat_weights = xr.DataArray(np.cos(np.deg2rad(lat)), dims=("lat",), coords={"lat": lat})
        lon_weights = xr.DataArray(np.ones_like(lon, dtype=float), dims=("lon",), coords={"lon": lon})
        cell_column = (ozone_subset * pressure_weights * 7.88e5).where(region_mask)
        area_weights = lat_weights * lon_weights
        region_mean_du = (
            (cell_column * area_weights).sum(dim=("lev", "lat", "lon"))
            / area_weights.sum(dim=("lat", "lon"))
        )
        years = time_to_years(ds["time"].values)
        annual = [
            (int(year), float(np.nanmean(np.asarray(region_mean_du.values[years == year], dtype=float))))
            for year in np.unique(years)
        ]
    return (
        np.asarray([year for year, _ in annual], dtype=int),
        np.asarray([value for _, value in annual], dtype=float),
    )


def ozone_scenario_series(
    paths: Iterable[str],
    troposphere: bool,
    stratosphere_mask: xr.DataArray,
) -> Tuple[np.ndarray, np.ndarray]:
    annual_by_year: Dict[int, List[float]] = {}
    for path in paths:
        years, values = process_scenario_file(path, troposphere, stratosphere_mask)
        for year, value in zip(years, values):
            annual_by_year.setdefault(int(year), []).append(float(value))
    years = np.array(sorted(annual_by_year), dtype=int)
    values = np.array([np.nanmean(annual_by_year[year]) for year in years], dtype=float)
    return years, values


def pressure_weighted_column_dobson(
    ds: xr.Dataset,
    ozone_data: xr.DataArray,
    month_index: int,
) -> Tuple[np.ndarray, np.ndarray]:
    lat = ds["lat"].values
    lon = ds["lon"].values
    lev = decode_pressure_levels(ds["lev"].values)
    time = ds["time"].values
    month_numbers = np.mod(np.asarray(time, dtype=float), 100.0).astype(int)
    time_idx = np.where(month_numbers == month_index)[0]
    if time_idx.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    lat_mask = (lat >= -90.0) & (lat <= -60.0)
    lat_sel = lat[lat_mask]
    monthly = ozone_data.isel(lat=lat_mask, time=time_idx).transpose("time", "lev", "lat", "lon")
    monthly = monthly.assign_coords(lev=np.arange(monthly.sizes["lev"]))
    pressure_weights = xr.DataArray(
        pressure_layer_thickness(lev), dims=("lev",), coords={"lev": np.arange(lev.size)}
    )
    lat_weights = xr.DataArray(
        np.cos(np.deg2rad(lat_sel)), dims=("lat",), coords={"lat": lat_sel}
    )
    lon_weights = xr.DataArray(np.ones_like(lon, dtype=float), dims=("lon",), coords={"lon": lon})
    cell_column = monthly * pressure_weights * 7.88e5
    area_weights = lat_weights * lon_weights
    region_mean_du = (
        (cell_column * area_weights).sum(dim=("lev", "lat", "lon"))
        / area_weights.sum(dim=("lat", "lon"))
    )
    selected_years = np.floor(np.asarray(time[time_idx], dtype=float) / 100.0).astype(int)
    years = np.array(sorted(set(selected_years.tolist())), dtype=int)
    values = np.array(
        [np.nanmean(region_mean_du.values[selected_years == year]) for year in years],
        dtype=float,
    )
    return years, values


def season_total_column_series(paths: Iterable[str], month_number: int) -> Tuple[np.ndarray, np.ndarray]:
    annual_by_year: Dict[int, List[float]] = {}
    for path in paths:
        with xr.open_dataset(path) as ds:
            years, values = pressure_weighted_column_dobson(ds, ds["OZ"], month_number)
        for year, value in zip(years, values):
            annual_by_year.setdefault(int(year), []).append(float(value))
    years = np.array(sorted(annual_by_year), dtype=int)
    values = np.array([np.nanmean(annual_by_year[year]) for year in years], dtype=float)
    return years, values


def find_level_index(
    plev_pa: np.ndarray,
    target_pa: float,
    tolerance_pa: float,
) -> Tuple[int, float]:
    plev_pa = np.asarray(plev_pa, dtype=float).squeeze()
    if plev_pa.ndim != 1:
        raise ValueError("Expected 1D pressure coordinate")
    idx = int(np.argmin(np.abs(plev_pa - target_pa)))
    found_pa = float(plev_pa[idx])
    if abs(found_pa - target_pa) > tolerance_pa:
        raise ValueError(
            f"Could not find pressure level close to {target_pa} Pa. Nearest level is {found_pa} Pa."
        )
    return idx, found_pa


def member_sort_key(member: str) -> Tuple[int, str]:
    try:
        index = int(member.split("i", 1)[0].lstrip("r"))
    except ValueError:
        index = 10**9
    return index, member


def first_n_members(member_names: List[str], n_members: int) -> List[str]:
    sorted_members = sorted(member_names, key=member_sort_key)
    if n_members <= 0:
        return sorted_members[:10]
    return sorted_members[: min(n_members, 10)]


def scenario_mean(
    members: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    if not members:
        return np.array([], dtype=int), np.array([], dtype=float)
    all_years = sorted({int(year) for years, _ in members.values() for year in years})
    if not all_years:
        return np.array([], dtype=int), np.array([], dtype=float)
    year_to_idx = {year: index for index, year in enumerate(all_years)}
    stack = np.full((len(members), len(all_years)), np.nan, dtype=float)
    for row, (years, values) in enumerate(members.values()):
        for year, value in zip(years, values):
            stack[row, year_to_idx[int(year)]] = float(value)
    return np.asarray(all_years, dtype=int), np.nanmean(stack, axis=0)


def print_ensemble_size(panel: str, scenario: str, members: Dict) -> None:
    print(f"Panel {panel} ensemble size for {scenario}: {len(members)} members")


def annual_mean_from_monthly(
    dates: np.ndarray,
    values: np.ndarray,
    season: str | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    years = np.array([int(date.year) for date in dates], dtype=int)
    months = np.array([int(date.month) for date in dates], dtype=int)
    valid = np.isfinite(values)
    years, months, values = years[valid], months[valid], values[valid]
    if years.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    if season is None:
        annual_years = np.array(sorted(set(years.tolist())), dtype=int)
        annual = np.array([np.nanmean(values[years == year]) for year in annual_years], dtype=float)
        return annual_years, annual
    if season == "DJF":
        season_months = np.array([12, 1, 2], dtype=int)
    elif season == "SON":
        season_months = np.array([9, 10, 11], dtype=int)
    else:
        raise ValueError(f"Unsupported season: {season}")
    season_mask = np.isin(months, season_months)
    years, months, values = years[season_mask], months[season_mask], values[season_mask]
    season_years = years.copy()
    if season == "DJF":
        season_years[months == 12] += 1
    results = []
    for year in sorted(set(season_years.tolist())):
        selected = season_years == year
        if set(months[selected].tolist()) == set(season_months.tolist()):
            results.append((int(year), float(np.nanmean(values[selected]))))
    return (
        np.asarray([year for year, _ in results], dtype=int),
        np.asarray([value for _, value in results], dtype=float),
    )


def read_monthly_global_mean_series(
    nc_files: List[str],
    target_pa: float,
    tolerance_pa: float,
    lat_min: float | None = None,
    lat_max: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    all_dates: List[object] = []
    all_means: List[np.ndarray] = []
    for nc_file in nc_files:
        with Dataset(nc_file, mode="r") as ds:
            for variable in ("ta", "plev", "lat", "time"):
                if variable not in ds.variables:
                    raise KeyError(f"{variable} variable not found in file: {nc_file}")
            plev = np.asarray(ds.variables["plev"][:], dtype=float)
            level_idx, _ = find_level_index(plev, target_pa, tolerance_pa)
            lat = np.asarray(ds.variables["lat"][:], dtype=float)
            lat_mask = np.ones_like(lat, dtype=bool)
            if lat_min is not None:
                lat_mask &= lat >= lat_min
            if lat_max is not None:
                lat_mask &= lat <= lat_max
            if not np.any(lat_mask):
                raise ValueError(f"No latitude points match the requested range for file: {nc_file}")
            ta_data = np.asarray(ds.variables["ta"][:, level_idx, lat_mask, :], dtype=float)
            for attr in ("_FillValue", "missing_value"):
                value = getattr(ds.variables["ta"], attr, None)
                if value is not None:
                    ta_data = np.where(ta_data == value, np.nan, ta_data)
            weights = np.cos(np.deg2rad(lat[lat_mask]))[:, None]
            weighted_sum = np.nansum(ta_data * weights[None, :, :], axis=(1, 2))
            weight_sum = np.nansum(np.isfinite(ta_data) * weights[None, :, :], axis=(1, 2))
            means = weighted_sum / weight_sum
            time_var = ds.variables["time"]
            units = getattr(time_var, "units", None)
            if units is None:
                raise KeyError(f"time units attribute missing in file: {nc_file}")
            all_dates.extend(
                num2date(np.asarray(time_var[:]), units=units, calendar=getattr(time_var, "calendar", "standard"))
            )
            all_means.append(means)
    if not all_means:
        raise ValueError("No time-series data found in input files")
    return np.asarray(all_dates, dtype=object), np.concatenate(all_means)


def discover_cmip6_files(
    cmip6_root: str,
    scenario: str,
    model: str,
    member: str,
    grid: str,
) -> List[str]:
    member_dir = os.path.join(cmip6_root, member) if scenario == "historical" else os.path.join(cmip6_root, scenario, member)
    version_dirs = sorted(glob.glob(os.path.join(member_dir, "Amon", "ta", grid, "v*")))
    if not version_dirs:
        return []
    pattern = f"ta_Amon_{model}_{scenario}_{member}_{grid}_*.nc"
    return sorted(glob.glob(os.path.join(version_dirs[-1], pattern)))


def discover_cmip6_ta_member_files(
    cmip6_hist_root: str,
    cmip6_scen_root: str,
    scenario: str,
    model: str,
    grid: str,
    member_glob: str = CMIP6_MEMBER_GLOB,
) -> Dict[str, List[str]]:
    if scenario == "historical":
        base_dirs = [cmip6_hist_root]
    else:
        base_dirs = [os.path.join(cmip6_scen_root, scenario)]
        extra_glob = CMIP6_EXTRA_SCEN_DIR_GLOBS.get(scenario)
        if extra_glob:
            base_dirs.extend(sorted(glob.glob(extra_glob)))

    files_by_member: Dict[str, List[str]] = {}
    for base_dir in base_dirs:
        for member_dir in sorted(glob.glob(os.path.join(base_dir, member_glob))):
            member = os.path.basename(member_dir)
            if member in files_by_member:
                continue
            version_dirs = sorted(glob.glob(os.path.join(member_dir, "Amon", "ta", grid, "v*")))
            if not version_dirs:
                continue
            pattern = f"ta_Amon_{model}_{scenario}_{member}_{grid}_*.nc"
            files = sorted(glob.glob(os.path.join(version_dirs[-1], pattern)))
            if files:
                files_by_member[member] = files
    return files_by_member


def discover_cmip7_ta_member_files(
    member_dir_glob: str,
    scenario: str,
    model: str,
) -> Dict[str, List[str]]:
    files_by_member: Dict[str, List[str]] = {}
    for member_dir in sorted(glob.glob(member_dir_glob)):
        path_parts = os.path.normpath(member_dir).split(os.sep)
        if scenario not in path_parts:
            continue
        scenario_idx = path_parts.index(scenario)
        if scenario_idx + 1 >= len(path_parts):
            continue
        member = path_parts[scenario_idx + 1]
        pattern = f"ta_tavg-p19-hxy-air_mon_glb_g120_{model}_{scenario}_{member}_*.nc"
        files = sorted(glob.glob(os.path.join(member_dir, pattern)))
        if files:
            files_by_member[member] = files
    return files_by_member


def discover_cmip6_sam_member_files(
    cmip6_hist_root: str,
    cmip6_scen_root: str,
    scenario: str,
    model: str,
    grid: str,
) -> Dict[str, List[str]]:
    if scenario == "historical":
        base_dirs = [cmip6_hist_root]
    else:
        base_dirs = [os.path.join(cmip6_scen_root, scenario)]
        extra_glob = CMIP6_EXTRA_SCEN_DIR_GLOBS.get(scenario)
        if extra_glob:
            base_dirs.extend(sorted(glob.glob(extra_glob)))

    files_by_member: Dict[str, List[str]] = {}
    for base_dir in base_dirs:
        for member_dir in sorted(glob.glob(os.path.join(base_dir, CMIP6_MEMBER_GLOB))):
            member = os.path.basename(member_dir)
            if member in files_by_member:
                continue
            version_dirs = sorted(glob.glob(os.path.join(member_dir, "Amon", PSL_VAR, grid, "v*")))
            if not version_dirs:
                continue
            pattern = f"{PSL_VAR}_Amon_{model}_{scenario}_{member}_{grid}_*.nc"
            files = sorted(glob.glob(os.path.join(version_dirs[-1], pattern)))
            if files:
                files_by_member[member] = files
    return files_by_member


def discover_cmip7_sam_member_files(
    member_dir_glob: str,
    scenario: str,
    model: str,
) -> Dict[str, List[str]]:
    files_by_member: Dict[str, List[str]] = {}
    for member_dir in sorted(glob.glob(member_dir_glob)):
        path_parts = os.path.normpath(member_dir).split(os.sep)
        if scenario not in path_parts:
            continue
        scenario_idx = path_parts.index(scenario)
        if scenario_idx + 1 >= len(path_parts):
            continue
        member = path_parts[scenario_idx + 1]
        pattern = f"{PSL_VAR}_{CMIP7_PSL_STREAM}_mon_glb_g120_{model}_{scenario}_{member}_*.nc"
        files = sorted(glob.glob(os.path.join(member_dir, pattern)))
        if files:
            files_by_member[member] = files
    return files_by_member


def read_monthly_sam_index_series(
    nc_files: List[str],
    lat_40s: float = -40.0,
    lat_65s: float = -70.0,
) -> Tuple[np.ndarray, np.ndarray]:
    all_dates: List[object] = []
    all_differences: List[np.ndarray] = []
    for nc_file in nc_files:
        with Dataset(nc_file, mode="r") as ds:
            for variable in (PSL_VAR, "lat", "lon", "time"):
                if variable not in ds.variables:
                    raise KeyError(f"{variable} variable not found in file: {nc_file}")
            lat = np.asarray(ds.variables["lat"][:], dtype=float)
            psl = np.asarray(ds.variables[PSL_VAR][:, :, :], dtype=float)
            for attr in ("_FillValue", "missing_value"):
                value = getattr(ds.variables[PSL_VAR], attr, None)
                if value is not None:
                    psl = np.where(psl == value, np.nan, psl)
            idx_40s = int(np.argmin(np.abs(lat - lat_40s)))
            idx_65s = int(np.argmin(np.abs(lat - lat_65s)))
            zonal_40s = np.nanmean(psl[:, idx_40s, :], axis=1)
            zonal_65s = np.nanmean(psl[:, idx_65s, :], axis=1)
            all_differences.append(zonal_40s - zonal_65s)
            time_var = ds.variables["time"]
            units = getattr(time_var, "units", None)
            if units is None:
                raise KeyError(f"time units attribute missing in file: {nc_file}")
            all_dates.extend(
                num2date(np.asarray(time_var[:]), units=units, calendar=getattr(time_var, "calendar", "standard"))
            )
    if not all_differences:
        raise ValueError("No SAM-index time-series data found in input files")
    return np.asarray(all_dates, dtype=object), np.concatenate(all_differences)


def sam_series_from_files(nc_files: List[str], season: str = "DJF") -> Tuple[np.ndarray, np.ndarray]:
    dates, sam_pa = read_monthly_sam_index_series(nc_files)
    years, seasonal_pa = annual_mean_from_monthly(dates, sam_pa, season=season)
    return years, seasonal_pa * PSL_UNIT_SCALE


def boxcar_smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values.copy()

    kernel = np.ones(window, dtype=float) / float(window)
    valid = np.isfinite(values).astype(float)
    filled = np.where(np.isfinite(values), values, 0.0)
    padding = (window // 2, window - 1 - window // 2)
    numer = np.convolve(np.pad(filled, padding), kernel, mode="valid")
    denom = np.convolve(np.pad(valid, padding), kernel, mode="valid")

    out = np.full(values.shape, np.nan, dtype=float)
    ok = denom > 0.0
    out[ok] = numer[ok] / denom[ok]
    return out


def build_cmip7_ensemble_sam_series(
    hist_member_files: Dict[str, List[str]],
    scen7h_member_files: Dict[str, List[str]],
    scen7m_member_files: Dict[str, List[str]],
    scen7vl_member_files: Dict[str, List[str]],
    n_members: int,
    season: str,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    selected_hist = first_n_members(list(hist_member_files), n_members)
    scenario_members = {}
    for key, files_by_member in (
        ("historical", hist_member_files),
        ("scen7-h", scen7h_member_files),
        ("scen7-m", scen7m_member_files),
        ("scen7-vl", scen7vl_member_files),
    ):
        selected = selected_hist if key == "historical" else first_n_members(list(files_by_member), n_members)
        members: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for member in selected:
            if key != "historical" and member not in hist_member_files:
                continue
            files = files_by_member[member]
            if key != "historical":
                files = hist_member_files[member] + files
            members[member] = sam_series_from_files(files, season=season)
        scenario_label = {
            "historical": "CMIP7 historical",
            "scen7-h": "CMIP7 H",
            "scen7-m": "CMIP7 M",
            "scen7-vl": "CMIP7 VL",
        }[key]
        print_ensemble_size("d", scenario_label, members)
        scenario_members[key] = scenario_mean(members)
    return scenario_members


def build_cmip6_ensemble_sam_series(
    hist_member_files: Dict[str, List[str]],
    scenario_member_files: Dict[str, Dict[str, List[str]]],
    season: str,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    selected_hist = first_n_members(list(hist_member_files), 10)
    selected_hist_files = {member: hist_member_files[member] for member in selected_hist}
    historical = {
        member: sam_series_from_files(files, season=season)
        for member, files in selected_hist_files.items()
    }
    print_ensemble_size("d", "CMIP6 historical", historical)
    out = {"historical": scenario_mean(historical)}
    for scenario in CMIP6_SCENARIOS[1:]:
        members: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        files_by_member = scenario_member_files.get(scenario, {})
        for member in first_n_members(list(files_by_member), 10):
            if member not in selected_hist_files:
                continue
            members[member] = sam_series_from_files(
                selected_hist_files[member] + files_by_member[member],
                season=season,
            )
        print_ensemble_size("d", CMIP6_LABELS[scenario], members)
        out[scenario] = scenario_mean(members)
    return out


def relabel_ta_series(series: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> OrderedDict:
    out: OrderedDict = OrderedDict()
    mapping = {
        "historical": "CMIP6 historical",
        "ssp119": "SSP1-1.9",
        "ssp126": "SSP1-2.6",
        "ssp245": "SSP2-4.5",
        "ssp460": "SSP4-6.0",
        "ssp370": "SSP3-7.0",
        "ssp585": "SSP5-8.5",
        "scen7-h": "CMIP7 H",
        "scen7-vl": "CMIP7 VL",
        "scen7-m": "CMIP7 M",
        "historical_cmip7": "CMIP7 historical",
    }
    for key, value in series.items():
        label = mapping.get(key)
        if label is not None:
            out[label] = value
    return out


def seasonal_ta_series_from_files(
    nc_files: list[str],
    target_pa: float,
    tolerance_pa: float,
    lat_min: float | None,
    lat_max: float | None,
    season: str,
) -> Tuple[np.ndarray, np.ndarray]:
    dates, values_k = read_monthly_global_mean_series(
        nc_files=nc_files,
        target_pa=target_pa,
        tolerance_pa=tolerance_pa,
        lat_min=lat_min,
        lat_max=lat_max,
    )
    years, seasonal_k = annual_mean_from_monthly(dates, values_k, season=season)
    return years, seasonal_k - 273.15


def seasonal_ta_member_mean(
    member_files: Dict[str, list[str]],
    target_pa: float,
    tolerance_pa: float,
    lat_min: float | None,
    lat_max: float | None,
    season: str,
) -> Tuple[np.ndarray, np.ndarray]:
    member_series: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for member, files in member_files.items():
        years, vals = seasonal_ta_series_from_files(
            nc_files=files,
            target_pa=target_pa,
            tolerance_pa=tolerance_pa,
            lat_min=lat_min,
            lat_max=lat_max,
            season=season,
        )
        member_series[member] = (years, vals)
    return scenario_mean(member_series)


def build_ozone_column_series(input_dir: str, region: str) -> OrderedDict:
    """Annual global-mean ozone column (DU) for region 'stratosphere', 'troposphere' or 'total'."""
    datasets = read_scenario_data(input_dir)
    reference_path = reference_scenario_path(datasets)
    _, stratosphere_mask = build_stratosphere_mask(reference_path)
    if region == "total":
        mask = xr.ones_like(stratosphere_mask, dtype=bool)
    else:
        mask = stratosphere_mask

    out: OrderedDict = OrderedDict()
    for scenario in CMIP6_ORDER + CMIP7_ORDER:
        if scenario in datasets:
            out[scenario] = ozone_scenario_series(
                datasets[scenario],
                troposphere=(region == "troposphere"),
                stratosphere_mask=mask,
            )
    return out


def build_ozone_sep_series(input_dir: str) -> OrderedDict:
    datasets = read_scenario_data(input_dir)
    out: OrderedDict = OrderedDict()
    for scenario in CMIP6_ORDER + CMIP7_ORDER:
        if scenario in datasets:
            out[scenario] = season_total_column_series(datasets[scenario], month_number=9)
    return out


def build_ta_series(args) -> OrderedDict:
    cmip6_hist_member_files = discover_cmip6_ta_member_files(
        args.cmip6_hist_root, args.cmip6_scen_root, "historical", args.model, args.grid
    )
    selected_cmip6_hist_members = first_n_members(
        list(cmip6_hist_member_files), MAX_ENSEMBLE_MEMBERS
    )
    cmip6_hist_member_subset = {
        member: cmip6_hist_member_files[member] for member in selected_cmip6_hist_members
    }
    cmip6_scenario_member_files: Dict[str, Dict[str, List[str]]] = {}
    for scenario in CMIP6_SCENARIOS[1:]:
        cmip6_scenario_member_files[scenario] = discover_cmip6_ta_member_files(
            args.cmip6_hist_root,
            args.cmip6_scen_root,
            scenario,
            args.model,
            args.grid,
        )

    cmip7_hist_member_files = discover_cmip7_ta_member_files(args.cmip7_hist_member_dir_glob, "historical", args.model)
    cmip7_scen7h_member_files = discover_cmip7_ta_member_files(args.cmip7_scen7h_member_dir_glob, "scen7-h", args.model)
    cmip7_scen7m_member_files = discover_cmip7_ta_member_files(args.cmip7_scen7m_member_dir_glob, "scen7-m", args.model)
    cmip7_scen7vl_member_files = discover_cmip7_ta_member_files(args.cmip7_scen7vl_member_dir_glob, "scen7-vl", args.model)

    selected_hist_members = first_n_members(list(cmip7_hist_member_files.keys()), args.cmip7_ensemble_members)
    selected_h_members = first_n_members(list(cmip7_scen7h_member_files.keys()), args.cmip7_ensemble_members)
    selected_m_members = first_n_members(list(cmip7_scen7m_member_files.keys()), args.cmip7_ensemble_members)
    selected_vl_members = first_n_members(list(cmip7_scen7vl_member_files.keys()), args.cmip7_ensemble_members)

    hist_member_subset = {member: cmip7_hist_member_files[member] for member in selected_hist_members}
    h_member_subset = {
        member: cmip7_hist_member_files[member] + cmip7_scen7h_member_files[member]
        for member in selected_h_members
        if member in cmip7_hist_member_files
    }
    m_member_subset = {
        member: cmip7_hist_member_files[member] + cmip7_scen7m_member_files[member]
        for member in selected_m_members
        if member in cmip7_hist_member_files
    }
    vl_member_subset = {
        member: cmip7_hist_member_files[member] + cmip7_scen7vl_member_files[member]
        for member in selected_vl_members
        if member in cmip7_hist_member_files
    }

    print_ensemble_size("c", "CMIP7 historical", hist_member_subset)
    print_ensemble_size("c", "CMIP7 H", h_member_subset)
    print_ensemble_size("c", "CMIP7 M", m_member_subset)
    print_ensemble_size("c", "CMIP7 VL", vl_member_subset)
    out: OrderedDict = OrderedDict()
    out["CMIP7 historical"] = seasonal_ta_member_mean(
        member_files=hist_member_subset,
        target_pa=10000.0,
        tolerance_pa=args.level_tolerance_pa,
        lat_min=None,
        lat_max=-60.0,
        season="SON",
    )
    out["CMIP7 H"] = seasonal_ta_member_mean(
        member_files=h_member_subset,
        target_pa=10000.0,
        tolerance_pa=args.level_tolerance_pa,
        lat_min=None,
        lat_max=-60.0,
        season="SON",
    )
    out["CMIP7 M"] = seasonal_ta_member_mean(
        member_files=m_member_subset,
        target_pa=10000.0,
        tolerance_pa=args.level_tolerance_pa,
        lat_min=None,
        lat_max=-60.0,
        season="SON",
    )
    out["CMIP7 VL"] = seasonal_ta_member_mean(
        member_files=vl_member_subset,
        target_pa=10000.0,
        tolerance_pa=args.level_tolerance_pa,
        lat_min=None,
        lat_max=-60.0,
        season="SON",
    )

    out["CMIP6 historical"] = seasonal_ta_member_mean(
        member_files=cmip6_hist_member_subset,
        target_pa=10000.0,
        tolerance_pa=args.level_tolerance_pa,
        lat_min=None,
        lat_max=-60.0,
        season="SON",
    )
    print_ensemble_size("c", "CMIP6 historical", cmip6_hist_member_subset)
    for scenario, label in CMIP6_LABELS.items():
        if scenario == "historical":
            continue
        scen_files_by_member = cmip6_scenario_member_files[scenario]
        members_to_use = first_n_members(
            list(scen_files_by_member), MAX_ENSEMBLE_MEMBERS
        )
        paired_files = {
            member: cmip6_hist_member_files[member] + scen_files_by_member[member]
            for member in members_to_use
            if member in cmip6_hist_member_files
        }
        print_ensemble_size("c", label, paired_files)
        out[label] = seasonal_ta_member_mean(
            member_files=paired_files,
            target_pa=10000.0,
            tolerance_pa=args.level_tolerance_pa,
            lat_min=None,
            lat_max=-60.0,
            season="SON",
        )

    return out


def build_sam_series(args) -> OrderedDict:
    cmip6_hist_member_files = discover_cmip6_sam_member_files(
        cmip6_hist_root=args.sam_cmip6_hist_root,
        cmip6_scen_root=args.sam_cmip6_scen_root,
        scenario="historical",
        model=args.model,
        grid=args.grid,
    )
    cmip6_scenario_member_files: Dict[str, Dict[str, list[str]]] = {}
    for scenario in CMIP6_SCENARIOS[1:]:
        cmip6_scenario_member_files[scenario] = discover_cmip6_sam_member_files(
            cmip6_hist_root=args.sam_cmip6_hist_root,
            cmip6_scen_root=args.sam_cmip6_scen_root,
            scenario=scenario,
            model=args.model,
            grid=args.grid,
        )

    cmip7_hist_member_files = discover_cmip7_sam_member_files(args.sam_cmip7_hist_member_dir_glob, "historical", args.model)
    cmip7_scen7h_member_files = discover_cmip7_sam_member_files(args.sam_cmip7_scen7h_member_dir_glob, "scen7-h", args.model)
    cmip7_scen7m_member_files = discover_cmip7_sam_member_files(args.sam_cmip7_scen7m_member_dir_glob, "scen7-m", args.model)
    cmip7_scen7vl_member_files = discover_cmip7_sam_member_files(args.sam_cmip7_scen7vl_member_dir_glob, "scen7-vl", args.model)

    cmip7_series = build_cmip7_ensemble_sam_series(
        hist_member_files=cmip7_hist_member_files,
        scen7h_member_files=cmip7_scen7h_member_files,
        scen7m_member_files=cmip7_scen7m_member_files,
        scen7vl_member_files=cmip7_scen7vl_member_files,
        n_members=args.cmip7_ensemble_members,
        season="DJF",
    )
    cmip6_series = build_cmip6_ensemble_sam_series(
        hist_member_files=cmip6_hist_member_files,
        scenario_member_files=cmip6_scenario_member_files,
        season="DJF",
    )

    out: OrderedDict = OrderedDict()
    for key, label in CMIP6_LABELS.items():
        years, vals = cmip6_series.get(key, (np.array([], dtype=int), np.array([], dtype=float)))
        out[label] = (years, vals)
    out["CMIP7 historical"] = cmip7_series.get("historical", (np.array([], dtype=int), np.array([], dtype=float)))
    out["CMIP7 H"] = cmip7_series.get("scen7-h", (np.array([], dtype=int), np.array([], dtype=float)))
    out["CMIP7 M"] = cmip7_series.get("scen7-m", (np.array([], dtype=int), np.array([], dtype=float)))
    out["CMIP7 VL"] = cmip7_series.get("scen7-vl", (np.array([], dtype=int), np.array([], dtype=float)))
    return out


def plot_panel(
    ax,
    series_by_scenario: OrderedDict,
    title: str,
    ylabel: str,
    legend_scenarios: set[str] | None = None,
    year_start: int | None = None,
    year_end: int | None = None,
    y_limits: tuple[float, float] | None = None,
    y_min: float | None = None,
):
    historical_labels = ["CMIP6 historical", "CMIP7 historical"]
    cmip6_scenario_labels = {"SSP1-1.9", "SSP1-2.6", "SSP2-4.5", "SSP4-6.0", "SSP3-7.0", "SSP5-8.5"}

    for scenario in SCENARIO_ORDER:
        if scenario in historical_labels:
            continue
        if scenario not in series_by_scenario:
            continue
        years, values = series_by_scenario[scenario]
        if years.size == 0:
            continue
        if year_start is not None and year_end is not None:
            if scenario in cmip6_scenario_labels:
                mask = (years >= year_start) & (years <= year_end)
                years = years[mask]
                values = values[mask]
            elif scenario.startswith("CMIP7"):
                mask = (years >= 2000) & (years <= 2100)
                years = years[mask]
                values = values[mask]
        if years.size == 0:
            continue
        linewidth = 3.0 if scenario.startswith("CMIP7") else 2.6
        label = scenario if legend_scenarios is None or scenario in legend_scenarios else "_nolegend_"
        ax.plot(years, values, color=SCENARIO_COLORS.get(scenario, "tab:gray"), linewidth=linewidth, label=label, zorder=2)

    for scenario in historical_labels:
        if scenario not in series_by_scenario:
            continue
        years, values = series_by_scenario[scenario]
        if years.size == 0:
            continue
        if year_start is not None and year_end is not None and scenario == "CMIP6 historical":
            mask = (years <= year_start - 1)
            years = years[mask]
            values = values[mask]
        if years.size == 0:
            continue
        label = scenario if legend_scenarios is None or scenario in legend_scenarios else "_nolegend_"
        ax.plot(years, values, color=SCENARIO_COLORS.get(scenario, "tab:gray"), linewidth=3.2, label=label, zorder=4)

    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_xlim(2000, 2100)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    if y_min is not None:
        ax.set_ylim(bottom=y_min)
    ax.grid(True, alpha=0.3)


LEGEND_COLUMN_1 = ["CMIP6 historical", "SSP5-8.5", "SSP3-7.0", "SSP4-6.0", "SSP2-4.5", "SSP1-2.6", "SSP1-1.9"]
LEGEND_COLUMN_2 = ["CMIP7 historical", "CMIP7 H", "CMIP7 M", "CMIP7 VL"]
LEGEND_LABELS = {"CMIP7 H": "H", "CMIP7 M": "M", "CMIP7 VL": "VL"}


def add_scenario_legend(ax) -> None:
    """Two-column legend: CMIP6 scenarios on the left, CMIP7 on the right."""
    handles = dict(zip(ax.get_legend_handles_labels()[1], ax.get_legend_handles_labels()[0]))
    blank = Line2D([], [], linestyle="none", label=" ")
    col1_keys = [key for key in LEGEND_COLUMN_1 if key in handles]
    col2_keys = [key for key in LEGEND_COLUMN_2 if key in handles]
    col1 = [handles[key] for key in col1_keys]
    col2 = [handles[key] for key in col2_keys]
    # Matplotlib fills legend columns top to bottom, so pad both to equal length.
    rows = max(len(col1), len(col2))
    col1 += [blank] * (rows - len(col1))
    col2 += [blank] * (rows - len(col2))
    labels = [LEGEND_LABELS.get(key, key) for key in col1_keys]
    labels += [" "] * (rows - len(col1_keys))
    labels += [LEGEND_LABELS.get(key, key) for key in col2_keys]
    labels += [" "] * (rows - len(col2_keys))
    ax.legend(handles=col1 + col2, labels=labels, ncol=2, loc="upper left", frameon=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a combined four-panel ozone, temperature, and SAM scenario figure.")
    parser.add_argument("--ozone-input-dir", default="/home/rng001/site7/ozone_forcing", help="Directory containing ozone NetCDF files")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output PNG file path for the four-panel figure")
    parser.add_argument("--ozone-output", default=DEFAULT_OZONE_OUTPUT, help="Output PNG file path for the stratosphere/troposphere ozone figure")
    parser.add_argument("--cmip6-hist-root", default=DEFAULT_CMIP6_HIST_ROOT, help="CMIP6 historical ta root directory")
    parser.add_argument("--cmip6-scen-root", default=DEFAULT_CMIP6_SCEN_ROOT, help="CMIP6 ScenarioMIP ta root directory")
    parser.add_argument("--cmip7-hist-member-dir-glob", default=DEFAULT_CMIP7_HIST_MEMBER_DIR_GLOB, help="Glob for CMIP7 historical ta member directories")
    parser.add_argument("--cmip7-scen7h-member-dir-glob", default=DEFAULT_CMIP7_SCEN7H_MEMBER_DIR_GLOB, help="Glob for CMIP7 scen7-h ta member directories")
    parser.add_argument("--cmip7-scen7m-member-dir-glob", default=DEFAULT_CMIP7_SCEN7M_MEMBER_DIR_GLOB, help="Glob for CMIP7 scen7-m ta member directories")
    parser.add_argument("--cmip7-scen7vl-member-dir-glob", default=DEFAULT_CMIP7_SCEN7VL_MEMBER_DIR_GLOB, help="Glob for CMIP7 scen7-vl ta member directories")
    parser.add_argument("--cmip7-scen7vl-root", default=DEFAULT_CMIP7_SCEN7VL_ROOT_CANDIDATES[0], help="Primary CMIP7 scen7-vl ta root directory")
    parser.add_argument("--sam-cmip6-hist-root", default=DEFAULT_SAM_CMIP6_HIST_ROOT, help="CMIP6 historical psl root directory")
    parser.add_argument("--sam-cmip6-scen-root", default=DEFAULT_SAM_CMIP6_SCEN_ROOT, help="CMIP6 ScenarioMIP psl root directory")
    parser.add_argument("--sam-cmip7-hist-member-dir-glob", default=DEFAULT_SAM_CMIP7_HIST_MEMBER_DIR_GLOB, help="Glob for CMIP7 historical psl member directories")
    parser.add_argument("--sam-cmip7-scen7h-member-dir-glob", default=DEFAULT_SAM_CMIP7_SCEN7H_MEMBER_DIR_GLOB, help="Glob for CMIP7 scen7-h psl member directories")
    parser.add_argument("--sam-cmip7-scen7m-member-dir-glob", default=DEFAULT_SAM_CMIP7_SCEN7M_MEMBER_DIR_GLOB, help="Glob for CMIP7 scen7-m psl member directories")
    parser.add_argument("--sam-cmip7-scen7vl-member-dir-glob", default=DEFAULT_SAM_CMIP7_SCEN7VL_MEMBER_DIR_GLOB, help="Glob for CMIP7 scen7-vl psl member directories")
    parser.add_argument("--cmip7-ensemble-members", type=int, default=10, help="Number of CMIP7 members used to compute ensemble means (maximum 10)")
    parser.add_argument("--member", default="r1i1p2f1", help="CMIP6 ensemble member")
    parser.add_argument("--model", default="CanESM5-1", help="Model name")
    parser.add_argument("--grid", default="gn", help="CMIP6 grid label")
    parser.add_argument("--level-tolerance-pa", type=float, default=200.0, help="Allowed absolute distance from target pressure level in Pa")
    parser.add_argument("--baseline-start", type=int, default=1850, help="Baseline start year for SAM anomalies")
    parser.add_argument("--baseline-end", type=int, default=1900, help="Baseline end year for SAM anomalies")
    args = parser.parse_args()

    ozone_total = build_ozone_column_series(args.ozone_input_dir, "total")
    ozone_strat = build_ozone_column_series(args.ozone_input_dir, "stratosphere")
    ozone_trop = build_ozone_column_series(args.ozone_input_dir, "troposphere")
    ozone_sep = build_ozone_sep_series(args.ozone_input_dir)
    ta_son = build_ta_series(args)
    sam_djf = OrderedDict(
        (scenario, (years, boxcar_smooth(values, SAM_SMOOTH_WINDOW)))
        for scenario, (years, values) in build_sam_series(args).items()
    )

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), sharex=True)
    plot_panel(
        axes[0, 0],
        ozone_total,
        "a) Global-mean total column ozone",
        "DU",
    )
    plot_panel(
        axes[0, 1],
        ozone_sep,
        "b) September 60\u00b0S\u201390\u00b0S total column ozone",
        "DU",
        legend_scenarios=set(),
    )
    plot_panel(
        axes[1, 0],
        ta_son,
        "c) September\u2013November 60\u00b0S\u201390\u00b0S 100-hPa temperature",
        "Temperature (\u00b0C)",
        year_start=2015,
        year_end=2100,
        y_limits=(-75, -64),
    )
    plot_panel(
        axes[1, 1],
        sam_djf,
        "d) December\u2013February SAM index",
        "SAM index (hPa)",
        year_start=2015,
        year_end=2100,
        y_min=24,
    )

    add_scenario_legend(axes[0, 0])
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    save_figure(fig, args.output)

    fig2, axes2 = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True)
    plot_panel(axes2[0], ozone_strat, "a) Stratospheric column ozone", "DU")
    plot_panel(axes2[1], ozone_trop, "b) Tropospheric column ozone", "DU", legend_scenarios=set())
    add_scenario_legend(axes2[0])
    fig2.tight_layout()
    save_figure(fig2, args.ozone_output)


def save_figure(fig, path: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved figure to {path}")


if __name__ == "__main__":
    main()
