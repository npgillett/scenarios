#!/usr/bin/env python3
"""Plot annual regional tas and precip anomaly comparisons for CMIP6 and CMIP7.

The figure shows:
    a) Canada mean tas
    b) Canada mean precip
    c) Brazil mean tas
    d) Brazil mean precip
    e) Antarctica mean tas
    f) Antarctica mean precip

Global mean series are retained as references for the per-global-warming plots,
but are not displayed as panels.

The script uses up to ten simulations per ensemble discovered on disk, computes annual means,
then plots anomalies relative to 1850-1900. Regional means use Natural Earth country
boundaries and land-fraction-weighted cells.
Precipitation is reported in mm/year.
"""

import argparse
import csv
import glob
import multiprocessing as mp
import os
import re
import tempfile
import time
import urllib.request
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset, num2date
from scipy import stats
from shapely import contains_xy, union_all
from shapely.geometry.base import BaseGeometry

DEFAULT_CMIP6_HIST_ROOT = "/fs/site7/eccc/crd/cccma/model_output/CMIP6/final/CMIP6/CMIP/CCCma/CanESM5-1/historical"
DEFAULT_CMIP6_SCEN_ROOT = "/fs/site7/eccc/crd/cccma/model_output/CMIP6/final/CMIP6/ScenarioMIP/CCCma/CanESM5-1"

DEFAULT_CMIP7_HIST_ROOT = (
    "/space/hall8/sitestore/eccc/crd/cccma/users/scrd103/canesm_runs/"
    "v51p2-c7hist-r1/data/nc_output/MIP-DRS7/CMIP7/CMIP/CCCma/CanESM5-1/"
    "historical"
)
DEFAULT_CMIP7_SCEN7H_ROOT = (
    "/space/hall8/sitestore/eccc/crd/cccma/users/scrd103/canesm_runs/"
    "v51p2-c7scen7h-r1/data/nc_output/MIP-DRS7/CMIP7/ScenarioMIP/CCCma/CanESM5-1/"
    "scen7-h"
)
DEFAULT_CMIP7_SCEN7M_ROOT = (
    "/space/hall8/sitestore/eccc/crd/cccma/users/scrd103/canesm_runs/"
    "v51p2-c7scen7m-r1/data/nc_output/MIP-DRS7/CMIP7/ScenarioMIP/CCCma/CanESM5-1/"
    "scen7-m/r1i1p2f1"
)
DEFAULT_CMIP7_SCEN7VL_ROOT_CANDIDATES = [
    "/space/hall8/sitestore/eccc/crd/cccma/users/scrd103/canesm_runs/"
    "v51p2-c7scen7vl-r1/data/nc_output/MIP-DRS7/CMIP7/ScenarioMIP/CCCma/CanESM5-1/"
    "scen7-vl",
    "/space/hall8/sitestore/eccc/crd/cccma/users/scrd103/canesm_runs/"
    "v51p2-c7scen7vl-r1.cd/data/nc_output/MIP-DRS7/CMIP7/ScenarioMIP/CCCma/CanESM5-1/"
    "scen7-vl",
]
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

DEFAULT_OUTPUT = "/home/rng001/plots/scenarios/canesm5-1_tas_precip_regional_anoms_2x2.png"
DEFAULT_LAND_MASK = (
    "/fs/site7/eccc/crd/cccma/model_output/CMIP6/final/CMIP6/CMIP/CCCma/CanESM5-1/"
    "historical/r10i1p2f1/fx/sftlf/gn/v20190429/"
    "sftlf_fx_CanESM5-1_historical_r10i1p2f1_gn.nc"
)
DEFAULT_COUNTRIES_SHAPEFILE = (
    "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"
)

CMIP6_SCENARIOS = ["historical", "ssp119", "ssp126", "ssp245", "ssp460", "ssp370", "ssp585"]
CMIP6_LABELS = {
    "historical": "CMIP6 historical",
    "ssp119": "SSP1-1.9",
    "ssp126": "SSP1-2.6",
    "ssp245": "SSP2-4.5",
    "ssp460": "SSP4-6.0",
    "ssp370": "SSP3-7.0",
    "ssp585": "SSP5-8.5",
}
CMIP6_GREYS = {
    "historical": "0.45",
    "ssp119": "0.85",
    "ssp126": "0.78",
    "ssp245": "0.70",
    "ssp460": "0.62",
    "ssp370": "0.54",
    "ssp585": "0.36",
}

CMIP7_SCENARIOS = ["historical", "scen7-h", "scen7-m", "scen7-vl"]
CMIP7_LABELS = {
    "historical": "CMIP7 historical",
    "scen7-h": "H",
    "scen7-m": "M",
    "scen7-vl": "VL",
}
CMIP7_COLORS = {
    "historical": "black",
    "scen7-h": "tab:red",
    "scen7-m": "tab:orange",
    "scen7-vl": "tab:blue",
}

CMIP6_SCENARIO_START_YEAR = 2015
CMIP7_SCENARIO_START_YEAR = 2022
CMIP6_MEMBER_GLOB = "r*i1p2f*"
MAX_ENSEMBLE_MEMBERS = 10
SCENARIO_REQUIRED_END_YYYYMM = 210012
SIGNIFICANCE_LEVEL = 0.05
# Additional CMIP6 ensemble members (r2-r10) run separately for these scenarios.
CMIP6_EXTRA_SCEN_DIR_GLOBS = {
    scenario: (
        f"/home/scrd103/site8/canesm_runs/v51p2-c6s{tag}-r*/data/nc_output/"
        f"CMIP6/ScenarioMIP/CCCma/CanESM5-1/{scenario}"
    )
    for scenario, tag in (("ssp119", "119"), ("ssp126", "126"), ("ssp460", "460"))
}
CMIP7_REQUIRED_FREQUENCY = "mon"
CMIP7_REQUIRED_STAT = "tavg"

RegionKey = Tuple[float | None, float | None, float | None, float | None, str | None]
FILE_REGION_CACHE: Dict[Tuple[str, str], Tuple[np.ndarray, Dict[RegionKey, np.ndarray]]] = {}
FILE_BRANCH_TIMES: Dict[str, str | None] = {}
GLOBAL_REFERENCE_PANEL_COUNT = 2

PANEL_SPECS = [
    {
        "title": "Global mean temperature (reference)",
        "country": None,
        "region_label": "Global",
        "lat_min": None,
        "lat_max": None,
        "lon_min": None,
        "lon_max": None,
        "variable": "tas",
        "ylabel": "Anomaly (°C, vs 1850-1900)",
    },
    {
        "title": "Global mean precipitation (reference)",
        "country": None,
        "region_label": "Global",
        "lat_min": None,
        "lat_max": None,
        "lon_min": None,
        "lon_max": None,
        "variable": "pr",
        "ylabel": "Anomaly (mm/year, vs 1850-1900)",
    },
    {
        "title": "a) Canada mean temperature",
        "country": "Canada",
        "region_label": "Canada",
        "lat_min": None,
        "lat_max": None,
        "lon_min": None,
        "lon_max": None,
        "variable": "tas",
        "ylabel": "Anomaly (°C, vs 1850-1900)",
    },
    {
        "title": "b) Canada mean precipitation",
        "country": "Canada",
        "region_label": "Canada",
        "lat_min": None,
        "lat_max": None,
        "lon_min": None,
        "lon_max": None,
        "variable": "pr",
        "ylabel": "Anomaly (mm/year, vs 1850-1900)",
    },
    {
        "title": "c) Brazil mean temperature",
        "country": "Brazil",
        "region_label": "Brazil",
        "lat_min": None,
        "lat_max": None,
        "lon_min": None,
        "lon_max": None,
        "variable": "tas",
        "ylabel": "Anomaly (°C, vs 1850-1900)",
    },
    {
        "title": "d) Brazil mean precipitation",
        "country": "Brazil",
        "region_label": "Brazil",
        "lat_min": None,
        "lat_max": None,
        "lon_min": None,
        "lon_max": None,
        "variable": "pr",
        "ylabel": "Anomaly (mm/year, vs 1850-1900)",
    },
    {
        "title": "e) Antarctica mean temperature",
        "country": "Antarctica",
        "region_label": "Antarctica",
        "lat_min": None,
        "lat_max": None,
        "lon_min": None,
        "lon_max": None,
        "variable": "tas",
        "ylabel": "Anomaly (°C, vs 1850-1900)",
    },
    {
        "title": "f) Antarctica mean precipitation",
        "country": "Antarctica",
        "region_label": "Antarctica",
        "lat_min": None,
        "lat_max": None,
        "lon_min": None,
        "lon_max": None,
        "variable": "pr",
        "ylabel": "Anomaly (mm/year, vs 1850-1900)",
    },
]


def pick_existing_path(candidates: List[str]) -> str:
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("None of the candidate paths exist: " + ", ".join(candidates))


def read_land_fraction(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Land-fraction NetCDF file not found: {path}")

    with Dataset(path, mode="r") as ds:
        for name in ("sftlf", "lat", "lon"):
            if name not in ds.variables:
                raise KeyError(f"{name} variable not found in land-fraction file: {path}")

        lat = np.asarray(ds.variables["lat"][:], dtype=float)
        lon = np.asarray(ds.variables["lon"][:], dtype=float)
        land_fraction = np.ma.filled(ds.variables["sftlf"][:], np.nan).astype(float)
        units = str(getattr(ds.variables["sftlf"], "units", "%")).strip()

    valid = np.isfinite(land_fraction)
    if not np.any(valid):
        raise ValueError(f"Land-fraction file contains no finite values: {path}")
    if units == "%" or np.nanmax(land_fraction) > 1.0:
        land_fraction /= 100.0
    if np.any((land_fraction[valid] < 0.0) | (land_fraction[valid] > 1.0)):
        raise ValueError(f"Land fractions must be in the range 0-1 or 0-100 percent: {path}")
    return lat, lon, land_fraction


def country_shapefile_path(source: str) -> str:
    if not source.startswith(("http://", "https://")):
        if not os.path.isfile(source):
            raise FileNotFoundError(f"Country shapefile archive not found: {source}")
        return source

    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "scenarios")
    os.makedirs(cache_dir, exist_ok=True)
    archive_path = os.path.join(cache_dir, "ne_10m_admin_0_countries.zip")
    if not os.path.isfile(archive_path):
        with tempfile.TemporaryDirectory(dir=cache_dir) as temporary_dir:
            download_path = os.path.join(temporary_dir, "countries.zip")
            urllib.request.urlretrieve(source, download_path)
            os.replace(download_path, archive_path)
    return archive_path


def read_country_geometries(source: str) -> Dict[str, BaseGeometry]:
    import geopandas as gpd

    countries = gpd.read_file(country_shapefile_path(source))
    if "ADMIN" not in countries.columns or "geometry" not in countries.columns:
        raise KeyError("Country shapefile must contain ADMIN and geometry fields")

    geometries: Dict[str, BaseGeometry] = {}
    for country in ("Canada", "Brazil", "Antarctica"):
        country_rows = countries.loc[countries["ADMIN"] == country, "geometry"]
        if country_rows.empty:
            raise ValueError(f"Country {country!r} not found in the supplied shapefile")
        geometries[country] = union_all(country_rows.array)
    return geometries


def read_coastline_geometry(source: str) -> BaseGeometry:
    import geopandas as gpd

    countries = gpd.read_file(country_shapefile_path(source))
    if "geometry" not in countries.columns:
        raise KeyError("Country shapefile must contain a geometry field")
    return union_all(countries.geometry.array).boundary


def align_land_fraction(
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    source_fraction: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> np.ndarray:
    lat_indices = np.abs(source_lat[:, None] - target_lat[None, :]).argmin(axis=0)
    lon_distance = np.abs(
        ((normalize_lon_to_180(source_lon[:, None]) - normalize_lon_to_180(target_lon)[None, :] + 180.0) % 360.0)
        - 180.0
    )
    lon_indices = lon_distance.argmin(axis=0)
    return source_fraction[np.ix_(lat_indices, lon_indices)]


def log(msg: str) -> None:
    print(msg, flush=True)


def is_historical_file(path: str) -> bool:
    return "_historical_" in os.path.basename(path) or f"{os.sep}historical{os.sep}" in path


def read_branch_time(nc_file: str) -> str | None:
    if nc_file not in FILE_BRANCH_TIMES:
        with Dataset(nc_file, mode="r") as ds:
            value = getattr(ds, "YMDH_branch_time_in_parent", None)
            FILE_BRANCH_TIMES[nc_file] = None if value is None else str(value)
    return FILE_BRANCH_TIMES[nc_file]


def write_historical_branch_times(
    output_path: str,
    cmip6_member_files: Dict[str, List[str]],
    cmip7_member_files: Dict[str, List[str]],
) -> None:
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("family", "member", "YMDH_branch_time_in_parent"))
        for family, members in (("CMIP6", cmip6_member_files), ("CMIP7", cmip7_member_files)):
            for member, files in sorted(members.items(), key=lambda item: member_sort_key(item[0])):
                branch_times = [read_branch_time(path) for path in files]
                branch_time = next((value for value in branch_times if value is not None), "")
                writer.writerow((family, member, branch_time))


def member_sort_key(member: str) -> Tuple[int, str]:
    try:
        idx = int(member.split("i", 1)[0].lstrip("r"))
    except ValueError:
        idx = 10**9
    return idx, member


def first_n_members(member_names: List[str], n_members: int | None) -> List[str]:
    sorted_members = sorted(member_names, key=member_sort_key)
    if n_members is None or n_members <= 0:
        return sorted_members[:MAX_ENSEMBLE_MEMBERS]
    return sorted_members[: min(n_members, MAX_ENSEMBLE_MEMBERS)]


def annual_mean_from_monthly(
    dates: np.ndarray,
    values: np.ndarray,
    season: str | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    years = np.array([int(d.year) for d in dates], dtype=int)
    months = np.array([int(d.month) for d in dates], dtype=int)
    valid = np.isfinite(years) & np.isfinite(values)
    years = years[valid]
    months = months[valid]
    values = values[valid]

    if years.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    if season is None:
        years_sorted = np.array(sorted(set(years.tolist())), dtype=int)
        annual = np.array([np.nanmean(values[years == yr]) for yr in years_sorted], dtype=float)
        return years_sorted, annual

    if season != "DJF":
        raise ValueError(f"Unsupported season: {season}")

    djf_mask = np.isin(months, np.array([12, 1, 2], dtype=int))
    years = years[djf_mask]
    months = months[djf_mask]
    values = values[djf_mask]
    if years.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    season_years = years.copy()
    season_years[months == 12] += 1

    out_years: List[int] = []
    out_vals: List[float] = []
    for yr in sorted(set(season_years.tolist())):
        sel = season_years == yr
        months_this = set(months[sel].tolist())
        if months_this != {12, 1, 2}:
            continue
        out_years.append(int(yr))
        out_vals.append(float(np.nanmean(values[sel])))

    if not out_years:
        return np.array([], dtype=int), np.array([], dtype=float)

    return np.array(out_years, dtype=int), np.array(out_vals, dtype=float)


def normalize_lon_to_180(lon_deg: np.ndarray) -> np.ndarray:
    return ((lon_deg + 180.0) % 360.0) - 180.0


def make_region_key(
    lat_min: float | None,
    lat_max: float | None,
    lon_min: float | None,
    lon_max: float | None,
    country: str | None = None,
) -> RegionKey:
    return (lat_min, lat_max, lon_min, lon_max, country)


def unique_region_keys_from_panel_specs(panel_specs: List[Dict[str, object]]) -> List[RegionKey]:
    seen: set[RegionKey] = set()
    keys: List[RegionKey] = []
    for spec in panel_specs:
        key = make_region_key(
            spec.get("lat_min"),
            spec.get("lat_max"),
            spec.get("lon_min"),
            spec.get("lon_max"),
            spec.get("country"),
        )
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return keys


def build_region_masks(
    lat: np.ndarray,
    lon: np.ndarray,
    lat_min: float | None,
    lat_max: float | None,
    lon_min: float | None,
    lon_max: float | None,
) -> Tuple[np.ndarray, np.ndarray]:
    lat_mask = np.ones_like(lat, dtype=bool)
    if lat_min is not None:
        lat_mask &= lat >= lat_min
    if lat_max is not None:
        lat_mask &= lat <= lat_max

    lon_mask = np.ones_like(lon, dtype=bool)
    if lon_min is not None and lon_max is not None:
        lon_180 = normalize_lon_to_180(lon)
        if lon_min <= lon_max:
            lon_mask &= (lon_180 >= lon_min) & (lon_180 <= lon_max)
        else:
            lon_mask &= (lon_180 >= lon_min) | (lon_180 <= lon_max)

    return lat_mask, lon_mask


def area_weighted_mean(
    field_tyx: np.ndarray,
    cell_weights: np.ndarray,
    region_mask: np.ndarray,
) -> np.ndarray:
    if field_tyx.ndim != 3:
        raise ValueError(f"Expected data with 3 dimensions (time, lat, lon); got shape {field_tyx.shape}")
    if cell_weights.shape != field_tyx.shape[1:] or region_mask.shape != field_tyx.shape[1:]:
        raise ValueError("Cell weights and region mask must match the field's latitude/longitude grid")
    if not np.any(region_mask):
        raise ValueError("No grid cells match the requested region")

    weights = np.where(region_mask & np.isfinite(cell_weights), cell_weights, 0.0)
    valid = np.isfinite(field_tyx)
    weighted_values = np.where(valid, field_tyx, 0.0) * weights[None, :, :]
    effective_weights = valid * weights[None, :, :]
    weight_sum = effective_weights.sum(axis=(1, 2))
    weighted_sum = weighted_values.sum(axis=(1, 2))
    return np.divide(
        weighted_sum,
        weight_sum,
        out=np.full(weight_sum.shape, np.nan, dtype=float),
        where=weight_sum > 0.0,
    )


MAP_BASELINE_PERIOD = (1850, 1900)
MAP_PERIODS = [(2081, 2100), (2041, 2060)]
MAP_YEARS = frozenset(
    year
    for start_year, end_year in [MAP_BASELINE_PERIOD, *MAP_PERIODS]
    for year in range(start_year, end_year + 1)
)
FILE_MAP_CACHE: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray, Dict[int, Tuple[np.ndarray, np.ndarray]]]] = {}
_WORKER_CONTEXT: Dict[str, object] = {}


def yearly_sums(
    values: np.ndarray,
    years: np.ndarray,
    requested_years,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for year in np.unique(years):
        year = int(year)
        if year not in requested_years:
            continue
        year_values = values[years == year]
        out[year] = (np.nansum(year_values, axis=0), np.sum(np.isfinite(year_values), axis=0))
    return out


def compute_file_data(
    nc_file: str,
    variable_name: str,
    all_region_keys: List[RegionKey],
    land_fraction_source: Tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    country_geometries: Dict[str, BaseGeometry] | None,
):
    """Read one file once; return regional monthly means, branch time and map year sums."""
    with Dataset(nc_file, mode="r") as ds:
        branch_time = None
        if variable_name == "tas" and is_historical_file(nc_file):
            value = getattr(ds, "YMDH_branch_time_in_parent", None)
            branch_time = None if value is None else str(value)

        data_var = data_variable_name(variable_name, ds)
        for needed_var in (data_var, "lat", "lon", "time"):
            if needed_var not in ds.variables:
                raise KeyError(f"{needed_var} variable not found in file: {nc_file}")

        lat = np.array(ds.variables["lat"][:], dtype=float)
        lon = np.array(ds.variables["lon"][:], dtype=float)
        variable = ds.variables[data_var]
        raw = variable[:, :, :]
        fill_value = getattr(variable, "_FillValue", None)
        missing_value = getattr(variable, "missing_value", None)

        values = np.array(raw, dtype=float)
        if fill_value is not None:
            values = np.where(values == fill_value, np.nan, values)
        if missing_value is not None:
            values = np.where(values == missing_value, np.nan, values)

        time_var = ds.variables["time"]
        units = getattr(time_var, "units", None)
        calendar = getattr(time_var, "calendar", "standard")
        if units is None:
            raise KeyError(f"time units attribute missing in file: {nc_file}")
        time_values = np.array(time_var[:])

    dates = np.array(num2date(time_values, units=units, calendar=calendar), dtype=object)

    region_means: Dict[RegionKey, np.ndarray] = {}
    for key in all_region_keys:
        lat_min, lat_max, lon_min, lon_max, country = key
        lat_mask, lon_mask = build_region_masks(
            lat=lat,
            lon=lon,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
        )
        if not np.any(lat_mask):
            raise ValueError(f"No latitude points match region {key}")
        if not np.any(lon_mask):
            raise ValueError(f"No longitude points match region {key}")

        region_values = values[:, lat_mask, :][:, :, lon_mask]
        region_lat = lat[lat_mask]
        region_lon = lon[lon_mask]
        cell_weights = np.cos(np.deg2rad(region_lat))[:, None] * np.ones(
            (1, region_lon.size), dtype=float
        )
        region_mask = np.ones(cell_weights.shape, dtype=bool)

        if country is not None:
            if land_fraction_source is None:
                raise RuntimeError("Land-fraction data must be loaded for regional means")
            if country_geometries is None or country not in country_geometries:
                raise KeyError(f"No country geometry loaded for {country!r}")

            source_lat, source_lon, source_fraction = land_fraction_source
            land_fraction = align_land_fraction(
                source_lat,
                source_lon,
                source_fraction,
                lat,
                lon,
            )[np.ix_(lat_mask, lon_mask)]
            cell_weights *= land_fraction

            lon_grid, lat_grid = np.meshgrid(
                normalize_lon_to_180(region_lon),
                region_lat,
            )
            region_mask = contains_xy(country_geometries[country], lon_grid, lat_grid)

        region_means[key] = area_weighted_mean(
            field_tyx=region_values,
            cell_weights=cell_weights,
            region_mask=region_mask,
        )

    years = np.asarray([int(date.year) for date in dates], dtype=int)
    map_sums: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    if MAP_YEARS.intersection(years.tolist()):
        map_values = np.ma.filled(raw, np.nan).astype(float)
        if fill_value is not None:
            map_values[map_values == fill_value] = np.nan
        if missing_value is not None:
            map_values[map_values == missing_value] = np.nan
        if map_values.shape[1:] == (lat.size, lon.size):
            map_sums = yearly_sums(map_values, years, MAP_YEARS)
    map_data = (lat, lon, map_sums)
    return dates, region_means, branch_time, map_data


def _file_data_task(task: Tuple[str, str]):
    nc_file, variable_name = task
    ctx = _WORKER_CONTEXT
    return task, compute_file_data(
        nc_file,
        variable_name,
        ctx["all_region_keys"],
        ctx["land_fraction_source"],
        ctx["country_geometries"],
    )


def store_file_data(task: Tuple[str, str], result) -> None:
    nc_file, _ = task
    dates, region_means, branch_time, map_data = result
    FILE_REGION_CACHE[task] = (dates, region_means)
    if branch_time is not None or task[1] == "tas" and is_historical_file(nc_file):
        FILE_BRANCH_TIMES[nc_file] = branch_time
    if map_data is not None:
        FILE_MAP_CACHE[task] = map_data


def prefetch_file_data(
    tasks: List[Tuple[str, str]],
    all_region_keys: List[RegionKey],
    land_fraction_source: Tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    country_geometries: Dict[str, BaseGeometry] | None,
    workers: int,
) -> None:
    """Fill the per-file caches in parallel; each file is read exactly once."""
    pending = sorted({t for t in tasks if t not in FILE_REGION_CACHE})
    if not pending:
        return
    t0 = time.perf_counter()
    workers = max(1, min(workers, len(pending)))
    log(f"[prefetch] processing {len(pending)} files with {workers} workers")
    _WORKER_CONTEXT.update(
        all_region_keys=all_region_keys,
        land_fraction_source=land_fraction_source,
        country_geometries=country_geometries,
    )
    if workers == 1:
        for task in pending:
            store_file_data(*_file_data_task(task))
    else:
        step = max(1, len(pending) // 20)
        with mp.get_context("fork").Pool(workers) as pool:
            for i, (task, result) in enumerate(pool.imap_unordered(_file_data_task, pending, chunksize=1), start=1):
                store_file_data(task, result)
                if i % step == 0 or i == len(pending):
                    log(f"[prefetch] {i}/{len(pending)} files done ({time.perf_counter() - t0:.0f}s)")
    log(f"[prefetch] done in {time.perf_counter() - t0:.1f}s")


def read_monthly_region_mean_series(
    nc_files: List[str],
    variable_name: str = "tas",
    region_key: RegionKey | None = None,
    all_region_keys: List[RegionKey] | None = None,
    land_fraction_source: Tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    country_geometries: Dict[str, BaseGeometry] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    t0 = time.perf_counter()
    all_dates: List[object] = []
    all_means: List[np.ndarray] = []
    total_files = len(nc_files)
    if total_files == 0:
        raise ValueError("No input NetCDF files provided")

    log(f"[read {variable_name}] reading {total_files} files")

    if region_key is None:
        region_key = make_region_key(None, None, None, None)
    if all_region_keys is None:
        all_region_keys = [region_key]

    for nc_file in nc_files:
        cache_key = (nc_file, variable_name)
        if cache_key not in FILE_REGION_CACHE:
            store_file_data(
                cache_key,
                compute_file_data(
                    nc_file, variable_name, all_region_keys, land_fraction_source, country_geometries
                ),
            )

        dates, region_means = FILE_REGION_CACHE[cache_key]
        if region_key not in region_means:
            raise KeyError(f"Region key {region_key} not cached for file {nc_file}")
        all_dates.extend(dates.tolist())
        all_means.append(region_means[region_key])

    if not all_means:
        raise ValueError("No time-series data found in input files")

    log(f"[read {variable_name}] done in {time.perf_counter() - t0:.1f}s")
    return np.array(all_dates, dtype=object), np.concatenate(all_means)


def annual_series_from_files(
    nc_files: List[str],
    variable_name: str = "tas",
    lat_min: float | None = None,
    lat_max: float | None = None,
    lon_min: float | None = None,
    lon_max: float | None = None,
    country: str | None = None,
    all_region_keys: List[RegionKey] | None = None,
    season: str | None = None,
    land_fraction_source: Tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    country_geometries: Dict[str, BaseGeometry] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    t0 = time.perf_counter()
    region_key = make_region_key(lat_min, lat_max, lon_min, lon_max, country)
    dates, values = read_monthly_region_mean_series(
        nc_files=nc_files,
        variable_name=variable_name,
        region_key=region_key,
        all_region_keys=all_region_keys,
        land_fraction_source=land_fraction_source,
        country_geometries=country_geometries,
    )
    years, annual_values = annual_mean_from_monthly(dates, values, season=season)
    log(
        f"[annual {variable_name}] {len(nc_files)} files -> {years.size} years "
        f"in {time.perf_counter() - t0:.1f}s"
    )

    if variable_name == "tas":
        return years, annual_values - 273.15
    if variable_name == "pr":
        return years, annual_values * 86400.0 * 365.0
    return years, annual_values


def to_anomaly(
    years: np.ndarray,
    values: np.ndarray,
    baseline_start: int,
    baseline_end: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if years.size == 0:
        return years, values

    baseline_mask = (years >= baseline_start) & (years <= baseline_end)
    if not np.any(baseline_mask):
        raise ValueError(
            f"No baseline years found in range {baseline_start}-{baseline_end}. "
            f"Data span is {int(np.nanmin(years))}-{int(np.nanmax(years))}."
        )

    baseline = float(np.nanmean(values[baseline_mask]))
    return years, values - baseline


def scenario_mean(members: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    if not members:
        return np.array([], dtype=int), np.array([], dtype=float)

    all_years = sorted({int(y) for years, _ in members.values() for y in years})
    if not all_years:
        return np.array([], dtype=int), np.array([], dtype=float)

    year_to_idx = {y: i for i, y in enumerate(all_years)}
    stack = np.full((len(members), len(all_years)), np.nan, dtype=float)

    for row, (years, vals) in enumerate(members.values()):
        for y, v in zip(years, vals):
            stack[row, year_to_idx[int(y)]] = float(v)

    mean_vals = np.nanmean(stack, axis=0)
    return np.array(all_years, dtype=int), mean_vals


def reaches_end_of_century(files: List[str]) -> bool:
    """True if the member's monthly files (named *_YYYYMM-YYYYMM.nc) extend through Dec 2100."""
    end_dates = []
    for path in files:
        match = re.search(r"_(\d{6})-(\d{6})\.nc$", os.path.basename(path))
        if match:
            end_dates.append(int(match.group(2)))
    return bool(end_dates) and max(end_dates) >= SCENARIO_REQUIRED_END_YYYYMM


def drop_incomplete_members(
    files_by_member: Dict[str, List[str]], label: str
) -> Dict[str, List[str]]:
    kept: Dict[str, List[str]] = {}
    for member, files in files_by_member.items():
        if reaches_end_of_century(files):
            kept[member] = files
        else:
            log(f"[members] excluding {label} {member}: data do not extend to 2100")
    return kept


def discover_cmip6_member_files(
    cmip6_hist_root: str,
    cmip6_scen_root: str,
    scenario: str,
    model: str,
    grid: str,
    variable_name: str = "tas",
    member_glob: str = CMIP6_MEMBER_GLOB,
) -> Dict[str, List[str]]:
    if scenario == "historical":
        base_dirs = [cmip6_hist_root]
    else:
        base_dirs = [os.path.join(cmip6_scen_root, scenario)]
        extra_glob = CMIP6_EXTRA_SCEN_DIR_GLOBS.get(scenario)
        if extra_glob:
            base_dirs.extend(sorted(glob.glob(extra_glob)))

    member_dirs: List[str] = []
    seen_members = set()
    for base_dir in base_dirs:
        for member_dir in sorted(glob.glob(os.path.join(base_dir, member_glob))):
            member_name = os.path.basename(member_dir)
            if member_name in seen_members:
                continue
            seen_members.add(member_name)
            member_dirs.append(member_dir)
    files_by_member: Dict[str, List[str]] = {}

    for member_dir in member_dirs:
        member = os.path.basename(member_dir)
        version_dirs = sorted(glob.glob(os.path.join(member_dir, "Amon", variable_name, grid, "v*")))
        if not version_dirs:
            continue
        latest_version = version_dirs[-1]
        pattern = f"{variable_name}_Amon_{model}_{scenario}_{member}_{grid}_*.nc"
        files = sorted(glob.glob(os.path.join(latest_version, pattern)))
        if files:
            files_by_member[member] = files

    if scenario != "historical":
        files_by_member = drop_incomplete_members(files_by_member, f"CMIP6 {scenario}")
    return files_by_member


def cmip7_variable_prefix(variable_name: str) -> str:
    return variable_name


def data_variable_name(variable_name: str, ds) -> str:
    return variable_name


def is_cmip7_target_file(file_name: str, variable_name: str) -> bool:
    """Keep only monthly mean files needed for annual-mean calculations."""
    if not file_name.startswith(f"{variable_name}_"):
        return False
    token = f"_{CMIP7_REQUIRED_FREQUENCY}_"
    if token not in file_name:
        return False
    if variable_name in {"tas", "pr"} and f"_{CMIP7_REQUIRED_STAT}" not in file_name:
        return False
    return True


def discover_cmip7_files(
    member_root: str,
    scenario: str,
    model: str,
    variable_name: str = "tas",
    member: str = "r1i1p2f1",
) -> List[str]:
    matches: List[str] = []
    for nc_file in sorted(glob.glob(os.path.join(member_root, "**", "*.nc"), recursive=True)):
        name = os.path.basename(nc_file)
        if not is_cmip7_target_file(name, variable_name):
            continue
        if model not in name:
            continue
        if f"_{scenario}_" not in name and f"{scenario}/" not in nc_file:
            continue
        if f"_{member}_" not in name and f"/{member}/" not in nc_file:
            continue
        matches.append(nc_file)
    return matches


def discover_cmip7_member_files(
    member_dir_glob: str,
    scenario: str,
    model: str,
    variable_name: str = "tas",
) -> Dict[str, List[str]]:
    member_dirs = sorted(glob.glob(member_dir_glob))
    files_by_member: Dict[str, List[str]] = {}

    for member_dir in member_dirs:
        if not os.path.isdir(member_dir):
            continue
        matches: List[str] = []
        for nc_file in sorted(glob.glob(os.path.join(member_dir, "**", "*.nc"), recursive=True)):
            name = os.path.basename(nc_file)
            if not is_cmip7_target_file(name, variable_name):
                continue
            if model not in name:
                continue
            if f"_{scenario}_" not in name:
                continue
            matches.append(nc_file)

        if not matches:
            continue

        member = name_to_member_id(matches[0], scenario)
        if member:
            files_by_member[member] = matches

    if scenario != "historical":
        files_by_member = drop_incomplete_members(files_by_member, f"CMIP7 {scenario}")
    return files_by_member


def name_to_member_id(file_name: str, scenario: str) -> str | None:
    marker = f"_{scenario}_"
    if marker not in file_name:
        return None
    tail = file_name.split(marker, 1)[1]
    member = tail.split("_", 1)[0]
    return member if member else None


def build_cmip7_ensemble_series(
    variable_name: str,
    lat_min: float | None,
    lat_max: float | None,
    lon_min: float | None,
    lon_max: float | None,
    all_region_keys: List[RegionKey],
    season: str | None,
    hist_member_files: Dict[str, List[str]],
    scen7h_member_files: Dict[str, List[str]],
    scen7m_member_files: Dict[str, List[str]],
    scen7vl_member_files: Dict[str, List[str]],
    country: str | None = None,
    land_fraction_source: Tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    country_geometries: Dict[str, BaseGeometry] | None = None,
    n_members: int | None = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    t0 = time.perf_counter()
    selected_hist_members = first_n_members(list(hist_member_files.keys()), n_members)
    selected_h_members = first_n_members(list(scen7h_member_files.keys()), n_members)
    selected_m_members = first_n_members(list(scen7m_member_files.keys()), n_members)
    selected_vl_members = first_n_members(list(scen7vl_member_files.keys()), n_members)
    log(
        f"[CMIP7 {variable_name}] selected members: historical={len(selected_hist_members)}, "
        f"scen7-h={len(selected_h_members)}, scen7-m={len(selected_m_members)}, "
        f"scen7-vl={len(selected_vl_members)}"
    )

    hist_members: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for idx, member in enumerate(selected_hist_members, start=1):
        log(f"[CMIP7 {variable_name}] historical member {idx}/{len(selected_hist_members)}: {member}")
        years, vals = annual_series_from_files(
            nc_files=hist_member_files[member],
            variable_name=variable_name,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            country=country,
            all_region_keys=all_region_keys,
            season=season,
            land_fraction_source=land_fraction_source,
            country_geometries=country_geometries,
        )
        hist_members[member] = (years, vals)

    h_members: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for idx, member in enumerate(selected_h_members, start=1):
        if member not in hist_member_files:
            continue
        log(f"[CMIP7 {variable_name}] scen7-h member {idx}/{len(selected_h_members)}: {member}")
        years, vals = annual_series_from_files(
            nc_files=hist_member_files[member] + scen7h_member_files[member],
            variable_name=variable_name,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            country=country,
            all_region_keys=all_region_keys,
            season=season,
            land_fraction_source=land_fraction_source,
            country_geometries=country_geometries,
        )
        h_members[member] = (years, vals)

    m_members: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for idx, member in enumerate(selected_m_members, start=1):
        if member not in hist_member_files:
            continue
        log(f"[CMIP7 {variable_name}] scen7-m member {idx}/{len(selected_m_members)}: {member}")
        years, vals = annual_series_from_files(
            nc_files=hist_member_files[member] + scen7m_member_files[member],
            variable_name=variable_name,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            country=country,
            all_region_keys=all_region_keys,
            season=season,
            land_fraction_source=land_fraction_source,
            country_geometries=country_geometries,
        )
        m_members[member] = (years, vals)

    vl_members: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for idx, member in enumerate(selected_vl_members, start=1):
        if member not in hist_member_files:
            continue
        log(f"[CMIP7 {variable_name}] scen7-vl member {idx}/{len(selected_vl_members)}: {member}")
        years, vals = annual_series_from_files(
            nc_files=hist_member_files[member] + scen7vl_member_files[member],
            variable_name=variable_name,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            country=country,
            all_region_keys=all_region_keys,
            season=season,
            land_fraction_source=land_fraction_source,
            country_geometries=country_geometries,
        )
        vl_members[member] = (years, vals)

    out = {
        "historical": scenario_mean(hist_members),
        "scen7-h": scenario_mean(h_members),
        "scen7-m": scenario_mean(m_members),
        "scen7-vl": scenario_mean(vl_members),
    }
    log(f"[CMIP7 {variable_name}] ensemble done in {time.perf_counter() - t0:.1f}s")
    return out


def build_cmip6_ensemble_series(
    variable_name: str,
    lat_min: float | None,
    lat_max: float | None,
    lon_min: float | None,
    lon_max: float | None,
    all_region_keys: List[RegionKey],
    season: str | None,
    hist_member_files: Dict[str, List[str]],
    scenario_member_files: Dict[str, Dict[str, List[str]]],
    country: str | None = None,
    land_fraction_source: Tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    country_geometries: Dict[str, BaseGeometry] | None = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    t0 = time.perf_counter()
    log(f"[CMIP6 {variable_name}] building ensemble from {len(hist_member_files)} historical members")
    hist_members: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    hist_items = [
        (member, hist_member_files[member])
        for member in first_n_members(list(hist_member_files), MAX_ENSEMBLE_MEMBERS)
    ]
    for idx, (member, files) in enumerate(hist_items, start=1):
        log(f"[CMIP6 {variable_name}] historical member {idx}/{len(hist_items)}: {member}")
        years, vals = annual_series_from_files(
            nc_files=files,
            variable_name=variable_name,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            country=country,
            all_region_keys=all_region_keys,
            season=season,
            land_fraction_source=land_fraction_source,
            country_geometries=country_geometries,
        )
        hist_members[member] = (years, vals)

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {"historical": scenario_mean(hist_members)}

    for scenario in CMIP6_SCENARIOS[1:]:
        log(f"[CMIP6 {variable_name}] scenario {scenario}")
        scen_members: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        files_by_member = scenario_member_files.get(scenario, {}).get(variable_name, {})
        scen_items = [
            (member, files_by_member[member])
            for member in first_n_members(list(files_by_member), MAX_ENSEMBLE_MEMBERS)
        ]
        for idx, (member, scen_files) in enumerate(scen_items, start=1):
            if member not in hist_member_files:
                continue
            log(f"[CMIP6 {variable_name}] {scenario} member {idx}/{len(scen_items)}: {member}")
            years, vals = annual_series_from_files(
                nc_files=hist_member_files[member] + scen_files,
                variable_name=variable_name,
                lat_min=lat_min,
                lat_max=lat_max,
                lon_min=lon_min,
                lon_max=lon_max,
                country=country,
                all_region_keys=all_region_keys,
                season=season,
                land_fraction_source=land_fraction_source,
                country_geometries=country_geometries,
            )
            scen_members[member] = (years, vals)
        out[scenario] = scenario_mean(scen_members)

    log(f"[CMIP6 {variable_name}] ensemble done in {time.perf_counter() - t0:.1f}s")
    return out


def plot_ensemble(
    ax,
    family: str,
    panel_idx: int,
    baseline_start: int,
    baseline_end: int,
    scenario_series_override: Dict[str, Tuple[np.ndarray, np.ndarray]],
    variable_name: str,
) -> None:
    del variable_name
    if family == "cmip7":
        colors = CMIP7_COLORS
        labels = CMIP7_LABELS
        order = CMIP7_SCENARIOS
        linewidth = 3.0
        zorder = 3.0
        legend_panel = 0
    else:
        colors = CMIP6_GREYS
        labels = CMIP6_LABELS
        order = CMIP6_SCENARIOS
        linewidth = 2.6
        zorder = 1.5
        legend_panel = 0

    for scenario in order:
        years, vals = scenario_series_override.get(scenario, (np.array([]), np.array([])))
        years, vals = to_anomaly(years, vals, baseline_start, baseline_end)
        if years.size == 0:
            continue

        years, vals = apply_family_scenario_plot_window(years, vals, family, scenario)
        if years.size == 0:
            continue

        label = labels[scenario] if panel_idx == legend_panel else "_nolegend_"
        ax.plot(years, vals, color=colors[scenario], linewidth=linewidth, zorder=zorder, label=label)


def divide_by_reference_for_plot_years(
    years: np.ndarray,
    values: np.ndarray,
    ref_years: np.ndarray,
    ref_values: np.ndarray,
    min_year: int = 2000,
    max_year: int = 2100,
) -> Tuple[np.ndarray, np.ndarray]:
    years = np.asarray(years, dtype=int)
    values = np.asarray(values, dtype=float)
    ref_years = np.asarray(ref_years, dtype=int)
    ref_values = np.asarray(ref_values, dtype=float)

    common_years, idx_vals, idx_ref = np.intersect1d(years, ref_years, return_indices=True)
    if common_years.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    vals = values[idx_vals]
    refs = ref_values[idx_ref]
    valid = (
        np.isfinite(vals)
        & np.isfinite(refs)
        & (np.abs(refs) > 1e-12)
        & (common_years >= min_year)
        & (common_years <= max_year)
    )
    return common_years[valid], vals[valid] / refs[valid]


def apply_family_scenario_plot_window(
    years: np.ndarray,
    values: np.ndarray,
    family: str,
    scenario: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if family == "cmip6":
        start_year = CMIP6_SCENARIO_START_YEAR
    else:
        start_year = CMIP7_SCENARIO_START_YEAR

    if scenario == "historical":
        mask = years < start_year
    else:
        mask = years >= start_year

    return years[mask], values[mask]


def plot_ensemble_per_gsat(
    ax,
    family: str,
    panel_idx: int,
    baseline_start: int,
    baseline_end: int,
    scenario_series_override: Dict[str, Tuple[np.ndarray, np.ndarray]],
    reference_series_override: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> None:
    if family == "cmip7":
        colors = CMIP7_COLORS
        labels = CMIP7_LABELS
        order = CMIP7_SCENARIOS
        linewidth = 3.0
        zorder = 3.0
        legend_panel = 0
    else:
        colors = CMIP6_GREYS
        labels = CMIP6_LABELS
        order = CMIP6_SCENARIOS
        linewidth = 2.6
        zorder = 1.5
        legend_panel = 0

    for scenario in order:
        years, vals = scenario_series_override.get(scenario, (np.array([]), np.array([])))
        years, vals = to_anomaly(years, vals, baseline_start, baseline_end)
        if years.size == 0:
            continue

        years, vals = apply_family_scenario_plot_window(years, vals, family, scenario)
        if years.size == 0:
            continue

        ref_years, ref_vals = reference_series_override.get(scenario, (np.array([]), np.array([])))
        ref_years, ref_vals = to_anomaly(ref_years, ref_vals, baseline_start, baseline_end)
        if ref_years.size == 0:
            continue

        ref_years, ref_vals = apply_family_scenario_plot_window(ref_years, ref_vals, family, scenario)
        if ref_years.size == 0:
            continue

        ratio_years, ratio_vals = divide_by_reference_for_plot_years(years, vals, ref_years, ref_vals, min_year=2000, max_year=2100)
        if ratio_years.size == 0:
            continue

        label = labels[scenario] if panel_idx == legend_panel else "_nolegend_"
        ax.plot(ratio_years, ratio_vals, color=colors[scenario], linewidth=linewidth, zorder=zorder, label=label)


def output_path_with_suffix(output_path: str, suffix: str) -> str:
    root, ext = os.path.splitext(output_path)
    if not ext:
        ext = ".png"
    return f"{root}{suffix}{ext}"


def read_file_year_sums(
    nc_file: str,
    variable_name: str,
    requested_years,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    with Dataset(nc_file, mode="r") as ds:
        for name in (variable_name, "lat", "lon", "time"):
            if name not in ds.variables:
                raise KeyError(f"{name} variable not found in file: {nc_file}")

        lat = np.asarray(ds.variables["lat"][:], dtype=float)
        lon = np.asarray(ds.variables["lon"][:], dtype=float)
        variable = ds.variables[variable_name]
        values = np.ma.filled(variable[:, :, :], np.nan).astype(float)
        fill_value = getattr(variable, "_FillValue", None)
        if fill_value is not None:
            values[values == fill_value] = np.nan
        missing_value = getattr(variable, "missing_value", None)
        if missing_value is not None:
            values[values == missing_value] = np.nan

        time_var = ds.variables["time"]
        units = getattr(time_var, "units", None)
        if units is None:
            raise KeyError(f"time units attribute missing in file: {nc_file}")
        calendar = getattr(time_var, "calendar", "standard")
        dates = num2date(np.asarray(time_var[:]), units=units, calendar=calendar)
        years = np.asarray([int(date.year) for date in dates], dtype=int)

    if values.shape[1:] != (lat.size, lon.size):
        raise ValueError(
            f"{variable_name} grid shape {values.shape[1:]} does not match "
            f"lat/lon dimensions {(lat.size, lon.size)} in {nc_file}"
        )
    return lat, lon, yearly_sums(values, years, requested_years)


def same_grid(
    lat_a: np.ndarray,
    lon_a: np.ndarray,
    lat_b: np.ndarray,
    lon_b: np.ndarray,
) -> bool:
    return (
        lat_a.shape == lat_b.shape
        and lon_a.shape == lon_b.shape
        and np.allclose(lat_a, lat_b, rtol=0.0, atol=1.0e-6)
        and np.allclose(lon_a, lon_b, rtol=0.0, atol=1.0e-6)
    )


def mean_grid_for_periods(
    nc_files: List[str],
    variable_name: str,
    periods: List[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray, Dict[Tuple[int, int], np.ndarray]]:
    if not nc_files:
        raise ValueError(f"No NetCDF files provided for {variable_name}")

    requested_years = {
        year
        for start_year, end_year in periods
        for year in range(start_year, end_year + 1)
    }
    annual_totals: Dict[int, np.ndarray] = {}
    annual_counts: Dict[int, np.ndarray] = {}
    reference_lat = None
    reference_lon = None
    for nc_file in nc_files:
        cached = FILE_MAP_CACHE.get((nc_file, variable_name))
        if cached is not None and requested_years <= MAP_YEARS:
            lat, lon, file_sums = cached
            file_sums = {year: s for year, s in file_sums.items() if year in requested_years}
        else:
            lat, lon, file_sums = read_file_year_sums(nc_file, variable_name, requested_years)

        if reference_lat is None:
            reference_lat, reference_lon = lat, lon
        elif not same_grid(reference_lat, reference_lon, lat, lon):
            raise ValueError(f"Grid coordinates differ between input files: {nc_file}")

        for year, (year_total, year_count) in file_sums.items():
            if year in annual_totals:
                annual_totals[year] = annual_totals[year] + year_total
                annual_counts[year] = annual_counts[year] + year_count
            else:
                annual_totals[year] = year_total
                annual_counts[year] = year_count

    annual_grids = {
        year: np.divide(
            total,
            annual_counts[year],
            out=np.full(total.shape, np.nan, dtype=float),
            where=annual_counts[year] > 0,
        )
        for year, total in annual_totals.items()
    }
    period_means: Dict[Tuple[int, int], np.ndarray] = {}
    for start_year, end_year in periods:
        period_years = [
            year for year in range(start_year, end_year + 1) if year in annual_grids
        ]
        if len(period_years) != end_year - start_year + 1:
            raise ValueError(
                f"{variable_name} data cover {len(period_years)} of "
                f"{end_year - start_year + 1} years in {start_year}-{end_year}"
            )
        period_values = np.stack([annual_grids[year] for year in period_years])
        if variable_name == "tas":
            period_values = period_values - 273.15
        elif variable_name == "pr":
            period_values = period_values * 86400.0 * 365.0
        period_means[(start_year, end_year)] = np.nanmean(period_values, axis=0)

    return reference_lat, reference_lon, period_means


def build_period_ensemble_maps(
    historical_files: Dict[str, List[str]],
    scenario_files: Dict[str, List[str]],
    variable_name: str,
    periods: List[Tuple[int, int]],
    n_members: int | None = None,
) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    selected_members = first_n_members(list(scenario_files), n_members)
    period_member_grids: Dict[Tuple[int, int], List[np.ndarray]] = {
        period: [] for period in periods
    }
    reference_lat = None
    reference_lon = None

    for member in selected_members:
        if member not in historical_files:
            log(f"[map {variable_name}] skipping {member}: no historical files")
            continue
        lat, lon, means = mean_grid_for_periods(
            historical_files[member] + scenario_files[member],
            variable_name,
            [MAP_BASELINE_PERIOD, *periods],
        )
        if reference_lat is None:
            reference_lat, reference_lon = lat, lon
        elif not same_grid(reference_lat, reference_lon, lat, lon):
            raise ValueError(f"Grid coordinates differ for {variable_name} member {member}")
        for period in periods:
            period_member_grids[period].append(means[period] - means[MAP_BASELINE_PERIOD])

    if reference_lat is None or reference_lon is None:
        raise ValueError(f"No paired historical/scenario members found for {variable_name}")

    return {
        period: (
            reference_lat,
            reference_lon,
            np.nanmean(np.stack(member_grids), axis=0),
            np.stack(member_grids),
        )
        for period, member_grids in period_member_grids.items()
        if member_grids
    }


def significant_difference_mask(
    members_a: np.ndarray, members_b: np.ndarray, alpha: float = SIGNIFICANCE_LEVEL
) -> np.ndarray:
    """Two-sample (Welch) t-test across members at each grid point; True where p < alpha."""
    if members_a.shape[0] < 2 or members_b.shape[0] < 2:
        return np.zeros(members_a.shape[1:], dtype=bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        _, p_values = stats.ttest_ind(members_a, members_b, axis=0, equal_var=False, nan_policy="omit")
    return np.isfinite(p_values) & (p_values < alpha)


def plot_scenario_difference_maps(
    fields: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    period: Tuple[int, int],
    output_path: str,
    coastline: BaseGeometry,
) -> None:
    panel_specs = [
        ("a) H − SSP4-6.0: Temperature", "tas_h_ssp460", "°C"),
        ("b) H − SSP4-6.0: Precipitation", "pr_h_ssp460", "mm/year"),
        ("c) VL − SSP1-1.9: Temperature", "tas_vl_ssp119", "°C"),
        ("d) VL − SSP1-1.9: Precipitation", "pr_vl_ssp119", "mm/year"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), constrained_layout=True)
    plotted = {}

    for ax, (title, key, units) in zip(axes.ravel(), panel_specs):
        lat, lon, values, significant = fields[key]
        limit = max(
            float(np.nanmax(np.abs(fields[field_key][2])))
            for _, field_key, field_units in panel_specs
            if field_units == units
        )
        if not np.isfinite(limit) or limit == 0.0:
            limit = 1.0
        image = ax.pcolormesh(
            lon,
            lat,
            values,
            shading="auto",
            cmap="RdBu" if units == "mm/year" else "RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        plotted[units] = image
        if np.any(significant):
            with plt.rc_context({"hatch.color": "0.15", "hatch.linewidth": 0.5}):
                ax.contourf(
                    lon,
                    lat,
                    significant.astype(float),
                    levels=[0.5, 1.5],
                    colors="none",
                    hatches=["////"],
                    zorder=2,
                )
        ax.set_title(title)
        ax.set_xlim(float(np.nanmin(lon)), float(np.nanmax(lon)))
        ax.set_ylim(float(np.nanmin(lat)), float(np.nanmax(lat)))
        ax.set_xticks([])
        ax.set_yticks([])
        plot_coastlines(ax, coastline, longitude_360=float(np.nanmax(lon)) > 180.0)

    for column, units in enumerate(("°C", "mm/year")):
        fig.colorbar(plotted[units], ax=axes[:, column], label=f"Difference in anomalies ({units})")

    fig.suptitle(
        f"Difference in scenario anomalies, {period[0]}–{period[1]} "
        "(relative to 1850–1900)",
        fontsize=14,
    )
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    log(f"Saved scenario difference maps to {output_path}")


def plot_coastlines(ax, coastline: BaseGeometry, longitude_360: bool) -> None:
    line_geometries = [coastline]
    while line_geometries:
        geometry = line_geometries.pop()
        if geometry.geom_type != "LineString":
            line_geometries.extend(getattr(geometry, "geoms", []))
            continue

        coordinates = np.asarray(geometry.coords, dtype=float)
        longitude = coordinates[:, 0].copy()
        if longitude_360:
            longitude %= 360.0
        split_points = (
            np.flatnonzero(np.abs(np.diff(longitude)) > 180.0) + 1
            if longitude_360
            else np.array([], dtype=int)
        )
        for segment in np.split(np.arange(coordinates.shape[0]), split_points):
            if segment.size < 2:
                continue
            ax.plot(
                longitude[segment],
                coordinates[segment, 1],
                color="black",
                linewidth=0.45,
                zorder=3,
            )


def main() -> None:
    t0_main = time.perf_counter()
    parser = argparse.ArgumentParser(
        description=(
            "Plot annual regional tas and precipitation anomaly comparisons for CMIP6 and CMIP7 "
            "(baseline 1850-1900)"
        )
    )
    parser.add_argument("--cmip6-hist-root", default=DEFAULT_CMIP6_HIST_ROOT, help="CMIP6 historical root directory")
    parser.add_argument("--cmip6-scen-root", default=DEFAULT_CMIP6_SCEN_ROOT, help="CMIP6 ScenarioMIP root directory")
    parser.add_argument("--cmip7-hist-root", default=DEFAULT_CMIP7_HIST_ROOT, help="CMIP7 historical root directory")
    parser.add_argument("--cmip7-scen7h-root", default=DEFAULT_CMIP7_SCEN7H_ROOT, help="CMIP7 scen7-h member root directory")
    parser.add_argument("--cmip7-scen7m-root", default=DEFAULT_CMIP7_SCEN7M_ROOT, help="CMIP7 scen7-m member root directory")
    parser.add_argument(
        "--cmip7-scen7vl-root",
        default=DEFAULT_CMIP7_SCEN7VL_ROOT_CANDIDATES[0],
        help="Primary CMIP7 scen7-vl member root directory",
    )
    parser.add_argument(
        "--cmip7-hist-member-dir-glob",
        default=DEFAULT_CMIP7_HIST_MEMBER_DIR_GLOB,
        help="Glob that expands to CMIP7 historical member data directories",
    )
    parser.add_argument(
        "--cmip7-scen7h-member-dir-glob",
        default=DEFAULT_CMIP7_SCEN7H_MEMBER_DIR_GLOB,
        help="Glob that expands to CMIP7 scen7-h member data directories",
    )
    parser.add_argument(
        "--cmip7-scen7m-member-dir-glob",
        default=DEFAULT_CMIP7_SCEN7M_MEMBER_DIR_GLOB,
        help="Glob that expands to CMIP7 scen7-m member data directories",
    )
    parser.add_argument(
        "--cmip7-scen7vl-member-dir-glob",
        default=DEFAULT_CMIP7_SCEN7VL_MEMBER_DIR_GLOB,
        help="Glob that expands to CMIP7 scen7-vl member data directories",
    )
    parser.add_argument(
        "--cmip7-ensemble-members",
        type=int,
        default=10,
        help="Maximum number of CMIP7 members to use; default is 10 for a practical runtime",
    )
    parser.add_argument("--member", default="r1i1p2f1", help="Ensemble member used for CMIP7 single-member fallback paths")
    parser.add_argument("--model", default="CanESM5-1", help="Model name")
    parser.add_argument("--grid", default="gn", help="CMIP6 grid label")
    parser.add_argument("--baseline-start", type=int, default=1850, help="Baseline period start year")
    parser.add_argument("--baseline-end", type=int, default=1900, help="Baseline period end year")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output PNG file path")
    parser.add_argument("--branch-times-output", default=None, help="CSV output path for historical ensemble branch times")
    parser.add_argument("--land-mask", default=DEFAULT_LAND_MASK, help="NetCDF file containing the sftlf land fraction")
    parser.add_argument(
        "--countries-shapefile",
        default=DEFAULT_COUNTRIES_SHAPEFILE,
        help="Natural Earth admin-0 countries archive or its URL",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, len(os.sched_getaffinity(0))),
        help="Number of parallel worker processes used to read NetCDF files",
    )
    args = parser.parse_args()
    branch_times_output = args.branch_times_output
    if branch_times_output is None:
        branch_times_output = os.path.splitext(args.output)[0] + "_branch_times.csv"

    log("[main] Starting plot assembly")
    cmip7_member_limit = (
        min(args.cmip7_ensemble_members, MAX_ENSEMBLE_MEMBERS)
        if args.cmip7_ensemble_members > 0
        else MAX_ENSEMBLE_MEMBERS
    )
    log(f"[main] Using CMIP7 ensemble cap: {cmip7_member_limit}")

    land_fraction_source = read_land_fraction(args.land_mask)
    country_geometries = read_country_geometries(args.countries_shapefile)
    coastline = read_coastline_geometry(args.countries_shapefile)
    cmip7_scen7vl_root = pick_existing_path([args.cmip7_scen7vl_root] + DEFAULT_CMIP7_SCEN7VL_ROOT_CANDIDATES[1:])

    cmip6_hist_member_files = discover_cmip6_member_files(
        cmip6_hist_root=args.cmip6_hist_root,
        cmip6_scen_root=args.cmip6_scen_root,
        scenario="historical",
        model=args.model,
        grid=args.grid,
        variable_name="tas",
    )
    if not cmip6_hist_member_files:
        raise FileNotFoundError("No CMIP6 historical tas member files found")

    cmip6_hist_pr_member_files = discover_cmip6_member_files(
        cmip6_hist_root=args.cmip6_hist_root,
        cmip6_scen_root=args.cmip6_scen_root,
        scenario="historical",
        model=args.model,
        grid=args.grid,
        variable_name="pr",
    )
    if not cmip6_hist_pr_member_files:
        raise FileNotFoundError("No CMIP6 historical pr member files found")

    cmip6_scenario_member_files: Dict[str, Dict[str, List[str]]] = {}
    for scenario in CMIP6_SCENARIOS[1:]:
        tas_files_by_member = discover_cmip6_member_files(
            cmip6_hist_root=args.cmip6_hist_root,
            cmip6_scen_root=args.cmip6_scen_root,
            scenario=scenario,
            model=args.model,
            grid=args.grid,
            variable_name="tas",
        )
        if not tas_files_by_member:
            raise FileNotFoundError(f"No CMIP6 tas member files found for scenario {scenario}")
        pr_files_by_member = discover_cmip6_member_files(
            cmip6_hist_root=args.cmip6_hist_root,
            cmip6_scen_root=args.cmip6_scen_root,
            scenario=scenario,
            model=args.model,
            grid=args.grid,
            variable_name="pr",
        )
        if not pr_files_by_member:
            raise FileNotFoundError(f"No CMIP6 pr member files found for scenario {scenario}")

        cmip6_scenario_member_files[scenario] = {"tas": tas_files_by_member, "pr": pr_files_by_member}

    cmip7_historical_tas = discover_cmip7_files(args.cmip7_hist_root, "historical", args.model, variable_name="tas", member=args.member)
    cmip7_historical_pr = discover_cmip7_files(args.cmip7_hist_root, "historical", args.model, variable_name="pr", member=args.member)
    if not cmip7_historical_tas or not cmip7_historical_pr:
        raise FileNotFoundError("No CMIP7 historical tas/pr files found")

    scen7h_tas = discover_cmip7_files(args.cmip7_scen7h_root, "scen7-h", args.model, variable_name="tas", member=args.member)
    scen7h_pr = discover_cmip7_files(args.cmip7_scen7h_root, "scen7-h", args.model, variable_name="pr", member=args.member)
    if not scen7h_tas or not scen7h_pr:
        raise FileNotFoundError("No CMIP7 scen7-h tas/pr files found")

    scen7m_tas = discover_cmip7_files(args.cmip7_scen7m_root, "scen7-m", args.model, variable_name="tas", member=args.member)
    scen7m_pr = discover_cmip7_files(args.cmip7_scen7m_root, "scen7-m", args.model, variable_name="pr", member=args.member)
    if not scen7m_tas or not scen7m_pr:
        raise FileNotFoundError("No CMIP7 scen7-m tas/pr files found")

    scen7vl_tas = discover_cmip7_files(cmip7_scen7vl_root, "scen7-vl", args.model, variable_name="tas", member=args.member)
    scen7vl_pr = discover_cmip7_files(cmip7_scen7vl_root, "scen7-vl", args.model, variable_name="pr", member=args.member)
    if not scen7vl_tas or not scen7vl_pr:
        raise FileNotFoundError("No CMIP7 scen7-vl tas/pr files found")

    cmip7_hist_member_files_tas = discover_cmip7_member_files(args.cmip7_hist_member_dir_glob, "historical", args.model, variable_name="tas")
    cmip7_hist_member_files_pr = discover_cmip7_member_files(args.cmip7_hist_member_dir_glob, "historical", args.model, variable_name="pr")
    if not cmip7_hist_member_files_tas or not cmip7_hist_member_files_pr:
        raise FileNotFoundError("No CMIP7 historical member tas/pr directories/files found")

    cmip7_scen7h_member_files_tas = discover_cmip7_member_files(args.cmip7_scen7h_member_dir_glob, "scen7-h", args.model, variable_name="tas")
    cmip7_scen7h_member_files_pr = discover_cmip7_member_files(args.cmip7_scen7h_member_dir_glob, "scen7-h", args.model, variable_name="pr")
    if not cmip7_scen7h_member_files_tas or not cmip7_scen7h_member_files_pr:
        raise FileNotFoundError("No CMIP7 scen7-h member tas/pr directories/files found")

    cmip7_scen7m_member_files_tas = discover_cmip7_member_files(args.cmip7_scen7m_member_dir_glob, "scen7-m", args.model, variable_name="tas")
    cmip7_scen7m_member_files_pr = discover_cmip7_member_files(args.cmip7_scen7m_member_dir_glob, "scen7-m", args.model, variable_name="pr")
    if not cmip7_scen7m_member_files_tas or not cmip7_scen7m_member_files_pr:
        raise FileNotFoundError("No CMIP7 scen7-m member tas/pr directories/files found")

    cmip7_scen7vl_member_files_tas = discover_cmip7_member_files(args.cmip7_scen7vl_member_dir_glob, "scen7-vl", args.model, variable_name="tas")
    cmip7_scen7vl_member_files_pr = discover_cmip7_member_files(args.cmip7_scen7vl_member_dir_glob, "scen7-vl", args.model, variable_name="pr")
    if not cmip7_scen7vl_member_files_tas or not cmip7_scen7vl_member_files_pr:
        raise FileNotFoundError("No CMIP7 scen7-vl member tas/pr directories/files found")

    log(f"[main] CMIP6 historical tas members: {len(cmip6_hist_member_files)}")
    log(f"[main] CMIP6 historical pr members: {len(cmip6_hist_pr_member_files)}")
    for scen in CMIP6_SCENARIOS[1:]:
        members = cmip6_scenario_member_files.get(scen, {})
        log(f"[main] CMIP6 {scen} tas members: {len(members.get('tas', {}))}")
        log(f"[main] CMIP6 {scen} pr members: {len(members.get('pr', {}))}")
    log(f"[main] CMIP7 historical tas members: {len(cmip7_hist_member_files_tas)}")
    log(f"[main] CMIP7 historical pr members: {len(cmip7_hist_member_files_pr)}")
    log(f"[main] CMIP7 scen7-h tas members: {len(cmip7_scen7h_member_files_tas)}")
    log(f"[main] CMIP7 scen7-h pr members: {len(cmip7_scen7h_member_files_pr)}")
    log(f"[main] CMIP7 scen7-m tas members: {len(cmip7_scen7m_member_files_tas)}")
    log(f"[main] CMIP7 scen7-m pr members: {len(cmip7_scen7m_member_files_pr)}")
    log(f"[main] CMIP7 scen7-vl tas members: {len(cmip7_scen7vl_member_files_tas)}")
    log(f"[main] CMIP7 scen7-vl pr members: {len(cmip7_scen7vl_member_files_pr)}")
    log("[main] Building figure panels")

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
        }
    )

    ncols = 2
    displayed_panel_count = len(PANEL_SPECS) - GLOBAL_REFERENCE_PANEL_COUNT
    nrows = (displayed_panel_count + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.8, 3.7 * nrows), sharex=True)
    axes_flat = np.atleast_1d(axes).ravel()

    all_region_keys = unique_region_keys_from_panel_specs(PANEL_SPECS)

    def selected_pairs(hist_files, scen_files):
        files = []
        for member in first_n_members(list(hist_files), args.cmip7_ensemble_members):
            files.extend(hist_files[member])
        for member in first_n_members(list(scen_files), args.cmip7_ensemble_members):
            if member in hist_files:
                files.extend(hist_files[member] + scen_files[member])
        return files

    prefetch_tasks: List[Tuple[str, str]] = []
    for variable_name, hist7, hist6 in (
        ("tas", cmip7_hist_member_files_tas, cmip6_hist_member_files),
        ("pr", cmip7_hist_member_files_pr, cmip6_hist_pr_member_files),
    ):
        scen7_by_variable = {
            "tas": (cmip7_scen7h_member_files_tas, cmip7_scen7m_member_files_tas, cmip7_scen7vl_member_files_tas),
            "pr": (cmip7_scen7h_member_files_pr, cmip7_scen7m_member_files_pr, cmip7_scen7vl_member_files_pr),
        }[variable_name]
        for scen_files in scen7_by_variable:
            prefetch_tasks.extend((f, variable_name) for f in selected_pairs(hist7, scen_files))
        selected_hist6 = {
            member: hist6[member]
            for member in first_n_members(list(hist6), MAX_ENSEMBLE_MEMBERS)
        }
        for files in selected_hist6.values():
            prefetch_tasks.extend((f, variable_name) for f in files)
        for scenario in CMIP6_SCENARIOS[1:]:
            scenario_files = cmip6_scenario_member_files[scenario][variable_name]
            for member in first_n_members(list(scenario_files), MAX_ENSEMBLE_MEMBERS):
                if member in selected_hist6:
                    prefetch_tasks.extend((f, variable_name) for f in scenario_files[member])
    prefetch_file_data(
        prefetch_tasks,
        all_region_keys,
        land_fraction_source,
        country_geometries,
        args.workers,
    )

    panel_series: List[Dict[str, object]] = []

    for i, spec in enumerate(PANEL_SPECS):
        lat_min = spec.get("lat_min")
        lat_max = spec.get("lat_max")
        lon_min = spec.get("lon_min")
        lon_max = spec.get("lon_max")
        country = spec["country"]
        variable_name = spec["variable"]
        panel_t0 = time.perf_counter()
        log(f"[panel {i+1}/{len(PANEL_SPECS)}] {spec['title']} ({variable_name})")

        cmip7_series = build_cmip7_ensemble_series(
            variable_name=variable_name,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            all_region_keys=all_region_keys,
            season=None,
            hist_member_files=(cmip7_hist_member_files_tas if variable_name == "tas" else cmip7_hist_member_files_pr),
            scen7h_member_files=(cmip7_scen7h_member_files_tas if variable_name == "tas" else cmip7_scen7h_member_files_pr),
            scen7m_member_files=(cmip7_scen7m_member_files_tas if variable_name == "tas" else cmip7_scen7m_member_files_pr),
            scen7vl_member_files=(cmip7_scen7vl_member_files_tas if variable_name == "tas" else cmip7_scen7vl_member_files_pr),
            country=country,
            land_fraction_source=land_fraction_source,
            country_geometries=country_geometries,
            n_members=args.cmip7_ensemble_members,
        )

        cmip6_series = build_cmip6_ensemble_series(
            variable_name=variable_name,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            all_region_keys=all_region_keys,
            season=None,
            hist_member_files=(cmip6_hist_member_files if variable_name == "tas" else cmip6_hist_pr_member_files),
            scenario_member_files=cmip6_scenario_member_files,
            country=country,
            land_fraction_source=land_fraction_source,
            country_geometries=country_geometries,
        )

        panel_series.append({"spec": spec, "cmip7": cmip7_series, "cmip6": cmip6_series})

        if i < GLOBAL_REFERENCE_PANEL_COUNT:
            log(f"[panel {i+1}/{len(PANEL_SPECS)}] retained as per-GSAT reference")
            continue

        display_idx = i - GLOBAL_REFERENCE_PANEL_COUNT
        ax = axes_flat[display_idx]
        plot_ensemble(
            ax,
            family="cmip7",
            panel_idx=display_idx,
            baseline_start=args.baseline_start,
            baseline_end=args.baseline_end,
            scenario_series_override=cmip7_series,
            variable_name=variable_name,
        )
        plot_ensemble(
            ax,
            family="cmip6",
            panel_idx=display_idx,
            baseline_start=args.baseline_start,
            baseline_end=args.baseline_end,
            scenario_series_override=cmip6_series,
            variable_name=variable_name,
        )
        log(f"[panel {display_idx+1}/{displayed_panel_count}] done in {time.perf_counter() - panel_t0:.1f}s")

        ax.set_title(spec["title"])
        ax.set_xlabel("Year")
        ax.set_ylabel(spec["ylabel"])
        ax.set_xlim(2000, 2100)
        ax.axhline(0.0, color="0.7", linewidth=1.0, zorder=0)
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)

    if len(axes_flat) > 0:
        axes_flat[0].legend(loc="upper left", ncol=1, frameon=False)

    write_historical_branch_times(
        output_path=branch_times_output,
        cmip6_member_files=cmip6_hist_member_files,
        cmip7_member_files=cmip7_hist_member_files_tas,
    )
    log(f"Saved historical branch times to {branch_times_output}")

    fig.tight_layout(pad=1.0)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.output, dpi=180)
    plt.close(fig)

    ratio_output = output_path_with_suffix(args.output, "_per_gsat")
    fig_ratio, axes_ratio = plt.subplots(nrows, ncols, figsize=(10.8, 3.7 * nrows), sharex=True)
    axes_ratio_flat = np.atleast_1d(axes_ratio).ravel()

    if not panel_series:
        raise RuntimeError("No panel series were generated")

    global_ref_cmip7 = panel_series[0]["cmip7"]
    global_ref_cmip6 = panel_series[0]["cmip6"]

    displayed_panel_series = panel_series[GLOBAL_REFERENCE_PANEL_COUNT:]
    for i, panel_data in enumerate(displayed_panel_series):
        ax = axes_ratio_flat[i]
        spec = panel_data["spec"]
        variable_name = spec["variable"]

        plot_ensemble_per_gsat(
            ax,
            family="cmip7",
            panel_idx=i,
            baseline_start=args.baseline_start,
            baseline_end=args.baseline_end,
            scenario_series_override=panel_data["cmip7"],
            reference_series_override=global_ref_cmip7,
        )
        plot_ensemble_per_gsat(
            ax,
            family="cmip6",
            panel_idx=i,
            baseline_start=args.baseline_start,
            baseline_end=args.baseline_end,
            scenario_series_override=panel_data["cmip6"],
            reference_series_override=global_ref_cmip6,
        )

        panel_prefix = spec["title"].split(")", 1)[0] + ") "
        if variable_name == "tas":
            title = f"{panel_prefix}{spec['region_label']} warming per degree global warming"
        else:
            title = f"{panel_prefix}{spec['region_label']} precipitation change per degree global warming"
        ax.set_title(title)
        ax.set_xlabel("Year")
        if variable_name == "tas":
            ax.set_ylabel("°C per °C")
        else:
            ax.set_ylabel("mm/year per °C")
        ax.set_xlim(2000, 2100)
        ax.axhline(0.0, color="0.7", linewidth=1.0, zorder=0)
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)

    if len(axes_ratio_flat) > 0:
        axes_ratio_flat[0].legend(loc="lower left", ncol=1, frameon=False)

    fig_ratio.tight_layout(pad=1.0)
    fig_ratio.savefig(ratio_output, dpi=180)
    plt.close(fig_ratio)

    comparison_periods = MAP_PERIODS
    map_ensembles = {}
    for variable_name, cmip7_historical, cmip7_scenario, cmip6_scenario in (
        (
            "tas",
            cmip7_hist_member_files_tas,
            cmip7_scen7h_member_files_tas,
            "ssp460",
        ),
        (
            "pr",
            cmip7_hist_member_files_pr,
            cmip7_scen7h_member_files_pr,
            "ssp460",
        ),
    ):
        map_ensembles[("h_ssp460", variable_name, "cmip7")] = build_period_ensemble_maps(
            cmip7_historical,
            cmip7_scenario,
            variable_name,
            comparison_periods,
            n_members=args.cmip7_ensemble_members,
        )
        map_ensembles[("h_ssp460", variable_name, "cmip6")] = build_period_ensemble_maps(
            (cmip6_hist_member_files if variable_name == "tas" else cmip6_hist_pr_member_files),
            cmip6_scenario_member_files[cmip6_scenario][variable_name],
            variable_name,
            comparison_periods,
        )

    for variable_name, cmip7_historical, cmip7_scenario in (
        ("tas", cmip7_hist_member_files_tas, cmip7_scen7vl_member_files_tas),
        ("pr", cmip7_hist_member_files_pr, cmip7_scen7vl_member_files_pr),
    ):
        map_ensembles[("vl_ssp119", variable_name, "cmip7")] = build_period_ensemble_maps(
            cmip7_historical,
            cmip7_scenario,
            variable_name,
            comparison_periods,
            n_members=args.cmip7_ensemble_members,
        )
        map_ensembles[("vl_ssp119", variable_name, "cmip6")] = build_period_ensemble_maps(
            (cmip6_hist_member_files if variable_name == "tas" else cmip6_hist_pr_member_files),
            cmip6_scenario_member_files["ssp119"][variable_name],
            variable_name,
            comparison_periods,
        )

    for period in comparison_periods:
        difference_fields = {}
        for comparison in ("h_ssp460", "vl_ssp119"):
            for variable_name in ("tas", "pr"):
                cmip7_lat, cmip7_lon, cmip7_values, cmip7_members = map_ensembles[
                    (comparison, variable_name, "cmip7")
                ][period]
                cmip6_lat, cmip6_lon, cmip6_values, cmip6_members = map_ensembles[
                    (comparison, variable_name, "cmip6")
                ][period]
                if not (
                    cmip7_lat.shape == cmip6_lat.shape
                    and cmip7_lon.shape == cmip6_lon.shape
                    and np.allclose(cmip7_lat, cmip6_lat, rtol=0.0, atol=1.0e-6)
                    and np.allclose(cmip7_lon, cmip6_lon, rtol=0.0, atol=1.0e-6)
                ):
                    raise ValueError(
                        f"CMIP7 and CMIP6 {variable_name} grids differ; "
                        "scenario difference maps require matching grids"
                    )
                difference_fields[f"{variable_name}_{comparison}"] = (
                    cmip7_lat,
                    cmip7_lon,
                    cmip7_values - cmip6_values,
                    significant_difference_mask(cmip7_members, cmip6_members),
                )
                log(
                    f"[map {variable_name} {comparison} {period[0]}-{period[1]}] "
                    f"CMIP7 n={cmip7_members.shape[0]}, CMIP6 n={cmip6_members.shape[0]}"
                )

        maps_output = output_path_with_suffix(
            args.output,
            f"_scenario_differences_{period[0]}-{period[1]}",
        )
        plot_scenario_difference_maps(difference_fields, period, maps_output, coastline)

    log(f"Saved regional tas/precip comparison plot to {args.output}")
    log(f"Saved per-GSAT regional tas/precip comparison plot to {ratio_output}")
    log(f"[main] Total runtime: {time.perf_counter() - t0_main:.1f}s")


if __name__ == "__main__":
    main()
