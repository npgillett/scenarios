#!/usr/bin/env python3
"""Plot annual regional TXx and RX1day anomaly comparisons for CMIP6 and CMIP7.

The figure shows:
    a) global mean TXx          b) global mean RX1day
    c) Canada mean TXx          d) Canada mean RX1day
    e) Brazil mean TXx          f) Brazil mean RX1day
    g) Antarctica mean TXx      h) Antarctica mean RX1day

TXx is the annual maximum of daily tasmax and RX1day is the annual maximum of daily
precipitation (mm/day), both taken at each grid point. Regional means of these
gridded annual extremes are then processed exactly as in
plot_regional_mean_tas_pr_scenarios_annual_2x2.py (land-fraction weighting and
Natural Earth country masks for Canada/Brazil/Antarctica, ensemble means,
anomalies relative to 1850-1900, legends and the per-global-warming
figure; the RTD-file annual GSAT reference uses the same members as the
corresponding TXx or RX1day ensemble, see --rtd-dir).
No time-series smoothing is applied. Ensembles use at most 10 members.

Efficiency: each daily file is read one year at a time and reduced to annual
maximum fields; files are processed in parallel and the annual-maximum fields are
cached on disk, so reruns only redo the (cheap) regional averaging.
"""

import argparse
import glob
import hashlib
import multiprocessing as mp
import os
import re
import time
from functools import lru_cache
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset, num2date
from shapely import contains_xy

import plot_regional_mean_tas_pr_scenarios_annual_2x2 as base
from plot_regional_mean_tas_pr_scenarios_annual_2x2 import RegionKey, log

DEFAULT_OUTPUT = "/home/rng001/plots/scenarios/canesm5-1_txx_rx1day_regional_anoms_2x2.png"
DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/scenarios/extremes")

# CMIP6 variable -> (name in file, CMIP7 variable dir, CMIP7 statistic dir)
VARIABLES = {
    "txx": {"cmip6": "tasmax", "cmip7": ("tas", "tmax-h2m-hxy-u"), "file_var": {"cmip6": "tasmax", "cmip7": "tas"}},
    "rx1day": {"cmip6": "pr", "cmip7": ("pr", "tavg-u-hxy-u"), "file_var": {"cmip6": "pr", "cmip7": "pr"}},
}
MIN_DAYS_PER_YEAR = 300
MAX_ENSEMBLE_MEMBERS = 10
DEFAULT_CMIP6_EXTRA_HIST_ROOT = "/fs/site7/eccc/crd/cccma/users/ncs001/v51_histens-02-10"

DEFAULT_RTD_DIR = "/home/rng001/site7/rtd_nc"
RTD_GSAT_PREFIX = "SCREEN_TEMPERATURE_GLOBAL__K__________________ANN_00001"
# family -> scenario -> RTD file glob; "historical" is prepended to every scenario member.
RTD_PATTERNS = {
    "cmip7": {
        "historical": "sc_v51p2-c7hist-r*_185001_202112_rtd074.nc",
        "scen7-h": "sc_v51p2-c7scen7h-r*_202201_210012_rtd074.nc",
        "scen7-m": "sc_v51p2-c7scen7m-r*_202201_210012_rtd074.nc",
        "scen7-vl": "sc_v51p2-c7scen7vl-r*_202201_210012_rtd074.nc",
    },
    "cmip6": {
        "historical": "sc_v5.1-p2-histens-*_185001_201412_rtd074.nc",
        "ssp119": "sc_v5.1.p2-s119-*_rtd074.nc",
        "ssp126": "sc_v5.1.p2-s126-*_rtd074.nc",
        "ssp245": "sc_v5.1.p2-s245-*_rtd074.nc",
        "ssp460": "sc_v5.1.p2-s460-*_rtd074.nc",
        "ssp370": "sc_v5.1.p2-s370-*_rtd074.nc",
        "ssp585": "sc_v5.1-p2-ssp585-*_rtd074.nc",
    },
}
CMIP6_EXTRA_RTD_PATTERNS = {
    "ssp119": "sc_v51p2-c6s119-r*_201501_210012_rtd074.nc",
    "ssp126": "sc_v51p2-c6s126-r*_201501_210012_rtd074.nc",
    "ssp460": "sc_v51p2-c6s460-r*_201501_210012_rtd074.nc",
}

PANEL_SPECS = [
    {"title": "a) Global mean TXx", "country": None, "region_label": "Global", "variable": "txx"},
    {"title": "b) Global mean RX1day", "country": None, "region_label": "Global", "variable": "rx1day"},
    {"title": "c) Canada mean TXx", "country": "Canada", "region_label": "Canada", "variable": "txx"},
    {"title": "d) Canada mean RX1day", "country": "Canada", "region_label": "Canada", "variable": "rx1day"},
    {"title": "e) Brazil mean TXx", "country": "Brazil", "region_label": "Brazil", "variable": "txx"},
    {"title": "f) Brazil mean RX1day", "country": "Brazil", "region_label": "Brazil", "variable": "rx1day"},
    {"title": "g) Antarctica mean TXx", "country": "Antarctica", "region_label": "Antarctica", "variable": "txx"},
    {"title": "h) Antarctica mean RX1day", "country": "Antarctica", "region_label": "Antarctica", "variable": "rx1day"},
]
YLABELS = {"txx": "Anomaly (°C, vs 1850-1900)", "rx1day": "Anomaly (mm/day, vs 1850-1900)"}
RATIO_YLABELS = {"txx": "°C per °C", "rx1day": "mm/day per °C"}
REGION_KEYS: List[RegionKey] = [
    (None, None, None, None, country) for country in (None, "Canada", "Brazil", "Antarctica")
]

# Set before the worker pool is created so forked workers inherit them.
LAND_FRACTION_SOURCE = None
COUNTRY_GEOMETRIES = None
CACHE_DIR: str | None = DEFAULT_CACHE_DIR


# ----------------------------------------------------------------------------
# Annual maxima
# ----------------------------------------------------------------------------
def annual_max_fields(nc_file: str, file_var: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (years, annual-max fields [year, lat, lon], lat, lon), reading one year at a time."""
    with Dataset(nc_file, mode="r") as ds:
        ds.set_auto_maskandscale(False)
        var = ds.variables[file_var]
        lat = np.array(ds.variables["lat"][:], dtype=float)
        lon = np.array(ds.variables["lon"][:], dtype=float)
        time_var = ds.variables["time"]
        dates = num2date(
            np.array(time_var[:]),
            units=time_var.units,
            calendar=getattr(time_var, "calendar", "standard"),
        )
        time_years = np.array([d.year for d in dates], dtype=int)
        bounds = np.concatenate(([0], np.flatnonzero(np.diff(time_years)) + 1, [time_years.size]))

        years: List[int] = []
        fields: List[np.ndarray] = []
        for start, stop in zip(bounds[:-1], bounds[1:]):
            if stop - start < MIN_DAYS_PER_YEAR:
                continue
            chunk = np.asarray(var[start:stop], dtype=np.float32)
            chunk[np.abs(chunk) > 1.0e19] = np.nan
            fields.append(np.fmax.reduce(chunk, axis=0))
            years.append(int(time_years[start]))
    return np.array(years, dtype=int), np.stack(fields), lat, lon


def cached_annual_max_fields(nc_file: str, file_var: str):
    if CACHE_DIR is None:
        return annual_max_fields(nc_file, file_var)
    stat = os.stat(nc_file)
    digest = hashlib.md5(f"{nc_file}|{file_var}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{digest}.npz")
    if os.path.exists(cache_path):
        try:
            with np.load(cache_path) as npz:
                return npz["years"], npz["fields"], npz["lat"], npz["lon"]
        except Exception:
            pass
    result = annual_max_fields(nc_file, file_var)
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp_path = f"{cache_path}.{os.getpid()}.tmp.npz"
    np.savez(tmp_path, years=result[0], fields=result[1], lat=result[2], lon=result[3])
    os.replace(tmp_path, cache_path)
    return result


# ----------------------------------------------------------------------------
# Region weights and regional means
# ----------------------------------------------------------------------------
@lru_cache(maxsize=8)
def region_weights(lat_bytes: bytes, lon_bytes: bytes) -> Dict[RegionKey, Tuple[np.ndarray, np.ndarray]]:
    """Per-region (cell weights, polygon mask) on a grid; computed once per grid."""
    lat = np.frombuffer(lat_bytes, dtype=float)
    lon = np.frombuffer(lon_bytes, dtype=float)
    lon_grid, lat_grid = np.meshgrid(base.normalize_lon_to_180(lon), lat)
    area = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, lon.size))
    land = None
    out: Dict[RegionKey, Tuple[np.ndarray, np.ndarray]] = {}
    for key in REGION_KEYS:
        country = key[4]
        if country is None:
            out[key] = (area, np.ones(area.shape, dtype=bool))
            continue
        if land is None:
            src_lat, src_lon, src_fraction = LAND_FRACTION_SOURCE
            land = base.align_land_fraction(src_lat, src_lon, src_fraction, lat, lon)
        out[key] = (area * land, contains_xy(COUNTRY_GEOMETRIES[country], lon_grid, lat_grid))
    return out


def file_task(task: Tuple[str, str, str]) -> Tuple[Tuple[str, str], Tuple[np.ndarray, Dict[RegionKey, np.ndarray]]]:
    nc_file, family, variable = task
    spec = VARIABLES[variable]
    years, fields, lat, lon = cached_annual_max_fields(nc_file, spec["file_var"][family])
    weights = region_weights(lat.tobytes(), lon.tobytes())
    series = {
        key: base.area_weighted_mean(fields, cell_weights, region_mask)
        for key, (cell_weights, region_mask) in weights.items()
    }
    return (nc_file, variable), (years, series)


def compute_file_series(
    tasks: List[Tuple[str, str, str]],
    workers: int,
) -> Dict[Tuple[str, str], Tuple[np.ndarray, Dict[RegionKey, np.ndarray]]]:
    t0 = time.perf_counter()
    tasks = sorted(set(tasks))
    log(f"[extremes] processing {len(tasks)} daily files with {workers} workers")
    results: Dict[Tuple[str, str], Tuple[np.ndarray, Dict[RegionKey, np.ndarray]]] = {}
    if workers <= 1:
        for i, task in enumerate(tasks, start=1):
            key, value = file_task(task)
            results[key] = value
    else:
        step = max(1, len(tasks) // 20)
        with mp.get_context("fork").Pool(workers) as pool:
            for i, (key, value) in enumerate(pool.imap_unordered(file_task, tasks, chunksize=1), start=1):
                results[key] = value
                if i % step == 0 or i == len(tasks):
                    log(f"[extremes] {i}/{len(tasks)} files done ({time.perf_counter() - t0:.0f}s)")
    log(f"[extremes] all files done in {time.perf_counter() - t0:.1f}s")
    return results


# ----------------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------------
def latest_version_files(pattern_dir: str, name_glob: str) -> List[str]:
    version_dirs = sorted(glob.glob(os.path.join(pattern_dir, "v*")))
    if not version_dirs:
        return []
    return sorted(glob.glob(os.path.join(version_dirs[-1], name_glob)))


def discover_cmip6(base_dir: str, variable: str, grid: str) -> Dict[str, List[str]]:
    cmip6_var = VARIABLES[variable]["cmip6"]
    out: Dict[str, List[str]] = {}
    for member_dir in sorted(glob.glob(os.path.join(base_dir, base.CMIP6_MEMBER_GLOB))):
        files = latest_version_files(
            os.path.join(member_dir, "day", cmip6_var, grid), f"{cmip6_var}_day_*.nc"
        )
        if files:
            out[os.path.basename(member_dir)] = files
    return out


def discover_cmip6_scenario(
    scenarios_root: str, scenario: str, variable: str, grid: str
) -> Dict[str, List[str]]:
    cmip6_var = VARIABLES[variable]["cmip6"]
    scenario_dirs = [os.path.join(scenarios_root, scenario)]
    extra_glob = base.CMIP6_EXTRA_SCEN_DIR_GLOBS.get(scenario)
    if extra_glob:
        scenario_dirs.extend(sorted(glob.glob(extra_glob)))

    out: Dict[str, List[str]] = {}
    for scenario_dir in scenario_dirs:
        for member_dir in sorted(glob.glob(os.path.join(scenario_dir, base.CMIP6_MEMBER_GLOB))):
            member = os.path.basename(member_dir)
            if member in out:
                continue
            files = latest_version_files(
                os.path.join(member_dir, "day", cmip6_var, grid), f"{cmip6_var}_day_*.nc"
            )
            if files:
                out[member] = files
    return out


def discover_cmip6_extra_historical(base_dir: str, variable: str, grid: str) -> Dict[str, List[str]]:
    cmip6_var = VARIABLES[variable]["cmip6"]
    member_dirs = glob.glob(
        os.path.join(
            base_dir,
            "v5.1-p2-histens-*",
            "nc_output",
            "CMIP6",
            "CCCma",
            "CCCma",
            "CanESM5-v5.1-p2-histens-*",
            "historical",
            base.CMIP6_MEMBER_GLOB,
        )
    )
    out: Dict[str, List[str]] = {}
    for member_dir in sorted(member_dirs):
        member = os.path.basename(member_dir)
        match = re.fullmatch(r"r(\d+)i\d+p\d+f\d+", member)
        if match is None or not 2 <= int(match.group(1)) <= 10:
            continue
        files = latest_version_files(
            os.path.join(member_dir, "day", cmip6_var, grid), f"{cmip6_var}_day_*.nc"
        )
        if files:
            out[member] = files
    return out


def discover_cmip7(member_dir_glob: str, variable: str) -> Dict[str, List[str]]:
    var_dir, stat_dir = VARIABLES[variable]["cmip7"]
    out: Dict[str, List[str]] = {}
    for member_dir in sorted(glob.glob(member_dir_glob)):
        files = latest_version_files(
            os.path.join(member_dir, "glb", "day", var_dir, stat_dir, "g120"), f"{var_dir}_{stat_dir}_day_*.nc"
        )
        if files:
            out[os.path.basename(member_dir)] = files
    return out


# ----------------------------------------------------------------------------
# Ensemble assembly
# ----------------------------------------------------------------------------
def member_series(
    files: List[str],
    variable: str,
    region_key: RegionKey,
    file_results: Dict[Tuple[str, str], Tuple[np.ndarray, Dict[RegionKey, np.ndarray]]],
) -> Tuple[np.ndarray, np.ndarray]:
    years = np.concatenate([file_results[(f, variable)][0] for f in files])
    values = np.concatenate([file_results[(f, variable)][1][region_key] for f in files])
    order = np.argsort(years, kind="stable")
    years, values = years[order], values[order]
    if variable == "txx":
        values = values - 273.15
    else:
        values = values * 86400.0
    return years, values


def selected_members(
    hist_files: Dict[str, List[str]],
    scenario_files: Dict[str, Dict[str, List[str]]],
    n_members: int | None,
) -> Dict[str, List[str]]:
    member_limit = (
        MAX_ENSEMBLE_MEMBERS
        if n_members is None or n_members <= 0
        else min(n_members, MAX_ENSEMBLE_MEMBERS)
    )
    selected = {"historical": base.first_n_members(list(hist_files), member_limit)}
    for scenario, files_by_member in scenario_files.items():
        selected[scenario] = [
            member
            for member in base.first_n_members(list(files_by_member), member_limit)
            if member in hist_files
        ]
    return selected


def build_ensemble(
    family: str,
    variable: str,
    region_key: RegionKey,
    hist_files: Dict[str, List[str]],
    scenario_files: Dict[str, Dict[str, List[str]]],
    scenarios: List[str],
    n_members: int | None,
    file_results,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    members_by_scenario = selected_members(hist_files, scenario_files, n_members)
    hist_members = members_by_scenario["historical"]
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
        "historical": base.scenario_mean(
            {m: member_series(hist_files[m], variable, region_key, file_results) for m in hist_members}
        )
    }
    for scenario in scenarios:
        files_by_member = scenario_files.get(scenario, {})
        members = members_by_scenario.get(scenario, [])
        out[scenario] = base.scenario_mean(
            {
                m: member_series(hist_files[m] + files_by_member[m], variable, region_key, file_results)
                for m in members
            }
        )
    return out


def log_member_counts(
    family: str,
    variable: str,
    hist_files: Dict[str, List[str]],
    scenario_files: Dict[str, Dict[str, List[str]]],
    n_members: int | None,
) -> None:
    """Log how many members build_ensemble uses for each scenario."""
    counts = {
        scenario: len(members)
        for scenario, members in selected_members(hist_files, scenario_files, n_members).items()
    }
    log(f"[members used] {family} {variable}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))


def read_rtd_gsat(nc_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Annual global-mean screen temperature (K) from an RTD file."""
    with Dataset(nc_file, mode="r") as ds:
        names = list(ds.variables)
        temp_var = next((v for v in names if v.startswith(RTD_GSAT_PREFIX)), None)
        year_var = next((v for v in names if v.startswith("YEAR")), None)
        if temp_var is None or year_var is None:
            raise KeyError(f"No global screen temperature/YEAR variable in {nc_file}")
        years = np.array(ds.variables[year_var][:], dtype=float).squeeze()
        values = np.array(ds.variables[temp_var][:], dtype=float).squeeze()
    valid = np.isfinite(years) & np.isfinite(values)
    years, values = years[valid].astype(int), values[valid]
    unique_years = np.unique(years)
    return unique_years, np.array([values[years == y].mean() for y in unique_years])


def rtd_members(rtd_dir: str, pattern: str | List[str]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    patterns = [pattern] if isinstance(pattern, str) else pattern
    for file_pattern in patterns:
        for path in glob.glob(os.path.join(rtd_dir, file_pattern)):
            match = re.search(r"-r?(\d+)(?:_\d{6}_\d{6})?_rtd074\.nc$", os.path.basename(path))
            if match:
                out[int(match.group(1))] = path
    return out


def build_rtd_gsat(
    rtd_dir: str, family: str, variable: str, members_by_scenario: Dict[str, List[str]]
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Build RTD GSAT ensembles from the exact member selections used by the data."""
    patterns = RTD_PATTERNS[family]
    hist_files = rtd_members(rtd_dir, patterns["historical"])
    if not hist_files:
        raise FileNotFoundError(f"No {family} historical RTD files found in {rtd_dir}")

    def rtd_ids(member_names: List[str], scenario: str) -> List[int]:
        ids = []
        for member in member_names:
            match = re.match(r"r(\d+)", member)
            if not match:
                raise ValueError(f"Cannot map {family} {scenario} member {member!r} to an RTD member ID")
            member_id = int(match.group(1))
            if member_id not in hist_files:
                raise FileNotFoundError(
                    f"No {family} historical RTD file for {scenario} member {member} in {rtd_dir}"
                )
            ids.append(member_id)
        return ids

    hist_ids = rtd_ids(members_by_scenario.get("historical", []), "historical")
    if not hist_ids:
        raise FileNotFoundError(f"No {family} historical members selected for RTD GSAT")
    selected_rtd_ids = set(hist_ids)
    for scenario, members in members_by_scenario.items():
        if scenario != "historical":
            selected_rtd_ids.update(rtd_ids(members, scenario))
    hist_series = {
        member_id: read_rtd_gsat(hist_files[member_id])
        for member_id in sorted(selected_rtd_ids)
    }
    out = {"historical": base.scenario_mean({str(m): hist_series[m] for m in hist_ids})}
    counts = {"historical": len(hist_ids)}
    for scenario, pattern in patterns.items():
        if scenario == "historical":
            continue
        scenario_patterns = [pattern]
        if family == "cmip6" and scenario in CMIP6_EXTRA_RTD_PATTERNS:
            scenario_patterns.append(CMIP6_EXTRA_RTD_PATTERNS[scenario])
        scen_files = rtd_members(rtd_dir, scenario_patterns)
        ids = rtd_ids(members_by_scenario.get(scenario, []), scenario)
        if not ids:
            raise FileNotFoundError(f"No {family} {scenario} members selected for RTD GSAT")
        members = {}
        for m in ids:
            if m not in scen_files:
                raise FileNotFoundError(
                    f"No {family} {scenario} RTD file for selected member r{m} in {rtd_dir}; "
                    "matching RTD references are required for every selected ensemble member"
                )
            years, vals = read_rtd_gsat(scen_files[m])
            h_years, h_vals = hist_series[m]
            members[str(m)] = (np.concatenate([h_years, years]), np.concatenate([h_vals, vals]))
        out[scenario] = base.scenario_mean(members)
        counts[scenario] = len(ids)
    log(
        f"[members used] {family} {variable} RTD GSAT: "
        + ", ".join(f"{scenario}={count}" for scenario, count in counts.items())
    )
    return out


def selected_tasks(
    family: str,
    variable: str,
    hist_files: Dict[str, List[str]],
    scenario_files: Dict[str, Dict[str, List[str]]],
    n_members: int | None,
) -> List[Tuple[str, str, str]]:
    tasks: List[Tuple[str, str, str]] = []
    members_by_scenario = selected_members(hist_files, scenario_files, n_members)
    hist_members = members_by_scenario["historical"]
    for m in hist_members:
        tasks.extend((f, family, variable) for f in hist_files[m])
    for scenario, files_by_member in scenario_files.items():
        for m in members_by_scenario[scenario]:
            tasks.extend((f, family, variable) for f in files_by_member[m])
            tasks.extend((f, family, variable) for f in hist_files[m])
    return tasks


def main() -> None:
    global LAND_FRACTION_SOURCE, COUNTRY_GEOMETRIES, CACHE_DIR
    t0_main = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Plot annual regional TXx and RX1day anomaly comparisons for CMIP6 and CMIP7 (baseline 1850-1900)"
    )
    parser.add_argument("--cmip6-hist-root", default=base.DEFAULT_CMIP6_HIST_ROOT)
    parser.add_argument(
        "--cmip6-extra-hist-root",
        default=DEFAULT_CMIP6_EXTRA_HIST_ROOT,
        help="Root containing additional CMIP6 historical members r2-r10",
    )
    parser.add_argument("--cmip6-scen-root", default=base.DEFAULT_CMIP6_SCEN_ROOT)
    parser.add_argument("--cmip7-hist-member-dir-glob", default=base.DEFAULT_CMIP7_HIST_MEMBER_DIR_GLOB)
    parser.add_argument("--cmip7-scen7h-member-dir-glob", default=base.DEFAULT_CMIP7_SCEN7H_MEMBER_DIR_GLOB)
    parser.add_argument("--cmip7-scen7m-member-dir-glob", default=base.DEFAULT_CMIP7_SCEN7M_MEMBER_DIR_GLOB)
    parser.add_argument("--cmip7-scen7vl-member-dir-glob", default=base.DEFAULT_CMIP7_SCEN7VL_MEMBER_DIR_GLOB)
    parser.add_argument(
        "--cmip7-ensemble-members",
        type=int,
        default=MAX_ENSEMBLE_MEMBERS,
        help="Maximum number of CMIP7 members to use (hard cap: 10)",
    )
    parser.add_argument("--rtd-dir", default=DEFAULT_RTD_DIR, help="Directory with RTD NetCDF files (GSAT)")
    parser.add_argument("--grid", default="gn", help="CMIP6 grid label")
    parser.add_argument("--baseline-start", type=int, default=1850)
    parser.add_argument("--baseline-end", type=int, default=1900)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output PNG file path")
    parser.add_argument("--land-mask", default=base.DEFAULT_LAND_MASK)
    parser.add_argument("--countries-shapefile", default=base.DEFAULT_COUNTRIES_SHAPEFILE)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1), help="Parallel worker processes")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="Cache directory for annual-maximum fields")
    parser.add_argument("--no-cache", action="store_true", help="Do not read/write the annual-maximum cache")
    args = parser.parse_args()

    CACHE_DIR = None if args.no_cache else args.cache_dir
    LAND_FRACTION_SOURCE = base.read_land_fraction(args.land_mask)
    COUNTRY_GEOMETRIES = base.read_country_geometries(args.countries_shapefile)

    cmip7_globs = {
        "historical": args.cmip7_hist_member_dir_glob,
        "scen7-h": args.cmip7_scen7h_member_dir_glob,
        "scen7-m": args.cmip7_scen7m_member_dir_glob,
        "scen7-vl": args.cmip7_scen7vl_member_dir_glob,
    }

    files: Dict[str, Dict[str, object]] = {}
    tasks: List[Tuple[str, str, str]] = []
    for variable in VARIABLES:
        cmip6_hist = discover_cmip6(args.cmip6_hist_root, variable, args.grid)
        extra_cmip6_hist = discover_cmip6_extra_historical(
            args.cmip6_extra_hist_root, variable, args.grid
        )
        for member, member_files in extra_cmip6_hist.items():
            if member in cmip6_hist:
                log(f"[main] retaining configured CMIP6 historical files for duplicate member {member}")
                continue
            cmip6_hist[member] = member_files
        if not cmip6_hist:
            raise FileNotFoundError(f"No CMIP6 historical daily {variable} files found")
        cmip6_scen = {
            s: discover_cmip6_scenario(args.cmip6_scen_root, s, variable, args.grid)
            for s in base.CMIP6_SCENARIOS[1:]
        }
        for s, found in cmip6_scen.items():
            if not found:
                raise FileNotFoundError(f"No CMIP6 {s} daily {variable} files found")
        cmip7 = {name: discover_cmip7(pattern, variable) for name, pattern in cmip7_globs.items()}
        for name, found in cmip7.items():
            if not found:
                raise FileNotFoundError(f"No CMIP7 {name} daily {variable} files found")
        files[variable] = {
            "cmip6_hist": cmip6_hist,
            "cmip6_scen": cmip6_scen,
            "cmip7_hist": cmip7["historical"],
            "cmip7_scen": {k: v for k, v in cmip7.items() if k != "historical"},
        }
        log(
            f"[main] {variable}: CMIP6 hist members={len(cmip6_hist)}, "
            + ", ".join(f"{s}={len(v)}" for s, v in cmip6_scen.items())
            + "; CMIP7 "
            + ", ".join(f"{k}={len(v)}" for k, v in cmip7.items())
        )
        log_member_counts("cmip6", variable, cmip6_hist, cmip6_scen, None)
        log_member_counts("cmip7", variable, cmip7["historical"], files[variable]["cmip7_scen"], args.cmip7_ensemble_members)
        tasks += selected_tasks("cmip6", variable, cmip6_hist, cmip6_scen, None)
        tasks += selected_tasks("cmip7", variable, cmip7["historical"], files[variable]["cmip7_scen"], args.cmip7_ensemble_members)

    gsat = {}
    for family in ("cmip7", "cmip6"):
        gsat[family] = {}
        n_members = args.cmip7_ensemble_members if family == "cmip7" else None
        for variable in VARIABLES:
            variable_files = files[variable]
            hist_files = variable_files[f"{family}_hist"]
            scenario_files = variable_files[f"{family}_scen"]
            scenario_gsat = build_rtd_gsat(
                args.rtd_dir,
                family,
                variable,
                selected_members(hist_files, scenario_files, n_members),
            )
            gsat[family][variable] = scenario_gsat

    file_results = compute_file_series(tasks, args.workers)

    panel_series: List[Dict[str, object]] = []
    for spec in PANEL_SPECS:
        variable = spec["variable"]
        key: RegionKey = (None, None, None, None, spec["country"])
        f = files[variable]
        cmip7_series = build_ensemble(
            "cmip7", variable, key, f["cmip7_hist"], f["cmip7_scen"],
            base.CMIP7_SCENARIOS[1:], args.cmip7_ensemble_members, file_results,
        )
        cmip6_series = build_ensemble(
            "cmip6", variable, key, f["cmip6_hist"], f["cmip6_scen"],
            base.CMIP6_SCENARIOS[1:], None, file_results,
        )
        panel_series.append(
            {
                "spec": spec,
                "cmip7": cmip7_series,
                "cmip6": cmip6_series,
            }
        )

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8})
    ncols = 2
    nrows = (len(PANEL_SPECS) + ncols - 1) // ncols

    def finish_axes(ax, title: str, ylabel: str) -> None:
        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.set_ylabel(ylabel)
        ax.set_xlim(2000, 2100)
        ax.axhline(0.0, color="0.7", linewidth=1.0, zorder=0)
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)

    fig, axes = plt.subplots(nrows, ncols, figsize=(10.8, 3.7 * nrows), sharex=True)
    axes_flat = np.atleast_1d(axes).ravel()
    for i, data in enumerate(panel_series):
        spec = data["spec"]
        for family in ("cmip7", "cmip6"):
            base.plot_ensemble(
                axes_flat[i], family, i, args.baseline_start, args.baseline_end, data[family], spec["variable"]
            )
        finish_axes(axes_flat[i], spec["title"], YLABELS[spec["variable"]])
    axes_flat[0].legend(loc="upper left", ncol=1, frameon=False)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout(pad=1.0)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    ratio_output = base.output_path_with_suffix(args.output, "_per_gsat")
    fig_r, axes_r = plt.subplots(nrows, ncols, figsize=(10.8, 3.7 * nrows), sharex=True)
    axes_r_flat = np.atleast_1d(axes_r).ravel()
    for i, data in enumerate(panel_series):
        spec = data["spec"]
        for family in ("cmip7", "cmip6"):
            base.plot_ensemble_per_gsat(
                axes_r_flat[i], family, i, args.baseline_start, args.baseline_end,
                data[family], gsat[family][spec["variable"]],
            )
        prefix = spec["title"].split(")", 1)[0] + ") "
        name = "TXx" if spec["variable"] == "txx" else "RX1day"
        finish_axes(
            axes_r_flat[i],
            f"{prefix}{spec['region_label']} {name} change per degree global warming",
            RATIO_YLABELS[spec["variable"]],
        )
    axes_r_flat[0].legend(loc="lower left", ncol=1, frameon=False)
    fig_r.tight_layout(pad=1.0)
    fig_r.savefig(ratio_output, dpi=180)
    plt.close(fig_r)

    log(f"Saved regional TXx/RX1day plot to {args.output}")
    log(f"Saved per-global-warming plot to {ratio_output}")
    log(f"[main] Total runtime: {time.perf_counter() - t0_main:.1f}s")


if __name__ == "__main__":
    main()
