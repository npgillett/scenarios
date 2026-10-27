#!/usr/bin/env python3
"""
Plot greenhouse-gas and aerosol figures from RTD NetCDF files.

The greenhouse-gas figure contains:
- CO2 concentration
- CO2 emissions
- CH4 concentration
- N2O concentration

The aerosol figure contains sulfur emissions, sulfate burden, black carbon
emissions, and organic carbon emissions. Both use the same scenario families
as the climate-response figure.
"""

import argparse
import csv
import glob
import os
import re
from typing import Callable, Dict, List, Tuple

import numpy as np
from netCDF4 import Dataset

DEFAULT_BRANCH_TIMES_CSV = "/home/rng001/plots/scenarios/canesm5-1_tas_precip_regional_anoms_2x2_branch_times.csv"
DEFAULT_CMIP6_PICONTROL = "/home/rng001/site7/rtd_nc/sc_v510b-p2-pict-joined_600001_901712_rtd074.nc"
DEFAULT_CMIP7_PICONTROL = "/home/rng001/site7/rtd_nc/sc_v51p2-c7pic-r1_650001_749912_rtd074.nc"
DEFAULT_CO2_ANOMALY_SUMMARY = "/home/rng001/plots/scenarios/cmip7_cmip6_diagnosed_co2_emissions_anomaly_2081-2100.csv"
HISTORICAL_START_YEAR = 1850
SUMMARY_START_YEAR = 2081
SUMMARY_END_YEAR = 2100
CMIP6_SCENARIO_START_YEAR = 2015
CMIP7_SCENARIO_START_YEAR = 2022
MAX_ENSEMBLE_SIZE = 10

LUC_VARS = [
    "LUC_VEGd_COMBUSTd_EMISSION__PG_CpYR___________ANN_00001",
    "LUC_LITTER_INPUTS__PG_CpYR____________________ANN_00001",
    "LUC_SOIL_CARBON_INPUTS__PG_CpYR_______________ANN_00001",
]

CMIP7_SCENARIO_PATTERNS = {
    "historical": "sc_v51p2-c7hist-r*_185001_202112_rtd074.nc",
    "H": "sc_v51p2-c7scen7h-r*_202201_210012_rtd074.nc",
    "M": "sc_v51p2-c7scen7m-r*_202201_210012_rtd074.nc",
    "VL": "sc_v51p2-c7scen7vl-r*_202201_210012_rtd074.nc",
}

CMIP6_SCENARIO_PATTERNS = {
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

CMIP6_SCENARIO_GREYS = {
    "CMIP6 historical": "0.45",
    "SSP1-1.9": "0.85",
    "SSP1-2.6": "0.78",
    "SSP2-4.5": "0.70",
    "SSP4-6.0": "0.62",
    "SSP3-7.0": "0.54",
    "SSP5-8.5": "0.36",
}

CMIP7_COLORS = {
    "historical": "black",
    "H": "tab:red",
    "M": "tab:orange",
    "VL": "tab:blue",
}

CMIP7_ORDER = ["historical", "H", "M", "VL"]
CMIP6_ORDER = [
    "CMIP6 historical",
    "SSP1-1.9",
    "SSP1-2.6",
    "SSP2-4.5",
    "SSP4-6.0",
    "SSP3-7.0",
    "SSP5-8.5",
]

EARTH_SURFACE_AREA_M2 = 5.10072e14
SECONDS_PER_365DAY_YEAR = 365.0 * 24.0 * 60.0 * 60.0
KG_TO_MT = 1e-9
KG_M2_S_TO_MT_YR = EARTH_SURFACE_AREA_M2 * SECONDS_PER_365DAY_YEAR * KG_TO_MT

PANEL_SPECS = [
    {
        "title": "a) Surface CO₂ concentration",
        "var": "SURFACE_CO2_CONCENTRATION__PPMV_______________ANN_00001",
        "ylabel": "ppmv",
    },
    {
        "title": "b) Diagnosed CO₂ emissions",
        "var": None,
        "ylabel": "Pg C/yr",
        "diagnosed_co2_emissions": True,
    },
    {
        "title": "c) Surface CH₄ concentration",
        "var": "SURFACE_CH4_CONCENTRATION__PPB________________ANN_00001",
        "ylabel": "ppb",
    },
    {
        "title": "d) Surface N₂O concentration",
        "var": "SURFACE_N2O_CONCENTRATION__PPB________________ANN_00001",
        "ylabel": "ppb",
    },
]

AEROSOL_PANEL_SPECS = [
    {
        "title": "a) Total sulphur dioxide emissions",
        "var": "TOTAL_SULFUR_EMISSIONS__KG_SpM2pS_____________ANN_00001",
        "ylabel": "TgS/yr",
        "scale_factor": KG_M2_S_TO_MT_YR,
    },
    {
        "title": "b) Global sulphate burden",
        "var": "SO4_BURDEN_GLOBAL__1E6KGmS____________________ANN_00001",
        "ylabel": "TgS",
        "scale_factor": 1.0 / 1000.0,
    },
    {
        "title": "c) Total black carbon emissions",
        "var": "TOTAL_BLACK_CARBON_EMISSIONS__KGpM2pS_________ANN_00001",
        "ylabel": "Tg/yr",
        "scale_factor": KG_M2_S_TO_MT_YR,
    },
    {
        "title": "d) Total organic carbon emissions",
        "var": "TOTAL_ORGANIC_CARBON_EMISSIONS__KGpM2pS_______ANN_00001",
        "ylabel": "Tg/yr",
        "scale_factor": KG_M2_S_TO_MT_YR,
    },
]


def extract_member_number(filename: str) -> int:
    match = re.search(r"-r(\d+)_", filename)
    return int(match.group(1)) if match else 9999


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

    unique_years, inverse = np.unique(years, return_inverse=True)
    annual_vals = np.full(unique_years.shape, np.nan, dtype=float)

    for i in range(unique_years.size):
        member_vals = values[inverse == i]
        annual_vals[i] = np.nanmean(member_vals)

    return unique_years, annual_vals


def read_member_variable_series(nc_path: str, variable_name: str) -> Tuple[np.ndarray, np.ndarray]:
    with Dataset(nc_path, mode="r") as ds:
        var_names = list(ds.variables.keys())
        year_var = pick_year_variable(var_names)

        if variable_name not in ds.variables:
            raise KeyError(f"Variable '{variable_name}' not found in {nc_path}")

        years = np.array(ds.variables[year_var][:]).squeeze()
        values = np.array(ds.variables[variable_name][:]).squeeze()

    return to_annual_series(years, values)


def read_luc_emissions_series(nc_path: str) -> Tuple[np.ndarray, np.ndarray]:
    luc_series = [read_member_variable_series(nc_path, var) for var in LUC_VARS]
    all_years = sorted({int(year) for years, _ in luc_series for year in years})
    year_to_idx = {year: idx for idx, year in enumerate(all_years)}
    luc_components = [np.full(len(all_years), np.nan, dtype=float) for _ in LUC_VARS]

    for component, (years, values) in zip(luc_components, luc_series):
        for year, value in zip(years, values):
            component[year_to_idx[int(year)]] = value

    return np.array(all_years, dtype=int), np.sum(luc_components, axis=0)


def read_diagnosed_co2_emissions_series(nc_path: str) -> Tuple[np.ndarray, np.ndarray]:
    burden_var = "ATMOSPHERIC_CO2_BURDEN_JAN_1__PG_C__SAMPLED___ANN_00001"
    ocean_flux_var = "OCEANmATMOS_CO2_FLUX__PG_CpYR_________________ANN_00001"
    nbp_var = "NET_BIOME_PRODUCTIVITY__PG_CpYR_______________ANN_00001"

    burden_years, burden_vals = read_member_variable_series(nc_path, burden_var)
    ocean_years, ocean_vals = read_member_variable_series(nc_path, ocean_flux_var)
    nbp_years, nbp_vals = read_member_variable_series(nc_path, nbp_var)
    luc_years, luc_vals = read_luc_emissions_series(nc_path)

    all_years = sorted(
        {int(y) for y in burden_years}
        | {int(y) for y in ocean_years}
        | {int(y) for y in nbp_years}
        | {int(y) for y in luc_years}
    )
    year_to_idx = {year: idx for idx, year in enumerate(all_years)}

    burden = np.full(len(all_years), np.nan, dtype=float)
    ocean_flux = np.full(len(all_years), np.nan, dtype=float)
    nbp = np.full(len(all_years), np.nan, dtype=float)
    luc = np.full(len(all_years), np.nan, dtype=float)

    for year, value in zip(burden_years, burden_vals):
        burden[year_to_idx[int(year)]] = value
    for year, value in zip(ocean_years, ocean_vals):
        ocean_flux[year_to_idx[int(year)]] = value
    for year, value in zip(nbp_years, nbp_vals):
        nbp[year_to_idx[int(year)]] = value
    for year, value in zip(luc_years, luc_vals):
        luc[year_to_idx[int(year)]] = value

    delta_burden = np.full(len(all_years), np.nan, dtype=float)
    if len(all_years) > 1:
        delta_burden[:-1] = np.diff(burden)

    diagnosed = delta_burden + ocean_flux + nbp + luc
    return np.array(all_years, dtype=int), diagnosed


def load_branch_times(csv_path: str) -> Dict[Tuple[str, str], int]:
    branch_times: Dict[Tuple[str, str], int] = {}
    with open(csv_path, newline="", encoding="utf-8") as branch_file:
        for row in csv.DictReader(branch_file):
            branch_value = row["YMDH_branch_time_in_parent"]
            branch_times[(row["family"], row["member"])] = int(branch_value.split(":", 1)[0])
    return branch_times


def member_id_from_path(path: str, family: str) -> str:
    filename = os.path.basename(path)
    if family == "CMIP6":
        match = re.search(r"-(?:r)?(\d+)_", filename)
        if match is None:
            raise ValueError(f"Could not identify CMIP6 member number in {filename}")
        return f"r{int(match.group(1))}i1p2f1"

    match = re.search(r"(r\d+(?:i\d+p\d+f\d+)?)", filename)
    if match is None:
        raise ValueError(f"Could not identify CMIP7 member in {filename}")
    member = match.group(1)
    return member if "i" in member else f"{member}i1p2f1"


def build_pi_control_anomaly_members(
    files_by_scenario: Dict[str, List[str]],
    family: str,
    control_series: Tuple[np.ndarray, np.ndarray],
    branch_times: Dict[Tuple[str, str], int],
    series_reader: Callable[[str], Tuple[np.ndarray, np.ndarray]] = read_diagnosed_co2_emissions_series,
) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    historical_scenario = "CMIP6 historical" if family == "CMIP6" else "historical"
    historical_files = {
        member_id_from_path(path, family): path
        for path in files_by_scenario.get(historical_scenario, [])
    }
    historical_series = {
        member: series_reader(path)
        for member, path in historical_files.items()
    }
    control_years, control_values = control_series
    control_by_year = {
        int(year): float(value)
        for year, value in zip(control_years, control_values)
        if np.isfinite(value)
    }

    out: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    for scenario, scenario_files in files_by_scenario.items():
        member_series: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for scenario_path in scenario_files:
            member = member_id_from_path(scenario_path, family)
            historical_member_series = historical_series.get(member)
            if historical_member_series is None:
                if scenario == historical_scenario:
                    raise ValueError(f"Historical file is missing for {family} member {member}")
                if member not in historical_files:
                    raise ValueError(
                        f"No {family} historical file found for member {member} ({scenario_path})"
                    )

            if (family, member) not in branch_times:
                raise KeyError(f"No piControl branch time for {family} member {member}")

            parts = [historical_member_series]
            if scenario != historical_scenario:
                parts.append(series_reader(scenario_path))
            years = np.concatenate([part[0] for part in parts])
            values = np.concatenate([part[1] for part in parts])
            order = np.argsort(years)
            years = years[order]
            values = values[order]

            control_years_for_member = branch_times[(family, member)] + (years - HISTORICAL_START_YEAR)
            reference = np.array(
                [control_by_year.get(int(year), np.nan) for year in control_years_for_member],
                dtype=float,
            )
            valid = np.isfinite(values) & np.isfinite(reference)
            anomalies = np.full(values.shape, np.nan, dtype=float)
            anomalies[valid] = values[valid] - reference[valid]
            if not np.any(valid):
                raise ValueError(
                    f"No overlapping piControl years for {family} member {member} ({scenario})"
                )
            member_series[member] = (years, anomalies)
        out[scenario] = member_series
    return out


def scenario_mean(members: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    if not members:
        return np.array([]), np.array([])

    all_years = sorted({int(y) for years, _ in members.values() for y in years})
    year_to_idx = {y: i for i, y in enumerate(all_years)}
    stack = np.full((len(members), len(all_years)), np.nan, dtype=float)

    for row, (_, (years, vals)) in enumerate(members.items()):
        for y, v in zip(years, vals):
            stack[row, year_to_idx[int(y)]] = v

    valid_counts = np.sum(np.isfinite(stack), axis=0)
    summed_vals = np.nansum(stack, axis=0)
    mean_vals = np.full(len(all_years), np.nan, dtype=float)
    valid_cols = valid_counts > 0
    mean_vals[valid_cols] = summed_vals[valid_cols] / valid_counts[valid_cols]
    return np.array(all_years, dtype=int), mean_vals


def write_co2_anomaly_summary(
    output_path: str,
    cmip6_members: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    cmip7_members: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

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
                "mean_diagnosed_co2_emissions_including_luc_anomaly_pg_c_per_year",
                "members_included",
            )
        )
        for family, scenarios in (("CMIP6", cmip6_members), ("CMIP7", cmip7_members)):
            for scenario, members in scenarios.items():
                member_periods = []
                for years, values in members.values():
                    period_mask = (years >= SUMMARY_START_YEAR) & (years <= SUMMARY_END_YEAR)
                    period_years = years[period_mask]
                    period_values = values[period_mask]
                    finite = np.isfinite(period_values)
                    member_periods.append(
                        {
                            int(year): float(value)
                            for year, value in zip(period_years[finite], period_values[finite])
                        }
                    )

                common_years = sorted(
                    set.intersection(*(set(period.keys()) for period in member_periods))
                ) if member_periods else []
                member_means = [
                    float(np.mean([period[year] for year in common_years]))
                    for period in member_periods
                ] if common_years else []

                mean_anomaly = float(np.mean(member_means)) if member_means else ""
                writer.writerow(
                    (
                        family,
                        scenario,
                        SUMMARY_START_YEAR,
                        SUMMARY_END_YEAR,
                        common_years[0] if common_years else "",
                        common_years[-1] if common_years else "",
                        len(common_years),
                        mean_anomaly,
                        len(member_means),
                    )
                )


def collect_cmip7_files(input_dir: str) -> Dict[str, List[str]]:
    files = {}
    for scenario, pattern in CMIP7_SCENARIO_PATTERNS.items():
        paths = sorted(
            glob.glob(os.path.join(input_dir, pattern)),
            key=lambda p: extract_member_number(os.path.basename(p)),
        )
        files[scenario] = paths[:MAX_ENSEMBLE_SIZE]
    return files


def cmip6_member_number(path: str) -> int:
    match = re.search(r"-r?(\d+)_\d{6}_", os.path.basename(path))
    return int(match.group(1)) if match else 9999


def collect_cmip6_files(input_dir: str) -> Dict[str, List[str]]:
    files_by_scenario: Dict[str, List[str]] = {}
    for scenario, patterns in CMIP6_SCENARIO_PATTERNS.items():
        matched = []
        for pattern in patterns:
            matched.extend(glob.glob(os.path.join(input_dir, pattern)))
        files_by_scenario[scenario] = sorted(
            set(matched), key=lambda p: (cmip6_member_number(p), p)
        )[:MAX_ENSEMBLE_SIZE]
    return files_by_scenario


def plot_aerosol_figure(
    output_path: str,
    cmip7_files_by_scenario: Dict[str, List[str]],
    cmip6_files_by_scenario: Dict[str, List[str]],
) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.2), sharex=True)
    axes_flat = axes.ravel()

    for i, spec in enumerate(AEROSOL_PANEL_SPECS):
        ax = axes_flat[i]
        variable_name = spec["var"]
        scale_factor = float(spec.get("scale_factor", 1.0))

        cmip7_members: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {
            scenario: {} for scenario in CMIP7_ORDER
        }
        for scenario in CMIP7_ORDER:
            for path in cmip7_files_by_scenario.get(scenario, []):
                filename = os.path.basename(path)
                member_id_match = re.search(r"(r\d+)", filename)
                member_id = member_id_match.group(1) if member_id_match else filename
                years, values = read_member_variable_series(path, variable_name)
                cmip7_members[scenario][member_id] = (years, values)

        for scenario in CMIP7_ORDER:
            members = cmip7_members[scenario]
            if not members:
                continue

            years, values = scenario_mean(members)
            label = "CMIP7 historical" if scenario == "historical" else scenario
            if i != 0:
                label = "_nolegend_"

            ax.plot(
                years,
                values * scale_factor,
                color=CMIP7_COLORS[scenario],
                linewidth=3.0,
                zorder=3,
                label=label,
            )

        cmip6_members: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {
            scenario: {} for scenario in CMIP6_ORDER
        }
        for scenario in CMIP6_ORDER:
            for path in cmip6_files_by_scenario.get(scenario, []):
                member_key = os.path.basename(path).replace("_rtd074.nc", "")
                years, values = read_member_variable_series(path, variable_name)
                cmip6_members[scenario][member_key] = (years, values)

        for scenario in CMIP6_ORDER:
            members = cmip6_members[scenario]
            if not members:
                continue

            years, values = scenario_mean(members)
            label = scenario
            if i != 0:
                label = "_nolegend_"

            ax.plot(
                years,
                values * scale_factor,
                color=CMIP6_SCENARIO_GREYS[scenario],
                linewidth=2.6,
                zorder=1.5,
                label=label,
            )

        ax.set_title(spec["title"])
        ax.set_xlabel("Year")
        ax.set_ylabel(spec["ylabel"])
        ax.set_xlim(2000, 2100)
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)

    panel_a = axes_flat[0]
    handles, labels = panel_a.get_legend_handles_labels()
    cmip7_entries = [
        (handle, label)
        for handle, label in zip(handles, labels)
        if label == "CMIP7 historical" or label in CMIP7_ORDER[1:]
    ]
    cmip6_entries = [
        (handle, label)
        for handle, label in zip(handles, labels)
        if label in CMIP6_ORDER
    ]

    cmip7_legend = panel_a.legend(
        [handle for handle, _ in cmip7_entries],
        [label for _, label in cmip7_entries],
        loc="upper right",
        bbox_to_anchor=(0.68, 1.0),
        frameon=False,
    )
    panel_a.add_artist(cmip7_legend)
    panel_a.legend(
        [handle for handle, _ in cmip6_entries],
        [label for _, label in cmip6_entries],
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        frameon=False,
    )

    fig.tight_layout(pad=1.0)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"Saved aerosol plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot greenhouse-gas and aerosol figures from RTD NetCDF ensembles"
    )
    parser.add_argument(
        "--input-dir",
        default="/home/rng001/site7/rtd_nc",
        help="Directory containing RTD NetCDF files",
    )
    parser.add_argument(
        "--output",
        default="/home/rng001/plots/scenarios/cmip7_cmip6_ghg_2x2.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--luc-output",
        default="/home/rng001/plots/scenarios/cmip7_cmip6_luc_emissions_1panel.png",
        help="Output PNG path for the standalone LUC emissions plot",
    )
    parser.add_argument(
        "--aerosol-output",
        default="/home/rng001/plots/scenarios/cmip7_cmip6_aerosol_2x2.png",
        help="Output PNG path for the aerosol panel figure",
    )
    parser.add_argument("--branch-times-csv", default=DEFAULT_BRANCH_TIMES_CSV, help="CSV of historical member piControl branch times")
    parser.add_argument("--cmip6-picontrol", default=DEFAULT_CMIP6_PICONTROL, help="CMIP6 piControl RTD NetCDF file")
    parser.add_argument("--cmip7-picontrol", default=DEFAULT_CMIP7_PICONTROL, help="CMIP7 piControl RTD NetCDF file")
    parser.add_argument("--co2-anomaly-summary", default=DEFAULT_CO2_ANOMALY_SUMMARY, help="CSV output for unsmoothed 2081-2100 diagnosed CO2 emission anomalies")
    parser.add_argument("--summary-only", action="store_true", help="Write the CO2 anomaly summary without plotting")
    args = parser.parse_args()

    cmip7_files_by_scenario = collect_cmip7_files(args.input_dir)
    cmip6_files_by_scenario = collect_cmip6_files(args.input_dir)

    for family, files_by_scenario in (
        ("CMIP7", cmip7_files_by_scenario),
        ("CMIP6", cmip6_files_by_scenario),
    ):
        for scenario, paths in files_by_scenario.items():
            print(f"{family} {scenario}: ensemble size = {len(paths)}")

    n_found = sum(len(v) for v in cmip7_files_by_scenario.values())
    if n_found == 0:
        raise FileNotFoundError(f"No matching CMIP7 RTD files found in {args.input_dir}")

    branch_times = load_branch_times(args.branch_times_csv)
    cmip6_control_series = read_diagnosed_co2_emissions_series(args.cmip6_picontrol)
    cmip7_control_series = read_diagnosed_co2_emissions_series(args.cmip7_picontrol)
    cmip7_emission_members = build_pi_control_anomaly_members(
        cmip7_files_by_scenario,
        "CMIP7",
        cmip7_control_series,
        branch_times,
    )
    cmip6_emission_members = build_pi_control_anomaly_members(
        cmip6_files_by_scenario,
        "CMIP6",
        cmip6_control_series,
        branch_times,
    )
    write_co2_anomaly_summary(
        args.co2_anomaly_summary,
        cmip6_emission_members,
        cmip7_emission_members,
    )
    print(f"Saved unsmoothed 2081-2100 CO2 emission anomaly summary to {args.co2_anomaly_summary}")
    if args.summary_only:
        return

    cmip7_luc_control_series = read_luc_emissions_series(args.cmip7_picontrol)
    cmip6_luc_control_series = read_luc_emissions_series(args.cmip6_picontrol)
    cmip7_luc_members = build_pi_control_anomaly_members(
        cmip7_files_by_scenario,
        "CMIP7",
        cmip7_luc_control_series,
        branch_times,
        series_reader=read_luc_emissions_series,
    )
    cmip6_luc_members = build_pi_control_anomaly_members(
        cmip6_files_by_scenario,
        "CMIP6",
        cmip6_luc_control_series,
        branch_times,
        series_reader=read_luc_emissions_series,
    )

    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.2), sharex=True)
    axes_flat = axes.ravel()

    for i, spec in enumerate(PANEL_SPECS):
        ax = axes_flat[i]
        variable_name = spec["var"]
        use_diagnosed_co2_emissions = spec.get("diagnosed_co2_emissions", False)

        if use_diagnosed_co2_emissions:
            cmip7_members = cmip7_emission_members
            cmip6_members = cmip6_emission_members
        else:
            cmip7_members: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {
                s: {} for s in CMIP7_ORDER
            }
            for scenario in CMIP7_ORDER:
                for path in cmip7_files_by_scenario.get(scenario, []):
                    fname = os.path.basename(path)
                    member_id_match = re.search(r"(r\d+)", fname)
                    member_id = member_id_match.group(1) if member_id_match else fname
                    years, vals = read_member_variable_series(path, variable_name)
                    cmip7_members[scenario][member_id] = (years, vals)

            cmip6_members = {s: {} for s in CMIP6_ORDER}
            for scenario in CMIP6_ORDER:
                for path in cmip6_files_by_scenario.get(scenario, []):
                    member_key = os.path.basename(path).replace("_rtd074.nc", "")
                    years, vals = read_member_variable_series(path, variable_name)
                    cmip6_members[scenario][member_key] = (years, vals)

        for scenario in CMIP7_ORDER:
            members = cmip7_members[scenario]
            if not members:
                continue

            years, vals = scenario_mean(members)
            if use_diagnosed_co2_emissions and scenario != "historical":
                scenario_mask = years >= CMIP7_SCENARIO_START_YEAR
                years, vals = years[scenario_mask], vals[scenario_mask]
                if years.size == 0:
                    continue

            label = (
                f"CMIP7 historical"
                if scenario == "historical"
                else f"{scenario}"
            )
            if i != 0:
                label = "_nolegend_"

            ax.plot(
                years,
                vals,
                color=CMIP7_COLORS[scenario],
                linewidth=3.0,
                zorder=3,
                label=label,
            )

        for scenario in CMIP6_ORDER:
            members = cmip6_members[scenario]
            if not members:
                continue

            years, vals = scenario_mean(members)

            if use_diagnosed_co2_emissions and scenario != "CMIP6 historical":
                scenario_mask = years >= CMIP6_SCENARIO_START_YEAR
                years, vals = years[scenario_mask], vals[scenario_mask]
                if years.size == 0:
                    continue

            label = f"{scenario}"
            if i != 0:
                label = "_nolegend_"

            ax.plot(
                years,
                vals,
                color=CMIP6_SCENARIO_GREYS[scenario],
                linewidth=2.6,
                zorder=1.5,
                label=label,
            )

        ax.set_title(spec["title"])
        ax.set_xlabel("Year")
        ax.set_ylabel(spec["ylabel"])
        ax.set_xlim(2000, 2100)
        ax.grid(True, alpha=0.3)

    axes_flat[0].legend(loc="upper left", ncol=1, frameon=False)

    # Requested lower bounds for selected concentration panels.
    axes_flat[0].set_ylim(bottom=350.0)
    axes_flat[2].set_ylim(bottom=1000.0)
    axes_flat[3].set_ylim(bottom=310.0)

    fig.tight_layout(pad=1.0)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.savefig(args.output, dpi=180)
    plt.close(fig)

    print(f"Saved plot to {args.output}")

    luc_fig, luc_ax = plt.subplots(figsize=(10.8, 4.8))
    for scenario in CMIP7_ORDER:
        members = cmip7_luc_members[scenario]
        if not members:
            continue

        years, vals = scenario_mean(members)
        if scenario != "historical":
            scenario_mask = years >= CMIP7_SCENARIO_START_YEAR
            years, vals = years[scenario_mask], vals[scenario_mask]
            if years.size == 0:
                continue

        label = "CMIP7 historical" if scenario == "historical" else scenario
        luc_ax.plot(
            years,
            vals,
            color=CMIP7_COLORS[scenario],
            linewidth=3.0,
            zorder=3,
            label=label,
        )

    for scenario in CMIP6_ORDER:
        members = cmip6_luc_members[scenario]
        if not members:
            continue

        years, vals = scenario_mean(members)
        if scenario != "CMIP6 historical":
            scenario_mask = years >= CMIP6_SCENARIO_START_YEAR
            years, vals = years[scenario_mask], vals[scenario_mask]
            if years.size == 0:
                continue

        luc_ax.plot(
            years,
            vals,
            color=CMIP6_SCENARIO_GREYS[scenario],
            linewidth=2.6,
            zorder=1.5,
            label=scenario,
        )

    luc_ax.axhline(0, color="0.35", linewidth=0.8, linestyle="--", zorder=0)
    luc_ax.set_title("Land-use change CO₂ emissions anomaly relative to piControl")
    luc_ax.set_xlabel("Year")
    luc_ax.set_ylabel("Pg C/yr")
    luc_ax.set_xlim(2000, 2100)
    luc_ax.grid(True, alpha=0.3)
    luc_ax.legend(loc="best", ncol=2, frameon=False)
    luc_fig.tight_layout(pad=1.0)

    luc_out_dir = os.path.dirname(args.luc_output)
    if luc_out_dir:
        os.makedirs(luc_out_dir, exist_ok=True)

    plt.savefig(args.luc_output, dpi=180)
    plt.close(luc_fig)

    print(f"Saved LUC emissions plot to {args.luc_output}")

    plot_aerosol_figure(
        args.aerosol_output,
        cmip7_files_by_scenario,
        cmip6_files_by_scenario,
    )


if __name__ == "__main__":
    main()
