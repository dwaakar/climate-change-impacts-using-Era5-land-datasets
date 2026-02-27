# -*- coding: utf-8 -*-
"""
GCM Selection Pipeline - Computation (Steps 1 to 4)
=====================================================
This script combines all computation steps (1a, 1b, 1c, 1d, Step 2, Step 3, Step 4)
into a single file. Each step saves its own output files exactly as before.

Steps Overview:
    Step 1a : Extract monthly average precipitation (pr) from CMIP6 GCMs
    Step 1b : Extract monthly average temperature (tas) from CMIP6 GCMs
    Step 1c : Extract monthly average precipitation (tp) from ERA5
    Step 1d : Extract monthly average temperature (temp) from ERA5
    Step 2  : GCM selection based on seasonal and annual bias (Step 1 & Step 2 selection)
    Step 3a : Future climate change analysis under SSP2-4.5 scenario
    Step 3b : Future climate change analysis under SSP5-8.5 scenario
    Step 4a : Percentile scatter plot and final selection under SSP2-4.5
    Step 4b : Percentile scatter plot and final selection under SSP5-8.5

Author   : Arshad / Diwakar Adhikari
"""

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import xarray as xr
import pandas as pd
from netCDF4 import Dataset
import os
import glob
import geopandas as gpd
import regionmask
from shapely.geometry import box
from scipy import stats
import calendar
import seaborn as sns
import matplotlib.pyplot as plt

# =============================================================================
# USER-DEFINED VARIABLES — Edit these before running
# =============================================================================
input_folder  = r"C:\Users\Diwakar Adhikari\Downloads\climate data\Model Selection\01_selection\01_selection\01_inputs"
output_folder = r"C:\Users\Diwakar Adhikari\Downloads\climate data\Model Selection\01_selection\01_selection\MRB"
shapefile_name = "MRB.shp"

# Baseline period
start_year = 1985
end_year   = 2014

# Future period (for Steps 3 & 4)
baseline_start = 1985
baseline_end   = 2014
future_start   = 2071
future_end     = 2100

# ERA5-Land CSV file paths (already area-weighted, local time, baseline ready)
era5_pr_csv  = r"C:\Users\Diwakar Adhikari\Documents\ERA5_Land\precipitation\era5land_localtime\area_weighted_output\era5land_monthly_basin_NST_BULK.csv"
era5_tas_csv = r"C:\Users\Diwakar Adhikari\Documents\ERA5_Land\temperature\for_GCM\monthly_avg_temperature_by_year_1984_2024_local.csv"

# Number of models to select at each step
select_step1 = 24
select_step2 = 16

# Location label (used in output filenames and plot titles)
location = 'MRB'

# Precision for percent_rank function
precision = 4

# =============================================================================
# SHARED SETUP
# =============================================================================
shapefile_path = os.path.join(input_folder, 'shapefiles', shapefile_name)
gdf = gpd.read_file(shapefile_path)
# Reproject to a projected CRS (UTM) for accurate area calculations
# EPSG:32644 = WGS 84 / UTM zone 44N (appropriate for MRB region)
# The intersection geometry boxes are also constructed in geographic coords,
# so we keep gdf_proj for area calculation only, and use gdf for intersection.
gdf_proj = gdf.to_crs(epsg=32645)
bbox = gdf.total_bounds
centroid = gdf.geometry.centroid.iloc[0]

# Month name mapping (numeric -> abbreviation)
month_mapping = {
    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr',
    5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
    9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def area_weighted_monthly_avg(ds_var, gdf, gdf_proj, lat_name='lat', lon_name='lon'):
    """
    Compute area-weighted monthly averages of a DataArray over a shapefile region.
    ds_var   : xarray DataArray with dimensions (month, lat, lon)
    gdf      : GeoDataFrame in geographic CRS (used for intersection geometry)
    gdf_proj : GeoDataFrame reprojected to a projected CRS (used for accurate area)
    Returns a list of 12 weighted averages (one per month).
    """
    lat_spacing = ds_var[lat_name][1] - ds_var[lat_name][0]
    lon_spacing = ds_var[lon_name][1] - ds_var[lon_name][0]
    monthly_weighted_averages = []

    for month in range(1, 13):
        data_array = ds_var.sel(month=month)
        weighted_sum = 0
        total_intersected_area = 0

        for i in range(data_array.sizes[lat_name] - 1):
            for j in range(data_array.sizes[lon_name] - 1):
                half_width  = lat_spacing / 2
                half_height = lon_spacing / 2
                cell = box(
                    data_array[lon_name][j] - half_height,
                    data_array[lat_name][i] - half_width,
                    data_array[lon_name][j] + half_height,
                    data_array[lat_name][i] + half_width
                )
                # Intersection in geographic CRS
                intersection = gdf.geometry.intersection(cell)
                if intersection.any():
                    # Reproject intersection to projected CRS for accurate area in m²
                    intersection_proj = gpd.GeoSeries(intersection, crs=gdf.crs).to_crs(gdf_proj.crs)
                    intersected_area  = intersection_proj.area
                    weighted_sum += (data_array[i, j].values * intersected_area)
                    total_intersected_area += intersected_area

        weighted_average = weighted_sum / total_intersected_area
        monthly_weighted_averages.append(weighted_average)

    return monthly_weighted_averages


def area_weighted_annual_avg(data_array, gdf, gdf_proj, lat_name='lat', lon_name='lon'):
    """
    Compute area-weighted annual average of a 2D DataArray (lat x lon) over a shapefile region.
    gdf_proj is used for accurate area calculation in projected CRS.
    Returns a single weighted average value.
    """
    lat_spacing = data_array[lat_name][1] - data_array[lat_name][0]
    lon_spacing = data_array[lon_name][1] - data_array[lon_name][0]
    weighted_sum = 0
    total_intersected_area = 0

    for i in range(data_array.sizes[lat_name] - 1):
        for j in range(data_array.sizes[lon_name] - 1):
            half_width  = lat_spacing / 2
            half_height = lon_spacing / 2
            cell = box(
                data_array[lon_name][j] - half_height,
                data_array[lat_name][i] - half_width,
                data_array[lon_name][j] + half_height,
                data_array[lat_name][i] + half_width
            )
            intersection = gdf.geometry.intersection(cell)
            if intersection.any():
                intersection_proj = gpd.GeoSeries(intersection, crs=gdf.crs).to_crs(gdf_proj.crs)
                intersected_area  = intersection_proj.area
                weighted_sum += (data_array[i, j].values * intersected_area)
                total_intersected_area += intersected_area

    weighted_average = round(weighted_sum / total_intersected_area, 1)
    return weighted_average


def percent_rank(pd_series, precision):
    """Compute percent rank for each value in a pandas Series."""
    return [
        np.round((pd_series < value).astype(int).sum() / (len(pd_series) - 1), precision)
        for value in pd_series
    ]


# =============================================================================
# STEP 1a: Monthly Average Precipitation from CMIP6 GCMs
# =============================================================================
print("\n" + "="*60)
print("STEP 1a: Extracting monthly avg precipitation from CMIP6 GCMs")
print("="*60)

df5 = pd.DataFrame()
folder_path = os.path.join(input_folder, 'GCMs', 'historical', 'pr')

for filename in os.listdir(folder_path):
    nc_file_path = os.path.join(folder_path, filename)
    ds = xr.open_dataset(nc_file_path)
    str1 = filename

    # Extract GCM name from filename (between "day_" and "_historical")
    start_index    = str1.index("day_") + len("day_")
    end_index      = str1.index("_historical")
    desired_portion = str1[start_index:end_index]

    # Select baseline period and resample to monthly sums
    ds1       = ds.sel(time=slice(str(start_year), str(end_year)))
    ds_monthly = ds1['pr'].resample(time='ME').sum()

    # Average across all years for each calendar month
    monthly_avg = ds_monthly.groupby('time.month').mean(dim='time')

    # Compute area-weighted monthly averages over the shapefile region
    monthly_weighted_averages = area_weighted_monthly_avg(monthly_avg, gdf, gdf_proj)
    monthly_weighted_averages_list = [area[0] for area in monthly_weighted_averages]

    # Build DataFrame for this model
    s3  = pd.Series(monthly_avg.month.values, name='Month')
    df4 = pd.DataFrame(s3)
    df4['PPT']   = monthly_weighted_averages_list
    df4['PPT']   = df4['PPT'] * 86400       # Convert kg/m2/s to mm/day
    df4['PPT']   = round(df4['PPT'], 1)
    df4['Month'] = df4['Month'].map(month_mapping)
    df4['Model'] = desired_portion
    df5 = pd.concat([df5, df4])
    print(f'  Done: {desired_portion}')

# Pivot and save
df6 = df5.pivot(index=['Model'], columns='Month', values='PPT')
df6.reset_index(inplace=True)
month_abbreviations = list(calendar.month_abbr)[1:]
month_abbreviations.insert(0, 'Model')
df7 = df6[month_abbreviations]
df7.to_excel(os.path.join(output_folder, 'GCMs_monthly_avg_pr_MRB.xlsx'), index=False)
print("  Saved: GCMs_monthly_avg_pr_MRB.xlsx")


# =============================================================================
# STEP 1b: Monthly Average Temperature from CMIP6 GCMs
# =============================================================================
print("\n" + "="*60)
print("STEP 1b: Extracting monthly avg temperature from CMIP6 GCMs")
print("="*60)

df5 = pd.DataFrame()
folder_path = os.path.join(input_folder, 'GCMs', 'historical', 'tas')

for filename in os.listdir(folder_path):
    nc_file_path = os.path.join(folder_path, filename)
    ds   = xr.open_dataset(nc_file_path)
    str1 = filename

    # Extract GCM name from filename
    start_index     = str1.index("day_") + len("day_")
    end_index       = str1.index("_historical")
    desired_portion = str1[start_index:end_index]

    # Select baseline period and resample to monthly means
    ds1        = ds.sel(time=slice(str(start_year), str(end_year)))
    ds_monthly = ds1['tas'].resample(time='ME').mean()

    # Average across all years for each calendar month
    monthly_avg = ds_monthly.groupby('time.month').mean(dim='time')

    # Compute area-weighted monthly averages over the shapefile region
    monthly_weighted_averages = area_weighted_monthly_avg(monthly_avg, gdf, gdf_proj)
    monthly_weighted_averages_list = [area[0] for area in monthly_weighted_averages]

    # Build DataFrame for this model
    s3   = pd.Series(monthly_avg.month.values, name='Month')
    df4  = pd.DataFrame(s3)
    df4['T']     = monthly_weighted_averages_list
    df4['T']     = df4['T'] - 273.15       # Convert Kelvin to Celsius
    df4['T']     = round(df4['T'], 1)
    df4['Month'] = df4['Month'].map(month_mapping)
    df4['Model'] = desired_portion
    df5 = pd.concat([df5, df4])
    print(f'  Done: {desired_portion}')

# Pivot and save
df6 = df5.pivot(index=['Model'], columns='Month', values='T')
df6.reset_index(inplace=True)
month_abbreviations = list(calendar.month_abbr)[1:]
month_abbreviations.insert(0, 'Model')
df7 = df6[month_abbreviations]
df7.to_excel(os.path.join(output_folder, 'GCMs_monthly_avg_tas_MRB.xlsx'), index=False)
print("  Saved: GCMs_monthly_avg_tas_MRB.xlsx")


# =============================================================================
# STEP 1c: Monthly Climatology Precipitation from ERA5-Land CSV
# Already area-weighted, already in local time (NST), already in mm
# Filter to baseline period, then average each month column across years
# =============================================================================
print("\n" + "="*60)
print("STEP 1c: Reading monthly avg precipitation from ERA5-Land CSV")
print("="*60)

df_era5_pr = pd.read_csv(era5_pr_csv)

# Filter to baseline period (1985-2014)
df_era5_pr = df_era5_pr[(df_era5_pr['year'] >= start_year) & (df_era5_pr['year'] <= end_year)]

# Compute monthly climatology: average (mean) each month column across all baseline years
# For precipitation, monthly climatology = mean of monthly totals across years
month_cols = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
climatology_pr = df_era5_pr[month_cols].mean()
climatology_pr = climatology_pr.round(1)

# Build output DataFrame in the same format as the rest of the pipeline
df7 = pd.DataFrame([['ERA5'] + climatology_pr.tolist()], columns=['Model'] + month_cols)
df7.to_excel(os.path.join(output_folder, 'ERA5_monthly_avg_pr_MRB.xlsx'), index=False)
print(f"  Baseline years used: {start_year}–{end_year} ({len(df_era5_pr)} years)")
print("  Saved: ERA5_monthly_avg_pr_MRB.xlsx")


# =============================================================================
# STEP 1d: Monthly Climatology Temperature from ERA5-Land CSV
# Already area-weighted, already in local time (NST), already in Celsius
# Filter to baseline period, then average each month column across years
# =============================================================================
print("\n" + "="*60)
print("STEP 1d: Reading monthly avg temperature from ERA5-Land CSV")
print("="*60)

df_era5_tas = pd.read_csv(era5_tas_csv)

# Filter to baseline period (1985-2014)
df_era5_tas = df_era5_tas[(df_era5_tas['year'] >= start_year) & (df_era5_tas['year'] <= end_year)]

# Compute monthly climatology: average (mean) each month column across all baseline years
# For temperature, monthly climatology = mean of monthly averages across years
climatology_tas = df_era5_tas[month_cols].mean()
climatology_tas = climatology_tas.round(1)

# Build output DataFrame in the same format as the rest of the pipeline
df7 = pd.DataFrame([['ERA5'] + climatology_tas.tolist()], columns=['Model'] + month_cols)
df7.to_excel(os.path.join(output_folder, 'ERA5_monthly_avg_tas_MRB.xlsx'), index=False)
print(f"  Baseline years used: {start_year}–{end_year} ({len(df_era5_tas)} years)")
print("  Saved: ERA5_monthly_avg_tas_MRB.xlsx")


# =============================================================================
# STEP 2: GCM Selection — Seasonal Bias (Step 1) and Annual Bias (Step 2)
# =============================================================================
print("\n" + "="*60)
print("STEP 2: GCM selection based on seasonal and annual bias")
print("="*60)

pr_ERA5  = pd.read_excel(os.path.join(output_folder, 'ERA5_monthly_avg_pr_MRB.xlsx'))
tas_ERA5 = pd.read_excel(os.path.join(output_folder, 'ERA5_monthly_avg_tas_MRB.xlsx'))
pr_GCMs  = pd.read_excel(os.path.join(output_folder, 'GCMs_monthly_avg_pr_MRB.xlsx'))
tas_GCMs = pd.read_excel(os.path.join(output_folder, 'GCMs_monthly_avg_tas_MRB.xlsx'))

df1 = pd.concat([pr_ERA5, pr_GCMs])
df2 = pd.concat([tas_ERA5, tas_GCMs])

season = {
    'DJF':    ['Dec', 'Jan', 'Feb'],
    'MAM':    ['Mar', 'Apr', 'May'],
    'JJAS':   ['Jun', 'Jul', 'Aug', 'Sep'],
    'ON':     ['Oct', 'Nov'],
    'Annual': df1.columns[1:]
}

for key in season.keys():
    df1[key] = df1[season[key]].sum(axis=1)
    df2[key] = round(df2[season[key]].mean(axis=1), 1)

df_ppt  = df1.copy()
df_temp = df2.copy()

season_contribution = ['DJF_C', 'MAM_C', 'JJAS_C', 'ON_C']
for item in season_contribution:
    item1       = item[:-2]
    df1[item]   = round(df1[item1] * 100 / df1['Annual'], 1)
    df2[item]   = round(df2['Annual'] - df2[item1], 1)

df1.set_index('Model', inplace=True)
df2.set_index('Model', inplace=True)

season_change = ['DJF_CH', 'MAM_CH', 'JJAS_CH', 'ON_CH']
models = [item for item in df1.index if item != 'ERA5']

for model in models:
    i = 0
    for item in season_change:
        df1.loc[model, item] = abs(df1.loc[model, season_contribution[i]] - df1.loc['ERA5', season_contribution[i]])
        df2.loc[model, item] = abs(df2.loc[model, season_contribution[i]] - df2.loc['ERA5', season_contribution[i]])
        i += 1

weightage  = df1.loc['ERA5', season_contribution].tolist()
weightage1 = [25, 25, 25, 25]   # Equal weightage

def total_change(a):
    return round(sum([x * y / 100 for x, y in zip(weightage, a)]), 1)

def total_change1(a):
    return round(sum([x * y / 100 for x, y in zip(weightage1, a)]), 1)

df1['Total_CH'] = df1[season_change].apply(total_change1, axis=1)
df2['Total_CH'] = round(df2[season_change].mean(axis=1), 1)

df1['Norm'] = round(df1['Total_CH'] / df1['Total_CH'].max(), 2)
df2['Norm'] = round(df2['Total_CH'] / df2['Total_CH'].max(), 2)

df3 = pd.DataFrame(index=models)
for model in models:
    df3.loc[model, 'PPT_Norm']  = df1.loc[model, 'Norm']
    df3.loc[model, 'Temp_Norm'] = df2.loc[model, 'Norm']

df3['Norm_Sum'] = df3['PPT_Norm'] + df3['Temp_Norm']
df3.reset_index(inplace=True)
df4 = df3.sort_values(by='Norm_Sum')
df4.reset_index(drop=True, inplace=True)

# Step 1 selection: top N models by seasonal bias
df41 = df4.iloc[0:select_step1]
df41 = df41.rename(columns={'index': 'GCMs'})
df41.to_excel(os.path.join(output_folder, 'step1_seasonal_bias_selected_GCMs.xlsx'), index=False)
print(f"  Saved: step1_seasonal_bias_selected_GCMs.xlsx  ({select_step1} models selected)")

l1 = ['ERA5']
model_selected       = df4.iloc[0:select_step1]['index'].tolist()
model_selected_step1 = l1 + model_selected

p1 = df1.loc[model_selected_step1]
for model in model_selected:
    p1.loc[model, 'Annual_bias'] = round(
        (abs(p1.loc[model, 'Annual'] - p1.loc['ERA5', 'Annual']) * 100) / p1.loc['ERA5', 'Annual'], 1
    )
p1['Annual_Norm'] = round(p1['Annual_bias'] / p1['Annual_bias'].max(), 2)

t1 = df2.loc[model_selected_step1]
for model in model_selected:
    t1.loc[model, 'Annual_bias'] = round(abs(t1.loc[model, 'Annual'] - t1.loc['ERA5', 'Annual']), 1)
t1['Annual_Norm'] = round(t1['Annual_bias'] / t1['Annual_bias'].max(), 2)

# Step 2 selection: top N models by annual bias
df5 = pd.DataFrame(index=model_selected)
for model in model_selected:
    df5.loc[model, 'PPT_Norm_Annual']  = p1.loc[model, 'Annual_Norm']
    df5.loc[model, 'Temp_Norm_Annual'] = t1.loc[model, 'Annual_Norm']

df5['Norm_Sum'] = df5['PPT_Norm_Annual'] + df5['Temp_Norm_Annual']
df5.reset_index(inplace=True)
df6 = df5.sort_values(by='Norm_Sum')
df6.reset_index(drop=True, inplace=True)
df7 = df6.iloc[0:select_step2]
df7 = df7.rename(columns={'index': 'GCMs'})
df7.to_excel(os.path.join(output_folder, 'step2_annual_bias_selected_GCMs.xlsx'), index=False)
print(f"  Saved: step2_annual_bias_selected_GCMs.xlsx  ({select_step2} models selected)")

# -------------------------------------------------------------------------
# Save Step 2 ranking table with full bias details for all Step 1 models
# Columns: Model, Annual_pr, Annual_tas, Bias_P_percent, Bias_T_degC,
#          abs_Bias_P_percent, abs_Bias_T_degC, rank_P, rank_T, Average_Rank
# -------------------------------------------------------------------------
ranking_table = pd.DataFrame()
ranking_table['Model']              = model_selected
ranking_table['Annual_pr']          = [round(p1.loc[m, 'Annual'], 1) for m in model_selected]
ranking_table['Annual_tas']         = [round(t1.loc[m, 'Annual'], 1) for m in model_selected]
ranking_table['Bias_P_percent']     = [round((p1.loc[m, 'Annual'] - p1.loc['ERA5', 'Annual']) * 100 / p1.loc['ERA5', 'Annual'], 2) for m in model_selected]
ranking_table['Bias_T_degC']        = [round(t1.loc[m, 'Annual'] - t1.loc['ERA5', 'Annual'], 2) for m in model_selected]
ranking_table['abs_Bias_P_percent'] = ranking_table['Bias_P_percent'].abs()
ranking_table['abs_Bias_T_degC']    = ranking_table['Bias_T_degC'].abs()
ranking_table['rank_P']             = ranking_table['abs_Bias_P_percent'].rank(method='min').astype(int)
ranking_table['rank_T']             = ranking_table['abs_Bias_T_degC'].rank(method='min').astype(int)
ranking_table['Average_Rank']       = (ranking_table['rank_P'] + ranking_table['rank_T']) / 2
ranking_table = ranking_table.sort_values('Average_Rank').reset_index(drop=True)
ranking_table.to_excel(os.path.join(output_folder, 'step2_annual_bias_ranking_table.xlsx'), index=False)
print("  Saved: step2_annual_bias_ranking_table.xlsx")



# =============================================================================
# STEP 3a: Future Climate Change Analysis — SSP2-4.5
# =============================================================================
print("\n" + "="*60)
print("STEP 3a: Future climate change analysis — SSP2-4.5")
print("="*60)

df1_step3 = pd.read_excel(os.path.join(output_folder, 'step2_annual_bias_selected_GCMs.xlsx'))
selected_GCM2 = df1_step3['GCMs'].unique().tolist()
data_dir      = os.path.join(input_folder, 'GCMs')
df2           = pd.DataFrame()

for variable in ['pr', 'tas']:
    for index, gcm in enumerate(selected_GCM2):
        df1 = pd.DataFrame(columns=['Model', 'historical', 'ssp245', 'Variable'])
        df1.loc[index, 'Model']    = gcm
        df1.loc[index, 'Variable'] = variable

        for period in ['historical', 'ssp245']:
            folder_path    = os.path.join(data_dir, period, variable)
            gcm1           = gcm + '_' + period
            matching_files = glob.glob(os.path.join(folder_path, f'*{gcm1}*.nc'))

            for file_path in matching_files:
                print(f'  Found: {gcm} | {variable} | {period}')
                ds         = xr.open_dataset(file_path)
                time_range = slice(str(baseline_start), str(baseline_end)) if period == 'historical' \
                             else slice(str(future_start), str(future_end))
                ds1 = ds.sel(time=time_range)

                if variable == 'pr':
                    annual_result = ds1[variable].resample(time='1Y').sum()
                    annual_avg    = annual_result.mean(dim='time')
                    annual_avg1   = annual_avg * 86400
                elif variable == 'tas':
                    annual_result = ds1[variable].resample(time='1Y').mean()
                    annual_avg    = annual_result.mean(dim='time')
                    annual_avg1   = annual_avg - 273.15

                data_array = annual_avg1
                weighted_average = area_weighted_annual_avg(data_array, gdf, gdf_proj)
                df1.loc[index, period] = weighted_average.values[0]

        df2 = pd.concat([df2, df1])

df2.reset_index(inplace=True)
df2['Change'] = 0
df2.loc[df2['Variable'] == 'pr',  'Change'] = (df2['ssp245'] - df2['historical']) * 100 / df2['historical']
df2.loc[df2['Variable'] == 'tas', 'Change'] = (df2['ssp245'] - df2['historical'])
df2['Change'] = df2['Change'].apply(lambda x: round(x, 1))

df3 = df2.pivot(index=['Model'], columns='Variable', values='Change')
df3.reset_index(inplace=True)
df3['pr_quantile']  = percent_rank(df3['pr'],  precision)
df3['tas_quantile'] = percent_rank(df3['tas'], precision)
df3['CD'] = np.sqrt(((df3['pr_quantile'] - 0.1)**2) + ((df3['tas_quantile'] - 0.1)**2))
df3['CW'] = np.sqrt(((df3['pr_quantile'] - 0.9)**2) + ((df3['tas_quantile'] - 0.1)**2))
df3['HD'] = np.sqrt(((df3['pr_quantile'] - 0.1)**2) + ((df3['tas_quantile'] - 0.9)**2))
df3['HW'] = np.sqrt(((df3['pr_quantile'] - 0.9)**2) + ((df3['tas_quantile'] - 0.9)**2))

min_model_cd = df3.loc[df3['CD'].idxmin()]['Model']
min_model_cw = df3.loc[df3['CW'].idxmin()]['Model']
min_model_hd = df3.loc[df3['HD'].idxmin()]['Model']
min_model_hw = df3.loc[df3['HW'].idxmin()]['Model']

new_df = pd.DataFrame({'CD': [min_model_cd], 'CW': [min_model_cw],
                        'WD': [min_model_hd], 'WW': [min_model_hw]})
new_df.to_excel(os.path.join(output_folder, 'Final_step_ssp245_corner_GCMs.xlsx'), index=False)
print("  Saved: Final_step_ssp245_corner_GCMs.xlsx")

# -------------------------------------------------------------------------
# Save SSP245 full future climate spectrum table
# Columns: Model, dP_percent, dT_degC, pr_quantile, tas_quantile,
#          CD, CW, WD, WW — with corner models marked
# -------------------------------------------------------------------------
corner_models_245 = [min_model_cd, min_model_cw, min_model_hd, min_model_hw]
df3_ssp245 = df3.copy()
df3_ssp245 = df3_ssp245.rename(columns={'pr': 'dP_percent', 'tas': 'dT_degC'})
df3_ssp245['Selected_Corner'] = df3_ssp245['Model'].apply(
    lambda m: 'CD' if m == min_model_cd else
              'CW' if m == min_model_cw else
              'WD' if m == min_model_hd else
              'WW' if m == min_model_hw else ''
)
df3_ssp245 = df3_ssp245[['Model', 'dP_percent', 'dT_degC', 'pr_quantile', 'tas_quantile',
                           'CD', 'CW', 'HD', 'HW', 'Selected_Corner']]
df3_ssp245 = df3_ssp245.sort_values('dT_degC').reset_index(drop=True)
df3_ssp245.to_excel(os.path.join(output_folder, 'step3_ssp245_future_climate_spectrum.xlsx'), index=False)
print("  Saved: step3_ssp245_future_climate_spectrum.xlsx")



# =============================================================================
# STEP 3b: Future Climate Change Analysis — SSP5-8.5
# =============================================================================
print("\n" + "="*60)
print("STEP 3b: Future climate change analysis — SSP5-8.5")
print("="*60)

df1_step3 = pd.read_excel(os.path.join(output_folder, 'step2_annual_bias_selected_GCMs.xlsx'))
selected_GCM2 = df1_step3['GCMs'].unique().tolist()
data_dir      = os.path.join(input_folder, 'GCMs')
df2           = pd.DataFrame()

for variable in ['pr', 'tas']:
    for index, gcm in enumerate(selected_GCM2):
        df1 = pd.DataFrame(columns=['Model', 'historical', 'ssp585', 'Variable'])
        df1.loc[index, 'Model']    = gcm
        df1.loc[index, 'Variable'] = variable

        for period in ['historical', 'ssp585']:
            folder_path    = os.path.join(data_dir, period, variable)
            gcm1           = gcm + '_' + period
            matching_files = glob.glob(os.path.join(folder_path, f'*{gcm1}*.nc'))

            for file_path in matching_files:
                print(f'  Found: {gcm} | {variable} | {period}')
                ds         = xr.open_dataset(file_path)
                time_range = slice(str(baseline_start), str(baseline_end)) if period == 'historical' \
                             else slice(str(future_start), str(future_end))
                ds1 = ds.sel(time=time_range)

                if variable == 'pr':
                    annual_result = ds1[variable].resample(time='1Y').sum()
                    annual_avg    = annual_result.mean(dim='time')
                    annual_avg1   = annual_avg * 86400
                elif variable == 'tas':
                    annual_result = ds1[variable].resample(time='1Y').mean()
                    annual_avg    = annual_result.mean(dim='time')
                    annual_avg1   = annual_avg - 273.15

                data_array = annual_avg1
                weighted_average = area_weighted_annual_avg(data_array, gdf, gdf_proj)
                df1.loc[index, period] = weighted_average.values[0]

        df2 = pd.concat([df2, df1])

df2.reset_index(inplace=True)
df2['Change'] = 0
df2.loc[df2['Variable'] == 'pr',  'Change'] = (df2['ssp585'] - df2['historical']) * 100 / df2['historical']
df2.loc[df2['Variable'] == 'tas', 'Change'] = (df2['ssp585'] - df2['historical'])
df2['Change'] = df2['Change'].apply(lambda x: round(x, 1))

df3 = df2.pivot(index=['Model'], columns='Variable', values='Change')
df3.reset_index(inplace=True)
df3['pr_quantile']  = percent_rank(df3['pr'],  precision)
df3['tas_quantile'] = percent_rank(df3['tas'], precision)
df3['CD'] = np.sqrt(((df3['pr_quantile'] - 0.1)**2) + ((df3['tas_quantile'] - 0.1)**2))
df3['CW'] = np.sqrt(((df3['pr_quantile'] - 0.9)**2) + ((df3['tas_quantile'] - 0.1)**2))
df3['HD'] = np.sqrt(((df3['pr_quantile'] - 0.1)**2) + ((df3['tas_quantile'] - 0.9)**2))
df3['HW'] = np.sqrt(((df3['pr_quantile'] - 0.9)**2) + ((df3['tas_quantile'] - 0.9)**2))

min_model_cd = df3.loc[df3['CD'].idxmin()]['Model']
min_model_cw = df3.loc[df3['CW'].idxmin()]['Model']
min_model_hd = df3.loc[df3['HD'].idxmin()]['Model']
min_model_hw = df3.loc[df3['HW'].idxmin()]['Model']

new_df = pd.DataFrame({'CD': [min_model_cd], 'CW': [min_model_cw],
                        'WD': [min_model_hd], 'WW': [min_model_hw]})
new_df.to_excel(os.path.join(output_folder, 'Final_step_ssp585_corner_GCMs.xlsx'), index=False)
print("  Saved: Final_step_ssp585_corner_GCMs.xlsx")

# -------------------------------------------------------------------------
# Save SSP585 full future climate spectrum table
# Columns: Model, dP_percent, dT_degC, pr_quantile, tas_quantile,
#          CD, CW, WD, WW — with corner models marked
# -------------------------------------------------------------------------
corner_models_585 = [min_model_cd, min_model_cw, min_model_hd, min_model_hw]
df3_ssp585 = df3.copy()
df3_ssp585 = df3_ssp585.rename(columns={'pr': 'dP_percent', 'tas': 'dT_degC'})
df3_ssp585['Selected_Corner'] = df3_ssp585['Model'].apply(
    lambda m: 'CD' if m == min_model_cd else
              'CW' if m == min_model_cw else
              'WD' if m == min_model_hd else
              'WW' if m == min_model_hw else ''
)
df3_ssp585 = df3_ssp585[['Model', 'dP_percent', 'dT_degC', 'pr_quantile', 'tas_quantile',
                           'CD', 'CW', 'HD', 'HW', 'Selected_Corner']]
df3_ssp585 = df3_ssp585.sort_values('dT_degC').reset_index(drop=True)
df3_ssp585.to_excel(os.path.join(output_folder, 'step3_ssp585_future_climate_spectrum.xlsx'), index=False)
print("  Saved: step3_ssp585_future_climate_spectrum.xlsx")



# =============================================================================
# STEP 4a: Percentile Scatter Plot and Final Selection — SSP2-4.5
# =============================================================================
print("\n" + "="*60)
print("STEP 4a: Percentile scatter plot and final selection — SSP2-4.5")
print("="*60)

def map_selection(model, model_ssp245):
    if model in model_ssp245['CD'].unique().tolist():
        return 'cold-dry [{}]'.format(model), 150, 'yellow'
    elif model in model_ssp245['CW'].unique().tolist():
        return 'cold-wet [{}]'.format(model), 150, 'blue'
    elif model in model_ssp245['WD'].unique().tolist():
        return 'warm-dry [{}]'.format(model), 150, 'green'
    elif model in model_ssp245['WW'].unique().tolist():
        return 'warm-wet [{}]'.format(model), 150, 'red'
    else:
        return 'All models', 100, 'grey'

model_ssp245 = pd.read_excel(os.path.join(output_folder, 'Final_step_ssp245_corner_GCMs.xlsx'))
data_dir     = os.path.join(input_folder, 'GCMs')
df2          = pd.DataFrame()

for variable in ['pr', 'tas']:
    df1 = pd.DataFrame(columns=['Model', 'historical', 'ssp245', 'Variable'])

    for period in ['historical', 'ssp245']:
        folder_path = os.path.join(data_dir, period, variable)
        for index, filename in enumerate(os.listdir(folder_path)):
            nc_file_path = os.path.join(folder_path, filename)
            print(nc_file_path)
            ds   = xr.open_dataset(nc_file_path)
            str1 = filename

            start_index = str1.index("day_") + len("day_")
            end_index   = str1.index("_" + str(period))
            gcm         = str1[start_index:end_index]

            df1.loc[index, 'Model']    = gcm
            df1.loc[index, 'Variable'] = variable

            time_range = slice(str(baseline_start), str(baseline_end)) if period == 'historical' \
                         else slice(str(future_start), str(future_end))
            ds1 = ds.sel(time=time_range)

            if variable == 'pr':
                annual_result = ds1[variable].resample(time='1Y').sum()
                annual_avg    = annual_result.mean(dim='time')
                annual_avg1   = annual_avg * 86400
            elif variable == 'tas':
                annual_result = ds1[variable].resample(time='1Y').mean()
                annual_avg    = annual_result.mean(dim='time')
                annual_avg1   = annual_avg - 273.15

            data_array = annual_avg1
            weighted_average = area_weighted_annual_avg(data_array, gdf, gdf_proj)
            df1.loc[index, period] = weighted_average.values[0]

    df2 = pd.concat([df2, df1])

df2.reset_index(inplace=True)
df2['Change'] = 0
df2.loc[df2['Variable'] == 'pr',  'Change'] = (df2['ssp245'] - df2['historical']) * 100 / df2['historical']
df2.loc[df2['Variable'] == 'tas', 'Change'] = (df2['ssp245'] - df2['historical'])
df2['Change'] = df2['Change'].apply(lambda x: round(x, 1))

df3 = df2.pivot(index=['Model'], columns='Variable', values='Change')
df3.reset_index(inplace=True)
df3['pr_quantile']  = percent_rank(df3['pr'],  precision)
df3['tas_quantile'] = percent_rank(df3['tas'], precision)
df3['CD'] = np.sqrt(((df3['pr_quantile'] - 0.1)**2) + ((df3['tas_quantile'] - 0.1)**2))
df3['CW'] = np.sqrt(((df3['pr_quantile'] - 0.9)**2) + ((df3['tas_quantile'] - 0.1)**2))
df3['HD'] = np.sqrt(((df3['pr_quantile'] - 0.1)**2) + ((df3['tas_quantile'] - 0.9)**2))
df3['HW'] = np.sqrt(((df3['pr_quantile'] - 0.9)**2) + ((df3['tas_quantile'] - 0.9)**2))

df3['Selection'], df3['Size'], df3['Color'] = zip(
    *df3['Model'].apply(map_selection, model_ssp245=model_ssp245)
)

plt.figure(figsize=(6, 4))
plt.subplots_adjust(top=0.95, left=0.1, right=0.95, bottom=0.15)
sns.scatterplot(x=[None], y=[None], size=[None], hue=[None], sizes=[100], legend=None)
scatter = sns.scatterplot(
    data=df3, x='pr', y='tas', hue='Selection',
    palette=dict(zip(df3['Selection'], df3['Color']))
)
pr_90th  = np.percentile(df3['pr'], 90)
pr_10th  = np.percentile(df3['pr'], 10)
tas_90th = np.percentile(df3['tas'], 90)
tas_10th = np.percentile(df3['tas'], 10)
plt.scatter(pr_90th, tas_90th, marker='x', color='black', s=150, label='10th and 90th Percentile')
plt.scatter(pr_10th, tas_10th, marker='x', color='black', s=150)
plt.scatter(pr_10th, tas_90th, marker='x', color='black', s=150)
plt.scatter(pr_90th, tas_10th, marker='x', color='black', s=150)
plt.xlabel('ΔP(%)')
plt.ylabel('ΔT(°C)')
plt.title(f'GCMs Selected for {location} under SSP245 Scenario')
scatter.legend_.remove()
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(title='', bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
plt.savefig(os.path.join(output_folder, f'SSP245_Final_Selection_{location}.png'), bbox_inches='tight', dpi=1000)
plt.show()
print(f"  Saved: SSP245_Final_Selection_{location}.png")


# =============================================================================
# STEP 4b: Percentile Scatter Plot and Final Selection — SSP5-8.5
# =============================================================================
print("\n" + "="*60)
print("STEP 4b: Percentile scatter plot and final selection — SSP5-8.5")
print("="*60)

model_ssp585 = pd.read_excel(os.path.join(output_folder, 'Final_step_ssp585_corner_GCMs.xlsx'))
data_dir     = os.path.join(input_folder, 'GCMs')
df2          = pd.DataFrame()

for variable in ['pr', 'tas']:
    df1 = pd.DataFrame(columns=['Model', 'historical', 'ssp585', 'Variable'])

    for period in ['historical', 'ssp585']:
        folder_path = os.path.join(data_dir, period, variable)
        for index, filename in enumerate(os.listdir(folder_path)):
            nc_file_path = os.path.join(folder_path, filename)
            ds   = xr.open_dataset(nc_file_path)
            str1 = filename

            start_index = str1.index("day_") + len("day_")
            end_index   = str1.index("_" + str(period))
            gcm         = str1[start_index:end_index]

            df1.loc[index, 'Model']    = gcm
            df1.loc[index, 'Variable'] = variable

            time_range = slice(str(baseline_start), str(baseline_end)) if period == 'historical' \
                         else slice(str(future_start), str(future_end))
            ds1 = ds.sel(time=time_range)

            if variable == 'pr':
                annual_result = ds1[variable].resample(time='1Y').sum()
                annual_avg    = annual_result.mean(dim='time')
                annual_avg1   = annual_avg * 86400
            elif variable == 'tas':
                annual_result = ds1[variable].resample(time='1Y').mean()
                annual_avg    = annual_result.mean(dim='time')
                annual_avg1   = annual_avg - 273.15

            data_array = annual_avg1
            weighted_average = area_weighted_annual_avg(data_array, gdf, gdf_proj)
            df1.loc[index, period] = weighted_average.values[0]

    df2 = pd.concat([df2, df1])

df2.reset_index(inplace=True)
df2['Change'] = 0
df2.loc[df2['Variable'] == 'pr',  'Change'] = (df2['ssp585'] - df2['historical']) * 100 / df2['historical']
df2.loc[df2['Variable'] == 'tas', 'Change'] = (df2['ssp585'] - df2['historical'])
df2['Change'] = df2['Change'].apply(lambda x: round(x, 1))

df3 = df2.pivot(index=['Model'], columns='Variable', values='Change')
df3.reset_index(inplace=True)
df3['pr_quantile']  = percent_rank(df3['pr'],  precision)
df3['tas_quantile'] = percent_rank(df3['tas'], precision)
df3['CD'] = np.sqrt(((df3['pr_quantile'] - 0.1)**2) + ((df3['tas_quantile'] - 0.1)**2))
df3['CW'] = np.sqrt(((df3['pr_quantile'] - 0.9)**2) + ((df3['tas_quantile'] - 0.1)**2))
df3['HD'] = np.sqrt(((df3['pr_quantile'] - 0.1)**2) + ((df3['tas_quantile'] - 0.9)**2))
df3['HW'] = np.sqrt(((df3['pr_quantile'] - 0.9)**2) + ((df3['tas_quantile'] - 0.9)**2))

df3['Selection'], df3['Size'], df3['Color'] = zip(
    *df3['Model'].apply(map_selection, model_ssp245=model_ssp585)
)

plt.figure(figsize=(6, 4))
plt.subplots_adjust(top=0.95, left=0.1, right=0.95, bottom=0.15)
sns.scatterplot(x=[None], y=[None], size=[None], hue=[None], sizes=[100], legend=None)
scatter = sns.scatterplot(
    data=df3, x='pr', y='tas', hue='Selection',
    palette=dict(zip(df3['Selection'], df3['Color']))
)
pr_90th  = np.percentile(df3['pr'], 90)
pr_10th  = np.percentile(df3['pr'], 10)
tas_90th = np.percentile(df3['tas'], 90)
tas_10th = np.percentile(df3['tas'], 10)
plt.scatter(pr_90th, tas_90th, marker='x', color='black', s=150, label='10th and 90th Percentile')
plt.scatter(pr_10th, tas_10th, marker='x', color='black', s=150)
plt.scatter(pr_10th, tas_90th, marker='x', color='black', s=150)
plt.scatter(pr_90th, tas_10th, marker='x', color='black', s=150)
plt.xlabel('ΔP(%)')
plt.ylabel('ΔT(°C)')
plt.title(f'GCMs Selected for {location} under SSP585 Scenario')
scatter.legend_.remove()
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(title='', bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
plt.savefig(os.path.join(output_folder, f'SSP585_Final_Selection_{location}.png'), bbox_inches='tight', dpi=1000)
plt.show()
print(f"  Saved: SSP585_Final_Selection_{location}.png")

print("\n" + "="*60)
print("ALL COMPUTATION STEPS COMPLETE.")
print("Run step5_visualization.py next to generate the line plots.")
print("="*60)
