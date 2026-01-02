import cdsapi
import os
import zipfile
import time
import logging
from multiprocessing import Pool

# === CONFIGURATION ===
output_base = r"C:/Users/Diwakar Adhikari/Downloads/Model Selection"
use_parallel = True # Set to False to disable parallel processing
max_retries = 1

# === LOGGING ===
os.makedirs(output_base, exist_ok=True)
log_file = os.path.join(output_base, "download_log.txt")
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === MODELS ===
models = {
    "NESM2": "nesm2",
    # "MPI-ESM1-2-LR": "mpi_esm1_2_lr",
    # "CMCC-ESM2": "cmcc_esm2",
    # "INM-CM5-0": "inm_cm5_0",
    # "MIROC-ES2L": "miroc_es2l",
    # "GFDL-ESM4": "gfdl_esm4",
    # "MIROC6": "miroc6",
    # "EC-Earth3-CC": "ec_earth3_cc",
    "KIOST-ESM": "kiost_esm"
    }

# === VARIABLES ===
variables = [
    "precipitation",
    "near_surface_specific_humidity",
    "near_surface_wind_speed",
    "near_surface_air_temperature",
    "sea_level_pressure",
    "daily_maximum_near_surface_air_temperature",
    "daily_minimum_near_surface_air_temperature"
]

# === SCENARIOS ===
scenarios = {
    "historical": [str(y) for y in range(1985, 2015)],
    "ssp5_8_5": [str(y) for y in range(2015, 2101)],
    "ssp2_4_5": [str(y) for y in range(2015, 2101)]
}

# === TIME PARAMETERS ===
months = [f"{m:02d}" for m in range(1, 13)]
days = [f"{d:02d}" for d in range(1, 32)]

# === AREA ===
area = [31, 79, 26, 90]  # [north, west, south, east]

# === SHAPEFILE AREA EXTRACTION (COMMENTED OUT) ===
# import geopandas as gpd
# def get_area_from_shapefile(shapefile_path):
#     gdf = gpd.read_file(shapefile_path)
#     bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
#     return [bounds[3], bounds[0], bounds[1], bounds[2]]  # [north, west, south, east]

# === DOWNLOAD FUNCTION ===
def download_data(args):
    model_id, variable, scenario, years = args
    client = cdsapi.Client()

    scenario_folder = os.path.join(output_base, scenario, variable)
    os.makedirs(scenario_folder, exist_ok=True)

    zip_path = os.path.join(scenario_folder, f"{model_id}_{variable}_{scenario}.zip")
    nc_path = os.path.join(scenario_folder, f"{model_id}_{variable}_{scenario}.nc")

    if os.path.exists(nc_path):
        logging.info(f"Already downloaded: {nc_path}")
        print(f"Already downloaded: {nc_path}")
        return

    request = {
        "temporal_resolution": "daily",
        "experiment": scenario,
        "variable": variable,
        "model": model_id,
        "year": years,
        "month": months,
        "day": days,
        "area": area,
    }

    success = False
    for attempt in range(max_retries):
        try:
            client.retrieve("projections-cmip6", request).download(zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(scenario_folder)
            os.remove(zip_path)
            logging.info(f"Downloaded and extracted: {nc_path}")
            print(f"Downloaded and extracted: {nc_path}")
            success = True
            break
        except Exception as e:
            logging.warning(f"Attempt {attempt+1} failed for {model_id} - {variable} - {scenario}: {e}")
            print(f"Attempt {attempt+1} failed for {model_id} - {variable} - {scenario}")
            time.sleep(5)

    if not success:
        logging.error(f"Failed to download after {max_retries} attempts: {model_id} - {variable} - {scenario}")
        print(f"Failed to download after {max_retries} attempts: {model_id} - {variable} - {scenario}")

# === TASK PREPARATION ===
if __name__ == "__main__":
    # Prepare download tasks
    tasks = []
    for model_name, model_id in models.items():
        for variable in variables:
            for scenario, years in scenarios.items():
                tasks.append((model_id, variable, scenario, years))

    # Execute downloads
    if use_parallel:
        with Pool(processes=5) as pool:
            pool.map(download_data, tasks)
    else:
        for task in tasks:
            download_data(task)

    print("✅ All downloads completed.")