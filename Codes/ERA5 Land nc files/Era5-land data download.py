import cdsapi
import os
import json
import zipfile
from multiprocessing.pool import ThreadPool

# Configuration
parallel_download = True
max_parallel = 5
max_retry = 2
resume_file = "download_resume_land_halfyear_split.json"
output_base = r"C:/Users/Diwakar Adhikari/Downloads/ERA5_Land"

# Create output directories
dewpoint_dir = os.path.join(output_base, "dewpoint")
temperature_dir = os.path.join(output_base, "temperature")
precipitation_dir = os.path.join(output_base, "precipitation")
os.makedirs(dewpoint_dir, exist_ok=True)
os.makedirs(temperature_dir, exist_ok=True)
os.makedirs(precipitation_dir, exist_ok=True)

# Dataset
dataset = "reanalysis-era5-land"

# Years to download
years = [str(y) for y in range(1985, 2025)]

# Half-year month groups
halves = {
    "H1": [f"{i:02d}" for i in range(1, 7)],
    "H2": [f"{i:02d}" for i in range(7, 13)]
}

# Load resume state
if os.path.exists(resume_file):
    with open(resume_file, "r") as f:
        resume_state = json.load(f)
else:
    resume_state = {}

# Counters
total_requests = len(years) * len(halves) * 3
success_downloads = 0
failed_downloads = 0
success_not_downloaded = 0

# Create CDS API client
client = cdsapi.Client()

# Function to download data for a specific year, half, and variable
def download_variable(year, half_label, months, variable, suffix, target_dir):
    global success_downloads, failed_downloads, success_not_downloaded

    key = f"{year}_{half_label}_{variable}"
    zip_filename = f"era5_land_{key}.zip"
    nc_filename = f"{year}{half_label}{suffix}.nc"

    if resume_state.get(key) == "done":
        print(f"Skipping {key}, already downloaded and processed.")
        return

    request = {
        "year": [year],
        "month": months,
        "day": [f"{i:02d}" for i in range(1, 32)],
        "time": [f"{i:02d}:00" for i in range(24)],
        "variable": [variable],
        "data_format": "netcdf",
        "download_format": "zip",
        "area": [31, 79, 26, 90]  # Nepal: [north, west, south, east]
    }

    for attempt in range(1, max_retry + 1):
        try:
            print(f"Downloading {key}, attempt {attempt}...")
            client.retrieve(dataset, request).download(zip_filename)
            if os.path.exists(zip_filename):
                with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                    zip_contents = zip_ref.namelist()
                    zip_ref.extractall()
                    for file in zip_contents:
                        if file.endswith(".nc"):
                            source_path = os.path.join(os.getcwd(), file)
                            target_path = os.path.join(target_dir, nc_filename)
                            os.rename(source_path, target_path)
                            break
                os.remove(zip_filename)
                resume_state[key] = "done"
                success_downloads += 1
                print(f"Successfully downloaded and processed {key}.")
            else:
                resume_state[key] = "success_not_downloaded"
                success_not_downloaded += 1
                print(f"Download reported success but file not found for {key}.")
            break
        except Exception as e:
            print(f"Failed to download {key} on attempt {attempt}: {e}")
            if attempt == max_retry:
                resume_state[key] = "failed"
                failed_downloads += 1

    with open(resume_file, "w") as f:
        json.dump(resume_state, f)

# Wrapper for parallel execution
def download_year_half(year_half):
    year, half_label = year_half
    months = halves[half_label]
    download_variable(year, half_label, months, "2m_dewpoint_temperature", "d", dewpoint_dir)
    download_variable(year, half_label, months, "2m_temperature", "t", temperature_dir)
    download_variable(year, half_label, months, "total_precipitation", "p", precipitation_dir)

# Prepare list of (year, half_label) tuples
year_half_list = [(y, h) for y in years for h in halves]

# Execute downloads
if parallel_download:
    with ThreadPool(max_parallel) as pool:
        pool.map(download_year_half, year_half_list)
else:
    for yh in year_half_list:
        download_year_half(yh)

# Print summary
print("\nDownload Summary:")
print(f"Total Requests: {total_requests}")
print(f"Successful Downloads: {success_downloads}")
print(f"Successful but Not Downloaded: {success_not_downloaded}")
print(f"Failed Downloads: {failed_downloads}")