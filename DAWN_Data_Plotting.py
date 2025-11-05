"""
Goal:
NASA-Gamma Web Scrapper designed to be able to pull any data from the NASA PDS
repository. Then using PDS4_Tools library be able to plot datas from a specified 
date and time, or from specific latitudes and longitudes. Once completed would like 
to use the latitudes and longitudes of the data to attempt to make a 3D imaging of 
each planetary body. Might need to use the ASTROpy library and the Plotly library.

Current issues: Cant figure out how to make a gui to specify which data sets to
pull. Tested all data with DAWN and it started pulling every known dataset calibrated
or uncalibrated. Swapped to specifying just CERES Calibrated for the time being. 

Another issue: is struggling to use the plot_data.py file within the DAWN_data folder
to actually plot each of the spectras might just be a local issue involved with my personal 
plotting function within this code.

Another Issue: Need to find a way to get around the time it takes to gather the data from 
the repository local machine is struggling to download individual data sets due to their 
size being over 100MB in some cases. Entire Calibrated Ceres data set is over 1GB with 
.xml and .tab files.

Time to run: 24:12.37
Absolute Must that it runs fully faster from downloading data to plotting the data.

AREAS OF SUCCESS:

November 5, 2025 ~ Was able to successfully scrap the calibrated DAWN Ceres files for the 
.xml and .tab files. Successfully able to plot all files for spectra on a counts versus channel 
graph.

Goal Finish Date: December 12, 2025
Finished: 
Author: Colin Kelley, Wyatt Ingram, Lucas Turner 
"""
import os
import requests
from bs4 import BeautifulSoup
import pds4_tools as pds
import numpy as np
import matplotlib.pyplot as plt

# ===========================================================================================
# CONFIGURATION ~ Make a GUI to choose data from at least CERES, LP, and DAN
# ===========================================================================================
BASE_URL = "https://sbnarchive.psi.edu/pds4/dawn/grand/dawn-grand-ceres_1.0/data_calibrated/"
LOCAL_DIR = os.path.join("NASA-gamma", "nasagamma", "DAWN", "DAWN_data")
os.makedirs(LOCAL_DIR, exist_ok=True)


# ===========================================================================================
# STEP 1. SCRAPE REPOSITORY FOR XML AND TAB FILES ~ Might take several minutes depending on the 
# processing power of local machine
# ===========================================================================================
def scrape_repository(base_url=BASE_URL, dest_dir=LOCAL_DIR):
    print(f" Scraping repository: {base_url}")
    r = requests.get(base_url)
    if r.status_code != 200:
        print(" Failed to connect to site:", r.status_code)
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    links = [a["href"] for a in soup.find_all("a", href=True)]
    xml_links = [l for l in links if l.lower().endswith(".xml")]
    tab_links = [l for l in links if l.lower().endswith(".tab")]
    all_links = xml_links + tab_links
    downloaded_files = []

    for link in all_links:
        file_url = base_url + link
        local_path = os.path.join(dest_dir, link)
        if not os.path.exists(local_path):
            print(f" Downloading: {link}")
            r = requests.get(file_url)
            with open(local_path, "wb") as f:
                f.write(r.content)
        downloaded_files.append(local_path)
    print(f" Downloaded {len(downloaded_files)} files to {dest_dir}")
    return downloaded_files


# =========================================================================================
# STEP 2. PDS4 PARSING
# =========================================================================================
def get_data(file):
    """Read PDS4 file and return a dictionary of data arrays."""
    struct = pds.read(file, lazy_load=False)
    iden = struct[0].id
    data = struct[iden].data
    names = data.dtype.names
    data_dict = {}
    for n in names:
        data_dict[n] = data[n]
    return data_dict


# =========================================================================================
# STEP 3. INSPECTION HELPER
# =========================================================================================
def inspect_single_file(file):
    """Print all keys and shapes for one file."""
    print(f"\n Inspecting structure of {os.path.basename(file)}")
    data = get_data(file)
    for key, val in data.items():
        print(f"  {key:40} -> shape {getattr(val, 'shape', None)} dtype {getattr(val, 'dtype', None)}")
    return data


# =========================================================================================
# STEP 4. SMART BGO PLOTTING ~ Need to update to include the other data types only works for 
# the BGOC detector on the DAWN craft. Might have to include different functions for each 
# detector type such as the ones on LP and DAN
# =========================================================================================
def plot_bgoc_spectra_individual(xml_files):
    """Plot each BGOC spectrum on its own figure."""
    for i, file in enumerate(xml_files):
        try:
            data_dict = get_data(file)
            # Find likely keys for BGO or spectral data
            key_candidates = [k for k in data_dict.keys() if any(tag in k.upper() for tag in ["BGO", "SPEC", "COUNTS"])]
            if not key_candidates:
                print(f" {os.path.basename(file)} has no BGO or spectral keys.")
                continue

            for key in key_candidates:
                value = data_dict[key]
                if not isinstance(value, np.ndarray):
                    continue
                # Handle 1D or 2D arrays
                if value.ndim == 2:
                    bgo_spe = value.sum(axis=0)
                elif value.ndim == 1:
                    bgo_spe = value
                else:
                    continue
                if np.sum(bgo_spe) == 0:
                    print(f" {os.path.basename(file)} {key}: all zeros, skipping.")
                    continue

                # Plotting individual figure
                plt.figure(figsize=(8,5))
                plt.plot(bgo_spe, drawstyle="steps", color='blue')
                plt.yscale("log")
                plt.xlabel("Channel / Bin")
                plt.ylabel("Counts (log scale)")
                plt.title(f"BGOC Spectrum: {os.path.basename(file)}")
                plt.grid(True, which="both", linestyle="--", alpha=0.5)
                plt.tight_layout()
                plt.show()
                break  # Only plot the first valid BGO array per file

        except Exception as e:
            print(f" Error reading {file}: {e}")
            continue



# =======================================================================================
# STEP 5. MAIN EXECUTION ~ If Data is preloaded will not retrieve again will skip to the 
# plot function
# =======================================================================================
if __name__ == "__main__":
    files = scrape_repository()
    xml_files = [f for f in files if f.endswith(".xml")]

    # Uncomment this to inspect a file structure before plotting
    # inspect_single_file(xml_files[0])

    plot_bgoc_spectra_individual(xml_files)

