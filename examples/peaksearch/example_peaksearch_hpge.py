"""
Example of peaksearch functionality for a HPGe detector
"""
from nasagamma import peaksearch as ps
from nasagamma import file_reader
import time


# dataset 
file = "../data/gui_test_data_hpge_NH3.txt"
# instantiate a Spectrum object
spect = file_reader.read_txt(file)

# Required input parameters (in channels)
fwhm_at_0 = 1
ref_fwhm = 5
ref_x = 420

# instantiate a peaksearch object with kernel method
# Note: kernel method is usually more precise, but takes longer
t0 = time.time()
search = ps.PeakSearch(spect, ref_x, ref_fwhm, fwhm_at_0, min_snr=20, method="km")
t1 = time.time() - t0
print(f"Kernel method: {t1} s")

search.plot()

# instantiate a peaksearch object with fast method
t2 = time.time()
search1 = ps.PeakSearch(
    spect, ref_x, ref_fwhm, fwhm_at_0, min_snr=20, method="fast"
)
t3 = time.time()-t2
print(f"Fast method: {t3} s")

search1.plot()



