"""
Example of peaksearch functionality for a CeBr3 detector
"""
from nasagamma import peaksearch as ps
from nasagamma import file_reader


# dataset
file = "../data/gui_test_data_cebr.csv"
# instantiate a Spectrum object
spect = file_reader.read_csv(file)

# Required input parameters (in channels)
fwhm_at_0 = 1
ref_fwhm = 20
ref_x = 420

# instantiate a peaksearch object
search = ps.PeakSearch(spect, ref_x, ref_fwhm, fwhm_at_0, min_snr=5, method="km")
# plot kernel, peaks, and components
search.plot_kernel()
search.plot_peaks()
search.plot_components()

# instantiate a peaksearch object with predetermined range
search1 = ps.PeakSearch(
    spect, ref_x, ref_fwhm, fwhm_at_0, min_snr=5, xrange=[1200, 1600], method="km"
)
search1.plot_peaks(snrs="off")

# Print parameters/metadata
print(search1.metadata())

# Print peak positions in a given range
search.peaks_in_range(x_min=0, x_max=300)

# Export peak positions found and other data to a csv file
#search.to_csv("tes_peaksearch.csv")


