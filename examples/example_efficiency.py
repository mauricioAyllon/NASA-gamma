"""
Efficiency curve example.
Data file: data/gui_test_data_lab_sources.cnf
"""
import numpy as np
from nasagamma import file_reader
from nasagamma import spectrum as sp
from nasagamma import peaksearch as ps
from nasagamma import peakfit as pf
from nasagamma import efficiency
import pandas as pd
%matplotlib qt

file = "data/gui_test_data_lab_sources.cnf"
spe = file_reader.read_cnf_to_spect(file)
# The energy calibration is off. Let's use a better one
erg_new = -0.8801 + 0.9328*spe.channels
spe.energies = erg_new
spe.x = erg_new
# find peaks for fitting
search = ps.PeakSearch(spe, ref_x=420, ref_fwhm=20, min_snr=4)
# fit several peaks
fit1 = pf.PeakFit(search, xrange=[600,720], bkg="linear") # 662 keV
fit2 = pf.PeakFit(search, xrange=[1077,1414], bkg="poly2") # 1173 keV, 1333 keV
fit3 = pf.PeakFit(search, xrange=[247,326], bkg="poly1") # 276 keV, 303 keV
fit4 = pf.PeakFit(search, xrange=[326,416], bkg="poly1") # # 356 keV, 384 keV

# activation parameters
A0 = 10e-6 * 3.7e10 # 10 uCi in Bq
t_elapsed = efficiency.calculate_t_elapsed(date0="2006-06-01", date1="2022-08-19")
t_half_Cs = 30.08 * 365 * 24 * 3600 # 137Cs half life
t_half_Co = 5.27 * 365 * 24 * 3600 # 60Co half life
t_half_Ba = 10.56 * 365 * 24 * 3600 # 133Ba half life

# Cs-137
eff_Cs = efficiency.Efficiency(t_half=t_half_Cs, A0=A0, Br=0.85, livetime=350,
                               t_elapsed=t_elapsed, which_peak=0)

eff_Cs.calculate_efficiency(fit1) 
eff_Cs.calculate_error(fit_obj=fit1, t_half_sig=0, A0_sig=0, Br_sig=0, livetime_sig=0,
                    t_elapsed_sig=0)
eff662 = eff_Cs.eff
eff662_sig = eff_Cs.error

# Co-60 (1173 kev)
eff_Co1 = efficiency.Efficiency(t_half=t_half_Co, A0=A0, Br=0.99, livetime=600,
                               t_elapsed=t_elapsed, which_peak=0)

eff_Co1.calculate_efficiency(fit2) 
eff_Co1.calculate_error(fit_obj=fit2, t_half_sig=0, A0_sig=0, Br_sig=0, livetime_sig=0,
                    t_elapsed_sig=0)
eff1173 = eff_Co1.eff
eff1173_sig = eff_Co1.error

# Co-60 (1333 keV)
eff_Co2 = efficiency.Efficiency(t_half=t_half_Co, A0=A0, Br=0.99, livetime=600,
                               t_elapsed=t_elapsed, which_peak=1)

eff_Co2.calculate_efficiency(fit2) 
eff_Co2.calculate_error(fit_obj=fit2, t_half_sig=0, A0_sig=0, Br_sig=0, livetime_sig=0,
                    t_elapsed_sig=0)
eff1333 = eff_Co2.eff
eff1333_sig = eff_Co2.error

# Ba-133 (276 keV)
eff_Ba1 = efficiency.Efficiency(t_half=t_half_Ba, A0=A0, Br=0.99, livetime=600,
                               t_elapsed=t_elapsed, which_peak=0)

eff_Co2.calculate_efficiency(fit2) 
eff_Co2.calculate_error(fit_obj=fit2, t_half_sig=0, A0_sig=0, Br_sig=0, livetime_sig=0,
                    t_elapsed_sig=0)
eff1333 = eff_Co2.eff
eff1333_sig = eff_Co2.error

efficiency.plot_points(e_vals=[662,1173,1333], eff_vals=[eff662, eff1173, eff1333],
                        err_vals=[eff662_sig, eff1173_sig, eff1333_sig], e_units="keV")

df1 = eff_Cs.to_df()
df2 = eff_Co1.to_df()
df3 = eff_Co2.to_df()

df_all = pd.concat((df1,df2,df3), ignore_index=True)