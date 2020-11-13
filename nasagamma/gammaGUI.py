"""
Usage:
  gammaGUI.py <file_name> [options]
  
  options:
      --x_ref_at_0=<xr0>        x reference value at 0
      --min_snr=<msnr>          min SNR
      --x_ref=<xref>            x reference for fwhm_ref
      --fwhm_ref=<fwhm_ref>     fwhm ref corresponding to x_ref
      --cebr                    detector type (cerium bromide)
      
  

Reads a csv file with the following column format: counts | energy_EUNITS, where
EUNITS can be for examle keV or MeV. No need to have channels because they are
automatically infered starting from channel = 0

If detector type is defined e.g. --cebr then the code guesses the x_ref and
fwhm_ref
"""
## run gammaGUI.py gui_test_data.csv --min_snr=3 --cebr


import docopt
if __name__ == "__main__":
    commands = docopt.docopt(__doc__)
print(commands)

file_name = commands["<file_name>"]
if commands["--cebr"]:
    fwhm_at_0 = 1.0
    ref_x = 1315
    ref_fwhm = 42
    min_snr = float(commands["--min_snr"])

# import the rest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
# from . import spectrum as sp
# from . import peaksearch as ps
# from . import peakfit as pf
from nasagamma import spectrum as sp
from nasagamma import peaksearch as ps
from nasagamma import peakfit as pf
from matplotlib.widgets import Button

df = pd.read_csv(file_name)
if df.shape[1] == 2:
    print("working with energy values")
    e_units = list(df.columns)[1][-3:]
    spect = sp.Spectrum(counts=df["counts"], energies=df.iloc[:,1],
                    e_units=e_units)
    x = spect.energies
elif df.shape[1] == 1:
    print("working with channel numbers")
    e_units = "channels"
    spect = sp.Spectrum(counts=df["counts"], e_units=e_units)
    x = spect.channels

# peaksearch class
search = ps.PeakSearch(spect, ref_x, ref_fwhm, fwhm_at_0, min_snr=min_snr)

# initialize figure 
plt.rc("font", size=14)
plt.style.use("seaborn-darkgrid")
fig = plt.figure(constrained_layout=True, figsize=(16,8))
# gs = fig.add_gridspec(3, 2, width_ratios=[1,0.8],
#                        height_ratios=[0.2,1,0.7])
# ax_s = fig.add_subplot(gs[:, 0])
# ax_res = fig.add_subplot(gs[0, 1])
# ax_fit = fig.add_subplot(gs[1, 1])
# ax_tab = fig.add_subplot(gs[2, 1])

gs = fig.add_gridspec(4, 4, width_ratios=[1,0.33,0.33,0.33],
                       height_ratios=[0.2,1.2,0.1,1])
ax_s = fig.add_subplot(gs[:, 0])
ax_res = fig.add_subplot(gs[0, 1:])
ax_fit = fig.add_subplot(gs[1, 1:])
ax_b1 = fig.add_subplot(gs[2, 1])
ax_b2 = fig.add_subplot(gs[2, 2])
ax_b3 = fig.add_subplot(gs[2, 3])
ax_tab = fig.add_subplot(gs[3, 1:])

# buttons
bpoly1 = Button(ax_b1, 'poly1', color="yellow", hovercolor="lightgreen")
bpoly2 = Button(ax_b2, 'poly2', color="yellow", hovercolor="lightgreen")
bexp = Button(ax_b3, 'exp', color="yellow", hovercolor="lightgreen")

ax_log = plt.axes([0.38, 0.85, 0.1, 0.05])
blog = Button(ax_log, "log-lin", color="yellow", hovercolor="lightgreen")

bpoly1.label.set_fontsize(16)
bpoly2.label.set_fontsize(16)
bexp.label.set_fontsize(16)


# spectrum with peaksearch
scale = "log"
ax_s.plot(x, spect.counts, label="Raw spectrum")
ax_s.set_yscale(scale)

for xc in search.peaks_idx:
    if spect.energies is None:
        x0 = xc
    else:
        x0 = spect.energies[xc]
    ax_s.axvline(x=x0, color='red', linestyle='-', alpha=0.2)
ax_s.legend(loc=1)
ax_s.set_title(f"SNR > {search.min_snr}")
ax_s.set_ylim(1e-1)
ax_s.set_ylabel("Cts")
ax_s.set_xlabel(spect.x_units)

# residual
ax_res.set_ylabel("Residual")
ax_res.set_xticks([])

# table
cols = ["mean", "net_area", "fwhm"]
#rs = np.array([mean, area, fwhm]).T
colors = [['lightblue']*len(cols)]
t = ax_tab.table(colLabels=cols,loc='center',
                cellLoc='center', 
                colColours =["palegreen"] * len(cols),
                cellColours=colors)
t.scale(1, 1.8)
t.auto_set_font_size(False)
t.set_fontsize(14)
ax_tab.axis('off')

list_xrange = []
def xSelect(xmin, xmax):
    idxmin = round(xmin, 4)
    idxmax = round(xmax, 4)
    xrange = [idxmin, idxmax]
    list_xrange.append(xrange)
    print('xmin: ', idxmin)
    print('xmax: ', idxmax)
    ax_res.clear()
    ax_fit.clear()
    ax_tab.clear()
    fit = pf.PeakFit(search, xrange, bkg="poly1")
    plot_fit(fit)
    fig.canvas.draw_idle()
    xspan.span_stays

# set useblit True on gtkagg for enhanced performance
xspan = SpanSelector(ax_s, xSelect, 'horizontal', useblit=True,
                     span_stays=True,
                     rectprops=dict(alpha=0.3, facecolor='green'))


def plot_fit(fit_ob):
    xf = fit_ob.x_data
    yf = fit_ob.y_data
    res = fit_ob.fit_result
    comps = res.eval_components()
    best_fit = res.best_fit
    if "poly" in fit_ob.bkg:
        n = [int(s) for s in list(fit_ob.bkg) if s.isdigit()][0]
        bkg_label = 'polynomial'
    else:
        n = "N/A"
        bkg_label = fit_ob.bkg
    mean = []
    area = []
    fwhm = []
    for i in fit_ob.peak_info:
        ls = list(i.values())
        mean.append(round(ls[0],3))
        area.append(round(ls[1],3))
        fwhm.append(round(ls[2],3))
    
    rs = np.array([mean, area, fwhm]).T
    colors = [['lightblue']*len(cols)]*len(rs)

    ax_res.plot(xf, res.residual, '.', ms=10, alpha=0.5)
    ax_res.hlines(y=0, xmin=xf.min(), xmax=xf.max(), lw=3)
    ax_res.set_ylabel("Residual")
    ax_res.set_xlim([xf.min(), xf.max()])
    ax_res.set_xticks([])
    
    ax_fit.set_title(f"Reduced $\chi^2$ = {round(res.redchi,4)}", fontsize=14)
    ax_fit.plot(xf,yf,'bo', alpha=0.5, label="data")
    ax_fit.plot(xf, best_fit, 'r', lw=3, alpha=0.5, label="Best fit")
    m = 1
    for cp in range(len(comps)-1):
        if m == 1:
            ax_fit.plot(xf, comps[f'{bkg_label}'] + comps[f'g{cp+1}_'],
                    'k--', lw=2,
                    label=f'Gaussian {cp+1} + {bkg_label}: n={n}')
            ax_fit.plot(xf, comps[f'{bkg_label}'], 'g--', label="bkg")
        else:
            ax_fit.plot(xf, comps[f'{bkg_label}'] + comps[f'g{cp+1}_'],
                    'k--', lw=2)
            ax_fit.plot(xf, comps[f'{bkg_label}'], 'g--')
        m = 0
            
    dely = res.eval_uncertainty(sigma=3)
    ax_fit.fill_between(xf, res.best_fit-dely, res.best_fit+dely,
                      color="#ABABAB",
                      label='3-$\sigma$ uncertainty band')
    ax_fit.set_xlabel(fit_ob.x_units)
    ax_fit.legend(loc='best', ncol=2)
    
    # errors
    vals_lst = []
    for element in fit_ob.peak_err:
        vals_lst.append(list(element.values()))
    maf = [float(item) for sublist in vals_lst for item in sublist]
    
    lst1 = []
    n = 1
    n2 = 0
    for i in rs:
        lst0 = []
        for j in i:
            str0 = f"{round(j,2)} +/- {round(maf[n2],2)}"
            lst0.append(str0)
            n2 += 1
        #lst0.insert(0,n)
        lst1.append(lst0)
        n += 1
    
    df = pd.DataFrame(lst1, columns=cols)
            
    t = ax_tab.table(cellText=df.values,colLabels=cols,loc='center',
                     cellLoc='center', 
                     colColours =["palegreen"] * len(cols),
                     cellColours=colors)
    t.scale(1, 1.8)
    t.auto_set_font_size(False)
    t.set_fontsize(14)
    ax_tab.axis('off')

def click_button1(event): 
    fit = pf.PeakFit(search, list_xrange[-1], bkg="poly1")
    ax_res.clear()
    ax_fit.clear()
    ax_tab.clear()
    plot_fit(fit)
    fig.canvas.draw_idle()
    
def click_button2(event): 
    fit = pf.PeakFit(search, list_xrange[-1], bkg="poly2")
    ax_res.clear()
    ax_fit.clear()
    ax_tab.clear()
    plot_fit(fit)
    fig.canvas.draw_idle()

def click_button3(event): 
    fit = pf.PeakFit(search, list_xrange[-1], bkg="exponential")
    ax_res.clear()
    ax_fit.clear()
    ax_tab.clear()
    plot_fit(fit)
    fig.canvas.draw_idle()
   
def click_log(event):
    global scale
    if scale == "log":
        ax_s.set_yscale("linear")
        scale = "linear"
    else:
        ax_s.set_yscale("log")
        scale = "log"
    fig.canvas.draw_idle()
    
bpoly1.on_clicked(click_button1)   
bpoly2.on_clicked(click_button2)  
bexp.on_clicked(click_button3)   
blog.on_clicked(click_log) 
plt.show()    
    
