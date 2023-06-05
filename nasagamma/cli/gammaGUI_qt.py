"""
Usage:
  gammaGUI-qt <file_name> [options]
  gammaGUI-qt [options]

  options:
      -o                        open a blank window
      --fwhm_at_0=<fwhm0>       fwhm value at x=0
      --min_snr=<msnr>          min SNR
      --ref_x=<xref>            x reference for fwhm_ref
      --ref_fwhm=<ref_fwhm>     fwhm ref corresponding to x_ref
      --cebr                    detector type (cerium bromide)
      --labr                    detector type (lanthanum bromide)
      --hpge                    detector type (HPGe)


Reads a csv file with the following column format: counts | energy_EUNITS,
where EUNITS can be for examle keV or MeV. It can also read a CSV file with a
single column named "counts". No need to have channels because
they are automatically infered starting from channel = 0.

If detector type is defined e.g. --cebr then the code guesses the x_ref and
fwhm_ref based on the known detector characteristics.

Note that the detector type input parameters must be changed depending on the
particular electronic gain used. The examples here are for our specific
detector configurations.
"""
import docopt
import pandas as pd
import numpy as np
import datetime
import re
from PyQt5.QtWidgets import *

# from PyQt5.QtGui import QKeySequence, QIcon, QPixmap
from PyQt5.uic import loadUi

# from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import SpanSelector, RectangleSelector
from nasagamma import param_handle
from nasagamma import peakfit as pf
from nasagamma import peaksearch as ps
from nasagamma import spectrum as sp
from nasagamma import peakarea as pa
from nasagamma import read_cnf
from nasagamma import tlist
from nasagamma import energy_calibration as ecal
from nasagamma import efficiency
from nasagamma import resolution
from nasagamma import file_reader
from nasagamma import read_parquet_api
from nasagamma import parse_NIST
import pkg_resources


class Dialog_from_UI(QDialog):
    """Create a dialog from a UI file with a given window title."""

    def __init__(self):
        super().__init__()
        self.define_ui_vars()
        ui_file = pkg_resources.resource_filename("nasagamma", self.ui_name)
        loadUi(ui_file, self)
        self.setWindowTitle(self.window_title)


class WindowCust(Dialog_from_UI):
    def define_ui_vars(self):
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.ui_name = "win_customize.ui"
        self.window_title = "Customize plot"


class WindowPeakFinder(Dialog_from_UI):
    def define_ui_vars(self):
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.ui_name = "win_peak_find.ui"
        self.window_title = "Peak finder"


class WindowPeakFinderInfo(Dialog_from_UI):
    def define_ui_vars(self):
        self.ui_name = "win_peak_find_info.ui"
        self.window_title = "Peak finder info"


class WindowCal(Dialog_from_UI):
    def define_ui_vars(self):
        self.ui_name = "win_erg_cal.ui"
        self.window_title = "Energy calibration"


class WindowCalEqns(Dialog_from_UI):
    def define_ui_vars(self):
        self.ui_name = "win_erg_cal_eqns.ui"
        self.window_title = "Energy calibration: set equations"


class WindowEff(Dialog_from_UI):
    def define_ui_vars(self):
        self.ui_name = "win_eff.ui"
        self.window_title = "Efficiency calibration"


class WindowInfoFile(Dialog_from_UI):
    def define_ui_vars(self):
        self.ui_name = "win_info_file.ui"
        self.window_title = "File information"


class WindowIsotID(Dialog_from_UI):
    def define_ui_vars(self):
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.ui_name = "win_isot_id.ui"
        self.window_title = "Isotope ID"


class WindowIsotIDInfo(Dialog_from_UI):
    def define_ui_vars(self):
        self.ui_name = "win_isot_id_info.ui"
        self.window_title = "Isotope ID info"


class WindowAdvFit(Dialog_from_UI):
    def define_ui_vars(self):
        self.ui_name = "win_adv_fit.ui"
        self.window_title = "Advanced Fitting"


class WindowAPIfilters(Dialog_from_UI):
    def define_ui_vars(self):
        self.ui_name = "win_api_filters.ui"
        self.window_title = "Apply API filters"


class TableModel(QtCore.QAbstractTableModel):
    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    def data(self, index, role):
        if role == Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            return str(value)

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]

    def headerData(self, section, orientation, role):
        # section is the index of the column/row.
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])

            if orientation == Qt.Vertical:
                return str(self._data.index[section])

    def sort(self, Ncol, order):
        """Sort table by given column number."""
        try:
            self.layoutAboutToBeChanged.emit()
            self._data = self._data.sort_values(
                self._data.columns[Ncol], ascending=not order
            )
            self.layoutChanged.emit()
        except Exception as e:
            print(e)


class NasaGammaApp(QMainWindow):
    def __init__(self, commands):
        print(super())
        super().__init__()
        ui_file = pkg_resources.resource_filename("nasagamma", "qt_gui.ui")
        loadUi(ui_file, self)
        # self.setMinimumSize(1800, 900)
        # pixmap = QPixmap("figs/nasa-logo.jpg")
        # self.label_nasa.setPixmap(pixmap)
        self.resize(1800, 950)
        self.setWindowTitle("NASA-gamma")
        self.setWindowIcon(QtGui.QIcon("figs/NASA-gamma-logo.png"))
        self.scale = "linear"
        self.pushButton_scale.clicked.connect(self.update_scale)
        self.pushButton_scale.setStyleSheet("background-color : lightgoldenrodyellow")
        self.button_add_peak.setStyleSheet("background-color : silver")
        self.button_add_peak.setCheckable(True)
        self.button_add_peak.clicked.connect(self.add_peak)
        # push buttons
        self.enable_fitButtons(False)
        self.bg = "poly1"
        self.pushButton_poly1.setStyleSheet("background-color : lightgreen")
        self.pushButton_poly1.setCheckable(True)
        self.pushButton_poly1.setChecked(True)
        self.pushButton_poly2.setStyleSheet("background-color : lightgreen")
        self.pushButton_poly2.setCheckable(True)
        self.pushButton_poly3.setStyleSheet("background-color : lightgreen")
        self.pushButton_poly3.setCheckable(True)
        self.pushButton_poly4.setStyleSheet("background-color : lightgreen")
        self.pushButton_poly4.setCheckable(True)
        self.pushButton_poly5.setStyleSheet("background-color : lightgreen")
        self.pushButton_poly5.setCheckable(True)
        self.pushButton_exp.setStyleSheet("background-color : lightgreen")
        self.pushButton_exp.setCheckable(True)
        self.btn_grp = QButtonGroup()
        self.btn_grp.setExclusive(True)
        self.btn_grp.addButton(self.pushButton_poly1)
        self.btn_grp.addButton(self.pushButton_poly2)
        self.btn_grp.addButton(self.pushButton_poly3)
        self.btn_grp.addButton(self.pushButton_poly4)
        self.btn_grp.addButton(self.pushButton_poly5)
        self.btn_grp.addButton(self.pushButton_exp)
        self.btn_grp.buttonClicked.connect(self.update_poly)

        # reset main figure
        self.button_reset.clicked.connect(self.reset_file)

        # Gaussian or skewed Gaussian?
        self.sk_gauss = False
        self.pushButton_gauss.setStyleSheet("background-color : silver")
        self.pushButton_gauss.setCheckable(True)
        self.pushButton_gauss.setChecked(True)
        self.pushButton_gauss2.setStyleSheet("background-color : silver")
        self.pushButton_gauss2.setCheckable(True)
        self.btn_grp_gauss = QButtonGroup()
        self.btn_grp_gauss.setExclusive(True)
        self.btn_grp_gauss.addButton(self.pushButton_gauss)
        self.btn_grp_gauss.addButton(self.pushButton_gauss2)
        self.btn_grp_gauss.buttonClicked.connect(self.update_gauss)
        self.button_remove_cal.setStyleSheet("background-color : lightcoral")

        # customize
        self.w_cust = WindowCust()
        self.button_customize.setStyleSheet("background-color : lightsteelblue")
        self.button_customize.clicked.connect(self.new_window_custom)
        self.w_cust.button_load1.clicked.connect(self.load_file1)
        self.w_cust.button_load2.clicked.connect(self.load_file2)
        self.w_cust.button_plot.clicked.connect(self.cust_plot)

        # navigation toolbars
        self.toolbars = []
        for tb in [
            self.main_plot,
            self.cal_plot,
            self.efficiency_plot,
            self.resolution_plot,
            self.lynx_plot,
            self.api_plot,
        ]:
            tmp = NavigationToolbar(tb.canvas, self)
            tmp.setVisible(False)
            tmp.setStyleSheet("font-size: 24px; background-color : wheat")
            self.addToolBar(tmp)
            self.toolbars.append(tmp)
        self.toolbars[0].setVisible(True)

        self.tabWidget.currentChanged.connect(self.switch_toolbar)
        self.tabWidget.setStyleSheet("background-color : whitesmoke")

        # menu bar
        self.saveFitReport.triggered.connect(self.saveReport)
        self.openFile.triggered.connect(self.load_spe_file)
        self.saveSpect.triggered.connect(self.save_spect)
        self.saveIDpeaks.triggered.connect(self.save_ID_peaks)
        self.info_file.triggered.connect(self.display_info_file)
        self.clearSpectrum.triggered.connect(self.clear_main_figure)

        self.reset_main_figure()

        self.list_xrange = []
        self.commands = commands
        self.search = 0  # initialize dummy search object

        get_input = param_handle.get_spect_search(self.commands)
        if get_input is not None:
            (
                self.spect,
                self.search,
                self.e_units,
                self.ref_x,
                self.fwhm_at_0,
                self.ref_fwhm,
            ) = get_input
            self.create_graph(fit=True)
        try:
            self.min_snr = float(self.commands["--min_snr"])
        except:
            pass

        self.button_remove_cal.clicked.connect(self.remove_cal)
        # peak search range
        self.x0 = None
        self.x1 = None

        # peak fitting
        self.span = None

        # advanced fitting
        # TODO
        self.parea = None
        self.button_adv_fit.setStyleSheet("background-color : limegreen")
        self.w_adv_fit = WindowAdvFit()
        self.button_adv_fit.clicked.connect(self.open_adv_fit)
        self.w_adv_fit.button_plot_area.clicked.connect(self.calculate_peak_area)

        ## energy calibration
        self.mean_vals = [0]
        self.e_vals = [0]
        self.pred_erg = 0
        self.mean_vals_not_fit = []
        self.e_vals_not_fit = []
        self.cal_e_units = None
        # push butons
        self.button_add_cal.setStyleSheet("background-color : lightblue")
        self.button_origin.setStyleSheet("background-color : lightgoldenrodyellow")
        self.button_apply_cal.setStyleSheet("background-color : sandybrown")
        self.button_reset_cal.setStyleSheet("background-color : lightcoral")
        self.button_cal1.setStyleSheet("background-color : lightgreen")
        self.button_cal2.setStyleSheet("background-color : lightgreen")
        self.button_cal3.setStyleSheet("background-color : lightgreen")
        self.button_cal_eqns.setStyleSheet("background-color : silver")
        self.button_origin.clicked.connect(self.set_origin)

        self.button_cal1.setCheckable(True)
        self.button_cal2.setCheckable(True)
        self.button_cal3.setCheckable(True)

        self.btn_grp2 = QButtonGroup()
        self.btn_grp2.setExclusive(True)
        self.btn_grp2.addButton(self.button_cal1)
        self.btn_grp2.addButton(self.button_cal2)
        self.btn_grp2.addButton(self.button_cal3)
        self.btn_grp2.buttonClicked.connect(self.update_cal)
        self.n = 1

        self.reset_cal_figure()

        txt = (
            "Select and add peaks in the 'Spectrum' tab.\n"
            "Separate peak energy values with commas and use '-'\n"
            "for peaks that you do not wish to include."
        )
        self.ax_cal.text(
            0.27,
            0.5,
            txt,
            style="italic",
            bbox={"facecolor": "blue", "alpha": 0.3, "pad": 20},
        )

        # energy textbox
        # self.button_add_cal.clicked.connect(self.add_cal)
        self.button_add_cal.clicked.connect(self.new_window_cal)
        # reset calibration
        self.button_reset_cal.clicked.connect(self.reset_cal)
        # apply calibration
        self.button_apply_cal.clicked.connect(self.apply_cal)
        # set calibration equation
        self.button_cal_eqns.clicked.connect(self.new_window_cal_eqns)

        ## Resolution
        # push butons
        self.button_fwhm1.setStyleSheet("background-color : lightblue")
        self.button_fwhm2.setStyleSheet("background-color : lightblue")
        self.button_add_fwhm.setStyleSheet("background-color : khaki")
        self.button_origin_fwhm.setStyleSheet("background-color : lightgoldenrodyellow")
        self.button_reset_fwhm.setStyleSheet("background-color : lightcoral")
        self.button_extrapolate.setStyleSheet("background-color : lightgreen")

        self.button_fwhm1.setCheckable(True)
        self.button_fwhm2.setCheckable(True)
        self.button_extrapolate.setCheckable(True)

        self.btn_grp3 = QButtonGroup()
        self.btn_grp3.setExclusive(True)
        self.btn_grp3.addButton(self.button_fwhm1)
        self.btn_grp3.addButton(self.button_fwhm2)

        self.btn_grp3.buttonClicked.connect(self.update_fwhm_figure)
        self.reset_fwhm_figure()
        self.fit_lst = []

        self.button_add_fwhm.clicked.connect(self.add_fwhm)
        self.button_origin_fwhm.clicked.connect(self.set_origin_fwhm)
        self.button_reset_fwhm.clicked.connect(self.reset_fwhm)
        self.button_extrapolate.clicked.connect(self.extrapolate_fwhm)

        self.fwhm_x = [0]
        self.fwhm = [0]

        ## Efficiency
        self.eff_vals = []
        self.eff_scale = "log"
        # push butons
        self.button_yscale_eff.setStyleSheet("background-color : lightgoldenrodyellow")
        self.button_yscale_eff.clicked.connect(self.eff_yscale)
        self.button_reset_eff.setStyleSheet("background-color : lightcoral")
        self.button_reset_eff.clicked.connect(self.reset_eff_all)
        self.button_fit1_eff.setStyleSheet("background-color : lightgreen")
        self.button_fit1_eff.clicked.connect(self.eff_fit1)
        self.button_fit2_eff.setStyleSheet("background-color : lightgreen")
        self.button_fit2_eff.clicked.connect(self.eff_fit2)
        self.button_add_eff.setStyleSheet("background-color : sandybrown")
        self.button_add_eff.clicked.connect(self.new_window_eff)
        self.reset_eff_figure()

        ## Lynx
        self.reset_lynx_figure()
        self.button_load_tlist.setStyleSheet("background-color : lightblue")
        self.button_load_tlist.clicked.connect(self.open_tlist_file)
        self.button_lynx_yscale.setStyleSheet("background-color : lightyellow")
        self.button_lynx_yscale.clicked.connect(self.lynx_yscale)
        self.button_apply_lynx.setStyleSheet("background-color : ivory")
        self.button_apply_lynx.clicked.connect(self.apply_period_tbins)
        self.button_filter_lynx.setStyleSheet("background-color : ivory")
        self.button_filter_lynx.clicked.connect(self.filter_time)
        self.button_reset_1.setStyleSheet("background-color : lightcoral")
        self.button_reset_1.clicked.connect(self.reset_plot1)
        self.button_send_to_spect.setStyleSheet("background-color : lightgreen")
        self.button_send_to_spect.clicked.connect(self.send_to_spect)
        # energy filter
        self.button_apply_lynx_e.setStyleSheet("background-color : ivory")
        self.button_apply_lynx_e.clicked.connect(self.apply_dt_bins)
        self.button_filter_lynx_e.setStyleSheet("background-color : ivory")
        self.button_filter_lynx_e.clicked.connect(self.filter_erg)

        ## API
        self.api_spect_scale = "linear"
        self.api_xy_scale = "linear"
        self.api_yscale.setStyleSheet("background-color : lightyellow")
        self.api_yscale.clicked.connect(self.api_spect_yscale)
        self.api_button_logxy.setStyleSheet("background-color : lightgoldenrodyellow")
        self.api_button_logxy.setCheckable(True)
        self.api_button_logxy.clicked.connect(self.api_xylog)
        self.api_vmax_txt.returnPressed.connect(self.api_on_returnXY)
        self.api_send_to_spect.setStyleSheet("background-color : lightgreen")
        self.api_send_to_spect.clicked.connect(self.send_to_spect_api)
        self.api_load_file.setStyleSheet("background-color : lightblue")
        self.api_load_file.clicked.connect(self.load_api_file)
        self.api_reset.setStyleSheet("background-color : lightcoral")
        self.api_reset.clicked.connect(self.reset_button_api)
        self.api_button_filters.setStyleSheet("background-color : silver")
        self.api_button_filters.clicked.connect(self.api_window_filters)

        ## Find peaks
        self.button_find_peaks.setStyleSheet("background-color : mediumaquamarine")
        self.button_find_peaks.clicked.connect(self.activate_peak_finder)
        self.w_peak_find = WindowPeakFinder()
        # self.check_peak_find_groups()
        self.w_peak_find.button_kernel_apply.clicked.connect(self.peakFind_kernel_apply)
        self.w_peak_find.radioButton_hpge.clicked.connect(self.peakFind_check_hpge)
        self.w_peak_find.radioButton_labr.clicked.connect(self.peakFind_check_labr)
        self.w_peak_find.radioButton_nai.clicked.connect(self.peakFind_check_nai)
        self.w_peak_find.radioButton_plastic.clicked.connect(
            self.peakFind_check_plastic
        )
        self.w_peak_find_info = WindowPeakFinderInfo()
        self.w_peak_find.button_info.clicked.connect(self.peakFind_info_activate)

        ## Isotope ID
        self.df_isotID_selected = pd.DataFrame()
        self.df_isotID = []
        self.isotID_vlines = []
        self.button_identify_peaks.setStyleSheet("background-color : navajowhite")
        self.button_identify_peaks.clicked.connect(self.activate_isotope_id)
        self.w_isot_id = WindowIsotID()
        self.w_isot_id.isotID_button_apply.setStyleSheet("background-color : lightblue")
        self.w_isot_id.isotID_button_clear.setStyleSheet(
            "background-color : lightcoral"
        )
        self.w_isot_id.isotID_button_apply.clicked.connect(self.isotID_apply)
        self.w_isot_id.isotID_button_clear.clicked.connect(self.isotID_clear)
        self.w_isot_id.button_remove_vlines.setStyleSheet(
            "background-color : lightcoral"
        )
        self.w_isot_id.button_plot_vlines.clicked.connect(self.isotID_plot_vlines)
        self.w_isot_id.button_remove_vlines.clicked.connect(self.isotID_remove_vlines)
        self.w_isot_id.edit_element_search.returnPressed.connect(self.isotID_textSearch)
        self.w_isotID_info = WindowIsotIDInfo()
        self.w_isot_id.button_info.clicked.connect(self.isotID_info_activate)

    def switch_toolbar(self):
        current = self.tabWidget.currentIndex()
        for i, tb in enumerate(self.toolbars):
            tb.setVisible(i == current)

    def create_graph(self, fit=True, reset=True):
        txt = f"Xrange (optional) [{self.e_units}]:"
        self.w_peak_find.label_units.setText(txt)
        if reset:
            self.reset_main_figure()
        else:
            self.ax_fit.clear()
            self.ax_fit.set_xticks([])
            self.ax_fit.set_yticks([])
        if fit:
            self.search.plot_peaks(
                yscale=self.scale, snrs=self.snr_state, fig=self.fig, ax=self.ax_main
            )
            self.span_select()
        else:
            self.spect.plot(fig=self.fig, ax=self.ax_main, scale=self.scale)
        self.fig.canvas.draw_idle()

    def clear_main_figure(self):
        self.spect = 0
        self.search = 0
        self.fit = 0
        self.reset_main_figure()

    def reset_main_figure(self):
        self.fig = self.main_plot.canvas.figure  # spectrum
        self.fig_fit = self.fit_plot.canvas.figure  # residual
        self.fig.clear()
        self.fig_fit.clear()
        self.fig.set_constrained_layout(True)
        self.fig_fit.set_constrained_layout(True)
        gs = self.fig_fit.add_gridspec(2, 1, height_ratios=[0.3, 1.5])
        self.fig.patch.set_alpha(0.3)

        # axes
        self.ax_main = self.fig.add_subplot()
        self.ax_res = self.fig_fit.add_subplot(gs[0, 0])
        self.ax_fit = self.fig_fit.add_subplot(gs[1, 0])

        # remove ticks
        # self.ax_main.set_xticks([])
        # self.ax_main.set_yticks([])
        self.ax_res.set_xticks([])
        self.ax_res.set_yticks([])
        self.ax_fit.set_xticks([])
        self.ax_fit.set_yticks([])

        if self.button_add_peak.isChecked():
            self.fig.canvas.mpl_disconnect(self.cid)
            self.button_add_peak.setChecked(False)

        self.fig.canvas.draw_idle()
        self.fig_fit.canvas.draw_idle()

    def reset_cal_figure(self):
        self.fig_cal = self.cal_plot.canvas.figure
        self.fig_cal.clear()
        self.fig_cal.set_constrained_layout(True)
        gs2 = self.fig_cal.add_gridspec(
            1, 2, width_ratios=[0.5, 0.5], height_ratios=[1]
        )

        # axes
        self.ax_cal = self.fig_cal.add_subplot(gs2[:, 0])
        self.ax_cal_tab = self.fig_cal.add_subplot(gs2[:, 1])

        # remove ticks
        self.ax_cal.set_xticks([])
        self.ax_cal.set_yticks([])
        self.ax_cal_tab.set_xticks([])
        self.ax_cal_tab.set_yticks([])

    def update_scale(self):
        if self.scale == "log":
            self.ax_main.set_yscale("linear")
            self.scale = "linear"
        else:
            self.ax_main.set_yscale("log")
            self.scale = "log"
        self.fig.canvas.draw_idle()

    def add_peak(self):
        if self.button_add_peak.isChecked():
            print("Waiting to add a new peak...")
            self.cid = self.fig.canvas.mpl_connect("button_press_event", self.onclick)
            if self.span is not None:
                self.span.set_active(False)
            # xnew = self.get_xvalue_from_click
        else:
            self.fig.canvas.mpl_disconnect(self.cid)

    def onclick(self, event):
        xnew = event.xdata
        if self.spect.energies is not None:
            xnew = np.where(self.spect.energies >= xnew)[0][0]
        else:
            xnew = np.where(self.spect.channels >= xnew)[0][0]
        print(xnew)
        # if no search object initialized, initialize a dummy one
        if self.search == 0:
            self.search = ps.PeakSearch(
                self.spect, 420, 20, min_snr=1e6, method="scipy"
            )
            self.span_select()
        x_idx = np.searchsorted(self.search.peaks_idx, xnew)
        self.search.peaks_idx = np.insert(self.search.peaks_idx, x_idx, xnew)
        fwhm_guess_new = self.search.fwhm(xnew)
        self.search.fwhm_guess = np.insert(
            self.search.fwhm_guess, x_idx, fwhm_guess_new
        )
        if self.spect.energies is not None:
            self.ax_main.axvline(
                x=self.spect.energies[xnew], color="red", linestyle="--", alpha=0.2
            )
        else:
            self.ax_main.axvline(
                x=self.spect.channels[xnew], color="red", linestyle="--", alpha=0.2
            )
        self.fig.canvas.draw_idle()
        self.fig.canvas.mpl_disconnect(self.cid)
        self.button_add_peak.setChecked(False)
        self.span.set_active(True)

    ## Advanced fitting
    ## TODO
    def open_adv_fit(self):
        self.w_adv_fit.activateWindow()
        self.parea = pa.PeakArea(spectrum=self.spect, xrange=self.fit.xrange)
        self.w_adv_fit.show()
        self.reset_plot_area()
        self.plot_area(bool_area=False)

    def reset_plot_area(self):
        self.fig_adv_fit = self.w_adv_fit.plot_area.canvas.figure
        self.fig_adv_fit.clear()
        self.fig_adv_fit.set_constrained_layout(True)
        self.ax_area = self.fig_adv_fit.add_subplot()

    def plot_area(self, bool_area=False):
        self.parea.plot(ax=self.ax_area, areas=bool_area)
        self.fig_adv_fit.canvas.draw_idle()

    def calculate_peak_area(self):
        x1 = self.w_adv_fit.x1.text()
        x2 = self.w_adv_fit.x2.text()
        if self.isevaluable(x1) and self.isevaluable(x2) and self.parea is not None:
            peak_range = [eval(x1), eval(x2)]
            self.parea.calculate_peak_area(prange=peak_range)
            self.reset_plot_area()
            self.plot_area(bool_area=True)

    ## Customize plots
    def new_window_custom(self):
        self.w_cust.activateWindow()
        self.w_cust.show()
        self.w_cust.accepted.connect(self.try_customize_plot)

    def try_customize_plot(self):
        try:
            self.customize_plot()
        except:
            pass

    def customize_plot(self):
        xlabel = self.w_cust.x_label_txt.text()
        ylabel = self.w_cust.y_label_txt.text()
        legend = self.w_cust.legend_txt.text().split(",")
        yconst = self.w_cust.yconst_txt.text()
        rebin = self.w_cust.rebin_txt.text()
        moving_avg = self.w_cust.smooth_txt.text()
        gain_shift = self.w_cust.shiftBy_txt.text()
        livetime = self.w_cust.livetime_txt.text()
        if self.w_cust.customize_checkBox_erg.isChecked():
            bool_erg = True
        else:
            bool_erg = False
        if xlabel != "":
            self.ax_main.set_xlabel(xlabel)
            self.spect.x_units = xlabel
            self.fig.canvas.draw_idle()
        if ylabel != "":
            self.ax_main.set_ylabel(ylabel)
            self.spect.y_label = ylabel
            self.fig.canvas.draw_idle()
        if legend != [""]:
            legend = [x.strip() for x in legend]
            self.ax_main.legend(self.ax_main.get_legend_handles_labels()[0], legend)
            # self.ax_main.legend(labels=legend)
            # self.spect.plot_label = legend
            self.fig.canvas.draw_idle()
        if yconst != "":
            self.spect.counts = self.spect.counts * eval(yconst)
            self.create_graph(fit=False, reset=True)
        if rebin != "":
            nbins = eval(rebin)
            if np.mod(nbins, 2) == 0:  # check if even
                n = nbins // 2
                for i in range(n):
                    self.spect.rebin()
                self.create_graph(fit=False, reset=True)
        if moving_avg != "":
            n = int(eval(moving_avg))
            self.spect.smooth(num=n)
            self.create_graph(fit=False, reset=True)
        if gain_shift != "":
            gs = eval(gain_shift)
            self.spect.gain_shift(by=gs, energy=bool_erg)
            self.create_graph(fit=False, reset=True)
        if self.w_cust.checkBox_CR.isChecked():
            yl = "CPS"
            self.ax_main.set_ylabel(yl)
            self.spect.y_label = yl
            self.spect.cps = True
            self.fig.canvas.draw_idle()
        if livetime != "" and self.isevaluable(livetime) and self.spect.cps:
            self.spect.livetime = eval(livetime)
            self.spect.plot_label = None
            self.fig.canvas.draw_idle()
        self.w_cust.accepted.disconnect(self.customize_plot)

    def load_file1(self):
        try:
            self.file1, self.e_units1, self.spect1 = self.open_file()
        except:
            self.file1 = "**ERROR loading file**"
        disp1 = self.file1.split("/")[-1]
        self.w_cust.filename1.setText(disp1)

    def load_file2(self):
        try:
            self.file2, self.e_units2, self.spect2 = self.open_file()
        except:
            self.file2 = "**ERROR loading file**"
        disp2 = self.file2.split("/")[-1]
        self.w_cust.filename2.setText(disp2)

    def cust_plot(self):
        try:
            if self.w_cust.checkBox_add.isChecked():
                cts = self.spect1.counts + self.spect2.counts
            elif self.w_cust.checkBox_sub.isChecked():
                cts = self.spect1.counts - self.spect2.counts

            if self.spect1.energies is not None:
                x = self.spect1.energies
                self.spect = sp.Spectrum(counts=cts, energies=x, e_units=self.e_units1)
                self.e_units = self.e_units1
            else:
                self.spect = sp.Spectrum(counts=cts, e_units=self.e_units1)
            self.create_graph(fit=False, reset=False)
        except:
            print("Could not perform operation")

    ## Fitting
    def which_button(self):
        if self.pushButton_poly1.isChecked():
            self.bg = "poly1"
        elif self.pushButton_poly2.isChecked():
            self.bg = "poly2"
        elif self.pushButton_poly3.isChecked():
            self.bg = "poly3"
        elif self.pushButton_poly4.isChecked():
            self.bg = "poly4"
        elif self.pushButton_poly5.isChecked():
            self.bg = "poly5"
        elif self.pushButton_exp.isChecked():
            self.bg = "exponential"

    def which_button_gauss(self):
        if self.pushButton_gauss2.isChecked():
            self.sk_gauss = True
        else:
            self.sk_gauss = False

    def update_poly(self):
        if len(self.list_xrange) != 0:
            self.which_button()
            try:
                self.fit = pf.PeakFit(
                    self.search, self.list_xrange[-1], bkg=self.bg, skew=self.sk_gauss
                )
                self.ax_res.clear()
                self.ax_fit.clear()
                self.fit.plot(
                    plot_type="simple",
                    fig=self.fig_fit,
                    ax_res=self.ax_res,
                    ax_fit=self.ax_fit,
                )
                data = self.get_values_table_fit()
                self.activate_fit_table(data)
            except:
                print("update_poly: could not perform fit")
            self.fig_fit.canvas.draw_idle()

    def update_gauss(self):
        if len(self.list_xrange) != 0:
            self.which_button_gauss()
            try:
                self.fit = pf.PeakFit(
                    self.search, self.list_xrange[-1], bkg=self.bg, skew=self.sk_gauss
                )
                self.ax_res.clear()
                self.ax_fit.clear()
                self.fit.plot(
                    plot_type="simple",
                    fig=self.fig,
                    ax_res=self.ax_res,
                    ax_fit=self.ax_fit,
                )
                data = self.get_values_table_fit()
                self.activate_fit_table(data)
            except:
                print("update_gauss:could not perform fit")
            self.fig_fit.canvas.draw_idle()

    def span_select(self):
        self.span = SpanSelector(
            self.ax_main,
            self.onselect,
            "horizontal",
            useblit=True,
            interactive=True,
            props=dict(alpha=0.3, facecolor="green"),
        )

    def enable_fitButtons(self, boolV):
        # self.button_adv_fit.setEnabled(boolV)
        self.button_add_cal.setEnabled(boolV)
        self.button_add_fwhm.setEnabled(boolV)
        self.button_add_eff.setEnabled(boolV)

    def onselect(self, xmin, xmax):
        self.idxmin_fit = round(xmin, 4)
        self.idxmax_fit = round(xmax, 4)
        xrange = [self.idxmin_fit, self.idxmax_fit]
        self.list_xrange.append(xrange)
        print("xmin: ", self.idxmin_fit)
        print("xmax: ", self.idxmax_fit)
        self.ax_res.clear()
        self.ax_fit.clear()
        try:
            self.fit = pf.PeakFit(self.search, xrange, bkg=self.bg, skew=self.sk_gauss)
            # plot_fit(fit)
            self.fit.plot(
                plot_type="simple",
                fig=self.fig_fit,
                ax_res=self.ax_res,
                ax_fit=self.ax_fit,
            )
            data = self.get_values_table_fit()
            self.activate_fit_table(data)
            self.enable_fitButtons(True)
        except:
            print("onselect: Could not perform fit")
        self.fig_fit.canvas.draw_idle()

    def get_values_table_fit(self):
        cols = [
            f"Mean ({self.spect.x_units})",
            "Peak area",
            "FWHM",
            "FWHM (%)",
            "Sigma Area",
            "Sigma Area (%)",
        ]
        mean = []
        area = []
        fwhm = []
        fwhm_p = []
        std_mu = []
        std_A = []
        std_A_p = []
        std_fwhm = []
        for i, e in zip(self.fit.peak_info, self.fit.peak_err):
            ls = list(i.values())
            lse = list(e.values())
            mean.append(round(ls[0], 3))
            area.append(round(ls[1], 3))
            fwhm.append(round(ls[2], 3))
            fwhm_p.append(round(ls[2] / ls[0] * 100, 3))
            std_A.append(round(lse[1], 3))
            std_A_p.append(round(lse[1] / ls[1] * 100, 3))

        rs = np.array([mean, area, fwhm, fwhm_p, std_A, std_A_p]).T
        df = pd.DataFrame(columns=cols, data=rs)
        return df

    def activate_fit_table(self, data):
        # self.selected_rows = []
        self.scroll_fit = self.table_fit_area
        self.table_fit = QtWidgets.QTableView()
        # self.table_fit.setAlternatingRowColors(True)
        self.table_fit.setSortingEnabled(True)
        stylesheet_header = "::section{Background-color:lightgreen}"
        self.table_fit.horizontalHeader().setStyleSheet(stylesheet_header)
        self.table_fit.setSelectionBehavior(QtWidgets.QTableView.SelectRows)
        # self.table_fit.resizeColumnsToContents()
        self.model_fit = TableModel(data)
        self.table_fit.setModel(self.model_fit)
        self.scroll_fit.setWidget(self.table_fit)
        # print(self.table_fit.columnWidth(0))
        stylesheet_ix = "::section{Background-color:lightgoldenrodyellow}"
        self.table_fit.setStyleSheet(stylesheet_ix)
        self.table_fit.setColumnWidth(0, 150)

    ## energy calibration
    def remove_cal(self):
        try:
            self.e_units = "channels"
            self.spect = sp.Spectrum(counts=self.spect.counts, e_units=self.e_units)
            self.create_graph(fit=False)
        except:
            pass

    @staticmethod
    def get_mean_vals(fit):
        mean_lst = []
        for d in fit.peak_info:
            keys = list(d)
            mean_ch = d[keys[0]]
            mean_lst.append(mean_ch)
        return mean_lst

    def append_to_list_not_repeated(self, list_e_in, list_m_in, list_e_out, list_m_out):
        for ch, e in zip(list_m_in, list_e_in):
            e2 = e.strip()  # remove white spaces
            if e2 == "_" or e2 == "-":
                next
            else:
                num_e = float(e2)
                if num_e not in list_e_out and ch not in list_m_out:
                    list_e_out.append(num_e)
                    list_m_out.append(ch)

    def perform_cal(self):
        self.pred_erg, self.efit = ecal.ecalibration(
            mean_vals=self.mean_vals,
            erg=self.e_vals,
            channels=self.spect.channels,
            n=self.n,
            e_units=self.e_units,
            plot=True,
            residual=False,
            fig=self.fig_cal,
            ax_fit=self.ax_cal,
        )

    def add_point_dont_fit(self):
        en = self.w_cal.energy_txt.text().split(",")
        mean_lst = self.get_mean_vals(self.fit)
        self.append_to_list_not_repeated(
            en, mean_lst, self.e_vals_not_fit, self.mean_vals_not_fit
        )
        self.ax_cal.clear()
        self.check_radioButton()
        self.perform_cal()
        self.update_cal_table()
        self.plot_point_dont_fit()

    def plot_point_dont_fit(self):
        if len(self.mean_vals_not_fit) > 0:
            self.ax_cal.plot(
                self.mean_vals_not_fit,
                self.e_vals_not_fit,
                "o",
                color="C0",
                ms=8,
                label="Not used for fit",
            )
            self.ax_cal.legend()
            self.mean_vals_not_fit.sort()
            self.e_vals_not_fit.sort()
            self.fig_cal.canvas.draw_idle()

    def add_cal(self):
        try:
            if self.w_cal.checkBox_cal.isChecked():
                self.add_point_dont_fit()
            else:
                self.ax_cal.clear()
                # self.reset_cal_figure()
                # en = self.text_add_cal.text().split(",")
                en = self.w_cal.energy_txt.text().split(",")
                mean_lst = self.get_mean_vals(self.fit)
                self.append_to_list_not_repeated(
                    en, mean_lst, self.e_vals, self.mean_vals
                )

                self.mean_vals.sort()
                self.e_vals.sort()
                self.check_radioButton()
                self.perform_cal()
                self.update_cal_table()
                self.fig_cal.canvas.draw_idle()
        except:
            print("Calibration not performed")

    def new_window_cal(self):
        self.w_cal = WindowCal()
        self.w_cal.show()
        self.w_cal.accepted.connect(self.add_cal)

    def check_radioButton(self):
        if self.w_cal.radio_button_ev.isChecked():
            self.e_units = "eV"
            self.cal_e_units = "eV"
        elif self.w_cal.radio_button_kev.isChecked():
            self.e_units = "keV"
            self.cal_e_units = "keV"
        elif self.w_cal.radio_button_mev.isChecked():
            self.e_units = "MeV"
            self.cal_e_units = "MeV"
        else:
            msg = "ERROR: energy units not determined. Will use channels instead"
            print(msg)

    def update_cal_table(self):
        self.ax_cal_tab.clear()
        sigs_not_fit = [0] * len(self.mean_vals_not_fit)
        ergs = self.e_vals + self.e_vals_not_fit
        # ergs = self.efit.data + self.e_vals_not_fit
        chs = self.mean_vals + self.mean_vals_not_fit
        sigs = list(self.efit.eval_uncertainty()) + sigs_not_fit
        ecal.cal_table(
            ch_lst=chs,
            e_lst=ergs,
            sig_lst=sigs,
            t_scale=[1, 2],
            decimals=3,
            e_units=self.cal_e_units,
            fig=self.fig_cal,
            ax=self.ax_cal_tab,
        )
        self.fig_cal.canvas.draw_idle()

    def set_origin(self):
        self.ax_cal.clear()
        if 0 in self.mean_vals:
            self.mean_vals.pop(0)
            self.e_vals.pop(0)
        else:
            self.mean_vals.insert(0, 0)
            self.e_vals.insert(0, 0)

        if len(self.mean_vals) > 1:
            self.perform_cal()
            self.plot_point_dont_fit()
            self.update_cal_table()
            self.fig_cal.canvas.draw_idle()

    def which_button_cal(self):
        if self.button_cal1.isChecked():
            self.n = 1
        elif self.button_cal2.isChecked():
            self.n = 2
        elif self.button_cal3.isChecked():
            self.n = 3

    def update_cal(self):
        try:
            self.which_button_cal()
            self.ax_cal.clear()
            self.perform_cal()
            self.plot_point_dont_fit()
            self.update_cal_table()
            self.fig_cal.canvas.draw_idle()
        except:
            pass

    def new_window_cal_eqns(self):
        self.w_cal_eqns = WindowCalEqns()
        self.w_cal_eqns.show()
        self.w_cal_eqns.accepted.connect(self.add_cal_eqns)

    def add_cal_eqns(self):
        try:
            a_txt = self.w_cal_eqns.a.text()
            b_txt = self.w_cal_eqns.b.text()
            c_txt = self.w_cal_eqns.c.text()
            d_txt = self.w_cal_eqns.d.text()
            self.which_button_cal_eqn()
            ch = self.spect.channels
            erg = 0
            if self.isevaluable(a_txt):
                erg = erg + eval(a_txt)
            if self.isevaluable(b_txt):
                erg = erg + eval(b_txt) * ch
            if self.isevaluable(c_txt):
                erg = erg + eval(c_txt) * ch**2
            if self.isevaluable(d_txt):
                erg = erg + eval(d_txt) * ch**3
            self.pred_erg = erg
            self.plot_cal_eqns()
        except:
            pass

    def plot_cal_eqns(self):
        self.ax_cal.clear()
        self.ax_cal.plot(self.spect.channels, self.pred_erg, lw=3)
        self.ax_cal.set_xlabel("Channels")
        self.ax_cal.set_ylabel(f"Energy ({self.cal_e_units})")
        self.fig_cal.canvas.draw_idle()

    def which_button_cal_eqn(self):
        if self.w_cal_eqns.radio_button_ev.isChecked():
            self.cal_e_units = "eV"
        elif self.w_cal_eqns.radio_button_kev.isChecked():
            self.cal_e_units = "keV"
        elif self.w_cal_eqns.radio_button_mev.isChecked():
            self.cal_e_units = "MeV"

    def reset_cal(self):
        print("Reseting calibration")
        self.ax_cal_tab.clear()
        self.ax_cal.clear()
        self.pred_erg = 0
        self.mean_vals = [0]
        self.e_vals = [0]
        self.mean_vals_not_fit = []
        self.e_vals_not_fit = []
        # remove ticks
        self.ax_cal.set_xticks([])
        self.ax_cal.set_yticks([])
        self.ax_cal_tab.set_xticks([])
        self.ax_cal_tab.set_yticks([])
        self.fig_cal.canvas.draw_idle()

    def apply_cal(self):
        try:
            if self.cal_e_units is None:
                self.spect = sp.Spectrum(
                    counts=self.spect.counts,
                    energies=self.pred_erg,
                    e_units=self.e_units,
                )
            else:
                self.spect = sp.Spectrum(
                    counts=self.spect.counts,
                    energies=self.pred_erg,
                    e_units=self.cal_e_units,
                )

            # if self.x0 is None:
            #     self.search = ps.PeakSearch(
            #         self.spect,
            #         self.ref_x,
            #         self.ref_fwhm,
            #         self.fwhm_at_0,
            #         min_snr=self.min_snr,
            #     )
            # else:
            #     self.search = ps.PeakSearch(
            #         self.spect,
            #         self.ref_x,
            #         self.ref_fwhm,
            #         self.fwhm_at_0,
            #         min_snr=self.min_snr,
            #         xrange=[self.x0, self.x1],
            #     )

            self.create_graph(fit=False)
        except:
            print("Could not apply calibration")

    ## efficiency calibration
    def reset_eff_figure(self):
        self.fig_eff = self.efficiency_plot.canvas.figure
        self.fig_eff.clear()
        self.fig_eff.set_constrained_layout(True)
        gs2 = self.fig_eff.add_gridspec(
            1, 2, width_ratios=[0.5, 0.5], height_ratios=[1]
        )

        # axes
        self.ax_eff = self.fig_eff.add_subplot(gs2[:, 0])
        self.ax_eff_tab = self.fig_eff.add_subplot(gs2[:, 1])

        # remove ticks
        self.ax_eff.set_xticks([])
        self.ax_eff.set_yticks([])
        self.ax_eff_tab.set_xticks([])
        self.ax_eff_tab.set_yticks([])

    def reset_eff_all(self):
        self.reset_eff_figure()
        self.fig_eff.canvas.draw_idle()
        self.eff_vals = []

    def new_window_eff(self):
        self.w_eff = WindowEff()
        self.w_eff.show()
        self.w_eff.accepted.connect(self.add_eff)

    def add_eff(self):
        try:
            self.get_eff_params()
            eff_class = efficiency.Efficiency(
                t_half=self.t_half,
                A0=self.A0,
                Br=self.B,
                livetime=self.acq_time,
                t_elapsed=self.delta_t_sec,
                which_peak=self.n_peak_eff,
            )
            eff_class.calculate_efficiency(self.fit)
            self.append_to_eff(eff_class.eff * 100, eff_class.mean_val)
            self.plot_eff_points()
        except:
            pass

    def plot_eff_points(self):
        x = np.array(self.eff_vals)[:, 0]
        y = np.array(self.eff_vals)[:, 1]
        self.ax_eff.clear()
        self.ax_eff_tab.clear()
        self.ax_eff.plot(x, y, "o", color="k")
        # self.ax_eff.legend(loc='best')
        self.ax_eff.set_xlabel(self.fit.x_units)
        self.ax_eff.set_ylabel("Efficiency [%]")
        self.ax_eff.set_yscale(self.eff_scale)
        self.ax_eff.set_title("Efficiency")
        efficiency.eff_table(
            x_lst=x,
            eff_lst=y,
            e_units=self.e_units,
            ax=self.ax_eff_tab,
            fig=self.fig_eff,
        )
        self.fig_eff.canvas.draw_idle()

    def append_to_eff(self, eff, mean):
        if (mean, eff) not in self.eff_vals:
            self.eff_vals.append((mean, eff))
        self.eff_vals.sort(key=lambda x: x[0])

    def eff_fit1(self):
        if len(self.eff_vals) > 3:
            x = np.array(self.eff_vals)[:, 0]
            y = np.array(self.eff_vals)[:, 1]
            self.ax_eff.clear()
            efficiency.eff_fit(
                x, y, order=1, plot_table=True, fig=self.fig_eff, ax=self.ax_eff
            )
            self.ax_eff.set_xlabel(self.fit.x_units)
            self.ax_eff.set_ylabel("Efficiency [%]")
            self.fig_eff.canvas.draw_idle()

    def eff_fit2(self):  # TODO
        pass

    def eff_yscale(self):
        if self.eff_scale == "log":
            self.ax_eff.set_yscale("linear")
            self.eff_scale = "linear"
        else:
            self.ax_eff.set_yscale("log")
            self.eff_scale = "log"
        self.fig_eff.canvas.draw_idle()

    def get_eff_params(self):
        # which peak?
        if self.w_eff.button_first.isChecked():
            self.n_peak_eff = 0
        elif self.w_eff.button_second.isChecked():
            self.n_peak_eff = 1
        elif self.w_eff.button_third.isChecked():
            self.n_peak_eff = 2

        # t_half
        if self.w_eff.button_seconds.isChecked():
            t_fact = 1
        elif self.w_eff.button_minutes.isChecked():
            t_fact = 60
        elif self.w_eff.button_years.isChecked():
            t_fact = 365 * 24 * 3600
        else:
            msg = "ERROR: time units not determined"
            print(msg)
        self.t_half = float(self.w_eff.txt_t_half.text()) * t_fact

        # A0
        if self.w_eff.button_Ci.isChecked():
            A_fact = 3.7e10
        elif self.w_eff.button_Sv.isChecked():
            A_fact = 1
        else:
            msg = "ERROR: activity units not determined"
            print(msg)
        self.A0 = float(self.w_eff.txt_activity.text()) * A_fact

        # dates
        date_str0 = self.w_eff.txt_init_date.text()
        fmt_str = "%Y-%m-%d"
        date0 = datetime.datetime.strptime(date_str0, fmt_str)
        date_str1 = self.w_eff.txt_end_date.text()
        date1 = datetime.datetime.strptime(date_str1, fmt_str)
        delta_t = date1 - date0
        self.delta_t_sec = delta_t.days * 24 * 3600

        # branching ratio
        self.B = float(self.w_eff.txt_B.text())

        # acquisition time
        self.acq_time = float(self.w_eff.txt_acq.text())

    ## Resolution (FWHM)
    def reset_fwhm_figure(self):
        self.fig_fwhm = self.resolution_plot.canvas.figure
        self.fig_fwhm.clear()
        self.fig_fwhm.set_constrained_layout(True)
        gs = self.fig_fwhm.add_gridspec(
            2, 2, width_ratios=[0.5, 0.5], height_ratios=[0.5, 0.5]
        )

        # axes
        self.ax_fwhm = self.fig_fwhm.add_subplot(gs[:, 0])
        self.ax_fwhm_gauss = self.fig_fwhm.add_subplot(gs[0, 1])
        self.ax_fwhm_tab = self.fig_fwhm.add_subplot(gs[1, 1])

        # remove ticks
        for ax in [self.ax_fwhm, self.ax_fwhm_tab, self.ax_fwhm_gauss]:
            ax.set_xticks([])
            ax.set_yticks([])

        # self.fig.canvas.draw_idle()

    @staticmethod
    def get_fwhm_vals(fit):
        fwhm_lst = []
        for d in fit.peak_info:
            keys = list(d)
            fwhm_val = d[keys[2]]
            fwhm_lst.append(fwhm_val)
        return fwhm_lst

    def which_button_fwhm(self):
        if self.button_fwhm1.isChecked():
            self.n_fwhm = 1
        elif self.button_fwhm2.isChecked():
            self.n_fwhm = 2
        else:
            self.n_fwhm = 1

    def perform_fwhm(self):
        try:
            self.which_button_fwhm()
            self.fwhm_fit = resolution.fwhm_vs_erg(
                energies=self.fwhm_x,
                fwhms=self.fwhm,
                x_units=self.spect.x_units,
                e_units=self.e_units,
                order=self.n_fwhm,
                fig=self.fig_fwhm,
                ax=self.ax_fwhm,
            )
            self.update_table_fwhm()
            self.update_gauss_fwhm()
            self.fig_fwhm.canvas.draw_idle()
        except:
            pass

    def update_table_fwhm(self):
        self.ax_fwhm_tab.clear()
        resolution.fwhm_table(
            x_lst=self.fwhm_x,
            fwhm_lst=self.fwhm,
            e_units=self.e_units,
            ax=self.ax_fwhm_tab,
            fig=self.fig_fwhm,
        )

    def update_gauss_fwhm(self):
        self.ax_fwhm_gauss.clear()
        gauss_comp = pf.GaussianComponents(fit_obj_lst=self.fit_lst)
        gauss_comp.plot_gauss(
            plot_type="fwhm", fig=self.fig_fwhm, ax=self.ax_fwhm_gauss
        )

    def add_fwhm(self):
        try:
            self.ax_fwhm.clear()
            xvals = self.get_mean_vals(self.fit)
            fwhms = self.get_fwhm_vals(self.fit)

            for x, f in zip(xvals, fwhms):
                if x not in self.fwhm_x and f not in self.fwhm:
                    self.fwhm_x.append(x)
                    self.fwhm.append(f)

            self.fit_lst.append(self.fit)
            self.fwhm_x.sort()
            self.fwhm.sort()
            self.perform_fwhm()
            # self.update_fwhm_table()
        except:
            pass

    def update_fwhm_figure(self):
        self.ax_fwhm.clear()
        self.which_button_fwhm()
        self.perform_fwhm()

    def set_origin_fwhm(self):
        self.ax_fwhm.clear()
        if 0 in self.fwhm_x:
            self.fwhm_x.pop(0)
            self.fwhm.pop(0)
        else:
            self.fwhm_x.insert(0, 0)
            self.fwhm.insert(0, 0)

        if len(self.fwhm_x) > 1:
            self.perform_fwhm()
            # self.update_cal_table()

    def reset_fwhm(self):
        print("Reseting calibration")
        self.fit_lst = []
        self.fwhm_x = [0]
        self.fwhm = [0]
        for ax in [self.ax_fwhm, self.ax_fwhm_tab, self.ax_fwhm_gauss]:
            ax.clear()
            # remove ticks
            ax.set_xticks([])
            ax.set_yticks([])
        self.fig_fwhm.canvas.draw_idle()

    def extrapolate_fwhm(self):
        try:
            if self.button_extrapolate.isChecked():
                resolution.fwhm_extrapolate(
                    energies=self.spect.x,
                    fit=self.fwhm_fit,
                    order=self.n_fwhm,
                    ax=self.ax_fwhm,
                    fig=self.fig_fwhm,
                )
                self.fig_fwhm.canvas.draw_idle()
            else:
                self.update_fwhm_figure()
        except:
            pass

    ## Lynx
    def open_tlist_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "QFileDialog.getOpenFileName()",
            "",
            "All Files (*);;",
            options=options,
        )

        if fileName == "":
            pass
        else:
            try:
                print("Open file: ", fileName)
                fileName = fileName.lower()

                self.reset_lynx_figure()
                self.tl = tlist.Tlist(period=1000, fname=fileName)
                self.tl.plot_time_hist(ax=self.ax_lynx_dt)
                self.tl.plot_spect_erg_all(ax=self.ax_lynx_spect1)
                self.fig_lynx.canvas.draw_idle()
                self.setWindowTitle(f"NASA-gamma: {fileName}")
            except:
                print("Could not open file")

    def reset_lynx_figure(self):
        try:
            self.fig_lynx = self.lynx_plot.canvas.figure
            self.fig_lynx.clear()
            self.fig_lynx.set_constrained_layout(True)
            gs = self.fig_lynx.add_gridspec(
                2, 2, width_ratios=[0.5, 0.5], height_ratios=[0.5, 0.5]
            )

            # axes
            self.ax_lynx_dt = self.fig_lynx.add_subplot(gs[0, 0])
            # self.ax_lynx_buttons = self.fig_lynx.add_subplot(gs[1, 0])
            self.ax_lynx_spect1 = self.fig_lynx.add_subplot(gs[0, 1])
            self.ax_lynx_spect2 = self.fig_lynx.add_subplot(gs[1, 1])
            self.lynx_scale = "linear"
            self.fig_lynx.canvas.draw_idle()
        except:
            pass

    def lynx_yscale(self):
        try:
            if self.lynx_scale == "log":
                self.ax_lynx_spect1.set_yscale("linear")
                self.lynx_scale = "linear"
            else:
                self.ax_lynx_spect1.set_yscale("log")
                self.lynx_scale = "log"
            self.fig_lynx.canvas.draw_idle()
        except:
            pass

    def apply_period_tbins(self):
        try:
            period = self.edit_period.text()
            tbins = self.edit_tbins.text()
            if period != "":
                self.tl.period = float(period)
                self.tl.change_period(self.tl.period)
            if tbins != "":
                self.tl.tbins = int(tbins)
            self.ax_lynx_dt.clear()
            self.tl.plot_time_hist(ax=self.ax_lynx_dt)
            self.fig_lynx.canvas.draw_idle()
        except:
            pass

    def filter_time(self):
        try:
            self.ax_lynx_dt.clear()
            t0 = float(self.edit_t0.text())
            t1 = float(self.edit_t1.text())
            self.tl.trange = [t0, t1]
            self.tl.filter_data(self.tl.trange)
            self.tl.plot_time_hist(ax=self.ax_lynx_dt)
            self.tl.plot_spect_erg_all(ax=self.ax_lynx_spect1)
            self.tl.plot_vlines(ax=self.ax_lynx_dt)
            self.fig_lynx.canvas.draw_idle()
        except:
            pass

    def filter_erg(self):
        try:
            self.ax_lynx_spect2.clear()
            ch0 = float(self.edit_ch0.text())
            ch1 = float(self.edit_ch1.text())
            self.tl.erange = [ch0, ch1]
            self.tl.filter_edata(self.tl.erange)
            self.tl.plot_spect_erg_range(ax=self.ax_lynx_spect2)
            self.fig_lynx.canvas.draw_idle()
        except:
            pass

    def apply_dt_bins(self):
        try:
            dt_bins = self.edit_dt_bins.text()
            if dt_bins != "":
                self.tl.dt_bins = int(dt_bins)
                self.ax_lynx_spect2.clear()
                self.tl.plot_spect_erg_range(ax=self.ax_lynx_spect2)
                self.fig_lynx.canvas.draw_idle()
        except:
            pass

    def reset_plot1(self):
        self.ax_lynx_spect1.clear()
        self.fig_lynx.canvas.draw_idle()

    def send_to_spect(self):
        try:
            self.spect = sp.Spectrum(counts=self.tl.spect_cts)
            self.create_graph(fit=False, reset=False)
        except:
            pass

    ## API
    def load_api_file(self):
        self.reset_api_figure()
        self.dt_flag = 0
        self.xy_flag = 0
        self.en_flag = 0
        try:
            ch = int(self.api_ch_txt.text())
            date = self.api_date_txt.text()
            runnr = int(self.api_run_txt.text())
            self.df_api = read_parquet_api.read_parquet_file(date, runnr, ch)
            self.df_current = self.df_api.copy()
            self.df_previous = self.df_api.copy()
            self.initialize_plots_api()
        except:
            pass
        try:
            filepath = self.api_path_txt.text()
            # self.df_api = read_parquet_api.read_parquet_file_from_path(filepath, ch)
            self.df_api = read_parquet_api.read_parquet_file(
                date, runnr, ch, flat_field=False, data_path_txt=filepath
            )
            self.df_current = self.df_api.copy()
            self.df_previous = self.df_api.copy()
            self.initialize_plots_api()
        except:
            pass

    def initialize_plots_api(self):
        self.trange = [-40, 120]  # dt range
        self.ebins = 2**16  # energy bins
        self.hexbins = 80  # x-y bins
        self.tbins = 512  # time bins
        self.xyplane = (-0.9, 0.9, -0.9, 0.9)  # x and y limits
        self.erange_api = [0, 2**16]  # [0,8]
        self.colormap = "plasma"
        self.plot_energy_hist_api(self.df_api)
        # self.ax_api_spe.set_yscale(self.api_spect_scale)
        self.plot_time_hist_api(self.df_api)
        self.plot_xy_api(self.df_api)
        self.espan_api()
        self.xyspan_api()
        self.tspan_api()
        self.fig_api.canvas.draw_idle()

    def reset_api_figure(self):
        self.fig_api = self.api_plot.canvas.figure
        self.fig_api.clear()
        self.fig_api.set_constrained_layout(True)
        gs = self.fig_api.add_gridspec(
            2, 2, width_ratios=[0.5, 0.5], height_ratios=[1, 1]
        )

        # axes
        # axcolor = "lightgoldenrodyellow"
        self.ax_api_spe = self.fig_api.add_subplot(gs[0, 0])
        self.ax_api_dt = self.fig_api.add_subplot(gs[1, 0])

        self.ax_api_xy = self.fig_api.add_subplot(gs[:, 1])
        self.api_xy_scale = "linear"
        self.api_button_logxy.setChecked(False)
        self.fig_api.canvas.draw_idle()  # to delete

    def reset_button_api(self):
        try:
            self.reset_api_figure()
            self.dt_flag = 0
            self.xy_flag = 0
            self.en_flag = 0
            self.df_current = self.df_api.copy()
            self.df_previous = self.df_api.copy()
            self.initialize_plots_api()
        except:
            pass

    def send_to_spect_api(self):
        try:
            self.spect = sp.Spectrum(counts=self.api_gam)
            self.e_units = "channels"
            self.create_graph(fit=False, reset=False)
        except:
            pass

    def plot_energy_hist_api(self, df):
        self.api_gam, self.api_edg = np.histogram(
            df["energy_orig"], bins=self.ebins, range=self.erange_api
        )
        self.api_ch = np.arange(0, self.ebins, 1)
        self.ax_api_spe.plot(self.api_ch, self.api_gam, color="green")
        self.ax_api_spe.set_yscale(self.api_spect_scale)
        self.ax_api_spe.set_xlabel("Channels")

    def plot_time_hist_api(self, df):
        df["dt"].plot.hist(
            bins=self.tbins,
            ax=self.ax_api_dt,
            range=self.trange,
            alpha=0.7,
            edgecolor="black",
        )
        self.ax_api_dt.set_xlabel("dt [ns]")

    def plot_xy_api(self, df, cbar=True, logxy=False, **kwargs):
        if logxy:
            kwargs["bins"] = "log"
        df.plot.hexbin(
            x="X2",
            y="Y2",
            gridsize=self.hexbins,
            cmap=self.colormap,
            ax=self.ax_api_xy,
            colorbar=cbar,
            extent=self.xyplane,
            **kwargs,
        )
        self.fig_api.canvas.draw_idle()

    def api_xylog(self):
        try:
            if self.api_xy_scale == "linear":
                self.plot_xy_api(self.df_current, cbar=False, logxy=True)
                self.api_xy_scale = "log"
            elif self.api_xy_scale == "log":
                self.plot_xy_api(self.df_current, cbar=False, logxy=False)
                self.api_xy_scale = "linear"
        except:
            pass

    def api_on_returnXY(self):
        try:
            self.vmax = int(self.api_vmax_txt.text())
            self.plot_xy_api(self.df_current, cbar=False, logxy=False, vmax=self.vmax)
            self.api_xy_scale = "linear"
            self.api_button_logxy.setChecked(False)
        except:
            pass

    def api_spect_yscale(self):
        try:
            if self.api_spect_scale == "log":
                self.ax_api_spe.set_yscale("linear")
                self.api_spect_scale = "linear"
            else:
                self.ax_api_spe.set_yscale("log")
                self.api_spect_scale = "log"
            self.fig_api.canvas.draw_idle()
        except:
            pass

    def api_window_filters(self):
        self.w_api_filt = WindowAPIfilters()
        self.w_api_filt.show()
        try:
            self.w_api_filt.button_api_apply.clicked.connect(self.apply_api_filters)
        except:
            pass

    def apply_api_filters(self):
        # TODO: test this function
        # Add vertical lines in plot indicating filter values
        xmin_txt = self.w_api_filt.xmin_txt.text()
        xmax_txt = self.w_api_filt.xmax_txt.text()
        ymin_txt = self.w_api_filt.ymin_txt.text()
        ymax_txt = self.w_api_filt.ymax_txt.text()
        tmin_txt = self.w_api_filt.tmin_txt.text()
        tmax_txt = self.w_api_filt.tmax_txt.text()
        emin_txt = self.w_api_filt.emin_txt.text()
        emax_txt = self.w_api_filt.emax_txt.text()
        if (
            (xmin_txt != "")
            and (xmax_txt != "")
            and (ymin_txt != "")
            and (ymax_txt != "")
        ):
            xmin = float(xmin_txt)
            xmax = float(xmax_txt)
            ymin = float(ymin_txt)
            ymax = float(ymax_txt)
            self.apply_xy_filter(xmin, xmax, ymin, ymax)
        if (tmin_txt != "") and (tmax_txt != ""):
            tmin = float(tmin_txt)
            tmax = float(tmax_txt)
            self.apply_t_filter(tmin, tmax)
        if (emin_txt != "") and (emax_txt != ""):
            emin = int(emin_txt)
            emax = int(emax_txt)
            self.apply_energy_filter(emin, emax)

    def espan_api(self):
        self.span_api = SpanSelector(
            self.ax_api_spe,
            self.enselect_api,
            "horizontal",
            useblit=True,
            interactive=True,
            props=dict(alpha=0.3, facecolor="green"),
        )

    def enselect_api(self, xmin, xmax):
        idxmin = int(round(xmin, 4))
        idxmax = int(round(xmax, 4))
        print("xmin: ", idxmin)
        print("xmax: ", idxmax)
        self.apply_energy_filter(idxmin, idxmax)

    def apply_energy_filter(self, idxmin, idxmax):
        self.ax_api_dt.clear()
        self.ax_api_xy.clear()
        if self.en_flag == 0:  # energy filter has not been used before
            elim = (self.df_current["energy_orig"] > self.api_edg[idxmin]) & (
                self.df_current["energy_orig"] < self.api_edg[idxmax]
            )
            self.df_previous = self.df_current.copy()
            self.df_current = self.df_current[elim]
            self.en_flag = 1  # set energy filter to used
        else:
            elim = (self.df_previous["energy_orig"] > self.api_edg[idxmin]) & (
                self.df_previous["energy_orig"] < self.api_edg[idxmax]
            )
            self.df_current = self.df_previous[elim]
        self.df_current.reset_index(drop=True, inplace=True)
        self.plot_time_hist_api(df=self.df_current)
        self.plot_xy_api(self.df_current, cbar=False)
        # self.span_api.update()
        self.fig_api.canvas.draw_idle()
        #     tspan.update()

    def xyspan_api(self):
        # drawtype is 'box' or 'line' or 'none'
        self.toggle_selector = RectangleSelector(
            self.ax_api_xy,
            self.xyselect_api,
            useblit=True,
            button=[1, 3],  # don't use middle button
            minspanx=1,
            minspany=1,
            spancoords="pixels",
            interactive=True,
            props=dict(facecolor="white", edgecolor="black", alpha=0.1, fill=True),
        )

    def xyselect_api(self, eclick, erelease):
        "eclick and erelease are the press and release events"
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
        # print(" The button you used were: %s %s" % (eclick.button, erelease.button))
        self.apply_xy_filter(x1, x2, y1, y2)

    def apply_xy_filter(self, x1, x2, y1, y2):
        self.ax_api_spe.clear()
        self.ax_api_dt.clear()
        if self.xy_flag == 0:  # xy filter not used before
            xlim = (self.df_current["X2"] > x1) & (self.df_current["X2"] < x2)
            ylim = (self.df_current["Y2"] > y1) & (self.df_current["Y2"] < y2)
            self.df_previous = self.df_current.copy()
            self.df_current = self.df_current[xlim & ylim]
            self.xy_flag = 1  # set energy filter to used
        else:
            xlim = (self.df_previous["X2"] > x1) & (self.df_previous["X2"] < x2)
            ylim = (self.df_previous["Y2"] > y1) & (self.df_previous["Y2"] < y2)
            self.df_current = self.df_previous[xlim & ylim]
        self.df_current.fillna(0)
        self.df_current.reset_index(drop=True, inplace=True)
        self.plot_energy_hist_api(df=self.df_current)
        self.plot_time_hist_api(df=self.df_current)
        self.plot_xy_api(df=self.df_current, cbar=False)
        # self.toggle_selector.update()
        self.fig_api.canvas.draw_idle()

    def tspan_api(self):
        self.tspan = SpanSelector(
            self.ax_api_dt,
            self.tselect_api,
            "horizontal",
            useblit=True,
            interactive=True,
            props=dict(alpha=0.4, facecolor="yellow"),
        )

    def tselect_api(self, tmin, tmax):
        print("tmin: ", round(tmin, 3))
        print("tmax: ", round(tmax, 3))
        self.apply_t_filter(tmin, tmax)

    def apply_t_filter(self, tmin, tmax):
        self.ax_api_spe.clear()
        self.ax_api_xy.clear()
        if self.dt_flag == 0:
            tlim = (self.df_current["dt"] > tmin) & (self.df_current["dt"] < tmax)
            self.df_previous = self.df_current.copy()
            self.df_current = self.df_current[tlim]
            self.dt_flag = 1
        else:
            tlim = (self.df_previous["dt"] > tmin) & (self.df_previous["dt"] < tmax)
            self.df_current = self.df_previous[tlim]
        self.df_current.reset_index(drop=True, inplace=True)
        self.plot_energy_hist_api(df=self.df_current)
        self.plot_xy_api(self.df_current, cbar=False)
        self.fig_api.canvas.draw_idle()
        # espan.update()
        # tspan.update()

    ## Peak finder
    def peakFind_info_activate(self):
        self.w_peak_find_info.activateWindow()
        self.w_peak_find_info.show()

    def activate_peak_finder(self):
        self.w_peak_find.activateWindow()
        self.w_peak_find.show()

    def peakFind_kernel_apply(self):
        # self.ax_main.clear()
        self.fwhm_at_0 = 1
        try:
            if self.w_peak_find.checkBox_km.isChecked():
                method = "km"
            else:
                method = "scipy"
            self.min_snr = float(self.w_peak_find.edit_snr.text())
            self.ref_x = float(self.w_peak_find.edit_ref_ch.text())
            self.ref_fwhm = float(self.w_peak_find.edit_ref_fwhm.text())

            x0 = self.w_peak_find.edit_x0.text()
            x1 = self.w_peak_find.edit_x1.text()
            if x0 and x1:
                self.x0 = float(x0)
                self.x1 = float(x1)
                self.search = ps.PeakSearch(
                    self.spect,
                    self.ref_x,
                    self.ref_fwhm,
                    self.fwhm_at_0,
                    min_snr=self.min_snr,
                    xrange=[self.x0, self.x1],
                    method=method,
                )
            else:
                if method == "km" and len(self.spect.channels) < 9000:
                    self.search = ps.PeakSearch(
                        self.spect,
                        self.ref_x,
                        self.ref_fwhm,
                        self.fwhm_at_0,
                        min_snr=self.min_snr,
                        method=method,
                    )
                elif method == "scipy":
                    self.search = ps.PeakSearch(
                        self.spect,
                        self.ref_x,
                        self.ref_fwhm,
                        self.fwhm_at_0,
                        min_snr=self.min_snr,
                        method=method,
                    )
                else:
                    pass
            if self.w_peak_find.checkBox_SNR.isChecked():
                self.snr_state = "on"
            else:
                self.snr_state = "off"
            self.create_graph(fit=True)
        except:
            print(
                (
                    "ERROR: non-numeric entry or if more than 9000 channels,"
                    "constrain the range using x0 and x1."
                )
            )
            # self.create_graph(fit=False)

    def peakFind_check_hpge(self):
        self.w_peak_find.edit_snr.setText("5")
        self.w_peak_find.edit_ref_ch.setText("420")
        self.w_peak_find.edit_ref_fwhm.setText("3")

    def peakFind_check_labr(self):
        self.w_peak_find.edit_snr.setText("5")
        self.w_peak_find.edit_ref_ch.setText("420")
        self.w_peak_find.edit_ref_fwhm.setText("12")

    def peakFind_check_nai(self):
        self.w_peak_find.edit_snr.setText("5")
        self.w_peak_find.edit_ref_ch.setText("420")
        self.w_peak_find.edit_ref_fwhm.setText("15")

    def peakFind_check_plastic(self):
        self.w_peak_find.edit_snr.setText("5")
        self.w_peak_find.edit_ref_ch.setText("420")
        self.w_peak_find.edit_ref_fwhm.setText("20")

    ## Isotope ID
    def isotID_info_activate(self):
        self.w_isotID_info.activateWindow()
        self.w_isotID_info.show()

    def activate_isotope_id(self):
        self.w_isot_id.activateWindow()
        self.w_isot_id.show()

    def isotID_apply(self):
        df_files = self.isotID_retrieve_data()
        if len(df_files) == 0:
            print("Cannot retrieve data")
        else:
            df_element = self.isotID_filter_by_element(df_files)
            df_energy = self.isotID_filter_by_energy(df_element)
            self.df_isotID = df_energy
            self.activate_table_gammas(self.df_isotID)

    def isotID_clear(self):
        "Clear entries, uncheck boxes"
        self.w_isot_id.edit_isot.clear()
        self.w_isot_id.edit_energy.clear()
        self.w_isot_id.edit_erange.clear()
        self.w_isot_id.lab_src.setChecked(False)
        self.w_isot_id.natural_rad.setChecked(False)
        self.w_isot_id.neutron_capt.setChecked(False)
        self.w_isot_id.neutron_inl_talys.setChecked(False)
        self.w_isot_id.neutron_inl_baghdad.setChecked(False)

    @staticmethod
    def join_gamma_files(files):
        data = pd.read_csv(files[0])
        if len(files) == 1:
            return data
        else:
            for f in files[1:]:
                df0 = pd.read_csv(f)
                data = pd.concat([data, df0], ignore_index=True)
            data["sort"] = (
                data["Isotope"].str.extract("(\d+)", expand=False).astype(int)
            )
            data.sort_values("sort", inplace=True, ascending=True)
            data = data.drop("sort", axis=1)
            data.reset_index(inplace=True, drop=True)
            return data

    def isotID_retrieve_data(self):
        lab_src_file = pkg_resources.resource_filename(
            "nasagamma", "data/Common_lab_sources.csv"
        )
        delay_act_file = pkg_resources.resource_filename(
            "nasagamma", "data/Delayed_activation_IAEA.csv"
        )
        nat_rad_file = pkg_resources.resource_filename(
            "nasagamma", "data/Natural_radiation.csv"
        )
        capt_file = pkg_resources.resource_filename(
            "nasagamma", "data/Capture_CapGam.csv"
        )
        capt_IAEA_file = pkg_resources.resource_filename(
            "nasagamma", "data/Capture_IAEA.csv"
        )
        inl_talys_file = pkg_resources.resource_filename(
            "nasagamma", "data/Inelastic_14MeV_Talys.csv"
        )
        inl_baghdad_file = pkg_resources.resource_filename(
            "nasagamma", "data/Inelastic_Baghdad.csv"
        )
        file = 0
        if self.w_isot_id.lab_src.isChecked():
            file = lab_src_file
        elif self.w_isot_id.delayed_activation.isChecked():
            file = delay_act_file
        elif self.w_isot_id.natural_rad.isChecked():
            file = nat_rad_file
        elif self.w_isot_id.neutron_capt.isChecked():
            file = capt_file
        elif self.w_isot_id.neutron_capt_IAEA.isChecked():
            file = capt_IAEA_file
        elif self.w_isot_id.neutron_inl_talys.isChecked():
            file = inl_talys_file
        elif self.w_isot_id.neutron_inl_baghdad.isChecked():
            file = inl_baghdad_file
        else:
            print("ERROR: Select a database from the menu")
        if file == 0:
            return []
        else:
            data = self.join_gamma_files([file])
            return data

    def isotID_filter_by_element(self, df):
        elements = self.w_isot_id.edit_isot.text()
        if elements == "":
            return df
        elements_lst = elements.split(",")
        ixs = []  # indices
        for el in elements_lst:
            elm = el.strip(" ").lower()
            isot = re.findall("\d+", elm)
            if len(isot) == 0:
                ix = list(
                    df.index[df["Isotope"].str.match(f"(\d+){elm}(\\b)", case=False)]
                )
                ixs.append(ix)
            else:
                ix = list(df.index[df["Isotope"].str.lower() == elm])
                ixs.append(ix)
        ixs_flat = [item for sublist in ixs for item in sublist]
        ixs_uniq = sorted(list(set(ixs_flat)))
        df = df.iloc[ixs_uniq]
        df.reset_index(inplace=True, drop=True)
        return df

    @staticmethod
    def isevaluable(s):
        "Check if eval(s) is possible"
        try:
            eval(s)
            return True
        except:
            return False

    def isotID_filter_by_energy(self, df):
        energy_txt = self.w_isot_id.edit_energy.text()
        if energy_txt == "" or self.isevaluable(energy_txt) is False:
            return df
        energy = eval(energy_txt)
        erange_txt = self.w_isot_id.edit_erange.text()
        if erange_txt == "" or self.isevaluable(erange_txt) is False:
            erange = 0.01  # keV
        else:
            erange = eval(erange_txt)
        filt = (df["Energy (keV)"] > energy - erange) & (
            df["Energy (keV)"] < energy + erange
        )
        df = df[filt]
        df.reset_index(inplace=True, drop=True)
        return df

    ## list of gamma rays
    def activate_table_gammas(self, data):
        self.selected_rows = []
        self.scroll = self.w_isot_id.scrollArea_gammas
        self.table = QtWidgets.QTableView()
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        stylesheet = "::section{Background-color:lightgreen}"
        self.table.horizontalHeader().setStyleSheet(stylesheet)
        self.table.setSelectionBehavior(QtWidgets.QTableView.SelectRows)
        self.model = TableModel(data)
        self.table.setModel(self.model)
        self.scroll.setWidget(self.table)
        # self.setCentralWidget(self.scroll)
        self.table.selectionModel().selectionChanged.connect(self.on_selectionChanged)

    def on_selectionChanged(self, selected, deselected):
        for ix in selected.indexes():
            self.selected_rows.append(ix.row())
        for ix in deselected.indexes():
            try:
                self.selected_rows.remove(ix.row())
            except ValueError:
                pass
        self.selected_rows = list(set(self.selected_rows))
        # print(self.selected_rows)
        self.selected_indexes = list(self.model._data.index[self.selected_rows])
        # print(self.selected_indexes)

    def isotID_getColor(self):
        if self.w_isot_id.button_blue.isChecked():
            return "blue"
        if self.w_isot_id.button_orange.isChecked():
            return "C1"
        if self.w_isot_id.button_green.isChecked():
            return "green"
        if self.w_isot_id.button_red.isChecked():
            return "red"

    def isotID_plot_vlines(self):
        "Plot vertical lines of energies of selected rows"
        if len(self.df_isotID) == 0:  # do nothing if empty
            pass
        else:
            if len(self.selected_rows) == 0:  # do all if none selected
                self.df_isotID_plot = self.df_isotID.copy()
                self.df_isotID_selected = self.df_isotID.copy()
            else:
                self.df_isotID_plot = self.df_isotID.loc[self.selected_indexes]
                self.df_isotID_selected = pd.concat(
                    [self.df_isotID_selected, self.df_isotID_plot], ignore_index=True
                )
                self.df_isotID_plot.drop_duplicates(inplace=True)
                self.df_isotID_selected.drop_duplicates(inplace=True)
            for row in self.df_isotID_plot.index:
                isot0 = self.df_isotID_plot.loc[row, "Isotope"]
                e0 = self.df_isotID_plot.loc[row, "Energy (keV)"]
                sep_checked = self.w_isot_id.checkBox_sep.isChecked()
                dep_checked = self.w_isot_id.checkBox_dep.isChecked()
                compton_edge = self.w_isot_id.checkBox_CE.isChecked()

                photopeak = self.ax_main.axvline(
                    x=e0,
                    color=self.isotID_getColor(),
                    linestyle="-",
                    alpha=0.5,
                    label=f"{isot0} : {round(e0,5)} keV",
                )
                if sep_checked and e0 > 1022:
                    e1 = e0 - 511
                    sep = self.ax_main.axvline(
                        x=e1,
                        color=self.isotID_getColor(),
                        linestyle="--",
                        alpha=0.5,
                        label=f"{isot0} : {round(e1,5)} keV (SEP)",
                    )
                    self.isotID_vlines.append(sep)
                if dep_checked and e0 > 1022:
                    e11 = e0 - 511 * 2
                    dep = self.ax_main.axvline(
                        x=e11,
                        color=self.isotID_getColor(),
                        linestyle="-.",
                        alpha=0.5,
                        label=f"{isot0} : {round(e11,5)} keV (DEP)",
                    )
                    self.isotID_vlines.append(dep)

                if compton_edge:
                    e111 = e0 - e0 / (1 + 2 * e0 / 511)
                    ce = self.ax_main.axvline(
                        x=e111,
                        color=self.isotID_getColor(),
                        linestyle=":",
                        alpha=0.5,
                        label=f"{isot0} : {round(e111,5)} keV (CE)",
                    )
                    self.isotID_vlines.append(ce)

                self.isotID_vlines.append(photopeak)
                if len(self.isotID_vlines) > 300:
                    print("Cannot display more than 300 vertical lines.")
                    break
            if self.w_isot_id.checkBox_labels.isChecked():
                self.ax_main.legend()
            self.fig.canvas.draw_idle()

    def isotID_remove_vlines(self):
        if len(self.isotID_vlines) != 0:
            self.df_isotID_selected = pd.DataFrame()
            for l in self.isotID_vlines:
                l.remove()
            self.isotID_vlines = []
            self.ax_main.legend()
            self.fig.canvas.draw_idle()

    def isotID_textSearch(self):
        element = self.w_isot_id.edit_element_search.text()
        if element == "":
            pass
        else:
            out_text = parse_NIST.isotopic_abundance_str(element)
            self.w_isot_id.text_search.setText(out_text)

    def saveReport(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(
            self, "Save", "", "All Files (*);;Text Files (*.txt)", options=options
        )
        if fileName and len(self.list_xrange) != 0:
            print("Report file: ", fileName)
            self.save_fit(fileName)

    def save_fit(self, fileName):
        try:
            res = self.fit.fit_result
            report = res.fit_report()
            fn = f"Data_filename: {self.fileName}\n"
            xunits = f"x_units: {self.spect.x_units}\n"
            ROI = f"ROI_start_stop: {self.idxmin_fit}, {self.idxmax_fit}\n"
            npeaks = f"Num_peaks: {len(self.fit.peak_info)}\n"
            bkgd = f"Background_type: {self.bg}\n"
            report2 = fn + xunits + ROI + npeaks + bkgd + "\n" + report

            with open(f"{fileName}", "w") as text_file:
                text_file.write(report2)
        except:
            print("Cannot save file with fit info")

    def save_spect(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(
            self, "Save", "", "All Files (*);;Text Files (*.txt)", options=options
        )
        try:
            print("Saved file: ", fileName)
            cts = self.spect.counts
            if self.spect.energies is not None:
                cols = ["counts", f"{self.spect.x_units}"]
                data = np.array((cts, self.spect.x)).T
            else:
                cols = ["counts"]
                data = cts

            df = pd.DataFrame(data=data, columns=cols)
            if ".csv" in fileName:
                df.to_csv(f"{fileName}", index=False)
            else:
                df.to_csv(f"{fileName}.csv", index=False)
        except:
            print("Cannot save file")

    def save_ID_peaks(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(
            self, "Save", "", "All Files (*);;Text Files (*.txt)", options=options
        )
        try:
            print("Saved file: ", fileName)
            df = self.df_isotID_selected
            if ".csv" in fileName:
                df.to_csv(f"{fileName}", index=False)
            else:
                df.to_csv(f"{fileName}.csv", index=False)
        except:
            print("Cannot save file with peak info")

    def display_info_file(self):
        self.w_info_file = WindowInfoFile()
        self.w_info_file.show()

        try:
            lt = float(self.lynxcsv.live_time[0:-4])
            rt = float(self.lynxcsv.real_time[0:-4])
            livetime = lt
            realtime = rt
            deadtime = round((1 - lt / rt) * 100, 2)
            total_cts = self.lynxcsv.counts
            count_rate = round(self.lynxcsv.count_rate, 3)
            nch = self.lynxcsv.nch
            energy_cal = self.lynxcsv.energy_calibration
            start_time = self.lynxcsv.start_time
            self.w_info_file.label_lt.setText(str(livetime))
            self.w_info_file.label_rt.setText(str(realtime))
            self.w_info_file.label_dt.setText(str(deadtime))
            self.w_info_file.label_tc.setText(str(total_cts))
            self.w_info_file.label_cr.setText(str(count_rate))
            self.w_info_file.label_nch.setText(str(nch))
            self.w_info_file.label_time.setText(start_time)
            self.w_info_file.label_energy_cal.setText(str(energy_cal))
        except:
            not_implemented_format = "File format not implemented"
            print(not_implemented_format)
            # self.w_info_file.label_tc.setText(not_implemented_format)

        try:
            self.cnf_dict = read_cnf.read_cnf_file(self.fileName, False)
            livetime = self.cnf_dict["Live time"]
            realtime = self.cnf_dict["Real time"]
            deadtime = (1 - livetime / realtime) * 100
            total_cts = self.cnf_dict["Total counts"]
            count_rate = total_cts / livetime
            ch_num = self.cnf_dict["Number of channels"]
            start_time = self.cnf_dict["Start time"]
            self.w_info_file.label_lt.setText(str(livetime))
            self.w_info_file.label_rt.setText(str(realtime))
            self.w_info_file.label_dt.setText(str(deadtime))
            self.w_info_file.label_tc.setText(str(total_cts))
            self.w_info_file.label_cr.setText(str(count_rate))
            self.w_info_file.label_nch.setText(str(ch_num))
            self.w_info_file.label_time.setText(start_time)
        except:
            not_implemented_format = "File format not implemented"
            print(not_implemented_format)
            # self.w_info_file.label_tc.setText(not_implemented_format)

        try:
            lt = self.multiscanPHA.live_time
            rt = self.multiscanPHA.real_time
            livetime = lt
            realtime = rt
            deadtime = round((1 - lt / rt) * 100, 2)
            total_cts = self.multiscanPHA.counts
            count_rate = round(self.multiscanPHA.count_rate, 3)
            nch = self.multiscanPHA.nch
            energy_cal = self.multiscanPHA.energy_calibration
            start_time = self.multiscanPHA.start_time
            self.w_info_file.label_lt.setText(str(livetime))
            self.w_info_file.label_rt.setText(str(realtime))
            self.w_info_file.label_dt.setText(str(deadtime))
            self.w_info_file.label_tc.setText(str(total_cts))
            self.w_info_file.label_cr.setText(str(count_rate))
            self.w_info_file.label_nch.setText(str(nch))
            self.w_info_file.label_time.setText(start_time)
            self.w_info_file.label_energy_cal.setText(str(energy_cal))
        except:
            not_implemented_format = "File format not implemented"
            print(not_implemented_format)
            # self.w_info_file.label_tc.setText(not_implemented_format)

    def reset_file(self):
        try:
            self.clear_main_figure()
            self.spect = self.spect_orig
            self.e_units = self.e_units_orig
            self.create_graph(fit=False, reset=False)
        except:
            print("Could not reset file")

    def load_spe_file(self):
        try:
            self.fileName, self.e_units, self.spect = self.open_file()
            self.e_units_orig = self.e_units
            self.spect_orig = self.spect
            self.create_graph(fit=False, reset=False)
            self.setWindowTitle(f"NASA-gamma: {self.fileName}")
        except:
            print("File could not be opened")

    def open_file(self):
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "Open spectrum file",
            "",
            "All Files (*);;",
            options=options,
        )
        if fileName == "":
            pass
        else:
            try:
                print("Opening file: ", fileName)
                fileName_lc = fileName.lower()
                if fileName_lc[-4:] == ".csv":
                    try:
                        e_units, spect = file_reader.read_csv_file(fileName)
                    except:
                        self.lynxcsv = file_reader.ReadLynxCsv(fileName)
                        spect = self.lynxcsv.spect
                        e_units = spect.e_units

                elif fileName_lc[-4:] == ".cnf":
                    e_units, spect = file_reader.read_cnf_to_spect(fileName)
                elif fileName_lc[-4:] == ".mca":
                    mca = file_reader.ReadMCA(fileName)
                    e_units = "channels"
                    spect = sp.Spectrum(
                        counts=mca.counts, e_units=e_units, livetime=mca.live_time
                    )
                elif fileName_lc[-8:] == ".pha.txt":
                    multiscanPHA = file_reader.ReadMultiscanPHA(fileName)
                    spect = multiscanPHA.spect
                    e_units = spect.e_units
                else:
                    print("Could not open file")
                    fileName = fileName + " ***INVALID FILE TYPE***"
            except:
                print("Could not open file")
        return fileName, e_units, spect


def main():
    commands = docopt.docopt(__doc__)
    print(commands)

    # initialize figure
    plt.rc("font", size=14)
    plt.style.use("seaborn-v0_8-darkgrid")

    # initialize app
    app = QApplication([])
    window = NasaGammaApp(commands)
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()
