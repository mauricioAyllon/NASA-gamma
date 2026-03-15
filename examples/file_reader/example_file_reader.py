"""
Example usage of file_reader
"""
from nasagamma import file_reader

file_txt = "../data/gui_test_data_hpge_NH3.txt"
file_csv = "../data/gui_test_data_cebr_cal.csv"
file_mca = "../data/gui_test_data_3He.mca"
file_spe = "../data/gui_test_data_hpge_Cu.Spe"
file_cnf = "../data/gui_test_data_lab_sources.cnf"

# Read the files and return spectrum objects
spe_txt = file_reader.read_txt(file_txt)
spe_csv = file_reader.read_csv(file_csv)
spe_mca = file_reader.read_mca(file_mca)
spe_spe = file_reader.read_spe(file_spe)
spe_cnf = file_reader.read_cnf(file_cnf)
