"""
Read MCA data from PIXIE formatted as npy files.
"""

from nasagamma import helper_api
import json
import matplotlib.pyplot as plt

date = "2024-05-15"
runnr = 0


mca = helper_api.read_mca(date="2024-05-15", runnr=runnr)

plt.figure()
for i, row in enumerate(mca):
    plt.plot(row, label=f"Spectrum #{i}")
plt.legend()
plt.yscale("log")

ch = 9
time = helper_api.read_mca_time(date, runnr, ch=ch, key="live")
print(f"Live time = {time} s")

file_path = helper_api.find_data_path(date, runnr)
files = sorted(list(file_path.glob("MCA-data/*-stats-*")))
d = helper_api.read_json(files[0])
info = []
for k in d.keys():
    if type(d[k]) is list:
        info.append((k, d[k][ch]))
    else:
        info.append((k, d[k]))
