"""
Script for reading a Canberra Nuclear File (CNF) form GENIE2000 software.

It can be used as a stand alone script or as a module.

Optionally, it generates a text file with the relevant information read from
the CNF file. The output file name is the input file plus the '.txt' extension.



Examples
--------
    >>> python read_cnf.py name_of_the_file.CNF

    ('name_of_the_file.CNF.txt' is automatically created)

References
----------

This script was made as a copy of the c program 'cnfconv' written for the same
purpose. That software can be found here:

https://github.com/messlinger/cnfconv

All the information of the binary file encoding was taken from the file
'cnf_file_format.txt' of the above repository.


"""

import sys
import numpy as np
import time
import struct
from nasagamma import spectrum as sp
from nasagamma import peaksearch as ps


def read_cnf_file(filename, write_output=False):
    """
    Reads data of a Canberra Nuclear File used by the Genie2000 software.

    Parameters
    ----------
    filename : string
        Name of the file to be read.
    write_output : boolean, optional
        Indicate weather to write an output file or not.

    Returns
    -------
    read_dic : dictionary
        Dictionary with all the magnitudes read. Depending on the data
        available,the dictionaries keys may change. Some possible keys are:
        Sample id
        Channels
        Sample unit
        Sample name
        Channels data
        Energy unit
        Energy coefficients
        Shape coefficients
        Left marker
        Total counts
        Number of channels
        Start time
        Counts in markers
        Right marker
        Sample type
        Sample description
        User name
        Live time
        Energy
        Real time
        Measurement mode
        MCA type
        Data source

    Examples
    --------
        >>> from read_cnf import lee_cnf_file
        >>> read_dic = read_cnf_file('name_of_the_file.CNF')
        >>> read_dic['Live time']

    TODO
    ----
    - Markers information is not being read correctly.
    - If the CNF file are obtained in a MCA mode, the live and real time are
    not read correctly.
    - Additional data must be read in case of a file from MCA mode
    (mainly dwell time).

    """

    # Name of the output file
    out_filename = filename + ".txt"

    # Dictionary with all the information read
    read_dic = {}
    with open(filename, "rb") as f:
        i = 0
        while True:
            # List of available section headers
            sec_header = 0x70 + i * 0x30
            i += 1
            # Section id in header
            sec_id_header = uint32_at(f, sec_header)

            # End of section list
            if sec_id_header == 0x00:
                break

            # Location of the begining of each sections
            sec_loc = uint32_at(f, sec_header + 0x0A)
            # Known section id's:
            # Parameter section (times, energy calibration, etc)
            if sec_id_header == 0x00012000:
                offs_param = sec_loc
                read_dic.update(get_energy_calibration(f, offs_param))
                read_dic.update(get_date_time(f, offs_param))
                read_dic.update(get_shape_calibration(f, offs_param))
            # String section
            elif sec_id_header == 0x00012001:
                offs_str = sec_loc
                read_dic.update(get_strings(f, offs_str))
            # Marker section
            elif sec_id_header == 0x00012004:
                offs_mark = sec_loc
                read_dic.update(get_markers(f, offs_mark))
            # Channel data section
            elif sec_id_header == 0x00012005:
                offs_chan = sec_loc
                read_dic.update(get_channel_data(f, offs_param, offs_chan))
            else:
                continue

            # For known sections: section header ir repeated in section block
            if sec_id_header != uint32_at(f, sec_loc):
                print("File {}: Format error\n".format(filename))

    # Once the file is read, some derived magnitudes can be obtained

    # Convert channels to energy
    if set(("Channels", "Energy coefficients")) <= set(read_dic):
        read_dic.update(chan_to_energy(read_dic))

    # Compute ingegration between markers
    if set(("Channels", "Left marker")) <= set(read_dic):
        read_dic.update(markers_integration(read_dic))

    # print(50 * "=")
    # print(10 * " " + "File " + str(filename) + " succesfully read!" + 10 * " ")
    # print(50 * "=")

    # If true, writes an text output file
    if write_output:
        write_to_file(out_filename, read_dic)

    return read_dic


##########################################################
# Definitions for reading some data types
##########################################################


def uint8_at(f, pos):
    f.seek(pos)
    return np.fromfile(f, dtype=np.dtype("<u1"), count=1)[0]


def uint16_at(f, pos):
    f.seek(pos)
    return np.fromfile(f, dtype=np.dtype("<u2"), count=1)[0]


def uint32_at(f, pos):
    f.seek(pos)
    return np.fromfile(f, dtype=np.dtype("<u4"), count=1)[0]


def uint64_at(f, pos):
    f.seek(pos)
    return np.fromfile(f, dtype=np.dtype("<u8"), count=1)[0]


def pdp11f_at(f, pos):
    """
    Convert PDP11 32bit floating point format to
    IEE 754 single precision (32bits)
    """
    f.seek(pos)
    # Read two int16 numbers
    tmp16 = np.fromfile(f, dtype=np.dtype("<u2"), count=2)
    # Swapp positions
    mypack = struct.pack("HH", tmp16[1], tmp16[0])
    f = struct.unpack("f", mypack)[0] / 4.0
    return f


def time_at(f, pos):
    return ~uint64_at(f, pos) * 1e-7


def datetime_at(f, pos):
    return uint64_at(f, pos) / 10000000 - 3506716800


def string_at(f, pos, length):
    f.seek(pos)
    # In order to avoid characters with not utf8 encoding
    return f.read(length).decode("utf8").rstrip("\00").rstrip()


###########################################################
# Definitions for locating and reading data inside the file
###########################################################


def get_strings(f, offs_str):
    """Read strings section."""

    sample_name = string_at(f, offs_str + 0x0030, 0x40)
    sample_id = string_at(f, offs_str + 0x0070, 0x10)
    # sample_id   = string_at(f, offs_str + 0x0070, 0x40)
    sample_type = string_at(f, offs_str + 0x00B0, 0x10)
    sample_unit = string_at(f, offs_str + 0x00C4, 0x40)
    user_name = string_at(f, offs_str + 0x02D6, 0x18)
    sample_desc = string_at(f, offs_str + 0x036E, 0x100)

    out_dic = {
        "Sample name": sample_name,
        "Sample id": sample_id,
        "Sample type": sample_type,
        "Sample unit": sample_unit,
        "User name": user_name,
        "Sample description": sample_desc,
    }

    return out_dic


def get_energy_calibration(f, offs_param):
    """Read energy calibration coefficients."""

    offs_calib = offs_param + 0x30 + uint16_at(f, offs_param + 0x22)
    A = np.empty(4)
    A[0] = pdp11f_at(f, offs_calib + 0x44)
    A[1] = pdp11f_at(f, offs_calib + 0x48)
    A[2] = pdp11f_at(f, offs_calib + 0x4C)
    A[3] = pdp11f_at(f, offs_calib + 0x50)

    # Assuming a maximum length of 0x11 for the energy unit
    energy_unit = string_at(f, offs_calib + 0x5C, 0x11)

    # MCA type
    MCA_type = string_at(f, offs_calib + 0x9C, 0x10)

    # Data source
    data_source = string_at(f, offs_calib + 0x108, 0x10)

    out_dic = {
        "Energy coefficients": A,
        "Energy unit": energy_unit,
        "MCA type": MCA_type,
        "Data source": data_source,
    }

    return out_dic


def get_shape_calibration(f, offs_param):
    """
    Read Shape Calibration Parameters :
        FWHM=B[0]+B[1]*E^(1/2)  . B[2] and B[3] probably tail parameters
    """

    offs_calib = offs_param + 0x30 + uint16_at(f, offs_param + 0x22)
    B = np.empty(4)
    B[0] = pdp11f_at(f, offs_calib + 0xDC)
    B[1] = pdp11f_at(f, offs_calib + 0xE0)
    B[2] = pdp11f_at(f, offs_calib + 0xE4)
    B[3] = pdp11f_at(f, offs_calib + 0xE8)

    out_dic = {"Shape coefficients": B}

    return out_dic


def get_channel_data(f, offs_param, offs_chan):
    """Read channel data."""

    # Total number of channels
    n_channels = uint8_at(f, offs_param + 0x00BA) * 256
    # Data in each channel
    f.seek(offs_chan + 0x200)
    chan_data = np.fromfile(f, dtype="<u4", count=n_channels)
    # Total counts of the channels
    total_counts = np.sum(chan_data)
    # Measurement mode
    meas_mode = string_at(f, offs_param + 0xB0, 0x03)

    # Create array with the correct channel numbering
    channels = np.arange(1, n_channels + 1, 1)

    out_dic = {
        "Number of channels": n_channels,
        "Channels data": chan_data,
        "Channels": channels,
        "Total counts": total_counts,
        "Measurement mode": meas_mode,
    }

    return out_dic


def get_date_time(f, offs_param):
    """Read date and time."""

    offs_times = offs_param + 0x30 + uint16_at(f, offs_param + 0x24)

    start_time = datetime_at(f, offs_times + 0x01)
    real_time = time_at(f, offs_times + 0x09)
    live_time = time_at(f, offs_times + 0x11)

    # Convert to formated date and time
    start_time_str = time.strftime("%d-%m-%Y, %H:%M:%S", time.gmtime(start_time))

    out_dic = {
        "Real time": real_time,
        "Live time": live_time,
        "Start time": start_time_str,
    }
    return out_dic


def get_markers(f, offs_mark):
    """Read left and right markers."""

    # TODO: not working properly
    marker_left = uint32_at(f, offs_mark + 0x007A)
    marker_right = uint32_at(f, offs_mark + 0x008A)

    out_dic = {
        "Left marker": marker_left,
        "Right marker": marker_right,
    }

    return out_dic


def chan_to_energy(dic):
    """Convert channels to energy using energy calibration coefficients."""

    A = dic["Energy coefficients"]
    ch = dic["Channels"]
    energy = A[0] + A[1] * ch + A[2] * ch * ch + A[3] * ch * ch * ch

    out_dic = {"Energy": energy}

    return out_dic


def markers_integration(dic):
    # Count between left and right markers
    # TODO: check integral counts limits
    chan_data = dic["Channels data"]
    l_marker = dic["Left marker"]
    r_marker = dic["Right marker"]
    marker_counts = np.sum(chan_data[l_marker - 1 : r_marker - 1])

    out_dic = {"Counts in markers": marker_counts}

    return out_dic


###########################################################
# Format of the output text file
###########################################################


def write_to_file(filename, dic):
    """Write data to a text file."""

    with open(filename, "w") as f:
        f.write("#\n")
        f.write("# Sample name: {}\n".format(dic["Sample name"]))
        f.write("\n")

        f.write("# Sample id: {}\n".format(dic["Sample id"]))
        f.write("# Sample type: {}\n".format(dic["Sample type"]))
        f.write("# User name: {}\n".format(dic["User name"]))
        f.write("# Sample description: {}\n".format(dic["Sample description"]))
        f.write("#\n")

        f.write("# Start time: {}\n".format(dic["Start time"]))
        f.write("# Real time (s): {:.3f}\n".format(dic["Real time"]))
        f.write("# Live time (s): {:.3f}\n".format(dic["Live time"]))
        f.write("#\n")

        f.write("# Total counts: {}\n".format(dic["Total counts"]))
        f.write("#\n")

        f.write("# Left marker: {}\n".format(dic["Left marker"]))
        f.write("# Right marker: {}\n".format(dic["Right marker"]))
        f.write("# Counts: {}\n".format(dic["Counts in markers"]))
        f.write("#\n")

        f.write("# Energy calibration coefficients (E = sum(Ai * n**i))\n")
        for j, co in enumerate(dic["Energy coefficients"]):
            f.write("#    A{} = {:.6e}\n".format(j, co))
        f.write("# Energy unit: {}\n".format(dic["Energy unit"]))
        f.write("#\n")

        f.write(
            "# Shape calibration coefficients (FWHM = B0 + B1*E^(1/2)  Low Tail= B2 + B3*E)\n"
        )
        for j, co in enumerate(dic["Shape coefficients"]):
            f.write("#    B{} = {:.6e}\n".format(j, co))
        f.write("# Energy unit: {}\n".format(dic["Energy unit"]))
        f.write("#\n")

        f.write("# Channel data\n")
        f.write(
            "#     n     energy({})     counts     rate(1/s)\n".format(
                dic["Energy unit"]
            )
        )
        f.write("#" + 50 * "-" + "\n")
        for i, j, k in zip(dic["Channels"], dic["Energy"], dic["Channels data"]):
            f.write("{:4d}\t{:.3e}\t{}\t{:.3e}\n".format(i, j, k, k / dic["Live time"]))


def read_cnf_to_spect(filename):
    dict_cnf = read_cnf_file(filename, False)

    counts = dict_cnf["Channels data"]
    energy = dict_cnf["Energy"]
    livetime = dict_cnf["Live time"]

    if energy is None:
        e_units = "channels"
        spect = sp.Spectrum(counts=counts, e_units=e_units, livetime=livetime)
        spect.x = spect.channels
    else:
        e_units = dict_cnf["Energy unit"]
        spect = sp.Spectrum(
            counts=counts, energies=energy, e_units=e_units, livetime=livetime
        )
        spect.x = spect.energies

    return e_units, spect
