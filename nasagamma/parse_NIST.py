"""
Parse NIST data file
"""
from collections import defaultdict
import re
import pandas as pd
import pkg_resources


def isotopic_abundance(element):
    """
    Parameters
    ----------
    element : string
        chemical symbol e.g 'H'.

    Returns
    -------
    result : dictionary
        stable isotopes and their respective natural abundance.

    """
    file = pkg_resources.resource_filename("nasagamma", "data/Isotopes-NIST.txt")
    file_symb = pkg_resources.resource_filename(
        "nasagamma", "data/elements_symbols.csv"
    )
    # file = "../isotID_docs/Isotopes-NIST.txt"
    # file_symb = "../isotID_docs/elements_symbols.csv"
    symbols = list(pd.read_csv(file_symb)["Sym"])
    element = element.title()
    if element not in symbols:
        warning_msg = "Element not found.\nMake sure to write the element symbol only"
        return warning_msg
    # read all lines and store in memory
    with open(file, "r") as f:
        lines = f.readlines()

    result = defaultdict()
    for i, l in enumerate(lines):
        tmp = l.split()
        if element in tmp:
            Z = re.findall("\d+", lines[i - 1])[0]
            symbol = element
            isotope = re.findall("\d+", lines[i + 1])[0]
            # account for isotopically pure elements
            try:
                tmp2 = lines[i + 3].split()
                comp = [float(tmp2[-1])]
            except ValueError:
                comp = re.findall("\d.+(?=\()", lines[i + 3])
            if len(comp) != 0:
                result[Z + symbol + "-" + isotope] = float(comp[0])
    return result


def isotopic_abundance_str(element):
    "Return string of isotopic abundances"
    res = isotopic_abundance(element)
    if type(res) == str:
        return res
    keys = list(res.keys())
    vals = list(res.values())
    string = ""
    for k, v in zip(keys, vals):
        v2 = round(v * 100, 3)
        string = string + k + " : " + str(v2) + "%" + "\n"
    return string
