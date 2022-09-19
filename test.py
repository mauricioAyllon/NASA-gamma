"""
Test nasagamma modules
"""
from nasagamma import spectrum as sp
from nasagamma import peaksearch as ps
from nasagamma import peakfit as pf
from nasagamma import file_reader
import traceback
import pandas as pd

# import sys, os

# # Disable print
# def blockPrint():
#     sys.stdout = open(os.devnull, 'w')

# # Restore print
# def enablePrint():
#     sys.stdout = sys.__stdout__


file_csv = "examples/data/gui_test_data_labr_uncalibrated.csv"
file_cnf = "examples/data/2021-03-23-MGS.cnf"
file_cal = "examples/data/gui_test_data_labr.csv"
file_hpge = "examples/data/gui_test_data_hpge.csv"


def green(s):
    return "\033[1;32m%s\033[m" % s


def yellow(s):
    return "\033[1;33m%s\033[m" % s


def red(s):
    return "\033[1;31m%s\033[m" % s


def log(*m):
    print(" ".join(map(str, m)))


def log_exit(*m):
    log(red("ERROR:"), *m)
    exit(1)


def check_nasagamma():
    try:
        import nasagamma

        log(green("PASS"), "nasagamma installed")
    except ModuleNotFoundError:
        log(red("FAIL"), "nasagamma not installed")


def check_csv_reader():
    try:
        e_units1, _ = file_reader.read_csv_file(file_csv)
        e_units2, _ = file_reader.read_csv_file(file_cal)
        if e_units1 == "channels" and e_units2 == "MeV":
            log(green("PASS"), "csv file reader OK")
        else:
            log(red("FAIL"), "csv file reader unable to read units")
    except:
        log(red("FAIL"), "csv file reader unable to run")


def check_cnf_reader():
    try:
        e_units, spect = file_reader.read_cnf_to_spect(file_cnf)
        if e_units == "keV" and spect.counts.shape[0] > 0:
            log(green("PASS"), "cnf file reader OK")
    except:
        log(red("FAIL"), "cnf file reader unable to run")


def check_spectrum():
    try:
        df = pd.read_csv(file_csv)
        cts_np = df["counts"].to_numpy()
        spect = sp.Spectrum(counts=cts_np)
        if len(spect.counts) == 0:
            log(red("FAIL"), "Empty counts in spectrum")
        else:
            log(green("PASS"), "Spectrum class OK")
    except:
        log(red("FAIL"), "Cannot instantiate a spectrum object")


def check_peaksearch():
    try:
        df = pd.read_csv(file_csv)
        cts_np = df["counts"].to_numpy()
        spect_csv = sp.Spectrum(counts=cts_np)
        _, spect_cnf = file_reader.read_cnf_to_spect(file_cnf)
        # without range
        try:
            search1 = ps.PeakSearch(spect_csv, ref_x=420, ref_fwhm=20)
            search2 = ps.PeakSearch(spect_cnf, ref_x=420, ref_fwhm=20)
            log(green("PASS"), "Peaksearch class ok before xrange")
        except:
            log(red("FAIL"), "Peaksearch class failed before testing xrange")
        # with defined range
        try:
            search3 = ps.PeakSearch(
                spect_csv, ref_x=420, ref_fwhm=20, xrange=[500, 800]
            )
            search4 = ps.PeakSearch(
                spect_cnf, ref_x=420, ref_fwhm=20, xrange=[523, 1400]
            )
            log(green("PASS"), "Peaksearch class ok after xrange")
        except:
            log(red("FAIL"), "Peaksearch class failed after testing xrange")
    except:
        log(red("FAIL"), "Cannot instantiate a peaksearch object")
    return search1, search2, search3, search4


def check_peakfit():
    search1, search2, search3, search4 = check_peaksearch()
    try:
        # blockPrint()
        fit1 = pf.PeakFit(search1, xrange=[600, 800], bkg="poly1")
        fit2 = pf.PeakFit(search2, xrange=[1080, 1400], bkg="poly1")
        fit3 = pf.PeakFit(search3, xrange=[700, 745], bkg="poly1")
        fit4 = pf.PeakFit(search4, xrange=[550, 700], bkg="poly1")
        # enablePrint()
        log(green("PASS"), "Peakfit class ok")
    except:
        log(red("FAIL"), "Cannot instantiate a peakfit object")


def main():
    try:
        check_nasagamma()
        check_csv_reader()
        check_cnf_reader()
        check_spectrum()
        check_peakfit()
    except Exception:
        log_exit(traceback.format_exc())


if __name__ == "__main__":
    main()


# import os
# import sys
# import time
# import traceback
# import project1 as p1
# import numpy as np

# verbose = False

# def green(s):
#     return '\033[1;32m%s\033[m' % s

# def yellow(s):
#     return '\033[1;33m%s\033[m' % s

# def red(s):
#     return '\033[1;31m%s\033[m' % s

# def log(*m):
#     print(" ".join(map(str, m)))

# def log_exit(*m):
#     log(red("ERROR:"), *m)
#     exit(1)


# def check_real(ex_name, f, exp_res, *args):
#     try:
#         res = f(*args)
#     except NotImplementedError:
#         log(red("FAIL"), ex_name, ": not implemented")
#         return True
#     if not np.isreal(res):
#         log(red("FAIL"), ex_name, ": does not return a real number, type: ", type(res))
#         return True
#     if res != exp_res:
#         log(red("FAIL"), ex_name, ": incorrect answer. Expected", exp_res, ", got: ", res)
#         return True


# def equals(x, y):
#     if type(y) == np.ndarray:
#         return (x == y).all()
#     return x == y

# def check_tuple(ex_name, f, exp_res, *args, **kwargs):
#     try:
#         res = f(*args, **kwargs)
#     except NotImplementedError:
#         log(red("FAIL"), ex_name, ": not implemented")
#         return True
#     if not type(res) == tuple:
#         log(red("FAIL"), ex_name, ": does not return a tuple, type: ", type(res))
#         return True
#     if not len(res) == len(exp_res):
#         log(red("FAIL"), ex_name, ": expected a tuple of size ", len(exp_res), " but got tuple of size", len(res))
#         return True
#     if not all(equals(x, y) for x, y in zip(res, exp_res)):
#         log(red("FAIL"), ex_name, ": incorrect answer. Expected", exp_res, ", got: ", res)
#         return True

# def check_array(ex_name, f, exp_res, *args):
#     try:
#         res = f(*args)
#     except NotImplementedError:
#         log(red("FAIL"), ex_name, ": not implemented")
#         return True
#     if not type(res) == np.ndarray:
#         log(red("FAIL"), ex_name, ": does not return a numpy array, type: ", type(res))
#         return True
#     if not len(res) == len(exp_res):
#         log(red("FAIL"), ex_name, ": expected an array of shape ", exp_res.shape, " but got array of shape", res.shape)
#         return True
#     if not all(equals(x, y) for x, y in zip(res, exp_res)):
#         log(red("FAIL"), ex_name, ": incorrect answer. Expected", exp_res, ", got: ", res)
#         return True

# def check_list(ex_name, f, exp_res, *args):
#     try:
#         res = f(*args)
#     except NotImplementedError:
#         log(red("FAIL"), ex_name, ": not implemented")
#         return True
#     if not type(res) == list:
#         log(red("FAIL"), ex_name, ": does not return a list, type: ", type(res))
#         return True
#     if not len(res) == len(exp_res):
#         log(red("FAIL"), ex_name, ": expected a list of size ", len(exp_res), " but got list of size", len(res))
#         return True
#     if not all(equals(x, y) for x, y in zip(res, exp_res)):
#         log(red("FAIL"), ex_name, ": incorrect answer. Expected", exp_res, ", got: ", res)
#         return True


# def check_get_order():
#     ex_name = "Get order"
#     if check_list(
#             ex_name, p1.get_order,
#             [0], 1):
#         log("You should revert `get_order` to its original implementation for this test to pass")
#         return
#     if check_list(
#             ex_name, p1.get_order,
#             [1, 0], 2):
#         log("You should revert `get_order` to its original implementation for this test to pass")
#         return
#     log(green("PASS"), ex_name, "")


# def check_hinge_loss_single():
#     ex_name = "Hinge loss single"

#     feature_vector = np.array([1, 2])
#     label, theta, theta_0 = 1, np.array([-1, 1]), -0.2
#     exp_res = 1 - 0.8
#     if check_real(
#             ex_name, p1.hinge_loss_single,
#             exp_res, feature_vector, label, theta, theta_0):
#         return
#     log(green("PASS"), ex_name, "")


# def check_hinge_loss_full():
#     ex_name = "Hinge loss full"

#     feature_vector = np.array([[1, 2], [1, 2]])
#     label, theta, theta_0 = np.array([1, 1]), np.array([-1, 1]), -0.2
#     exp_res = 1 - 0.8
#     if check_real(
#             ex_name, p1.hinge_loss_full,
#             exp_res, feature_vector, label, theta, theta_0):
#         return

#     log(green("PASS"), ex_name, "")


# def check_perceptron_single_update():
#     ex_name = "Perceptron single update"

#     feature_vector = np.array([1, 2])
#     label, theta, theta_0 = 1, np.array([-1, 1]), -1.5
#     exp_res = (np.array([0, 3]), -0.5)
#     if check_tuple(
#             ex_name, p1.perceptron_single_step_update,
#             exp_res, feature_vector, label, theta, theta_0):
#         return

#     feature_vector = np.array([1, 2])
#     label, theta, theta_0 = 1, np.array([-1, 1]), -1
#     exp_res = (np.array([0, 3]), 0)
#     if check_tuple(
#             ex_name + " (boundary case)", p1.perceptron_single_step_update,
#             exp_res, feature_vector, label, theta, theta_0):
#         return

#     log(green("PASS"), ex_name, "")


# def check_perceptron():
#     ex_name = "Perceptron"

#     feature_matrix = np.array([[1, 2]])
#     labels = np.array([1])
#     T = 1
#     exp_res = (np.array([1, 2]), 1)
#     if check_tuple(
#             ex_name, p1.perceptron,
#             exp_res, feature_matrix, labels, T):
#         return

#     feature_matrix = np.array([[1, 2], [-1, 0]])
#     labels = np.array([1, 1])
#     T = 1
#     exp_res = (np.array([0, 2]), 2)
#     if check_tuple(
#             ex_name, p1.perceptron,
#             exp_res, feature_matrix, labels, T):
#         return

#     feature_matrix = np.array([[1, 2]])
#     labels = np.array([1])
#     T = 2
#     exp_res = (np.array([1, 2]), 1)
#     if check_tuple(
#             ex_name, p1.perceptron,
#             exp_res, feature_matrix, labels, T):
#         return

#     feature_matrix = np.array([[1, 2], [-1, 0]])
#     labels = np.array([1, 1])
#     T = 2
#     exp_res = (np.array([0, 2]), 2)
#     if check_tuple(
#             ex_name, p1.perceptron,
#             exp_res, feature_matrix, labels, T):
#         return

#     log(green("PASS"), ex_name, "")


# def check_average_perceptron():
#     ex_name = "Average perceptron"

#     feature_matrix = np.array([[1, 2]])
#     labels = np.array([1])
#     T = 1
#     exp_res = (np.array([1, 2]), 1)
#     if check_tuple(
#             ex_name, p1.average_perceptron,
#             exp_res, feature_matrix, labels, T):
#         return

#     feature_matrix = np.array([[1, 2], [-1, 0]])
#     labels = np.array([1, 1])
#     T = 1
#     exp_res = (np.array([-0.5, 1]), 1.5)
#     if check_tuple(
#             ex_name, p1.average_perceptron,
#             exp_res, feature_matrix, labels, T):
#         return

#     feature_matrix = np.array([[1, 2]])
#     labels = np.array([1])
#     T = 2
#     exp_res = (np.array([1, 2]), 1)
#     if check_tuple(
#             ex_name, p1.average_perceptron,
#             exp_res, feature_matrix, labels, T):
#         return

#     feature_matrix = np.array([[1, 2], [-1, 0]])
#     labels = np.array([1, 1])
#     T = 2
#     exp_res = (np.array([-0.25, 1.5]), 1.75)
#     if check_tuple(
#             ex_name, p1.average_perceptron,
#             exp_res, feature_matrix, labels, T):
#         return

#     log(green("PASS"), ex_name, "")


# def check_pegasos_single_update():
#     ex_name = "Pegasos single update"

#     feature_vector = np.array([1, 2])
#     label, theta, theta_0 = 1, np.array([-1, 1]), -1.5
#     L = 0.2
#     eta = 0.1
#     exp_res = (np.array([-0.88, 1.18]), -1.4)
#     if check_tuple(
#             ex_name, p1.pegasos_single_step_update,
#             exp_res,
#             feature_vector, label, L, eta, theta, theta_0):
#         return

#     feature_vector = np.array([1, 1])
#     label, theta, theta_0 = 1, np.array([-1, 1]), 1
#     L = 0.2
#     eta = 0.1
#     exp_res = (np.array([-0.88, 1.08]), 1.1)
#     if check_tuple(
#             ex_name +  " (boundary case)", p1.pegasos_single_step_update,
#             exp_res,
#             feature_vector, label, L, eta, theta, theta_0):
#         return

#     feature_vector = np.array([1, 2])
#     label, theta, theta_0 = 1, np.array([-1, 1]), -2
#     L = 0.2
#     eta = 0.1
#     exp_res = (np.array([-0.88, 1.18]), -1.9)
#     if check_tuple(
#             ex_name, p1.pegasos_single_step_update,
#             exp_res,
#             feature_vector, label, L, eta, theta, theta_0):
#         return

#     log(green("PASS"), ex_name, "")


# def check_pegasos():
#     ex_name = "Pegasos"

#     feature_matrix = np.array([[1, 2]])
#     labels = np.array([1])
#     T = 1
#     L = 0.2
#     exp_res = (np.array([1, 2]), 1)
#     if check_tuple(
#             ex_name, p1.pegasos,
#             exp_res, feature_matrix, labels, T, L):
#         return

#     feature_matrix = np.array([[1, 1], [1, 1]])
#     labels = np.array([1, 1])
#     T = 1
#     L = 1
#     exp_res = (np.array([1-1/np.sqrt(2), 1-1/np.sqrt(2)]), 1)
#     if check_tuple(
#             ex_name, p1.pegasos,
#             exp_res, feature_matrix, labels, T, L):
#         return

#     log(green("PASS"), ex_name, "")


# def check_classify():
#     ex_name = "Classify"

#     feature_matrix = np.array([[1, 1], [1, 1], [1, 1]])
#     theta = np.array([1, 1])
#     theta_0 = 0
#     exp_res = np.array([1, 1, 1])
#     if check_array(
#             ex_name, p1.classify,
#             exp_res, feature_matrix, theta, theta_0):
#         return

#     feature_matrix = np.array([[-1, 1]])
#     theta = np.array([1, 1])
#     theta_0 = 0
#     exp_res = np.array([-1])
#     if check_array(
#             ex_name + " (boundary case)", p1.classify,
#             exp_res, feature_matrix, theta, theta_0):
#         return

#     log(green("PASS"), ex_name, "")

# def check_classifier_accuracy():
#     ex_name = "Classifier accuracy"

#     train_feature_matrix = np.array([[1, 0], [1, -1], [2, 3]])
#     val_feature_matrix = np.array([[1, 1], [2, -1]])
#     train_labels = np.array([1, -1, 1])
#     val_labels = np.array([-1, 1])
#     exp_res = 1, 0
#     T=1
#     if check_tuple(
#             ex_name, p1.classifier_accuracy,
#             exp_res,
#             p1.perceptron,
#             train_feature_matrix, val_feature_matrix,
#             train_labels, val_labels,
#             T=T):
#         return

#     train_feature_matrix = np.array([[1, 0], [1, -1], [2, 3]])
#     val_feature_matrix = np.array([[1, 1], [2, -1]])
#     train_labels = np.array([1, -1, 1])
#     val_labels = np.array([-1, 1])
#     exp_res = 1, 0
#     T=1
#     L=0.2
#     if check_tuple(
#             ex_name, p1.classifier_accuracy,
#             exp_res,
#             p1.pegasos,
#             train_feature_matrix, val_feature_matrix,
#             train_labels, val_labels,
#             T=T, L=L):
#         return

#     log(green("PASS"), ex_name, "")

# def check_bag_of_words():
#     ex_name = "Bag of words"

#     texts = [
#         "He loves to walk on the beach",
#         "There is nothing better"]

#     try:
#         res = p1.bag_of_words(texts)
#     except NotImplementedError:
#         log(red("FAIL"), ex_name, ": not implemented")
#         return
#     if not type(res) == dict:
#         log(red("FAIL"), ex_name, ": does not return a tuple, type: ", type(res))
#         return

#     vals = sorted(res.values())
#     exp_vals = list(range(len(res.keys())))
#     if not vals == exp_vals:
#         log(red("FAIL"), ex_name, ": wrong set of indices. Expected: ", exp_vals, " got ", vals)
#         return

#     log(green("PASS"), ex_name, "")

#     keys = sorted(res.keys())
#     exp_keys = ['beach', 'better', 'he', 'is', 'loves', 'nothing', 'on', 'the', 'there', 'to', 'walk']
#     stop_keys = ['beach', 'better', 'loves', 'nothing', 'walk']

#     if keys == exp_keys:
#         log(yellow("WARN"), ex_name, ": does not remove stopwords:", [k for k in keys if k not in stop_keys])
#     elif keys == stop_keys:
#         log(green("PASS"), ex_name, " stopwords removed")
#     else:
#         log(red("FAIL"), ex_name, ": keys are missing:", [k for k in stop_keys if k not in keys], " or are not unexpected:", [k for k in keys if k not in stop_keys])


# def check_extract_bow_feature_vectors():
#     ex_name = "Extract bow feature vectors"
#     texts = [
#         "He loves her ",
#         "He really really loves her"]
#     keys = ["he", "loves", "her", "really"]
#     dictionary = {k:i for i, k in enumerate(keys)}
#     exp_res = np.array(
#         [[1, 1, 1, 0],
#         [1, 1, 1, 1]])
#     non_bin_res = np.array(
#         [[1, 1, 1, 0],
#         [1, 1, 1, 2]])


#     try:
#         res = p1.extract_bow_feature_vectors(texts, dictionary)
#     except NotImplementedError:
#         log(red("FAIL"), ex_name, ": not implemented")
#         return

#     if not type(res) == np.ndarray:
#         log(red("FAIL"), ex_name, ": does not return a numpy array, type: ", type(res))
#         return
#     if not len(res) == len(exp_res):
#         log(red("FAIL"), ex_name, ": expected an array of shape ", exp_res.shape, " but got array of shape", res.shape)
#         return

#     log(green("PASS"), ex_name)

#     if (res == exp_res).all():
#         log(yellow("WARN"), ex_name, ": uses binary indicators as features")
#     elif (res == non_bin_res).all():
#         log(green("PASS"), ex_name, ": correct non binary features")
#     else:
#         log(red("FAIL"), ex_name, ": unexpected feature matrix")
#         return

# def main():
#     log(green("PASS"), "Import project1")
#     try:
#         check_get_order()
#         check_hinge_loss_single()
#         check_hinge_loss_full()
#         check_perceptron_single_update()
#         check_perceptron()
#         check_average_perceptron()
#         check_pegasos_single_update()
#         check_pegasos()
#         check_classify()
#         check_classifier_accuracy()
#         check_bag_of_words()
#         check_extract_bow_feature_vectors()
#     except Exception:
#         log_exit(traceback.format_exc())

# if __name__ == "__main__":
#     main()
