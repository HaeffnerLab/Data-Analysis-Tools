"""
Module for importing labrad-saved data from the experimental pc
Functions get_data, get_base_data, and get_parameters should work regardless of whether data was saved using SSV1 or SSV2.
Functions get_photon_counts and get_hist only work with data saved from SSV2.
Assumes some stuff about the directory structure of how the data is saved; depending on how your script scanner saves your data, results may vary.
"""

import numpy as np
import os
from labrad_data_tools_config import SCAN_LISTS_DIRECTORY, DATA_DIRECTORY, MULTID_SCAN_MAP


##############################################################################################
# MAIN FUNCTIONS
##############################################################################################

def get_data(date, scantime, exclude=[]):
    """
    Main function to use for importing data
    date and scantime are strings indicating which scan to use. Formatting examples:
    
    get_data('20180606', '1340_50')
    get_data('20180606', '1340.50', exclude=[-1, -2])

    Returns a tuple (x, y) of arrays containing the data. If there are multiple y-data sets then y is a 2-dimensional array.

    Input "scantime" can also be a list. This concatenates the datasets of several scans, and is intended to be used for datasets which span multiple (usually consecutive (but not necessarily)) scans.

    The optional argument "exclude" is a list of indices in the scan to exclude from being returned (e.g. bad data points that you don't want to inlcude).
    To exclude data points when using N time scans, "exclude" must be a list of length N, each element being a list containing the indices to exclude for the corresponding time scan. 
    """

    # Allow concatenation of data if input "scantime" is a list
    if isinstance(scantime, list):
        # Check that the argument "exclude" has been used correctly
        if exclude is not []:
            if not hasattr(exclude, "__len__"):
                raise ValueError('Argument "exclude" must be array-like.')
            if (len(exclude) != 0) and (len(exclude) != len(scantime)):
                raise ValueError('Argument "exclude" must be array-like with length equal to the number of time scans in the list of times. len(scantime) = {0}, len(exclude) = {1}'.format(len(scantime), len(exclude)))
        x = np.array([])
        y = np.array([])
        for (i, t) in enumerate(scantime):
            t = t.replace('.', '_')        # Change a period in the time string to an underscore
            if len(exclude) != 0:
                (xi, yi) = _get_data_single(date, t, exclude=exclude[i])
            else:
                (xi, yi) = _get_data_single(date, t, exclude=exclude)
            x = np.concatenate((x, xi))
            y = np.concatenate((y, yi)) if y.size else yi
        return (x, y)

    else:
        scantime = scantime.replace('.', '_')        # Change a period in the time string to an underscore
        (x, y) = _get_data_single(date, scantime, exclude=exclude)
        return (x, y)


def get_data_manual(directory, exclude=[]):
    """
    Returns x and y data for an experiment specified by a directory where the data is saved. Identical to _get_data_single except for the arguments.
    Useful for cases where scan_lists failed to index the dataset for some reason.
    """

    # Check the directory for csv files. If there are none, raise an error
    file_path = None
    for filename in os.listdir(directory):
        if filename.endswith('.csv') and filename.startswith('00001'):
            file_path = directory + filename
            break
    if file_path is None:
        raise IOError('No .csv files found in {}.'.format(directory))

    # Extract the data
    data_arr = np.loadtxt(file_path, delimiter=',')
    if data_arr.ndim == 1:
        # If only 1 data point, expant into a 2d array
        data_arr = np.expand_dims(data_arr, axis=0)

    # Sort the data into x and y arrays
    x = data_arr[:, 0]
    y = data_arr[:, 1:]

    # Remove data points to be excluded
    x = _remove_excludes(x, exclude)
    y = _remove_excludes(y, exclude)

    # Flatten the y array if there is only one set of y data
    if np.shape(y)[1] == 1:
        y = y.flatten()

    return (x, y)


def get_raw_number_dark(date, scantime, exclude=[]):
    """
    Input "scantime" can be a list. This concatenates the datasets of several scans, and is intended to be used for datasets which span multiple (usually consecutive) scans.

    The optional argument "exclude" is a list of indices in the scan to exclude from being returned (e.g. bad data points that you don't want to inlcude).
    To exclude data points when using N time scans, "exclude" must be a list of length N, each element being a list containing the indices to exclude for the corresponding time scan. 
    """

    # Allow concatenation of data if input "time" is a list
    if isinstance(scantime, list):
        # Check that the argument "exclude" has been used correctly
        if exclude is not []:
            if not hasattr(exclude, "__len__"):
                raise ValueError('Argument "exclude" must be array-like.')
            if (len(exclude) != 0) and (len(exclude) != len(scantime)):
                raise ValueError('Argument "exclude" must be array-like with length equal to the number of time scans in the list of times. len(scantime) = {0}, len(exclude) = {1}'.format(len(scantime), len(exclude)))
        x = np.array([])
        y = np.array([])
        for (i, t) in enumerate(scantime):
            t = t.replace('.', '_')        # Change a period in the time string to an underscore
            if len(exclude) != 0:
                (xi, yi) = _get_raw_number_dark_single(date, t, exclude=exclude[i])
            else:
                (xi, yi) = _get_raw_number_dark_single(date, t, exclude=exclude)
            x = np.concatenate((x, xi))
            y = np.concatenate((y, yi)) if y.size else yi
        return (x, y)

    else:
        scantime = scantime.replace('.', '_')        # Change a period in the time string to an underscore
        (x, y) = _get_raw_number_dark_single(date, scantime, exclude=exclude)
        return (x, y)


def get_consecutive_scantimes(date, first_scantime, last_scantime, exclude=[]):
    """
    Returns scantimes of multiple scans which have been taken consecutively.
    Is aware of experiment names, so doesn't count other types of scans in the set of consecutive scans (e.g. if you calibrated lines in between two of the scans, you don't have to worry about that)
    The scans first_scantime and last_scantime MUST be of the same experiment.

    Arguments:
    date: Date of the set of scans
    first_scantime: scantime of the first of the consecutive set
    last_scantime: scantime of the last of the consecutive set
    exclude: Indices of scans to exclude
    
    Returns:
    scantimes: List of scantimes from the result
    """

    first_scantime = first_scantime.replace('.', '_')
    last_scantime = last_scantime.replace('.', '_')

    # Get list of all scan times from that day
    try:
        with open(SCAN_LISTS_DIRECTORY + '{0}/scan_list_{0}'.format(date)) as f:
            lines = f.readlines()
    except IOError:
        raise ValueError('No scans found on date {}'.format(date))
    all_scantimes = [line[:7] for line in lines]
    
    # Remove duplicates and sort
    all_scantimes = list(set(all_scantimes))
    all_scantimes.sort()

    # Pick out the scans only from this experiment
    experiment_name = get_experiment_name(date, first_scantime)
    this_experiment_scantimes = [st for st in all_scantimes if get_experiment_name(date, st) == experiment_name]

    # Find the all scans with the same experiment name from the first to the last one
    i_first = this_experiment_scantimes.index(first_scantime)
    i_last = this_experiment_scantimes.index(last_scantime)
    scantimes = this_experiment_scantimes[i_first:i_last+1]

    # Remove scans to exclude
    scantimes = _remove_excludes(scantimes, exclude)

    return scantimes


def get_base_scantimes(date, superscantime, exclude=[], return_xvalues=False):
    """
    Returns list of scan times corresponding to the base scans of a superscan
    (e.g. the scantimes of the phase measurements of a Ramsey with contrast measurement)
    If return_xvalues=True, then returns a list of 2-tuples, the first element being the corresponding x-value of the superscan, and the second being the scantime.
    """

    # Allow concatenation of data if input "time" is a list
    if isinstance(superscantime, list):
        # Check that the argument "exclude" has been used correctly
        if exclude is not []:
            if not hasattr(exclude, "__len__"):
                raise ValueError('Argument "exclude" must be array-like.')
            if (len(exclude) != 0) and (len(exclude) != len(superscantime)):
                raise ValueError('Argument "exclude" must be array-like with length equal to the number of time scans in the list of times. len(superscantime) = {0}, len(exclude) = {1}'.format(len(superscantime), len(exclude)))
        xvalues = np.array([])
        subscantimes = []
        for (i, sst_i) in enumerate(superscantime):
            sst_i = sst_i.replace('.', '_')        # Change a period in the time string to an underscore
            if len(exclude) != 0:
                result_i = _get_base_scantimes_single(date, sst_i, exclude=exclude[i], return_xvalues=return_xvalues)
            else:
                result_i = _get_base_scantimes_single(date, sst_i, exclude=exclude, return_xvalues=return_xvalues)
            if return_xvalues:
                (xvalues_i, subscantimes_i) = result_i
                xvalues = np.concatenate((xvalues, xvalues_i))
            else:
                subscantimes_i = result_i
            subscantimes += subscantimes_i
    else:
        superscantime = superscantime.replace('.', '_')        # Change a period in the time string to an underscore
        result = _get_base_scantimes_single(date, superscantime, exclude=exclude, return_xvalues=return_xvalues)
        if return_xvalues:
            (xvalues, subscantimes) = result
        else:
            subscantimes = result

    if return_xvalues:
        return (xvalues, subscantimes)
    else:
        return subscantimes


def get_photon_counts(date, scantime, exclude=[]):
    """
    Main function to use for getting photon counts
    Only works with new data

    date and scantime are strings indicating which scan to use. Formatting example:
    
    get_photon_counts('20180606', '1340_50')
    get_photon_counts('20180606', '1340.50', exclude=[-1, -2])
    
    Returns an 2-D ndarray of shape (n_data_points, n_reps) containing the photon counts, where n_data_points is the number of data points in the scan and n_reps is the number of experiment repetitions for each data point

    Input "scantime" can also be a list. This concatenates the datasets of several scans, and is intended to be used for datasets which span multiple (usually consecutive) scans.

    The optional argument "exclude" is a list of indices in the scan to exclude from being returned (e.g. bad data points that you don't want to inlcude).
    To exclude data points when using N time scans, "exclude" must be a list of length N, each element being a list containing the indices to exclude for the corresponding time scan. 
    """
    
    # Allow concatenation of data if input "time" is a list
    if isinstance(scantime, list):
        # Check that the argument "exclude" has been used correctly
        if exclude is not []:
            if not hasattr(exclude, "__len__"):
                raise ValueError('Argument "exclude" must be array-like.')
            if (len(exclude) != 0) and (len(exclude) != len(scantime)):
                raise ValueError('Argument "exclude" must be array-like with length equal to the number of time scans in the list of times. len(scantime) = {0}, len(exclude) = {1}'.format(len(scantime), len(exclude)))
        counts_arr = np.array([])
        for (i, t) in enumerate(scantime):
            t = t.replace('.', '_')        # Change a period in the time string to an underscore
            if len(exclude) != 0:
                counts_arr_i = _get_photon_counts_single(date, t, exclude=exclude[i])
            else:
                counts_arr_i = _get_photon_counts_single(date, t, exclude=exclude)
            counts_arr = np.concatenate((counts_arr, counts_arr_i), axis=0) if counts_arr.size else counts_arr_i
        return counts_arr

    else:
        scantime = scantime.replace('.', '_')        # Change a period in the time string to an underscore
        counts_arr = _get_photon_counts_single(date, scantime, exclude=exclude)
        return counts_arr


def get_hist(date, scantime, histNum):
    """
    Imports a saved histogram of photon counts
    Only works with new data

    date and scantime are strings indicating which scan to use. histNum is the number of the histogram to look at. Formatting example:
    
    get_hist('20180606', '1340_50', [0, 1])

    Returns a tuple (bins, instances) of arrays containing the histogram.

    Input "histNum" can also be a list. This adds the listed histograms of that scan together.
    """

    # Change a period in the time string to an underscore
    scantime = scantime.replace('.', '_')

    # Allow concatenation of data if input "histNum" is a list
    if isinstance(histNum, list):
        bins      = np.array([])
        instances = np.array([])
        for (i, hist) in enumerate(histNum):
            (bins, instances_i) = _get_hist_single(date, scantime, histNum[i])
            instances = instances+instances_i if instances.size else instances_i
        return (bins, instances)

    else:
        scantime = scantime.replace('.', '_')        # Change a period in the time string to an underscore
        (bins, instances) = _get_hist_single(date, scantime, histNum)
        return (bins, instances)


def get_parameters(date, scantime, search=None):
    """
    Loads and returns the pickle file containing all the parameters.
    Works with both old and new data

    Returns an OrderedDict of parameters.
    Use the search argument to filter parameters: Only parameters whose name contains the string defined by "search" will be returned.
        e.g. can use this to see only Doppler cooling parameters by using search='DopplerCooling'
        Not limited to filtering parameter collections; can further filter e.g. with search='DopplerCooling.doppler_cooling_'
    """

    import pickle
    from collections import OrderedDict

    scantime = scantime.replace('.', '_')
    directory = get_data_directory(date, scantime)
    param_prefix = '00001'

    # Check for .pickle file
    file_path = None
    for filename in os.listdir(directory):
        if filename.endswith('.pickle') and filename.startswith(param_prefix):
            file_path = directory + filename
            break

    # Extract data differently depending on whether the dataset has an associated .pickle file
    if file_path is not None:
        # .pickle file found; use this
        # Extract the data
        with open(file_path, 'rb') as f:
            unsorted_dict = pickle.load(f, encoding='latin1')
        f.close()
    else:
        # .pickle file not found; instead use the .ini file
        for filename in os.listdir(directory):
            if filename.endswith('.ini') and filename.startswith(param_prefix):
                file_path = directory + filename
                break
        with open(file_path, 'r') as f:
            lines = f.readlines()
        # Now parse the lines of the .ini file
        unsorted_dict = {}
        for i in range(len(lines)-1):
            thisline = lines[i]
            nextline = lines[i+1]
            if thisline.startswith('label') and nextline.startswith('data'):
                label = thisline.split(' = ')[1][:-1]
                data = nextline.split(' = ')[1][:-1]
                unsorted_dict[label] = data

    parameters = OrderedDict(sorted(unsorted_dict.items()))
    if search is not None:
        parameters = OrderedDict([(p, parameters[p]) for p in parameters if p.find(search) >= 0])

    return parameters


def get_pmt_counts(number):
    """Returns x and y data for PMT count data."""

    # Define directories
    pmt_dir = DATA_DIRECTORY + 'PMT Counts.dir/'
    file_path = pmt_dir + '{} - PMT Counts.csv'.format(str(number).zfill(5))
    
    # Extract the data
    pmt_data = np.loadtxt(file_path, delimiter=',')

    # Sort the data into x and y arrays
    x = pmt_data[:, 0]
    y = pmt_data[:, 1]

    return (x, y)


def get_data_directory(date, scantime):
    # Allow for period instead of underscore in scantime
    scantime = scantime.replace('.', '_')
    
    # Check that folder for specified date exists in scan_lists
    try:
        with open(SCAN_LISTS_DIRECTORY + '{0}/scan_list_{0}'.format(date)) as f:
            lines = f.readlines()
    except IOError:
        raise ValueError('No scans found on date {}'.format(date))
    
    # Create list of all scan times on given date
    all_scantimes = [line[:7] for line in lines]

    # Determine which index corresponds to desired time
    try:
        # Get the highest indexed item of this scantime (seems to be necessary in some cases and harmless in others)
        index = len(all_scantimes) - all_scantimes[::-1].index(scantime) - 1
    except ValueError:
        raise ValueError('No scan at {0} found on {1}.'.format(scantime, date))

    # Strip off time and newline
    # These last 2 lines are a bandaid for having data in a new place
    data_directory = lines[index][8:][:-1] + '/'
    return data_directory


def get_experiment_name(date, scantime):
    """
    Extract the name of the experiment folder for the given scantime
    """

    # Bandaid for case when I'm trying to plot a scan made up of multiple different ones. Just take the first one
    if isinstance(scantime, list):
        scantime = scantime[0]

    # Find the full line in the scan list corresponding to the scan of interest
    scantime = scantime.replace('.', '_')
    try:
        with open(SCAN_LISTS_DIRECTORY + '{0}/scan_list_{0}'.format(date)) as f:
            lines = f.readlines()
    except IOError:
        raise ValueError('No scans found on date {}'.format(date))
    line = [el for el in lines if el[:7]==scantime][0]

    folders = line.split('.dir/')
    # The experiment name is either the second or third element of the array "folders", depending on how the data was saved. The other one is the date.
    # Use the date to figure out which it is by seeing which one isn't the date (uses year only to allow backwards compatbility with old way of saving dates)
    if folders[1][:4] == date[:4]:
        return folders[2]
    elif folders[2][:4] == date[:4]:
        return folders[1]
    else:
        ValueError('Error parsing the scan_lists for name of experiment {0}/{1}.'.format(date, scantime))


##############################################################################################
# HELPER FUNCTIONS
##############################################################################################

def _get_data_single(date, scantime, exclude=[]):
    """Returns x and y data for an experiment specified by date and scan time."""

    directory = get_data_directory(date, scantime)

    # Check the directory for csv files. If there are none, raise an error
    file_path = None
    for filename in os.listdir(directory):
        if filename.endswith('.csv') and filename.startswith('00001'):
            file_path = directory + filename
            break
    if file_path is None:
        raise IOError('No .csv files found in {}.'.format(directory))

    # Extract the data
    data_arr = np.loadtxt(file_path, delimiter=',')
    if data_arr.ndim == 1:
        # If only 1 data point, expant into a 2d array
        data_arr = np.expand_dims(data_arr, axis=0)

    # Sort the data into x and y arrays
    x = data_arr[:, 0]
    y = data_arr[:, 1:]

    # Remove data points to be excluded
    x = _remove_excludes(x, exclude)
    y = _remove_excludes(y, exclude)

    # Flatten the y array if there is only one set of y data
    if np.shape(y)[1] == 1:
        y = y.flatten()

    return (x, y)


def _get_raw_number_dark_single(date, scantime, exclude=[]):
    """
    Infers "number dark" for each datapoint from raw PMT counts and set photon count thresholds
    Returns (x, y), where x is the x-axis data and y is n-ion excitation probabilities
    For N ions (inferred from length of threshold_list), y-data has N+1 columns. Column j is the probability that j ions were found dark (excited)
    
    Assumes photon count thresholds were set appropriately in ParameterVault (does not do any fancy fitting of histograms or anything like that).
    """

    # Get x data
    (x, _) = get_data(date, scantime)

    # Check the directory for readout data. If not present, raise an error
    directory = get_data_directory(date, scantime)
    file_path = None
    for filename in os.listdir(directory):
        if filename.endswith('Readouts.csv'):
            file_path = directory + filename
            break
    if file_path is None:
        raise IOError('No readout data found in {}.'.format(directory))

    # Extract the raw readout data
    raw = np.loadtxt(file_path, delimiter=',')

    # Find the photon count thresholds
    threshold_list_strings = get_parameters(date, scantime)['StateReadout.threshold_list'].split(',')
    threshold_list = [0] + [int(count) for count in threshold_list_strings]
    n_ions = len(threshold_list)-1

    repeats = int(get_parameters(date, scantime)['StateReadout.repeat_each_measurement'])

    y = np.zeros((len(x), n_ions+1))
    for i in range(len(x)):
        # Just as a sanity check, make sure data point counts are as expected
        raw_x = raw[i*repeats:(i+1)*repeats, 0]
        if not all(raw_x == i):
            raise Exception("Raw PMT counts not stored as expected: each data point should have {} repetitions".format(repeats))

        y_i = [0]*(n_ions+1)
        raw_y = raw[i*repeats:(i+1)*repeats, 1]
        # Add this photon count to the bin corresponding to having j ions dark
        for count in raw_y:
            for j in range(n_ions):
                if threshold_list[j] <= count < threshold_list[j+1]:
                    y_i[j] += 1
            if count >= threshold_list[n_ions]:
                y_i[n_ions] += 1

        y_i.reverse()
        y[i,:] = np.array(y_i)/float(repeats)

    # Remove y data points to be excluded
    x = _remove_excludes(x, exclude)
    y = _remove_excludes(y, exclude)

    return (x, y)


def _get_base_scantimes_single(date, superscantime, exclude=[], return_xvalues=False):

    superscantime = superscantime.replace('.', '_')
    
    # Get list of all scan times from that day
    try:
        with open(SCAN_LISTS_DIRECTORY + '{0}/scan_list_{0}'.format(date)) as f:
            lines = f.readlines()
    except IOError:
        raise ValueError('No scans found on date {}'.format(date))
    all_scantimes = [line[:7] for line in lines]
    
    # Remove duplicates and sort
    all_scantimes = list(set(all_scantimes))
    all_scantimes.sort()

    # Prune "all_scantimes" to include only those whose experiment type are those which are subscans of the superscan (and the superscan itself)
    superscan_expt_name = get_experiment_name(date, superscantime)
    allowed_subscans = MULTID_SCAN_MAP[superscan_expt_name][0] + [superscan_expt_name]
    all_scantimes = [st for st in all_scantimes if get_experiment_name(date, st) in allowed_subscans]

    # Further prune "all_scantimes" to include only the scans which are subscans of superscantime
    # These should be the n scans prior to superscantime plus the n*(m-1) scans after superscantime,
    # where n is equal to the number of subscans per data point, and m is the number of data points in the super scan
     
    # Count number of datapoints
    (x, y) = get_data(date, superscantime)
    npoints = len(x)

    # Find index of superscantime
    index = len(all_scantimes) - all_scantimes[::-1].index(superscantime) - 1

    nsubscans = MULTID_SCAN_MAP[superscan_expt_name][1]  # Number of subscans per data point
    
    if MULTID_SCAN_MAP[superscan_expt_name][2] == 'old':
        # In data saved by the old script scanner, all subscans come after the super scan
        base_scantimes = all_scantimes[index+1:index+1+nsubscans*npoints]
    elif MULTID_SCAN_MAP[superscan_expt_name][2] == 'new':
        # In data saved by the new script scanner, the subscans for the first data point come before the super scan, and the rest come after
        base_scantimes = all_scantimes[index-nsubscans:index] + all_scantimes[index+1:index+1+nsubscans*(npoints-1)]
    
    # Remove scans to exclude
    base_scantimes = _remove_excludes(base_scantimes, exclude)

    if return_xvalues:
        x = _remove_excludes(x, exclude)
        return (x, base_scantimes)
    else:
        return base_scantimes


def _get_photon_counts_single(date, scantime, exclude=[]):
    """Returns photon counts for an experiment specified by date and scan time."""

    directory = get_data_directory(date, scantime)

    # Check the time directory for csv files. If there are none, raise an error
    file_path = None
    for filename in os.listdir(directory):
        if filename.endswith('Readouts.csv'):
            file_path = directory + filename
            break
    if file_path is None:
        raise IOError('No readout file found in {}.'.format(directory))

    # Extract the data
    counts_arr = np.loadtxt(file_path, delimiter=',')

    # Determine repetitions per data point
    n_data_points = int(counts_arr[-1, 0]) + 1
    n_reps = 0
    while True:
        try:
            if counts_arr[n_reps, 0] > 0:
                break
        except:
            break
        n_reps += 1

    # Reshape the counts array
    counts_arr = np.reshape(counts_arr[:, 1], (n_data_points, n_reps))

    # Remove data points to be excluded
    counts_arr = _remove_excludes(counts_arr, exclude)

    return counts_arr


def _get_hist_single(date, scantime, histNum):
    """Returns saved histogram for an experiment specified by an date, scan time, and number (as in general there may be multiple histograms saved for a scan."""

    directory = get_data_directory(date, scantime)
    hist_prefix = '0000' + str(histNum + 2)

    # Check the time directory for csv files. If there are none, raise an error
    file_path = None
    for filename in os.listdir(directory):
        if filename.endswith('.csv') and filename.startswith(hist_prefix):
            file_path = directory + filename
            break
    if file_path is None:
        raise IOError('No .csv files found for that histogram number in {}.'.format(directory))

    # Extract the data
    data_arr = np.loadtxt(file_path, delimiter=',')
    if data_arr.ndim == 1:
        # If only 1 data point, expant into a 2d array
        data_arr = np.expand_dims(data_arr, axis=0)

    # Sort the data into x and y arrays
    x = data_arr[:, 0]
    y = data_arr[:, 1:]

    # Flatten the y array if there is only one set of y data
    if np.shape(y)[1] == 1:
        y = y.flatten()

    return (x, y)


def _remove_excludes(array_in, exclude):
    """Handles deleting excluded indices for both standard lists and numpy arrays"""
    
    # Shift indices to be positive
    for (i, el) in enumerate(exclude):
        if el < 0: exclude[i] = el + len(array_in)

    if isinstance(array_in, np.ndarray):
        # array_in is a numpy array
        exclude = list(exclude)
        array_out = np.delete(array_in, exclude, axis=0)
    elif isinstance(array_in, list):
        # array is an ordinary list
        for item in exclude[::-1]:
            del array_in[item]
        array_out = array_in
    else:
        raise ValueError('array_in must be a list or numpy array')

    return array_out
