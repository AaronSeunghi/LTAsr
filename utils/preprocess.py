import numpy as np
import os
import pandas
import tables

from functools import partial
from multiprocessing.dummy import Pool
from utils.audio import compute_mfcc
from utils.text import text_to_char_array, get_start_sym_label, get_end_sym_label

def pmap(fun, iterable, threads=8):
    pool = Pool(threads)
    results = pool.map(fun, iterable)
    pool.close()
    return results


def process_single_file(row, numcep, numcontext, alphabet):
    # row = index, Series
    _, file = row
    features = compute_mfcc(file.wav_filename, numcep, numcontext)
    transcript = text_to_char_array(file.transcript, alphabet)

    if (2*numcontext + len(features)) < len(transcript):
        raise ValueError('Error: Audio file {} is too short for transcription.'.format(file.wav_filename))

    # One stride per time step in the input
    num_strides = len(features) - (numcontext * 2)

    # Create a view into the array with overlapping strides of size
    # numcontext (past) + 1 (present) + numcontext (future)
    window_size = 2*numcontext+1
    features_context = np.lib.stride_tricks.as_strided(
                              features,
                              (num_strides, window_size, numcep),
                              (features.strides[0], features.strides[0], features.strides[1]),
                              writeable=False)

    features_context = np.reshape(features_context, [num_strides, -1])

    X_length = len(features_context)
    X_indices = features_context
    Y_length = len(transcript) + 1  # added <sos> or <eos> symbols
    Y_input_indices = np.concatenate((get_start_sym_label(alphabet), transcript), axis=0)
    Y_target_indices = np.concatenate((transcript, get_end_sym_label(alphabet)), axis=0)

    return X_length, X_indices, Y_length, Y_input_indices, Y_target_indices


# load samples from CSV, compute features, optionally cache results on disk
def preprocess(csv_files, numcep, numcontext, alphabet):
    COLUMNS = ('X_length', 'X_indices', 'Y_length', 'Y_input_indices', 'Y_target_indices')

    print('Preprocessing', csv_files)
    source_data = None
    for csv in csv_files:
        file = pandas.read_csv(csv, encoding='utf-8', na_filter=False)
        csv_dir = os.path.dirname(os.path.abspath(csv))
        file['wav_filename'] = file['wav_filename'].str.replace(r'(^[^/])', lambda m: os.path.join(csv_dir, m.group(1)))
        if source_data is None:
            source_data = file
        else:
            source_data = source_data.append(file)

    # sort by the shortest length
    source_data.sort_values(by='wav_filesize', ascending=True, inplace=True)

    step_fn = partial(process_single_file,
                      numcep=numcep,
                      numcontext=numcontext,
                      alphabet=alphabet)
    out_data = pmap(step_fn, source_data.iterrows())

    print('Preprocessing done')
    return pandas.DataFrame(data=out_data, columns=COLUMNS)


