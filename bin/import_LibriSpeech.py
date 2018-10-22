#!/usr/bin/env python

import os
import sys

import codecs
import errno
import fnmatch
import pandas
import progressbar
import subprocess
import tarfile
import unicodedata

from os import path
from sox import Transformer

def _preprocess_data(data_dir):

    # Assume data is downloaded from OpenSLR - http://www.openslr.org/resources/12/
    LIBRIVOX_DIR = "LibriSpeech"
    work_dir = os.path.join(data_dir, LIBRIVOX_DIR)

    if not path.isdir(work_dir):
            print("Error :: Could not find extracted %s" %LIBRIVOX_DIR)
            print("Error :: File should be downloaded from OpenSLR and placed at:", work_dir)
            raise IOError(errno, "File not found", work_dir)
    else:
            # is path therefore continue
            print("Found extracted data in: ", work_dir)


    # Convert FLAC data to wav, from:
    #  data_dir/LibriSpeech/split/1/2/1-2-3.flac
    # to:
    #  data_dir/LibriSpeech/split-wav/1-2-3.wav
    #
    # And split LibriSpeech transcriptions, from:
    #  data_dir/LibriSpeech/split/1/2/1-2.trans.txt
    # to:
    #  data_dir/LibriSpeech/split-wav/1-2-0.txt
    #  data_dir/LibriSpeech/split-wav/1-2-1.txt
    #  data_dir/LibriSpeech/split-wav/1-2-2.txt
    #  ...
    print("Converting FLAC to WAV and splitting transcriptions...")
    with progressbar.ProgressBar(max_value=7, widget=progressbar.AdaptiveETA) as bar:
        train_100 = _convert_audio_and_split_sentences(work_dir, "train-clean-100", "train-clean-100-wav")
        bar.update(0)
        train_360 = _convert_audio_and_split_sentences(work_dir, "train-clean-360", "train-clean-360-wav")
        bar.update(1)
        train_500 = _convert_audio_and_split_sentences(work_dir, "train-other-500", "train-other-500-wav")
        bar.update(2)

        dev_clean = _convert_audio_and_split_sentences(work_dir, "dev-clean", "dev-clean-wav")
        bar.update(3)
        dev_other = _convert_audio_and_split_sentences(work_dir, "dev-other", "dev-other-wav")
        bar.update(4)

        test_clean = _convert_audio_and_split_sentences(work_dir, "test-clean", "test-clean-wav")
        bar.update(5)
        test_other = _convert_audio_and_split_sentences(work_dir, "test-other", "test-other-wav")
        bar.update(6)


    # Write sets to disk as CSV files
    print("Building CSV files")
    train_100.to_csv(os.path.join(data_dir, "librivox-train-clean-100.csv"), index=False)
    train_360.to_csv(os.path.join(data_dir, "librivox-train-clean-360.csv"), index=False)
    train_500.to_csv(os.path.join(data_dir, "librivox-train-other-500.csv"), index=False)

    dev_clean.to_csv(os.path.join(data_dir, "librivox-dev-clean.csv"), index=False)
    dev_other.to_csv(os.path.join(data_dir, "librivox-dev-other.csv"), index=False)

    test_clean.to_csv(os.path.join(data_dir, "librivox-test-clean.csv"), index=False)
    test_other.to_csv(os.path.join(data_dir, "librivox-test-other.csv"), index=False)


def _convert_audio_and_split_sentences(extracted_dir, data_set, dest_dir):
    source_dir = os.path.join(extracted_dir, data_set)
    target_dir = os.path.join(extracted_dir, dest_dir)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Loop over transcription files and split each one
    #
    # The format for each file 1-2.trans.txt is:
    #  1-2-0 transcription of 1-2-0.flac
    #  1-2-1 transcription of 1-2-1.flac
    #  ...
    #
    # Each file is then split into several files:
    #  1-2-0.txt (contains transcription of 1-2-0.flac)
    #  1-2-1.txt (contains transcription of 1-2-1.flac)
    #  ...
    #
    # We also convert the corresponding FLACs to WAV in the same pass
    files = []
    for root, dirnames, filenames in os.walk(source_dir):
        for filename in fnmatch.filter(filenames, '*.trans.txt'):
            trans_filename = os.path.join(root, filename)
            with codecs.open(trans_filename, "r", "utf-8") as fin:
                for line in fin:
                    # Parse each segment line
                    first_space = line.find(" ")
                    seqid, transcript = line[:first_space], line[first_space+1:]

                    # We need to do the encode-decode dance here because encode
                    # returns a bytes() object on Python 3, and text_to_char_array
                    # expects a string.
                    transcript = unicodedata.normalize("NFKD", transcript)  \
                                            .encode("ascii", "ignore")      \
                                            .decode("ascii", "ignore")

                    transcript = transcript.lower().strip()

                    # Convert corresponding FLAC to a WAV
                    flac_file = os.path.join(root, seqid + ".flac")
                    wav_file = os.path.join(target_dir, seqid + ".wav")
                    if not os.path.exists(wav_file):
                        Transformer().build(flac_file, wav_file)
                    wav_filesize = os.path.getsize(wav_file)

                    files.append((os.path.abspath(wav_file), wav_filesize, transcript))

    return pandas.DataFrame(data=files, columns=["wav_filename", "wav_filesize", "transcript"])


if __name__ == "__main__":
    _preprocess_data(sys.argv[1])
    print("Completed")

