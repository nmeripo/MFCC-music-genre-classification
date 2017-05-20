import os
import glob
import sys
import fnmatch
import numpy as np
import scipy
import scipy.io.wavfile
from scikits.talkbox.features import mfcc
import cPickle as pickle

genres_dir = '/home/michael/Documents/mgc/genres'
processed_dir = genres_dir[: genres_dir.rindex('/') + 1] + "/processed_genres/"

# The first 13 (the lower dimensions) of MFCC represents the envelope of spectra.
# And the discarded higher dimensions expresses the spectral details.
# For different phonemes, envelopes are enough to represent the difference, so we can recognize phonemes through MFCC.


def get_ceps(filepath):
    """    
    :param filepath: takes wav audio file path
    :return: returns first 13 Mel-frequency cepstral coefficients
    """
    sample_rate, X = scipy.io.wavfile.read(filepath)
    ceps, mspec, spec = mfcc(X)
    ceps_ = np.mean(ceps, axis=0)
    return ceps_


X = []
y = []


# Create trainable dataset
for dir_name, subdir_list, file_list in os.walk(processed_dir):
    current_genre =  dir_name[dir_name.rindex('/') + 1 : ]
    print('Found directory: %s' % dir_name)
    for fname in fnmatch.filter(file_list, '*.wav'):
        filepath = dir_name + "/" + fname
        ceps = get_ceps(filepath)
        if not np.isnan(ceps).any():
            X.append(ceps)
            y.append(current_genre)



pickle.dump(np.array(X), open("mfcc_features.pkl", "wb"))
pickle.dump(np.array(y), open("genre_targets.pkl", "wb"))

print np.array(X).shape
print np.array(y).shape