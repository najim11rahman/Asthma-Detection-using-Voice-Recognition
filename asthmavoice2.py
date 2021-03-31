# Features were extracted for all voice samples
# using Mel-frequency Cepstral Coefficients.
# After analyzing all extracted features for both asthmatic and
# normal persons it has been observed that there is a large
# variation in coefficients of asthmatic persons as compared to
# normal persons especially in the 1st and 2nd coefficients. 


import numpy as np
from librosa import cqt
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import statistics
import numpy, scipy, matplotlib.pyplot as plt, IPython.display as ipd
import librosa, librosa.display

audio = "nortest.wav"

frequency_sampling, audio_signal = wavfile.read(audio)

audio_signal = audio_signal[:15000]

features_mfcc = mfcc(audio_signal, frequency_sampling)

filterbank_features = logfbank(audio_signal, frequency_sampling)

sum = 0
for i in range(len(features_mfcc)):
    sum = sum + features_mfcc[i]
    mean = sum/len(features_mfcc)

variance = statistics.pvariance(mean)

if(variance < 170):
    print("MFCC",variance,"LOW POSSIBILITY FOR ASTHMA")

if((variance > 170) & (variance < 190)):
    print("MFCC",variance,"MEDIUM POSSIBILITY FOR ASTHMA")

if(variance > 190):
    print("MFCC",variance,"HIGH POSSIBILITY FOR ASTHMA")

x, sr = librosa.load(audio)
ipd.Audio(x, rate=sr)

fmin = librosa.midi_to_hz(36)
hop_length = 512
C = librosa.cqt(x, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
logC = librosa.amplitude_to_db(numpy.abs(C))

sum = 0
for i in range(len(logC)):
    var = int(np.var(logC[i]))
    sum = sum + var

mean = sum/len(logC)
print("CQCC",mean)
print("Value",variance - mean)
