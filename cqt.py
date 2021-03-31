import numpy, scipy, matplotlib.pyplot as plt, IPython.display as ipd
import librosa, librosa.display
import statistics
import numpy as np


x, sr = librosa.load('words2.wav')
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
print("N",mean)

