import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank


frequency_sampling, audio_signal = wavfile.read("normal1.wav")

audio_signal = audio_signal[:15000]

features_mfcc = mfcc(audio_signal, frequency_sampling)

filterbank_features = logfbank(audio_signal, frequency_sampling)
sum = 0
for i in range(len(filterbank_features)):
    sum = sum + filterbank_features[i]
    mean = sum/len(filterbank_features)

sum=0
for i in range(len(mean)):
    sum = sum + mean[i]
    mean2 = sum/len(mean)

print(mean2)
plt.hist(filterbank_features, bins=20)
plt.show() 
