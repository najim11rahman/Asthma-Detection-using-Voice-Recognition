import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import statistics

frequency_sampling, audio_signal = wavfile.read("cough7.wav")

audio_signal = audio_signal[:15000]

features_mfcc = mfcc(audio_signal, frequency_sampling)

filterbank_features = logfbank(audio_signal, frequency_sampling)

sum = 0
for i in range(len(features_mfcc)):
    sum = sum + features_mfcc[i]
    mean = sum/len(features_mfcc)

variance = statistics.pvariance(mean)

if(variance < 170):
    print(variance)
    print("LOW POSSIBILITY FOR ASTHMA")

if((variance > 170) & (variance < 180)):
    print(variance)
    print("MEDIUM POSSIBILITY FOR ASTHMA")

if((variance > 180) & (variance < 190)):
    print(variance)
    print("MEDIUM POSSIBILITY FOR ASTHMA")

if(variance > 190):
    print(variance)
    print("HIGH POSSIBILITY FOR ASTHMA")