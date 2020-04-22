import librosa as lb
from matplotlib import pyplot as plt
import librosa.display
import numpy as np
import sounddevice 

def plot_spectrogram(audioIn,fs,name):
    sr, audio= fs, audioIn
    #- Calculate the spectrogram of it.
    spectrogram=lb.core.stft (audio)
    #- Calculate the constant-Q spectrogram of it.
    constantQ_spectrogram=lb.core.cqt (audio, sr = sr)
    #- Calculate the chromagram of it.
    chromagram=lb.feature.chroma_stft (audio, sr)
    
    #- Plot and observe and report differences between the spectrogram,
    # constant-Q spectrogram and chromagram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(spectrogram,
                                                     ref=np.max),
                              y_axis='log', x_axis='time')
    plt.title('Power spectrogram '+str(name))
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(constantQ_spectrogram, ref=np.max),
                              sr=sr, x_axis='time', y_axis='cqt_note')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Constant-Q power spectrum '+str(name))
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chromagram, y_axis='chroma', x_axis='time')
    plt.colorbar()
    plt.title('Chromagram of '+str(name))
    plt.tight_layout()
    plt.show()

audioIn, fs=lb.load('brahms_hungarian_dance_5_short.wav', sr=None)
name='brahms_hungarian_dance_5_short.wav'
plot_spectrogram(audioIn,fs,name)

audioIn, fs=lb.load('classic_rock_beat.wav', sr=None)
name='classic_rock_beat.wav'
plot_spectrogram(audioIn,fs,name)

audioIn, fs=lb.load('conga_groove.wav', sr=None)
plot_spectrogram(audioIn,fs,name)
name='conga_groove.wav'

audioIn, fs=lb.load('latin_groove_short.wav', sr=None)
plot_spectrogram(audioIn,fs,name)
name='latin_groove_short.wav'

###############################################################################
###TASK2###############TASK2#########################################TASK2#####
############################################################TASK2##############
###############################################################################
##########TASK2##################TASK2############TASK2########################
########################################################################TASK2##
########################TASK2##################################################
############TASK2#########################################TASK2################
###############################TASK2#######TASK2###############################
###############################################################################
# 1. Load the audio file.
audioIn, fs=lb.load('classic_rock_beat.wav', sr=None)

# 2. Compute spectral novelty function using librosa library.
onset_env = librosa.onset.onset_strength(y=audioIn, sr=fs)

D = np.abs(librosa.stft(y=audioIn))
times = librosa.times_like(D)
plt.figure()
ax1 = plt.subplot(2, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                         y_axis='log', x_axis='time')
plt.title('Power spectrogram')
plt.subplot(2, 1, 2, sharex=ax1)
plt.plot(times, 2 + onset_env / onset_env.max(), alpha=0.8,
         label='Mean (mel)')

# 3. Pick peaks to detect the frame indexes of the onsets.
peaks_onsetFrames = librosa.util.peak_pick(onset_env, 1, 1, 1, 1, 3, 1)

# 4. Convert the frame indexes into time indexes in unit of seconds. 
beat_times = librosa.frames_to_time(peaks_onsetFrames, sr=fs)

# 5. Plot the onsets on top of the time domain signal and report your observations.
plt.figure()
librosa.display.waveplot(audioIn, x_axis='time')
plt.vlines(beat_times, -1,1, color='r', alpha=0.8,
           label='Selected peaks')
plt.legend(frameon=True, framealpha=0.8)
plt.axis('tight')
plt.tight_layout()
plt.title('parameter is equal 3')
plt.show()

# 6. Plot the onsets on top of the spectrogram and report your observations.
plt.figure()
librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                         y_axis='log', x_axis='time')
plt.vlines(beat_times, 0,8192, color='w')
plt.show()

# 7. Adjust the ​librosa.util.peak_pick​ parameters and observe how they affect the detected onsets​.
peaks_ADJUSTED = librosa.util.peak_pick(onset_env, 1, 1, 1, 1, 0.5, 1)

beat_times = librosa.frames_to_time(peaks_ADJUSTED, sr=fs)
plt.figure()
librosa.display.waveplot(audioIn, x_axis='time')
plt.vlines(beat_times, -1,1, color='r', alpha=0.8,
           label='Selected peaks')
plt.legend(frameon=True, framealpha=0.8)
plt.axis('tight')
plt.tight_layout()
plt.title('parameter is equal 0.5')
plt.show()

# At this point you have detected the onsets.
# Now let us add clicks at the detected onset frames.

# 1. Create a signal with the same length as the analyzed music example
clicks=librosa.clicks(frames=peaks_ADJUSTED, sr=fs, length=len(audioIn))
added_1=audioIn+clicks
#sounddevice.play(added_1)
added_2=np.vstack([audioIn, clicks]).T
sounddevice.play(added_2,fs)
