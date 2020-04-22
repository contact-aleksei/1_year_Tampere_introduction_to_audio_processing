from __future__ import print_function
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Load Audio Recordings

x_1, fs = librosa.load('antti_yesterday.wav')
plt.subplot(2, 2, 1)
librosa.display.waveplot(x_1, sr=fs)
plt.title('antti_yesterday $X_1$')
plt.tight_layout()

x_2, fs2 = librosa.load('ferenc_yesterday.wav')
plt.subplot(2, 2, 2)
librosa.display.waveplot(x_2, sr=fs)
plt.title('ferenc_yesterday $X_1$')
plt.tight_layout()

x_3, fs = librosa.load('johanna_yesterday.wav')
plt.subplot(2, 2, 3)
librosa.display.waveplot(x_3, sr=fs)
plt.title('johanna_yesterday $X_1$')
plt.tight_layout()

x_4, fs = librosa.load('outi_yesterday.wav')
plt.subplot(2, 2, 4)
librosa.display.waveplot(x_4, sr=fs)
plt.title('outi_yesterday $X_1$')
plt.tight_layout()

# Extract Chroma Features
# In music, the term chroma feature or chromagram closely relates
# to the twelve different pitch classes.
n_fft = 4410
hop_size = 2205

x_1_chroma = librosa.feature.chroma_stft(y=x_1, sr=fs, tuning=0, norm=2,
                                         hop_length=hop_size, n_fft=n_fft)
x_2_chroma = librosa.feature.chroma_stft(y=x_2, sr=fs, tuning=0, norm=2,
                                         hop_length=hop_size, n_fft=n_fft)
x_3_chroma = librosa.feature.chroma_stft(y=x_3, sr=fs, tuning=0, norm=2,
                                         hop_length=hop_size, n_fft=n_fft)
x_4_chroma = librosa.feature.chroma_stft(y=x_4, sr=fs, tuning=0, norm=2,
                                         hop_length=hop_size, n_fft=n_fft)

plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.title('Chroma Representation of $X_1$')
librosa.display.specshow(x_1_chroma, x_axis='time',
                         y_axis='chroma', cmap='plasma', hop_length=hop_size)
plt.colorbar()
plt.subplot(2, 2, 2)
plt.title('Chroma Representation of $X_2$')
librosa.display.specshow(x_2_chroma, x_axis='time',
                         y_axis='chroma', cmap='plasma', hop_length=hop_size)
plt.colorbar()
plt.subplot(2, 2, 3)
plt.title('Chroma Representation of $X_3$')
librosa.display.specshow(x_3_chroma, x_axis='time',
                         y_axis='chroma', cmap='plasma', hop_length=hop_size)
plt.colorbar()
plt.subplot(2, 2, 4)
plt.title('Chroma Representation of $X_4$')
librosa.display.specshow(x_4_chroma, x_axis='time',
                         y_axis='chroma', cmap='plasma', hop_length=hop_size)
plt.colorbar()
plt.tight_layout()


# Align Chroma Sequences

D12, wp12 = librosa.sequence.dtw(X=x_1_chroma, Y=x_2_chroma, metric='cosine')
wp_s12 = np.asarray(wp12) * hop_size / fs

D23, wp23 = librosa.sequence.dtw(X=x_2_chroma, Y=x_3_chroma, metric='cosine')
wp_s23 = np.asarray(wp23) * hop_size / fs

D34, wp34 = librosa.sequence.dtw(X=x_3_chroma, Y=x_4_chroma, metric='cosine')
wp_s34 = np.asarray(wp34) * hop_size / fs



fig = plt.figure(figsize=(16, 8))





fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))


fig = plt.figure(figsize=(16, 8))

# Plot x_1
librosa.display.waveplot(x_1, sr=fs, ax=ax1)
ax1.set(title='Slower Version $X_1$')

# Plot x_2
librosa.display.waveplot(x_2, sr=fs, ax=ax2)
ax2.set(title='Slower Version $X_2$')

# Plot x_3
librosa.display.waveplot(x_3, sr=fs, ax=ax3)
ax3.set(title='Slower Version $X_3$')

# Plot x_4
librosa.display.waveplot(x_4, sr=fs, ax=ax4)
ax4.set(title='Slower Version $X_4$')

plt.tight_layout()


trans_figure = fig.transFigure.inverted()
lines = []
arrows = 30
points_idx12 = np.int16(np.round(np.linspace(0, wp12.shape[0] - 1, arrows)))
points_idx23 = np.int16(np.round(np.linspace(0, wp23.shape[0] - 1, arrows)))
points_idx34 = np.int16(np.round(np.linspace(0, wp34.shape[0] - 1, arrows)))
# for tp1, tp2 in zip((wp[points_idx, 0]) * hop_size, (wp[points_idx, 1]) * hop_size):
for tp1, tp2 in wp12[points_idx12] * hop_size / fs:
    # get position on axis for a given index-pair
    coord1 = trans_figure.transform(ax1.transData.transform([tp1, 0]))
    coord2 = trans_figure.transform(ax2.transData.transform([tp2, 0]))

    # draw a line
    line = matplotlib.lines.Line2D((coord1[0], coord2[0]),
                                   (coord1[1], coord2[1]),
                                   transform=fig.transFigure,
                                   color='r')
    lines.append(line)
    
for tp2, tp3 in wp12[points_idx23] * hop_size / fs:
    # get position on axis for a given index-pair
    coord2 = trans_figure.transform(ax2.transData.transform([tp2, 0]))
    coord3 = trans_figure.transform(ax3.transData.transform([tp3, 0]))

    # draw a line
    line = matplotlib.lines.Line2D((coord2[0], coord3[0]),
                                   (coord2[1], coord3[1]),
                                   transform=fig.transFigure,
                                   color='r')
    lines.append(line)
    
fig.lines = lines
plt.tight_layout()






    