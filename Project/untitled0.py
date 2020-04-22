from __future__ import print_function
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import librosa


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

print ('going allright')

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

print ('going allright')


D, wp = librosa.core.dwt(X=x_1_chroma, Y=x_2_chroma, metric='cosine')
wp_s = np.asarray(wp) * hop_size / fs

fig = plt.figure(figsize=(16, 8))

# Plot x_1
plt.subplot(2, 1, 1)
librosa.display.waveplot(x_1, sr=fs)
plt.title('Slower Version $X_1$')
ax1 = plt.gca()

# Plot x_2
plt.subplot(2, 1, 2)
librosa.display.waveplot(x_2, sr=fs)
plt.title('Slower Version $X_2$')
ax2 = plt.gca()

plt.tight_layout()

trans_figure = fig.transFigure.inverted()
lines = []
arrows = 30
points_idx = np.int16(np.round(np.linspace(0, wp.shape[0] - 1, arrows)))

# for tp1, tp2 in zip((wp[points_idx, 0]) * hop_size, (wp[points_idx, 1]) * hop_size):
for tp1, tp2 in wp[points_idx] * hop_size / fs:
    # get position on axis for a given index-pair
    coord1 = trans_figure.transform(ax1.transData.transform([tp1, 0]))
    coord2 = trans_figure.transform(ax2.transData.transform([tp2, 0]))

    # draw a line
    line = matplotlib.lines.Line2D((coord1[0], coord2[0]),
                                   (coord1[1], coord2[1]),
                                   transform=fig.transFigure,
                                   color='r')
    lines.append(line)

fig.lines = lines
plt.tight_layout()