from __future__ import print_function
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Load Audio Recordings
plt.figure(figsize=(12, 3))
x_1, fs = librosa.load('antti_yesterday.wav')
x_1, index = librosa.effects.trim(x_1)
plt.subplot(2, 2, 1)
librosa.display.waveplot(x_1, sr=fs)
plt.title('antti_yesterday $X_1$')
plt.tight_layout()

x_2, fs2 = librosa.load('ferenc_yesterday.wav')
x_2, index = librosa.effects.trim(x_2)
plt.subplot(2, 2, 2)
librosa.display.waveplot(x_2, sr=fs)
plt.title('ferenc_yesterday $X_1$')
plt.tight_layout()

x_3, fs = librosa.load('johanna_yesterday.wav')
x_3, index = librosa.effects.trim(x_3)
plt.subplot(2, 2, 3)
librosa.display.waveplot(x_3, sr=fs)
plt.title('johanna_yesterday $X_1$')
plt.tight_layout()

x_4, fs = librosa.load('outi_yesterday.wav')
x_4, index = librosa.effects.trim(x_4)
plt.subplot(2, 2, 4)
librosa.display.waveplot(x_4, sr=fs)
plt.title('outi_yesterday $X_1$')
plt.tight_layout()
# Extract Chroma Features
# In music, the term chroma feature or chromagram closely relates
# to the twelve different pitch classes.
n_fft = 4410
hop_size = 2205

def plot_chroma(x_1, x_2, x_3, x_4, fs):
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
    return x_1_chroma, x_2_chroma, x_3_chroma, x_4_chroma


# Align Chroma Sequences
def plot_time_domain(x_1_chroma, x_2_chroma, x_3_chroma, x_4_chroma,x_1,x_2,x_3,x_4):
    D12, wp12 = librosa.sequence.dtw(X=x_1_chroma, Y=x_2_chroma, metric='cosine')
    #wp_s12 = np.asarray(wp12) * hop_size / fs
    
    fig = plt.figure(figsize=(12, 4))
    # Plot x_1
    plt.subplot(2, 1, 1)
    librosa.display.waveplot(x_1, sr=fs)
    plt.title('Version $antti$')
    ax1 = plt.gca()
    
    # Plot x_2
    plt.subplot(2, 1, 2)
    librosa.display.waveplot(x_2, sr=fs)
    plt.title('Version $ferenc$')
    ax2 = plt.gca()
    
    plt.tight_layout()
    
    trans_figure = fig.transFigure.inverted()
    lines = []
    arrows = 30
    points_idx12 = np.int16(np.round(np.linspace(0, wp12.shape[0] - 1, arrows)))
    
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
    
    trans_figure = fig.transFigure.inverted()
    fig.lines = lines
    plt.tight_layout()
    
    D13, wp13 = librosa.sequence.dtw(X=x_1_chroma, Y=x_3_chroma, metric='cosine')
    #wp_s13 = np.asarray(wp13) * hop_size / fs
    
    
    fig = plt.figure(figsize=(12, 4))
    # Plot x_1
    plt.subplot(2, 1, 1)
    librosa.display.waveplot(x_1, sr=fs)
    plt.title('Version $antti$')
    ax1 = plt.gca()
    
    # Plot x_3
    plt.subplot(2, 1, 2)
    librosa.display.waveplot(x_3, sr=fs)
    plt.title('Version $johanna$')
    ax3 = plt.gca()
       
    plt.tight_layout()
    
    trans_figure = fig.transFigure.inverted()
    lines = []
    arrows = 30
    points_idx13 = np.int16(np.round(np.linspace(0, wp13.shape[0] - 1, arrows)))
    for tp1, tp3 in wp13[points_idx13] * hop_size / fs:
        # get position on axis for a given index-pair
        coord1 = trans_figure.transform(ax1.transData.transform([tp1, 0]))
        coord3 = trans_figure.transform(ax3.transData.transform([tp3, 0]))
    
        # draw a line
        line = matplotlib.lines.Line2D((coord1[0], coord3[0]),
                                       (coord1[1], coord3[1]),
                                       transform=fig.transFigure,
                                       color='r')
        lines.append(line)
    fig.lines = lines
    plt.tight_layout()
       
    
    D14, wp14 = librosa.sequence.dtw(X=x_1_chroma, Y=x_4_chroma, metric='cosine')
    #wp_s14 = np.asarray(wp14) * hop_size / fs
    
    fig = plt.figure(figsize=(12, 4))
    # Plot x_1
    plt.subplot(2, 1, 1)
    librosa.display.waveplot(x_1, sr=fs)
    plt.title('Version $antti$')
    ax1 = plt.gca()
    
    # Plot x_4
    plt.subplot(2, 1, 2)
    librosa.display.waveplot(x_4, sr=fs)
    plt.title('Version $outi$')
    ax4 = plt.gca()
    
    plt.tight_layout()
    
    trans_figure = fig.transFigure.inverted()
    lines = []
    
    arrows = 30
    points_idx14 = np.int16(np.round(np.linspace(0, wp14.shape[0] - 1, arrows)))
    
    for tp1, tp4 in wp14[points_idx14] * hop_size / fs:
        # get position on axis for a given index-pair
        coord1 = trans_figure.transform(ax1.transData.transform([tp1, 0]))
        coord4 = trans_figure.transform(ax4.transData.transform([tp4, 0]))
    
        # draw a line
        line = matplotlib.lines.Line2D((coord1[0], coord4[0]),
                                       (coord1[1], coord4[1]),
                                       transform=fig.transFigure,
                                       color='r')
        lines.append(line)
    
    fig.lines = lines
    plt.tight_layout()


x_1_chroma, x_2_chroma, x_3_chroma, x_4_chroma=plot_chroma(x_1, x_2, x_3, x_4, fs)
plot_time_domain(x_1_chroma, x_2_chroma, x_3_chroma, x_4_chroma,x_1,x_2,x_3,x_4)
# Determine time-stretching factor based on the length of singing

d1=librosa.get_duration(y=x_1, sr=fs)
d2=librosa.get_duration(y=x_2, sr=fs)
d3=librosa.get_duration(y=x_3, sr=fs)
d4=librosa.get_duration(y=x_4, sr=fs)

k21=d2/d1
k31=d3/d1
k41=d4/d1

# You can use your own time stretching implementation, or a library version
x_21 = librosa.effects.time_stretch(x_2, k21)
x_31 = librosa.effects.time_stretch(x_3, k31)
x_41 = librosa.effects.time_stretch(x_4, k41)


x_1_chroma, x_21_chroma, x_31_chroma, x_41_chroma=plot_chroma(x_1, x_21, x_31, x_41, fs)
plot_time_domain(x_1_chroma, x_21_chroma, x_31_chroma, x_41_chroma,x_1,x_21,x_31,x_41)

def chromapaths(X,Y):

    D, wp = librosa.sequence.dtw(X, Y)
    P=wp
    N = X.shape[1]
    M = Y.shape[1]
    
    plt.figure(figsize=(10, 4))
    ax_X = plt.axes([0, 0.60, 1, 0.40])
    ax_X.imshow(X, origin='lower', aspect='auto', cmap='plasma')
    plt.yticks(np.arange(12), 'C C# D D# E F F# G G# A A# B'.split())
    ax_X.xaxis.tick_top()
    ax_X.set_ylabel('Sequence X')
    ax_X.set_xlim(0, X.shape[1])
    
    ax_Y = plt.axes([0, 0, 1, 0.40])
    ax_Y.imshow(Y, origin='lower', aspect='auto', cmap='plasma')
    plt.yticks(np.arange(12), 'C C# D D# E F F# G G# A A# B'.split())
    ax_Y.set_ylabel('Sequence Y')
    ax_Y.set_xlim(0, Y.shape[1])
    
    step = 10
    y_min_X, y_max_X = ax_X.get_ylim()
    y_min_Y, y_max_Y = ax_Y.get_ylim()
    for t in P[0:-1:step, :]: 
        ax_X.vlines(t[0], y_min_X, y_max_X, color='r')
        ax_Y.vlines(t[1], y_min_Y, y_max_Y, color='r')
    
    ax = plt.axes([0, 0.40, 1, 0.20])
    for t in P[0:-1:step, :]: 
        ax.plot((t[0]/N, t[1]/M), (1, -1), color='r')
        ax.set_xlim(0, 1)
        ax.set_ylim(-1, 1)
    ax.set_xticks([])
    ax.set_yticks([]);


X=x_1_chroma
Y=x_2_chroma
chromapaths(X,Y)
plt.title('ANTTI vs original FERENC', color= 'w',
          fontsize=14, fontweight='bold')

X=x_1_chroma
Y=x_21_chroma
chromapaths(X,Y)
plt.title('ANTTI vs stretched FERENC', color= 'w',
          fontsize=14, fontweight='bold')

X=x_1_chroma
Y=x_3_chroma
chromapaths(X,Y)
plt.title('ANTTI vs original JOHANNA', color= 'w',
          fontsize=14, fontweight='bold')

X=x_1_chroma
Y=x_31_chroma
chromapaths(X,Y)
plt.title('ANTTI vs stretched JOANNA', color= 'w',
          fontsize=14, fontweight='bold')

X=x_1_chroma
Y=x_4_chroma
chromapaths(X,Y)
plt.title('ANTTI vs original OUTI', color= 'w',
          fontsize=14, fontweight='bold')

X=x_1_chroma
Y=x_41_chroma
chromapaths(X,Y)
plt.title('ANTTI vs stretched OUTI', color= 'w',
          fontsize=14, fontweight='bold')

import scipy
scipy.io.wavfile.write('antti.wav', rate=fs, data=x_1)
scipy.io.wavfile.write('ferenc.wav', rate=fs, data=x_21)
scipy.io.wavfile.write('johanna.wav', rate=fs, data=x_31)
scipy.io.wavfile.write('outi.wav', rate=fs, data=x_41)
