import scipy
import librosa as lb
#from scipy.signal import hamming
from matplotlib import pyplot as plt
#from audiolazy import lazy_lpc as lpc
from librosa import filters
from librosa import display
import librosa.display
audioIn, fs=lb.load('audio.wav', sr=None)                    
N_fft = 512
n_fft=N_fft
N_mel = 40
Mel_filterbank = filters.mel(sr=fs,n_fft=N_fft,n_mels=N_mel)
display.specshow(Mel_filterbank,x_axis='linear')                        
plt.ylabel('Mel filter')
plt.title('Mel filter bank')
plt.colorbar()
plt.tight_layout()
plt.show()
import numpy as np
pre_emphasis=0.97

signalx=audioIn
emphasized_signal = np.append(signalx[0], signalx[1:] - pre_emphasis * signalx[:-1])


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

#from scipy.io.wavfile import read
import matplotlib.pyplot as plt  # For ploting
import numpy
import numpy as np # to work with numerical data efficiently
#import winsound
#import scipy
#import scipy.io
#import scipy.io.wavfile
#import sounddevice
from scipy.fftpack import fft
from numpy.lib.stride_tricks import as_strided
import soundfile

# implementation of spectrogram

#Implement a function to compute power spectrogram
def MFCC(s,n_fft,winsize,hopsize,Mel_filterbank):
    
    #win_funct = signal.hann(winsize,sym=False)
    
##############################################################################
    win_funct = scipy.signal.hamming(512, sym=False)
    #winsize=512
#   1   Window each frame (using hamming window) Hint: ​signal.hamming
   
    window_analysis = np.sqrt(win_funct)
    
# TASK 2 b)50 % overlap.    
    n_frames=int((len(s)-winsize)/hopsize)+1
    y_frames=as_strided(s,(winsize,n_frames),(s.itemsize,hopsize*s.itemsize))
                            #shape             #strides (step) 
                            # both are sequences of integer
    spectrogram_long = np.zeros((N_fft//2+1,n_frames))
    End_mel_spec = np.zeros((40,n_frames))
    End_log_mel_spec = np.zeros((40,n_frames))
    End_DCT = np.zeros((40,n_frames))
    
# TASK 2 a) Multiply each signal frame with a windowing function (use Hamming).
    for i in np.arange(n_frames):
        a=y_frames[:,i]
#   1   Window each frame (using hamming window) Hint: ​signal.hamming
        window_frame=a*window_analysis

#   2   Calculate the fft
        spectrum=fft(window_frame,N_fft)
        # we specify 
        
        
#   3   Collect the power spectrum (you will get power spectrum)
        spectrum=abs(spectrum[:N_fft//2+1].reshape((257,1)))
        pow_spectrum=np.power(spectrum,2)
        
#   4   Multiply it with mel filterbank you created in Question 1
        mel_spectrum = numpy.dot(Mel_filterbank,pow_spectrum)
        
#   5   Take log operation (you will get logarithmic mel spectrum) Hint: ​20 * np.log10
        logarithmic_mel_spectrum = 20 * np.log10(mel_spectrum)
#   6   DCT (Finally, you will get MFCC)
        DCT=scipy.fftpack.dct(logarithmic_mel_spectrum, type=2, n=None, axis=0, norm=None, overwrite_x=False)[:40]
        
        spectrogram_long[:,i]   =  spectrum.flatten() 
        End_mel_spec[:,i]       =  mel_spectrum.flatten()
        End_log_mel_spec[:,i]=logarithmic_mel_spectrum.flatten()
        End_DCT[:,i]=DCT.flatten()
        
    return spectrogram_long, End_mel_spec, End_log_mel_spec, End_DCT
############################################################################## 



def plot_spectrogram(spec,s,fs,audio,win_len):
    plt.figure()
    
    if xpectrogram == "Log Power spectrogram, magnitude,":
        plt.imshow(20*np.log10(np.abs(spec)**2),origin="lower",aspect="auto")
        #power spectrogram
        
        
    locs,labels = plt.xticks()
    plt.xticks(locs[1:-1],np.round(locs/locs[-1]*len(s)/fs,decimals=1)[1:-1])
    locs,labels = plt.yticks()
    locs_=[int((i/locs[-1]*fs//2)) for i in locs]
    plt.yticks(locs[1:-1],locs_[1:-1])
    plt.ylabel("Frequency [hz]")
    plt.xlabel("Time [sec]")
    plt.title("Log Power spectrogram")



##READING AUDIO AND EMPHASIZING SIGNAL
audioS,fs  = soundfile.read('audio.wav')
signalx=audioS
emphasized_signal = np.append(signalx[0], signalx[1:] - pre_emphasis * signalx[:-1])
s=emphasized_signal


##DEFINING WINDOW LENGTH AND HOP SIZE
window_length=512
n_fft = window_length
n_fft=N_fft
hop_size = window_length//2
winsize=window_length

##APPLYING MFFC FUNCTION
spectrogram_long, End_mel_spec, End_log_mel_spec, End_DCT=MFCC(s,n_fft,winsize,hop_size,Mel_filterbank)
spec=spectrogram_long

##PLOTTING SPECTROGRAM
xpectrogram = "Log Power spectrogram, magnitude,"
plot_spectrogram( spec,s,fs,audioS,window_length)
plt.show()

librosa.display.specshow(End_mel_spec)
plt.title("mel_spec")
plt.show()
librosa.display.specshow(End_log_mel_spec)
plt.title("log_mel_spec")
plt.show()

plt.figure()
librosa.display.specshow(End_DCT, x_axis='time')
plt.colorbar()
plt.title("own_MFCC")

MFCClibrosa=lb.feature.mfcc(y=s, sr=fs, S=None, n_mfcc=20, dct_type=2, norm='ortho', lifter=0)
plt.figure()
lb.display.specshow(MFCClibrosa, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()
plt.show()

# =============================================================================
# m_slaney = lb.feature.mfcc(y=s, sr=fs, dct_type=2)
# m_htk = lb.feature.mfcc(y=s, sr=fs, dct_type=3)
# plt.figure(figsize=(10, 6))
# plt.subplot(2, 1, 1)
# lb.display.specshow(m_slaney, x_axis='time')
# plt.title('RASTAMAT / Auditory toolbox (dct_type=2)')
# plt.colorbar()
# plt.subplot(2, 1, 2)
# lb.display.specshow(m_htk, x_axis='time')
# plt.title('HTK-style (dct_type=3)')
# plt.colorbar()
# plt.tight_layout()
# plt.show()
# =============================================================================
