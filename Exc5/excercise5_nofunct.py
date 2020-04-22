import scipy
import librosa as lb
#from scipy.signal import hamming
from matplotlib import pyplot as plt
#from audiolazy import lazy_lpc as lpc
from librosa import filters
from librosa import display
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


signal=audioIn
emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])


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
#pip install sounddevice
from scipy import signal 
from scipy.fftpack import fft,ifft
from numpy.lib.stride_tricks import as_strided
import librosa
import soundfile

# implementation of spectrogram


############################################################################## 



def plot_spectrogram(spec,s,fs,audio,win_len):
    plt.figure()
    
    if xpectrogram == "Power spectrogram, magnitude,":
        plt.imshow(np.abs(spec)**2,origin="lower",aspect="auto")
        #power spectrogram
    if xpectrogram == "Log spectrogram, magnitude,":
        plt.imshow(np.log(np.abs(spec)+0.0005),origin="lower",aspect="auto")
        #log spectrogram
        
        
    locs,labels = plt.xticks()
    plt.xticks(locs[1:-1],np.round(locs/locs[-1]*len(s)/fs,decimals=1)[1:-1])
    locs,labels = plt.yticks()
    locs_=[int((i/locs[-1]*fs//2)) for i in locs]
    plt.yticks(locs[1:-1],locs_[1:-1])
    plt.ylabel("Frequency [hz]")
    plt.xlabel("Time [sec]")
    plt.title(xpectrogram+" "+audio.split(".")[0]+' '+"with window size"+' '+str(win_len)+' '+'ms.')




def ISTFT(spectrogram_long,s,winsize,hopsize):
    
    win_funct = signal.hann(winsize,sym=False)
    window_synthesis = np.sqrt(win_funct)
    
    n_frames=spectrogram_long.shape[1]
    result=np.zeros((s.shape[0]))
    
    for i in np.arange(n_frames):
        
        a=spectrogram_long[:,i]
        b=np.conjugate(spectrogram_long[:,i][-2:0:-1])
        c=np.concatenate((a,b))
        out_long=ifft(c).real
        out_long_wnd=window_synthesis*out_long
        result[i*hopsize:i*hopsize+2*hopsize]=result[i*hopsize:i*hopsize+2*hopsize]+out_long_wnd
    return result

##TASK 2 A Now run your implementation in Problem 1 with different window sizes
# for different types of signals,
audio = ['audio.wav','audio.wav','audio.wav']
audio = audio[2]
s, fs = soundfile.read(audio)
#use 16ms, 32 ms, 64 ms, and 128 ms.
window_sizes = [16,32,64,128]
window_size = [int(i*0.001*fs) for i in window_sizes]
k=0 

for window in window_size:
    n_fft = window
    hop_size= window//2
    
    #TASK 2 A
    #Calculate spectrogram with a library function
    spec_library = librosa.core.stft(s,n_fft=window, win_length=window,hop_length=hop_size)
    #then: spec=spec_library
    
    xpectrogram ="Power spectrogram, magnitude,"
    plot_spectrogram( spec,s,fs,audio,window_sizes[k])       
    xpectrogram ="Log spectrogram, magnitude,"  
    plot_spectrogram( spec,s,fs,audio,window_sizes[k])
    
    
    k+=1
   
s,fs  = soundfile.read('audio.wav')

signal=s
emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
s=emphasized_signal

window_length=int(64*0.001*fs)

n_fft = window_length
n_fft=N_fft
hop_size = window_length//2



#s_=ISTFT(spec,s,window_length,hop_size)  





#win_funct = signal.hann(winsize,sym=False)
    
##############################################################################
win_funct = signal.hamming(winsize,sym=False)
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
    DCT=scipy.fftpack.dct(logarithmic_mel_spectrum, type=2, n=None, axis=-1, norm=None, overwrite_x=False)[:40]
        
    spectrogram_long[:,i]   =  spectrum.flatten() 
    End_mel_spec[:,i]       =  mel_spectrum.flatten()
    End_log_mel_spec[:,i]=logarithmic_mel_spectrum.flatten()
    End_DCT[:,i]=DCT.flatten()
xpectrogram ="Power spectrogram, magnitude,"
plot_spectrogram( spec,s_,fs,audio,64)
xpectrogram ="Log spectrogram, magnitude,"  
plot_spectrogram( spec,s_,fs,audio,64)
