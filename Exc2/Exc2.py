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

#Implement a function to compute power spectrogram
def STFT(s,n_fft,winsize,hopsize):
    
    win_funct = signal.hann(winsize,sym=False)
    window_analysis = np.sqrt(win_funct)
    
# TASK 2 b)50 % overlap.    
    n_frames=int((len(s)-winsize)/hopsize)+1
    y_frames=as_strided(s,(winsize,n_frames),(s.itemsize,hopsize*s.itemsize))
                            #shape             #strides (step) 
                            # both are sequences of integer
    spectrogram_long = np.zeros((n_fft//2+1,n_frames),dtype=np.complex_)
    
# TASK 2 a) Multiply each signal frame with a windowing function (use Hamming).
    for i in np.arange(n_frames):
        a=y_frames[:,i]
        window_frame=a*window_analysis
       
        spectrum=fft(window_frame)
        spectrum=spectrum[:n_fft//2+1]
        
        spectrogram_long[:,i]=spectrum
    return spectrogram_long,np.abs(spectrogram_long)




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




    #BONUS TASK

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
audio = ['audio1.wav','audio3.wav','sampled.wav']
audio = audio[2]
s, fs = soundfile.read(audio)
#use 16ms, 32 ms, 64 ms, and 128 ms.
window_sizes = [16,32,64,128]



window_size = [int(i*0.001*fs) for i in window_sizes]
k=0 

for window in window_size:
    n_fft = window
    hop_size= window//2
    spec,_= STFT(s, n_fft,window, hop_size)
    
    #TASK 2 A
    #Calculate spectrogram with a library function
    spec_library = librosa.core.stft(s,n_fft=window, win_length=window,hop_length=hop_size)
    #then: spec=spec_library
    
    xpectrogram ="Power spectrogram, magnitude,"
    plot_spectrogram( spec,s,fs,audio,window_sizes[k])
        
       
    xpectrogram ="Log spectrogram, magnitude,"  
    plot_spectrogram( spec,s,fs,audio,window_sizes[k])
    
    
    k+=1
   
    
         
    #BONUS TASK
s,fs  = soundfile.read('sampled.wav')
    
window_length=int(64*0.001*fs)
n_fft = window_length
hop_size = window_length//2
spec,_=STFT(s,n_fft,window_length,hop_size)
s_=ISTFT(spec,s,window_length,hop_size)
    
xpectrogram ="Power spectrogram, magnitude,"
plot_spectrogram( spec,s_,fs,audio,64)

xpectrogram ="Log spectrogram, magnitude,"  
plot_spectrogram( spec,s_,fs,audio,64)
   