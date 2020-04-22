# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 12:11:33 2019

@author: OWNER
"""
"""FIRST TASK"""
import numpy as np # to work with numerical data efficiently
import matplotlib.pyplot as plot # For ploting
#import winsound
import scipy
import scipy.io
import scipy.io.wavfile
import sounddevice
#pip install sounddevice

fs = 8000 # sample rate 
f1 = 100 # the frequency of the signal
f2 = 500 # the frequency of the signal
f3 = 1000 # the frequency of the signal
f4 = 2500 # the frequency of the signal
x = np.arange(fs) # the points on the x axis for plotting

plot.title('sine wave 1')
A=5
ph=np.pi
t=np.linspace(0,3,fs*3)
y1 = A*np.sin(2*np.pi*f1*t+ph)
plot.plot(t[:300],y1[:300])
plot.show()
scipy.io.wavfile.write('audio1.wav',8000,y1)
sounddevice.play(y1,fs)

plot.title('sine wave 2')
A=2
ph=np.pi/2
t=np.linspace(0,3,fs*3)
y2 = A*np.sin(2*np.pi*f2*t+ph)
plot.plot(t[:300],y2[:300])
plot.show()
sounddevice.play(y2,fs)

plot.title('sine wave 3')
A=3
ph=np.pi*3/14
t=np.linspace(0,3,fs*3)
y3 = A*np.sin(2*np.pi*f3*t+ph)
plot.plot(t[:300],y3[:300])
plot.show()
sounddevice.play(y3,fs)

plot.title('sine wave 4')
A=7
ph=np.pi*3/2
t=np.linspace(0,3,fs*3)
y4 = A*np.sin(2*np.pi*f4*t+ph)
plot.plot(t[:300],y4[:300])
plot.show()
sounddevice.play(y4,fs)

#c) Add them up to x(t). Plot and play x(t). Write the signal to a wav file.
plot.title('SUMMED UP wave of 4 sinusoids')
x=y2+y3+y4+y1;
t=np.linspace(0,3,fs*3)
plot.plot(t[:300],x[:300])
plot.show()
sounddevice.play(x,fs)

#d) Apply DFT. Plot magnitude DFT.
DFT=np.abs(scipy.fft(x, n=512))
plot.plot(DFT)
plot.show()



"""SECOND TASK""" 
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import numpy

# read audio samples
print("first audio")
input_data = read("audio3.wav")
audio = input_data[1]
fs=input_data[0]
sounddevice.play(audio,fs)

#  Plot signal between 0.5 and 1 s.
plt.plot(audio[int(0.5*fs):int(1*fs)])
# label the axes
plt.ylabel("Amplitude")
plt.xlabel("Time")
# set the title  
plt.title("Sample Wav")
# display the plot
plt.show()


increment=0.1
fs=input_data[0]
end=len(audio)/fs
start=1
#c) and d) unction to apply DFT to segments and plot the magnitude DFT
def printingDFT(start,audio,increment,end):
    #plotting the first segment
    plt.plot(audio[int(start*fs):int((start+increment)*fs)])
    plt.title("plotting the first segment")
    plt.show()
    #and corresponding magnitude DFT
    DFT=np.abs(scipy.fft(audio[int(start*fs):int((start+increment)*fs)], n=512))
    plt.plot(DFT)
    plt.title(" plotting corresponding magnitude DFT is enough")
    plt.show()
    
    #loop basically computes magnitude DFT for each of the 100 ms segments of the
    #audio
    for x in numpy.arange(1, end, increment):
        DFT=np.abs(scipy.fft(audio[int(start*fs):int((start+increment)*fs)], n=512))
        start=+increment

print("first audio, function applied")
# This function is for the first audio        
printingDFT(start,audio,increment,end)        
        
        

#e) Do the same for the other audio and analyze the difference in spectrum between the
#two audio examples. How does the spectrum of these signals differ from that of sum of
#sinusoids?
print("second audio")
#  Plot signal between 0.5 and 1 s.

input_data = read("audio2.wav")
audio = input_data[1]
#defining sampling frequence audio2
fs=input_data[0]

plt.plot(audio[int(0.5*fs):int(1*fs)])
# label the axes
plt.ylabel("Amplitude of the second audio")
plt.xlabel("Time of the second audio")
# set the title  
plt.title("Sample Wav of the second audio")
# display the plot OF THE SECOND AUDIO
plt.show()

print("second audio")
printingDFT(start,audio,increment,end)
sounddevice.play(audio,fs)


"""THIRD TASK"""

import scipy.signal
import scipy.io.wavfile
import scipy.io
from scipy.io import wavfile
from scipy.io.wavfile import read
#from scipy.io.wavfile import write
#BONUS TASK
#In problem1 downsample the sum of sinusoids x(t) by a factor of 2 using
#scipy.signal.resample​.
print("THIRD TASK THIRD TASK THIRD TASK THIRD TASK THIRD TASK THIRD TASK ")

x = y1+y2+y3+y4
A = 4
fs=8000
newsampling_frequency=int(fs/2)
newnumberof_samples = int(len(x)/2);
#In problem1 downsample the sum of sinusoids x(t) by a factor of 2 using
#scipy.signal.resample
downsample = scipy.signal.resample(x,newnumberof_samples)
#Write the downsampled signal into wav file.
#scipy.io.wavfile.write('downsample.wav', fs/2, downsample)

#writing original file into wav file.
wavfile.write("sampled.wav", fs, x)
#writing downsampled file into wav file.
wavfile.write("downsampled.wav", newsampling_frequency, downsample)


#Plot the magnitude
#DFT and compare it with DFT of x(t).
t=np.linspace(0,3,fs*3)
plot.plot(t[:300], downsample[:300])
# label the axes
plt.ylabel("Amplitude")
plt.xlabel("Time")
# set the title  
plt.title("DOWNSAMPLED Wav audio")
# display the plot
plt.show()


#d) Apply DFT. Plot magnitude DFT.
DFTdownsampled=np.abs(scipy.fft(downsample, n=512))
plt.title("Magnitude discrete Fourier transform (DFT) of DOWNSAMPLED x signal")
plot.plot(DFTdownsampled)
plot.show()



