import numpy as np
import scipy
from scipy.io import wavfile
from numpy.fft import fft, ifft, fftshift
import librosa as lb
#import sys
from scipy.signal import hann
import matplotlib.pyplot as plt  # For ploting
import sounddevice


# Problem 1: ​Frequency modulation (FM) sound synthesis
fs = 16000  # Sampling rate: 16000 Hz 
cf = 880    # Carrier frequency: 880 Hz
fmod = 220  # modulation frequency fmod=220 Hz!
A=1         # Amplitude: 1
I =2        # Modulation index: 2
sd = 1      # Signal duration: 1 s


plt.title('an FM synthesis signal ')
t=np.linspace(0,sd,fs*sd)
y = A*np.sin(2*np.pi*cf*t+I*np.sin(2*np.pi*fmod*t))

# Now plot the signal and its DFT spectrum
plt.plot(t[:300],y[:300])
# plt.grid(True, 'both')
plt.show()
# sounddevice.play(y,fs)
# Now plot the signal and its DFT spectrum
DFT=np.abs(scipy.fft(y))
plt.plot(DFT)

plt.show()




def princarg(phase_in):
  """
  Computes principle argument,  wraps phase to (-pi, pi]
  """
  phase = np.mod(phase_in + np.pi,-2*np.pi)+np.pi;
  return phase
  



def delta_phi_(Phase_current, Phase_previous, winHopAn, wLen):
    """
    Function for calculating unwrapped phase difference between consecutive frames
    
    Phase_current: current frame phase
    Phase_previous: previous frame phase
    winHopAn: Analysis hop length
    wLen: window length
    """
    
    # nominal phase increment for the analysis hop size for each bin
    omega = 2*np.pi*(winHopAn/wLen)*np.arange(0, wLen)
    delta_phi = omega + princarg(Phase_current-Phase_previous-omega)
    
    return delta_phi
    
R=1.4
# A Loop for overap add reconstruction  with no spectral processing in between    
audioIn, fs=lb.load('audio.wav', sr=None)   # read audio
audioOut = np.zeros(len(audioIn))        # placeholder for reconstructed audio
audioOut2 = np.zeros(int(len(audioIn)*R)) 
wLen = int(0.032*fs)                   # window length
winAn = np.sqrt(hann(wLen, sym=False)) # analysis window
winSyn =winAn
winHopAn = int(0.008*fs)             # Hop length or frame advance
inInd = 0
synInd = 0
winHopSyn = int(0.008*fs) 

Phase_previous=0
Phase_previous_S=0

while inInd< len(audioIn)-wLen:

##11111
  # selct the frame and multiply with window function
  frame = audioIn[inInd:inInd+wLen]* winAn 
  # compute DFT
  f = fft(frame)
  
##22222  
  # save magnitudes and phases
  mag_f = np.abs(f)          # Mag(t) 
  phi0 = np.angle(f)         # Phase(t) 
  
##33333 444444 555555
  ####################
  # processing in spectral domain
  Phase_current = phi0
  delta_phi=delta_phi_(Phase_current, Phase_previous, winHopAn, wLen)
  #DeltaPhase(t)
  
  
      
  Phase_previous=Phase_current
  synth_phase = Phase_previous_S+R* delta_phi
#  This adjusts the phase difference between adjacent frames to what
#  it must be for the modified hop size
  
  phi0 =  princarg(synth_phase)
  
  
  Phase_previous_S= phi0
  
  
 #######################


  
  # Recover the complex FFT back
  ft = (abs(f)* np.exp(1j*phi0))  


##66666
  # inverse DFT and windowing
  frame = np.real(ifft(ft))*winSyn
  
##77777  
  # Ovelap add
 
  audioOut2[synInd :synInd +wLen] =  audioOut2[synInd :synInd +wLen] + frame
  # frame advance by winHopAn
  inInd = inInd + winHopAn
  synInd = synInd + int(winHopAn*R)

# sounddevice.play(audioIn,fs)
# sounddevice.play(audioOut2,fs)

#  Plot signal Original
plt.plot(audioIn[int(0.0*fs):int(5*fs)])
# label the axes
plt.ylabel("Amplitude")
plt.xlabel("number of samples")
# set the title  
plt.title("Original signal")
# display the plot
plt.show()
#  Plot signal Stretched
plt.plot(audioOut2[int(0.0*fs):int(5*fs)])
# label the axes
plt.ylabel("Amplitude")
plt.xlabel("number of samples")
# set the title  
plt.title("Stretched signal")
# display the plot
plt.show()


##bonus 
# We resample stretched audio

#
pitched= lb.resample(audioOut2,fs*R,fs)
sounddevice.play(pitched,fs)

#  Plot signal Stretched
plt.plot(pitched[int(0.0*fs):int(5*fs)])
# label the axes
plt.ylabel("Amplitude")
plt.xlabel("number of samples")
# set the titles
plt.title("Pitch shifting")
# display the plot
plt.show()
