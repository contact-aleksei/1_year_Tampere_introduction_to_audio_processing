import numpy as np
import scipy
import librosa as lb
from scipy.signal import hamming
from matplotlib import pyplot as plt
from audiolazy import lazy_lpc as lpc

# interactive plotting
plt.ion()

def est_predictor_gain(x, a, p):
    '''
    A function to compute gain of the residual signal in LP analyis.
    x:  signal 
    a: LPC coefficients
    p: order of the filter
    '''
    cor = np.correlate(x, x, mode='full')
    
    rr = cor[len(cor)//2: len(cor)//2+p+1]
    g = np.sqrt(np.sum(a*rr))
    return g
   
def reject_outliers(data, m=2):
    '''
    Function to reject outliers. All values beyond m standard deviations from means are excluded
    '''
    return data[abs(data - np.mean(data)) < m * np.std(data)]
    

def LPA(audio, p):
    # # read audio
    audioIn, fs=lb.load(audio, sr=None)   

    # filter order
    # p = 4                    # has to be tuned

    # number of DFT points
    nfft = 1024

    inInd =0
    wLen = int(0.02*fs) # 20 ms window
    win = hamming(wLen) # hamming window for example

    cnt = 0
    numframes = np.ceil( (len(audioIn)-wLen)/(wLen/2)) # number of franes 
    formants  = []                                     # A placeholder for storing formants

    # choose a representative frame of the vowel
    plot_frame = int(numframes/2)  # middle of the vowel

    # The analysis loop
    while inInd< len(audioIn)-wLen:
        
        # audio frame
        frame = audioIn[inInd:inInd+wLen]* win    
        
        
        #frame = reject_outliers(frame, m=2)
        # compute LPC and gain 
        
        #LPC = lb.autocorrelate(frame)
        LPC=lpc.lpc.autocor(frame,order=p).numerator
        g=est_predictor_gain(frame, LPC, p)
        
        # Compute the filter tansfer function
        w, h = scipy.signal.freqz(g, a=LPC, fs=fs)
        
        # Compute DFT spectrum
        DFT=np.abs(scipy.fft(frame,n=nfft))
        
    # Problem 2: ​Formant analysis.    
        # Compute roots of
        roots=np.roots(LPC)
           
        #  LPC coefficients are real-valued, the roots occur in complex conjugate pairs.
        # Retain only the roots with +ve sign for the imaginary part 
        

        roots = roots[np.imag(roots) >= 0]
        
        # compute formants from roots
        angz=np.angle(roots)
    
        # convert to Hertz from angular frequencies
        angz = angz*(fs/(2*np.pi))
    
        # sort the formants in increasing order
        angz = np.sort(angz)
        
        # remove zero frequencies
        angz = angz[angz !=0]
        
        # First three formants
        if angz[:3].shape!=(0,):
            formants.append(angz[:3]) 
        
        inInd = inInd + int(wLen/2) # frame advance
        
        cnt = cnt+1 
        
        # plot the FFT spectrum and LPC spectrum here for chosen frame
        if cnt == plot_frame :
            # plot DFT spectrum (remember both in dB scale)
            line = np.linspace(0, fs/2, nfft//2)
            plt.plot(line, np.log(DFT[0:int(nfft/2)]))
            # plot LPC spectrum
            plt.plot(w, np.log(np.abs(h)))
            plt.show()
    formants = np.array(formants)
    print('------ The computed formants are :', np.mean(formants, 0))
        

# has to be tuned
# Refine formant estimations (optional)
LPA('e.wav', 4)


LPA('i.wav', 20)


LPA('o.wav', 50)
#BONUS
LPA('oboe59.wav', 20)






