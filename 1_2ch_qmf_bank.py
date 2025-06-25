import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

def design_qmf_filters(N):
    """
    Design low-pass and high-pass QMF filters with power-symmetric property.
    N: filter length (must be even)
    Returns: h0 (low-pass), h1 (high-pass)
    """
    # Design a simple low-pass filter using window method
    # For QMF, we aim for power-symmetric filter: |H(w)|^2 + |H(w + pi)|^2 = 1
    
    n = np.arange(0,N)

    fc = 0.5 # Normalized cutoff frequency
    h0 = np.sinc(fc * (n - (N-1)/2)) * np.hamming(N)
    h0 /= np.sqrt(np.sum(h0**2))  # Normalize energy

    h1 = np.copy(h0)
    h1 *= (-1)**n  # modulate the filter to create high pass

    return h0,h1

def plot_filters(h0, h1):
    # Frequency response
    w, H0 = freqz(h0, worN=1024)
    _, H1 = freqz(h1, worN=1024)

    # Convert magnitude to dB
    H0_dB = 20 * np.log10(np.abs(H0) + 1e-12)  # Add small value to avoid log(0)
    H1_dB = 20 * np.log10(np.abs(H1) + 1e-12)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(w / np.pi, H0_dB, label='Low-pass (h0)')
    plt.plot(w / np.pi, H1_dB, label='High-pass (h1)', linestyle='--')
    plt.title('Magnitude Response of QMF Filters (dB Scale)')
    plt.xlabel('Normalized Frequency (×π rad/sample)')
    plt.ylabel('Magnitude (dB)')
    plt.ylim([-80, 5])  # Limit y-axis to show stopband clearly
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_fft(x, Fs):
    Y0 = np.fft.fft(x)
    N = len(x)
    freqs = np.fft.fftfreq(N,1/Fs)

    plt.figure(figsize=(12, 6))
    plt.plot(freqs[:N//2], (np.abs(Y0)[:N//2]*2)/Fs, label='FFT of Signal')
    plt.title('FFT of Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()


def analysis_bank(x, h0, h1):
    """
    Analysis filter bank: Split input into low-pass and high-pass sub-bands.
    x: input signal
    h0: low-pass filter
    h1: high-pass filter
    Returns: y0 (low-pass output), y1 (high-pass output)
    """
    # Convolve input with filters
    y0 = np.convolve(x, h0, mode='valid')
    y1 = np.convolve(x, h1, mode='valid')
    
    # Downsample by 2
    y0 = y0[::2]
    y1 = y1[::2]

    return y0, y1

def synthesis_bank(y0, y1, f0, f1):
    """
    Synthesis filter bank: Reconstruct signal from sub-bands with alias cancellation.
    y0: low-pass sub-band
    y1: high-pass sub-band
    f0: low-pass synthesis filter
    f1: high-pass synthesis filter
    Returns: reconstructed signal
    """
    # Upsample by 2
    N = max(len(y0), len(y1)) * 2
    x0 = np.zeros(N)
    x1 = np.zeros(N)
    x0[::2] = y0
    x1[::2] = y1
    
    # Apply synthesis filters
    z0 = np.convolve(x0, f0, mode='valid')
    z1 = np.convolve(x1, f1, mode='valid')
    
    # Sum to reconstruct
    return z0 + z1

def qmf_bank(x, N=32):
    
    # Design Analysis Filters
    h0,h1 = design_qmf_filters(N)

    plot_filters(h0, h1)
    
    # Design Synthesis Filters
    f0 = np.copy(h0)
    f1 = -np.copy(h1)
    plot_filters(f0, f1)

    y0, y1 = analysis_bank(x,h0,h1)

    plot_fft(y0, 500)
    plot_fft(y1, 500)

    x_recon = synthesis_bank(y0, y1, f0, f1)

    return x_recon

    
if __name__ == "__main__":
    Fs = 1000  # Sampling frequency
    t = np.arange(0, 1, 1/Fs)
    x = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)#+ 0.5 * np.sin(2 * np.pi * 240 * t)

    X = np.fft.fft(x)
    N = len(x)
    freqs = np.fft.fftfreq(N,1/Fs)

    FN = 32
    x_recon = qmf_bank(x,FN)

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, x, label='Original Signal')
    plt.title('Original Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(freqs[:N//2], (np.abs(X)[:N//2]*2)/Fs, label='FFT of Signal')
    plt.title('FFT of Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()

    X_RECON = np.fft.fft(x_recon)
    N_RECON = len(x_recon)
    freqs = np.fft.fftfreq(N,1/Fs)

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t[:len(x_recon)], x_recon, label='Reconstructed Signal')
    plt.title('Reconstructed Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(freqs[:N_RECON//2], (np.abs(X_RECON)[:N_RECON//2]*2)/Fs, label='FFT of Signal')
    plt.title('FFT of ReconSignal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend() 

    plt.show()