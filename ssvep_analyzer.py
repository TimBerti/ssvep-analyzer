import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, lfilter, cwt, morlet2
from sklearn.cross_decomposition import CCA
from collections import Counter

sns.set_theme('notebook', 'whitegrid', 'dark')
figsize = (11, 7)

class SsvepAnalyzer:

    def __init__(self, sampling_rate, stimulus_frequency):
        self.sampling_rate = sampling_rate
        self.stimulus_frequency = stimulus_frequency
        self.nyquist_frequency = 0.5 * sampling_rate

    def plot_eeg(self, eeg_data):
        eeg_data = eeg_data - np.mean(eeg_data, axis=0)

        num_channels = eeg_data.shape[1]
        num_points = eeg_data.shape[0]
        time = np.arange(0, num_points) / self.sampling_rate

        _, ax = plt.subplots(figsize=(figsize[0], num_channels))

        max_diff = np.max(np.max(eeg_data, axis=0) - np.min(eeg_data, axis=0))
        offsets = np.arange(num_channels, 0, -1) * max_diff
        
        for i in range(num_channels):
            ax.plot(time, eeg_data[:, i] + offsets[i], label=f"Channel {i}", linewidth=0.5, color='black')
            ax.axhline(y=offsets[i], color='gray', linestyle='--', linewidth=0.5)

        ax.set_xlabel("Time (s)")
        ax.set_yticks(offsets)
        ax.set_yticklabels([f"Channel {i}" for i in range(num_channels)])
        plt.show()
        return ax
    
    def filter_extreme_values(self, eeg_data, threshold_factor=3, centering=np.mean):
        eeg_data = eeg_data - centering(eeg_data, axis=0)
        threshold = threshold_factor * np.std(eeg_data, axis=0)
        return np.where(np.abs(eeg_data) > threshold, 0, eeg_data)
    
    def apply_linear_detrending(self, eeg_data):
        return np.apply_along_axis(lambda x: x - np.polyval(np.polyfit(np.arange(len(x)), x, 1), np.arange(len(x))), 0, eeg_data)

    def apply_lowpass_filter(self, eeg_data, cutoff=35, filter_order=5):
        normalized_cutoff = cutoff / self.nyquist_frequency
        b, a = butter(filter_order, normalized_cutoff, btype='low')
        return np.apply_along_axis(lambda x: lfilter(b, a, x), 0, eeg_data)

    def apply_highpass_filter(self, eeg_data, cutoff=0.1, filter_order=5):
        normalized_cutoff = cutoff / self.nyquist_frequency
        b, a = butter(filter_order, normalized_cutoff, btype='high')
        return np.apply_along_axis(lambda x: lfilter(b, a, x), 0, eeg_data)

    def apply_notch_filter(self, eeg_data, notch_freq=50, bandwidth=3, filter_order=5):
        low = (notch_freq - bandwidth/2) / self.nyquist_frequency
        high = (notch_freq + bandwidth/2) / self.nyquist_frequency
        b, a = butter(filter_order, [low, high], btype='bandstop')
        return np.apply_along_axis(lambda x: lfilter(b, a, x), 0, eeg_data)

    def compute_cca(self, eeg_data, n_components=1, n_harmonics=2):
        """
        Computes Canonical Correlation Analysis between the EEG data and a generated reference signal 
        with n_harmonics harmonics of the stimulus frequency.
        """
        cca = CCA(n_components=n_components)
        target = self._generate_reference_signals(n_harmonics, eeg_data.shape[0])
        cca.fit(eeg_data, target)
        return cca, target
    
    def _generate_reference_signals(self, n_harmonics, length):
        time_values = np.arange(0, length / self.sampling_rate, 1 / self.sampling_rate)
        signals = []
        for harmonic in range(n_harmonics):
            signals.append(np.sin(2 * np.pi * self.stimulus_frequency * (harmonic + 1) * time_values))
            signals.append(np.cos(2 * np.pi * self.stimulus_frequency * (harmonic + 1) * time_values))
        return np.column_stack(signals)

    def compute_reduced_signal(self, eeg_data, n_components=1, n_harmonics=2):
        """
        Reduces EEG data to n_components dimensions using CCA.
        """
        cca, _ = self.compute_cca(eeg_data, n_components, n_harmonics)
        reduced_signal = cca.transform(eeg_data).flatten()
        return reduced_signal, cca.coef_
    
    def compute_power_spectrum(self, eeg_data):
        spectrum = np.abs(np.fft.rfft(eeg_data, axis=0))
        frequencies = np.fft.rfftfreq(eeg_data.shape[0], 1.0/self.sampling_rate)
        return frequencies, spectrum
    
    def plot_power_spectrum(self, frequencies, spectrum):
        plt.figure(figsize=figsize)
        plt.plot(frequencies, spectrum, color='black')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.show()
        return plt
    
    def plot_coefficient_matrix(self, coefficient_matrix):
        plt.figure(figsize=figsize)
        ax = sns.heatmap(coefficient_matrix, cmap='binary')
        ax.set_xticklabels([f'{i}' for i in range(coefficient_matrix.shape[1])])
        ax.set_yticklabels([f'$\sin ({i//2+1}\omega)$' if i%2==0 else f'$\cos ({i//2+1}\omega)$' for i in range(coefficient_matrix.shape[0])])
        plt.xlabel('Channel')
        plt.ylabel('Harmonic / Phase')
        plt.show()
        return ax

    def compute_running_r_values(self, eeg_data, marker=None, n_components=1, n_harmonics=2, window_duration=2, step_size=40):
        N_samples = eeg_data.shape[0]
        window_size = window_duration * self.sampling_rate
        step_size = step_size

        r_values = []
        marker_values = []
        for start_idx in range(0, N_samples - window_size, step_size):
            eeg_window = eeg_data[start_idx:start_idx+window_size, :]
            eeg_window = eeg_window - np.mean(eeg_window, axis=0)

            if marker is not None:
                # If there are multiple markers in the window, take the most common one
                marker_values.append(Counter(marker[start_idx:start_idx+window_size]).most_common(1)[0][0])
            
            cca, target = self.compute_cca(eeg_window, n_components, n_harmonics)
            x, y = cca.transform(eeg_window, target)
            r = np.corrcoef(x.T, y.T)[0, 1]
            
            r_values.append(r)

        times = np.arange(0, len(r_values) * step_size, step_size) / self.sampling_rate
        
        if marker_values:
            return r_values, times, marker_values
        return r_values, times
    
    def plot_r_values(self, r_values, times, marker_values=None):
        marker_map = self._initialize_marker_map(marker_values)

        plt.figure(figsize=(10, 6))
        for r_value, time, marker in zip(r_values, times, marker_values):
            plt.scatter(time, r_value, marker=marker_map[marker], s=30, color='black')
        
        # Adding a legend for each unique marker
        if marker_values:
            handles = [plt.Line2D([0], [0], marker=marker, color='black', markersize=8) 
                        for marker in marker_map.values()]
            labels = [marker for marker in marker_map.keys()]
            plt.legend(handles, labels, loc='upper right')
        
        plt.xlabel("Time (s)")
        plt.ylabel("r-Value")
        plt.show()
    
    def _initialize_marker_map(self, marker_values):
        unique_markers = np.unique(marker_values)
        markers = ['o', 'x', '^', 's', '+', 'v', 'D', '*', '<', 'p', '>']
        if len(unique_markers) > len(markers):
            raise ValueError("Not enough marker shapes to assign to unique markers.")
        return {marker_value: markers[i] for i, marker_value in enumerate(unique_markers)}

    
    def compute_wavelet_transform(self, eeg_data, w=50, frequencies = np.linspace(0, 35, 300), n_times=300):
        """
        Computes wavelet transform for 1D EEG data using the Morlet wavelet with parameter w.
        """
        widths = w*self.sampling_rate / (2*np.pi*frequencies + 1e-9)

        cwt_matrix = cwt(eeg_data, morlet2, widths=widths, w=w, dtype='complex128')

        idx_mask = cwt_matrix.shape[1] // n_times * np.arange(n_times)
        cwt_matrix = cwt_matrix[:, idx_mask]
        times = idx_mask / self.sampling_rate

        return frequencies, times, cwt_matrix
    
    def plot_wavelet_transform(self, frequencies, times, cwt_matrix):
        df = pd.DataFrame(cwt_matrix, index=frequencies.round(2), columns=times.round(2))
        plt.figure(figsize=figsize)
        ax = sns.heatmap(df, cmap='binary', xticklabels=10, yticklabels=10)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        plt.show()
        return ax