# SSVEP Analyzer

The SSVEP Analyzer is a Python class designed to analyze Steady-State Visual Evoked Potentials (SSVEP) in EEG data. It provides methods for plotting, filtering, transforming, and extracting features from EEG data.

A full example of the SSVEP Analyzer can be found in `example.ipynb`.

### Initialization

```python
analyzer = SsvepAnalyzer(eeg_data, sampling_rate=250, stimulus_frequency=10)
```

## Methods
```python
plot_eeg(eeg_data)
```
#### Parameters
- eeg_data: A 2D numpy array with shape (samples, channels).
#### Returns
- matplotlib figure


```python
apply_linear_detrending(eeg_data)
```
#### Parameters
- eeg_data: A 2D numpy array with shape (samples, channels).
#### Returns
- A 2D numpy array with shape (samples, channels).

```python
apply_lowpass_filter(eeg_data, cutoff=100, filter_order=5)
```
#### Parameters
- eeg_data: A 2D numpy array with shape (samples, channels).
- cutoff: The cutoff frequency (default 100 Hz).
- filter_order: The order of the filter (default 5).
#### Returns
- A 2D numpy array with shape (samples, channels).

```python
apply_highpass_filter(eeg_data, cutoff=1, filter_order=5)
```
#### Parameters
- eeg_data: A 2D numpy array with shape (samples, channels).
- cutoff: The cutoff frequency (default 1 Hz).
- filter_order: The order of the filter (default 5).
#### Returns
- A 2D numpy array with shape (samples, channels).

```python
apply_notch_filter(eeg_data, notch_freq=50, bandwidth=1, filter_order=5)
```
#### Parameters
- eeg_data: A 2D numpy array with shape (samples, channels).
- notch_freq: The central frequency to remove (default 50 Hz).
- bandwidth: The width of the notch (default 1 Hz).
- filter_order: The order of the filter (default 5).
#### Returns
- A 2D numpy array with shape (samples, channels).

```python
compute_cca(eeg_data, n_components=1, n_harmonics=2)
```
Computes Canonical Correlation Analysis between the EEG data and a generated reference signal.
#### Parameters
- eeg_data: A 2D numpy array with shape (samples, channels).
- n_components: Number of components to keep (default 1).
- n_harmonics: Number of harmonics to include in the reference signal (default 2).
#### Returns
- The fitted cca model and the reference signals.

```python
compute_reduced_signal(eeg_data, n_components=1, n_harmonics=2)
```

Reduces the dimensionality of the EEG data using CCA.
#### Parameters
- eeg_data: A 2D numpy array with shape (samples, channels).
- n_components: Number of components to keep (default 1).
- n_harmonics: Number of harmonics for CCA (default 2).
#### Returns
- A 2D numpy array with shape (samples, n_components).

```python
compute_power_spectrum(eeg_data)
```
#### Parameters
- eeg_data: A 1D numpy array of EEG data samples.
#### Returns
- A 1D numpy array of frequencies
- A 1D numpy array of the power spectrum.

```python
plot_power_spectrum(frequencies, spectrum)
```
#### Parameters
- frequencies: Frequencies of the power spectrum.
- spectrum: Amplitudes of the power spectrum.
#### Returns
- matplotlib figure

```python
plot_coefficient_matrix(coefficient_matrix)
```
#### Parameters
- coefficient_matrix: A 2D numpy array representing the coefficient matrix.
#### Returns
- matplotlib figure

```python
compute_running_r_values(eeg_data, marker=None, n_components=1, n_harmonics=2, window_duration=2, step_size=40)
```
Computes r values of a CCA per window between EEG data and a reference signal with n_harmonics harmonics of the stimulus frequency.
#### Parameters
- eeg_data: A 2D numpy array with shape (samples, channels).
- marker: Optional marker data to annotate segments (default None).
- n_components: Number of components to keep (default 1).
- n_harmonics: Number of harmonics for CCA (default 2).
- window_duration: Duration of the window for analysis (default 2 seconds).
- step_size: The step size for windowing (default 40 samples).
#### Returns
- A 1D numpy array of r values.
- A 1D numpy array of times.
- If marker is not None, a 1D numpy array of marker values.

```python
plot_r_values(r_values, marker_values=None)
```
#### Parameters
- r_values: The r values to plot.
- times: The times corresponding to the r values.
- marker_values: Optional marker data for coloring (default None).
#### Returns
- matplotlib figure

```python
compute_wavelet_transform(eeg_data, w=50, frequency_range=(1, 35), n_frequencies=100, n_times=100)
```

Computes the wavelet transform for EEG data using a Morelet wavelet with parameter w.
#### Parameters
- eeg_data: A 2D numpy array with shape (samples, channels).
- w: The parameter w for the Morlet wavelet (default 50).
- frequency_range: A tuple representing the frequency range (default (1, 30) Hz).
- n_frequencies: The number of frequencies to compute (default 100).
- n_times: The number of times to compute (default 100).
#### Returns
- A 1D numpy array of frequencies.
- A 1D numpy array of times.
- A 2D numpy array of wavelet coefficients.

```python
plot_wavelet_transform(frequencies, times, cwt_matrix)
```
#### Parameters
- frequencies: Frequencies corresponding to the wavelet coefficients.
- times: Times corresponding to the wavelet coefficients.
- cwt_matrix: The computed wavelet transform coefficients.
#### Returns
- matplotlib figure