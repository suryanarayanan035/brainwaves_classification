from scipy.signal import butter,sosfiltfilt

"""
This function creates a butterpass filter and applies it to the given signal
"""
def butter_bandpass_filter(signal,fs,order,band=[30,100]):
    nyq = fs*0.5
    start_cutoff = band[0]/nyq
    stop_cutoff = band[1]/nyq
    sos = butter(order,[start_cutoff,stop_cutoff],btype="bp",output="sos")
    y = sosfiltfilt(sos,signal)
    return y

"""
This function apllies filter on each channel of given list of signals
"""
def filter_signals(signals,sample_frequency,order,band):
    filtered_signals = []
    for signal in signals: #iterating through signals
        filtered_channels=[] #iterating though channels of signals
        for channel in signal:
            filtered_channel=butter_bandpass_filter(channel, sample_frequency, order, band)
            filtered_channels.append(filtered_channel)
        filtered_signals.append(filtered_channels)
    return filtered_signals

# filtered_signals=signals