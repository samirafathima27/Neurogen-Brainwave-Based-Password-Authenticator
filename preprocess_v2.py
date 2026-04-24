import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch

DATASET_PATH = r"C:\neurogen\datasets"
OUTPUT_PATH = r"C:\neurogen\output"
FS = 512

os.makedirs(OUTPUT_PATH, exist_ok=True)

def bandpass_filter(signal, fs, low=0.5, high=40):
    nyq = 0.5 * fs
    b, a = butter(4, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, signal)

def bandpower(signal, fs, fmin, fmax):
    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    idx = (freqs >= fmin) & (freqs <= fmax)
    return np.trapz(psd[idx], freqs[idx])

features = []

for file in os.listdir(DATASET_PATH):
    if not file.endswith(".csv"):
        continue

    df = pd.read_csv(os.path.join(DATASET_PATH, file))
    eeg = df["EEG"].values
    eeg = eeg[~np.isnan(eeg)]

    if len(eeg) < 50:
        continue

    eeg = bandpass_filter(eeg, FS)
    eeg = (eeg - np.mean(eeg)) / np.std(eeg)

    
    alpha = bandpower(eeg, FS, 8, 13)
    beta  = bandpower(eeg, FS, 13, 30)
    theta = bandpower(eeg, FS, 4, 8)

    
    mean_val = np.mean(eeg)
    std_val  = np.std(eeg)
    var_val  = np.var(eeg)

    features.append([alpha, beta, theta, mean_val, std_val, var_val])

df_feat = pd.DataFrame(
    features,
    columns=["Alpha", "Beta", "Theta", "Mean", "Std", "Var"]
)

df_feat.to_csv(
    r"C:\neurogen\output\eeg_features_all_v2.csv",
    index=False
)

print("Feature extraction v2 completed.")
print("Samples:", len(df_feat))
