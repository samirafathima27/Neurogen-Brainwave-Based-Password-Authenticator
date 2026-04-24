import pandas as pd
import numpy as np

# Load features
df = pd.read_csv(r"C:\neurogen\output\eeg_features_all_v2.csv")

# Genuine samples
genuine = df.copy()
genuine["Label"] = 1

# Impostor samples (add noise ONLY to features)
impostor = df.copy()

noise = np.random.normal(
    loc=0,
    scale=0.05,
    size=impostor.shape
)

impostor = impostor + noise
impostor["Label"] = 0

# Combine
final_df = pd.concat([genuine, impostor], ignore_index=True)

# Save
final_df.to_csv(
    r"C:\neurogen\output\eeg_final_labeled_v2.csv",
    index=False
)

print("Labeled v2 dataset created successfully.")
print("Total samples:", len(final_df))
