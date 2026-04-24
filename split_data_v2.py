import pandas as pd

df = pd.read_csv(r"C:\neurogen\output\eeg_final_labeled_v2.csv")

genuine = df[df["Label"] == 1].sample(frac=1, random_state=42)
impostor = df[df["Label"] == 0].sample(frac=1, random_state=42)

train_df = pd.concat([
    genuine.iloc[:100],
    impostor.iloc[:100]
]).sample(frac=1, random_state=42)

test_df = pd.concat([
    genuine.iloc[100:125],
    impostor.iloc[100:125]
]).sample(frac=1, random_state=42)

train_df.to_csv(r"C:\neurogen\output\train_v2.csv", index=False)
test_df.to_csv(r"C:\neurogen\output\test_v2.csv", index=False)

print("Split v2 completed.")
print("Train:", len(train_df), "Test:", len(test_df))
