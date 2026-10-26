import wfdb
import pandas as pd

# Read record (replace '100' with the record you want)
record = wfdb.rdrecord('101', sampfrom=0, sampto=3600)  # first 3600 samples (~10s)
signal = record.p_signal[:,0]  # if two channels, choose one lead

# Save as single-column CSV
pd.DataFrame(signal, columns=['amplitude']).to_csv("101_test.csv", index=False)
