"""Fixed properties of the G2Net data."""

FS = 2048.0          # sample rate [Hz]
DT = 1.0 / FS        # sample spacing [s]
N = 4096             # samples per detector per sample (2 s)
T = N * DT           # segment duration [s]
DF = 1.0 / T         # frequency resolution of the full-segment DFT [Hz]
N_FREQ = N // 2 + 1  # rfft bins
DETECTORS = ("H1", "L1", "V1")  # channel order in every G2Net .npy file
