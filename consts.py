import numpy as np

all_well_names = np.array([i + 1 for i in range(48) if not i % 8 in [0, 7]])
allWellNames = all_well_names
offWallWellNames = np.array([11, 12, 13, 14, 19, 20, 21, 22, 27, 28, 29, 30, 35, 36, 37, 38])
TRODES_SAMPLING_RATE = 30000
LFP_SAMPLING_RATE = 1500.0
CM_PER_FT = 30.48
