# -*- coding: utf-8 -*-

import getpass

# Base directory for data files
user = getpass.getuser()
DEEPC_BASE = 'D:\Research\deep-clustering'

# Base name of model files
MODEL_BASE = 'model_dcase2019_task3'

# FFT parameters, self-explanatory
FRAME_RATE = 8000
FRAME_LENGTH = .032
FRAME_SHIFT = .008

# Audio mixing parameters
TIMESTEPS = 100    # Number of time bins for the NNET
DB_THRESHOLD = 40  # Difference from max amplitude to be treated as silence

# Clustering parameters
EMBEDDINGS_DIMENSION = 40
MIN_MIX = 2  # Minimum number of mixed speakers for training
MAX_MIX = 2  # Maximum number of mixed speakers for training

# Neural net topology
SIZE_RLAYERS = 300  # Since we use BLSTMs, the number of neurons is doubled
NUM_RLAYERS = 2     # Number of layers

# Training parameter, self-explanatory
SMALL_TEST = False
if SMALL_TEST:
    BATCH_SIZE = 16
    SAMPLES_PER_EPOCH = 256
    VALID_SIZE = 64
    NUM_EPOCHS = 1
else:
    BATCH_SIZE = 128
    SAMPLES_PER_EPOCH = 8192
    VALID_SIZE = 512
    NUM_EPOCHS = 200

STEPS_PER_EPOCH = SAMPLES_PER_EPOCH // BATCH_SIZE
VALIDATION_STEPS = VALID_SIZE // BATCH_SIZE

# Regularization parameters
DROPOUT = 0.5     # Feed forward dropout
RDROPOUT = 0.2    # Recurrent dropout
L2R = 1e-6        # L2 regularization factor
CLIPNORM = 200    # Norm clipping for gradients
