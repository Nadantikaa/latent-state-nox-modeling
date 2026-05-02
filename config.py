class Config:

    # DATA
    DATA_PATH = "data/raw_000010_measurement_000000.csv"

    # TRAINING
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 1e-3

    # MODEL
    INPUT_DIM = 13
    LATENT_DIM = 32

    # DYNAMICS
    TAU = 1