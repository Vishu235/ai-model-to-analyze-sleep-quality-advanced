import os

IS_MODAL = os.environ.get("IS_MODAL", "False") == "True"

if IS_MODAL:
    SLEEP_CASSETTE_PATH = "/data/sleep-cassette"
else:
    # Local path to Sleep-EDF Cassette directory.
    # Override with the DATA_DIR env var if needed:
    #   set DATA_DIR=D:\path\to\sleep-cassette
    SLEEP_CASSETTE_PATH = os.environ.get(
        "DATA_DIR",
        r"D:\PES\Semester 3\Capstone Project\Jan'26\Project Work\sleep-cassette"
    )

EPOCH_LENGTH_SEC = 30
SAMPLING_RATE = 100