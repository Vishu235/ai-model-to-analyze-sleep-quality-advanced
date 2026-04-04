import os

IS_MODAL = os.environ.get("IS_MODAL", "False") == "True"

if IS_MODAL:
    SLEEP_CASSETTE_PATH = "/data/sleep-cassette"
else:
    home = os.path.expanduser("~")
    SLEEP_CASSETTE_PATH = os.path.join(home, "mne_data", "physionet-sleep-data")

EPOCH_LENGTH_SEC = 30
SAMPLING_RATE = 100 