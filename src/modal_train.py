"""
modal_train.py
"""
import modal

image = (
    modal.Image.from_registry("tensorflow/tensorflow:2.15.0-gpu")
    .pip_install(
        "numpy<2",
        "mne",
        "wandb",
        "scikit-learn",
        "pandas",
       
    )
    .add_local_dir(".", remote_path="/root")
)

data_volume = modal.Volume.from_name("sleep-edf-data", create_if_missing=True)

app = modal.App("sleep-staging-project")

@app.function(image=image, volumes={"/data": data_volume}, timeout=14400)
def download_data_remote():
    import os
    os.environ["IS_MODAL"] = "True"

    # Sleep-EDF Cassette: 20 subjects (IDs 0-19), 2 recordings each.
    # fetch_data stores files under /data/sleep-cassette/physionet-sleep-data/
    # recording=[1,2] fetches both nights per subject → up to 40 recordings total.
    print("Downloading Sleep-EDF Cassette to Modal Volume...")
    from mne.datasets.sleep_physionet.age import fetch_data

    fetch_data(
        subjects=range(20),
        recording=[1, 2],
        path="/data/sleep-cassette",
        on_missing="ignore",
    )
    print("Download complete.")


@app.function(
    image=image,
    gpu="T4",
    volumes={"/data": data_volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=21600,   # 6 hours — enough for 5 folds x 35 epochs on T4
)
def train_remote():
    import os
    os.environ["IS_MODAL"] = "True"

    from dl_main import train_model

    config = {
        "max_subjects": None,  # use all subjects found in the volume
        "seq_len":      5,
        "epochs":       35,
        "batch_size":   32,
    }

    print("Starting 5-fold CV training on Modal T4 GPU...")
    train_model(config)

@app.function(
    image=image,
    volumes={"/data": data_volume},
    gpu="T4",
    timeout=1800,
    secrets=[modal.Secret.from_name("wandb-secret")]
)
def export_clean_model():
    import os
    os.environ["IS_MODAL"] = "True"

    import wandb
    import tensorflow as tf

    print("Initializing WandB...")
    run = wandb.init(project="sleep-edf-5class")

    print("Downloading model artifact...")
    artifact = run.use_artifact(
        "vishal-bglr-pes-university/sleep-edf-5class/run_7l5s41bb_model:v3",
        type="model"
    )
    artifact_dir = artifact.download()

    model_path = os.path.join(artifact_dir, "sleep_best_ckpt.keras")

    print("Loading full model...")
    model = tf.keras.models.load_model(model_path, compile=False)

    print("Saving clean version...")
    model.save("/data/final_model_clean.h5", include_optimizer=False)
    print("Model saved to /data/final_model_clean.h5")

@app.local_entrypoint()
def main():
    download_data_remote.remote()
    train_remote.remote()