"""
load_data.py
Robust auto-loader for Sleep-EDF compatible with Modal and Local setups.
"""
import os
import numpy as np
import mne
from mne.datasets.sleep_physionet.age import fetch_data
from config import SLEEP_CASSETTE_PATH, EPOCH_LENGTH_SEC
FIXED_EVENT_ID = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,  
    "Sleep stage R": 4,
}

def load_all_subjects_epochs(max_subjects=None):
    """
    loads Sleep-EDF data.        
        
    """
    
    if max_subjects is None:
        subject_ids = range(50) 
    else:
        subject_ids = range(min(max_subjects, 50))

    print(f"verifying data for {len(subject_ids)} subjects...")
    
    files = fetch_data(subjects=subject_ids, recording=[1], path=SLEEP_CASSETTE_PATH, on_missing='ignore')

    all_epochs_list = []
    all_subject_ids = []

    for file_group in files:
        psg_path = file_group[0]
        hypno_path = file_group[1]
        
        fname = os.path.basename(psg_path)
        subject_id = fname[:6]  
        
        print(f"Processing: {subject_id}")

        raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
        
        mne.rename_channels(raw.info, {'EEG Fpz-Cz': 'EEG'}, allow_duplicates=False)
        
        try:
            raw.pick_channels(['EEG'])
        except ValueError:
            print(f"Warning: 'EEG Fpz-Cz' not found in {subject_id}. Picking first EEG.")
            raw.pick_types(eeg=True)
            raw = raw.pick_channels([raw.ch_names[0]])

        annotations = mne.read_annotations(hypno_path)
        raw.set_annotations(annotations, emit_warning=False)

        try:
            events, _ = mne.events_from_annotations(
                raw, event_id=FIXED_EVENT_ID, chunk_duration=30., verbose=False
            )
        except ValueError:
            print(f"Skipping {subject_id}: No matching sleep events found.")
            continue


        epochs = mne.Epochs(
            raw,
            events,
            event_id=FIXED_EVENT_ID,
            tmin=0,
            tmax=EPOCH_LENGTH_SEC - (1 / raw.info['sfreq']), # 0 to 29.99s
            baseline=None,
            preload=True,
            verbose=False,
            on_missing='ignore'
        )

        n_epochs = len(epochs)
        if n_epochs > 0:
            all_epochs_list.append(epochs)
            all_subject_ids.extend([subject_id] * n_epochs)

    if not all_epochs_list:
        raise RuntimeError("No epochs loaded. Check data path.")

    final_epochs = mne.concatenate_epochs(all_epochs_list)
    
    return final_epochs, np.array(all_subject_ids)