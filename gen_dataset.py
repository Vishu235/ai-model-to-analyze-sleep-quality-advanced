import sys, glob, os
import numpy as np
import mne, keras, keras.layers, pandas as pd
from scipy.signal import butter, filtfilt, welch
from scipy.ndimage import median_filter
from scipy.stats import skew, kurtosis

# Keras patch
_orig = keras.layers.Dense.from_config.__func__
@classmethod
def _p(cls, c):
    c.pop('quantization_config', None)
    return _orig(cls, c)
keras.layers.Dense.from_config = _p

MODEL_PATH = 'D:/ML_PROJECT/Vishal_Code/ai-model-to-analyze-sleep-quality/src/sleep_best_ckpt.keras'
DATA_DIR   = 'D:/ML_PROJECT/data/sleep-edf-database-expanded-1.0.0/sleep-edf-database-expanded-1.0.0/sleep-cassette'
OUT_CSV    = 'D:/ML_PROJECT/Vishal_Code/21-03-2026/ai-model-to-analyze-sleep-quality/regression_dataset.csv'
EVENT_ID   = {'Sleep stage W':0,'Sleep stage 1':1,'Sleep stage 2':2,
               'Sleep stage 3':3,'Sleep stage 4':3,'Sleep stage R':4}
FS, ES, SL = 100, 30, 5


def quality_score(stages, confidence):
    n = len(stages); sleep = [s for s in stages if s != 0]
    rec = n * ES / 60; tst = len(sleep) * ES / 60
    se  = tst / rec if rec > 0 else 0
    fi  = next((i for i, s in enumerate(stages) if s != 0), None)
    lat = fi * ES / 60 if fi is not None else rec
    waso = sum(1 for s in stages[fi:] if s == 0) * ES / 60 if fi else 0
    pn3  = stages.count(3) / len(sleep) if sleep else 0
    pr   = stages.count(4) / len(sleep) if sleep else 0
    trans = sum(1 for i in range(1, n) if stages[i] != stages[i-1])
    frag  = trans / n if n > 0 else 0
    def clip(x): return max(0., min(1., float(x)))
    return (0.30 * clip((se - 0.70) / 0.20)
          + 0.20 * clip((90 - waso) / 90)
          + 0.15 * clip((45 - lat) / 45)
          + 0.20 * 0.5 * (clip(pn3 / 0.20) + clip(pr / 0.22))
          + 0.15 * clip((0.12 - frag) / 0.12)) * 100


def extract_features(signal):
    freqs, psd = welch(signal, fs=FS, nperseg=min(len(signal), FS * 4))
    bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}
    bp = {}
    for b, (lo, hi) in bands.items():
        idx = np.where((freqs >= lo) & (freqs <= hi))[0]
        bp[b] = float(np.trapz(psd[idx], freqs[idx]))
    total = sum(bp.values()) + 1e-10
    f = {f'power_{b}': v for b, v in bp.items()}
    f.update({f'rel_{b}': v / total for b, v in bp.items()})
    f['ratio_delta_beta']  = bp['delta'] / (bp['beta'] + 1e-10)
    f['ratio_theta_alpha'] = bp['theta'] / (bp['alpha'] + 1e-10)
    f.update({'mean': float(signal.mean()), 'std': float(signal.std()),
              'skewness': float(skew(signal)), 'kurtosis': float(kurtosis(signal)),
              'rms': float(np.sqrt(np.mean(signal ** 2)))})
    return f


print('Loading CNN+LSTM model...', flush=True)
model = keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
print('Model loaded.\n', flush=True)

pairs = []
for p in sorted(glob.glob(f'{DATA_DIR}/*E0-PSG.edf')):
    sub = os.path.basename(p)[:6]
    hyp = glob.glob(f'{DATA_DIR}/{sub}*Hypnogram.edf')
    if hyp:
        pairs.append((p, hyp[0]))
print(f'Found {len(pairs)} recordings.\n', flush=True)

dataset = []
for idx, (psg, hyp) in enumerate(pairs):
    try:
        raw = mne.io.read_raw_edf(psg, preload=True, verbose=False)
        ch  = 'EEG Fpz-Cz' if 'EEG Fpz-Cz' in raw.ch_names else raw.copy().pick_types(eeg=True).ch_names[0]
        raw.pick_channels([ch])
        if raw.info['sfreq'] != FS:
            raw.resample(FS, verbose=False)

        ann = mne.read_annotations(hyp)
        raw.set_annotations(ann, emit_warning=False)
        events, _ = mne.events_from_annotations(raw, event_id=EVENT_ID, chunk_duration=30., verbose=False)
        ep = mne.Epochs(raw, events, event_id=EVENT_ID,
                        tmin=0, tmax=ES - (1/FS), baseline=None, preload=True, verbose=False)

        data   = ep.get_data()[:, 0, :].astype('float32')
        ne     = len(data)
        if ne < SL:
            continue

        m, s   = data.mean(axis=1, keepdims=True), data.std(axis=1, keepdims=True) + 1e-8
        norm   = (data - m) / s
        X      = np.stack([norm[i:i+SL] for i in range(ne - SL + 1)])[..., np.newaxis]
        pr     = model.predict(X, verbose=0)
        preds  = pr.argmax(axis=1)
        confs  = pr.max(axis=1)

        fp = np.zeros(ne, int)
        for i in range(len(preds)): fp[i + SL - 1] = preds[i]
        for i in range(SL - 1):    fp[i] = fp[SL - 1]
        fp = list(median_filter(fp, size=5))

        score = quality_score(fp, confs.tolist())

        sig = raw.get_data()[0].astype('float32')
        nyq = FS * 0.5
        b, a = butter(4, [0.5 / nyq, 30.0 / nyq], btype='band')
        feats = extract_features(filtfilt(b, a, sig).astype('float32'))

        for s2, name in enumerate(['wake', 'n1', 'n2', 'n3', 'rem']):
            feats[f'pct_{name}'] = fp.count(s2) / len(fp)
        feats['fragmentation']   = sum(1 for i in range(1, len(fp)) if fp[i] != fp[i-1]) / len(fp)
        feats['mean_confidence'] = float(confs.mean())
        feats['target_score']    = score
        dataset.append(feats)

        print(f'[{idx+1}/{len(pairs)}] {os.path.basename(psg)}: score={score:.1f}', flush=True)

    except Exception as e:
        print(f'  SKIP {os.path.basename(psg)}: {e}', flush=True)

df = pd.DataFrame(dataset)
df.to_csv(OUT_CSV, index=False)
print(f'\nDataset saved → {OUT_CSV}', flush=True)
print(f'Shape: {df.shape}  Score: {df.target_score.min():.1f}–{df.target_score.max():.1f}  Mean: {df.target_score.mean():.1f}', flush=True)
