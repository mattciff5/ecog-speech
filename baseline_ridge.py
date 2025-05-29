import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
from scipy import signal
from scipy.io import wavfile
from scipy.stats import pearsonr, zscore
from mne_bids import BIDSPath
from functools import partial
from nilearn.plotting import plot_markers
import torch
from torch import nn
import torchaudio
from transformers import WhisperProcessor, WhisperModel, AutoFeatureExtractor, AutoProcessor
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from himalaya.ridge import RidgeCV
from himalaya.backend import set_backend


device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
model = WhisperModel.from_pretrained("openai/whisper-base")
feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
processor = AutoProcessor.from_pretrained("openai/whisper-base")
model.eval()
base_path = "/srv/nfs-data/sisko"
bids_root = base_path + "/storage/ECoG_podcast/ds005574-1.0.2" 
subject = '03'
func = partial(zscore, axis=1)
ecog_sr = 512
ecog_sr_down = 128
whisper_sr = 16000
pre_stimulus = 1.0
pre_audio = 0.2

def preprocess_raw_audio(audio_wave, fs):
    if audio_wave.ndim > 1:
        audio_wave = audio_wave.mean(axis=1)
    audio_wave = audio_wave.astype(np.float32)
    audio_wave_model = audio_wave / np.max(np.abs(audio_wave))
    return audio_wave_model

file_path = BIDSPath(root=bids_root+"/derivatives/ecogprep",
                     subject=subject,
                     task="podcast",
                     datatype="ieeg",
                     description="highgamma",
                     suffix="ieeg",
                     extension="fif")

audio_path = f"{bids_root}/stimuli/podcast.wav"
audio_sf, audio_wave = wavfile.read(audio_path)
audio_wave_clean = preprocess_raw_audio(audio_wave, audio_sf)

transcript_path = f"{bids_root}/stimuli/podcast_transcript.csv"
df = pd.read_csv(transcript_path)
df.dropna(subset=['start'], inplace=True)
df.sort_values("start", inplace=True)
events = np.zeros((len(df), 3))
events[:, 0] = df.start

download_audio = False

def get_stimuli_and_brain(file_path, audio_wave_clean, audio_sf, df, events, 
                          tmax=2.0, pre_audio=0.5, pre_stimulus=0.2,
                          model=None, processor=None, whisper_sr=16000,
                          device=device):    
    model = model.to(device)
    raw = mne.io.read_raw_fif(file_path, verbose=False)
    raw.load_data(verbose=False)
    raw = raw.apply_function(func, channel_wise=False, verbose=False)

    epochs = mne.Epochs(
        raw,
        (events * raw.info['sfreq']).astype(int),
        tmin=-pre_stimulus,
        tmax=tmax,
        baseline=None,
        proj=None,
        event_id=None,
        preload=True,
        event_repeated="merge",
        verbose=False
    )
    good_idx = epochs.selection
    print(f"Epochs object has a shape of: {epochs._data.shape}")
    epochs = epochs.resample(sfreq=ecog_sr_down, npad='auto', method='fft', window='hamming')
    epochs_snippet = epochs._data
    print(f"Epochs object after down-sampling has a shape of: {epochs_snippet.shape}")

    if download_audio:
        audio_snip_whisper = []
        for idx, row in tqdm.tqdm(enumerate(good_idx)):
            row = df.iloc[idx]
            start_sample = int((row['start']) * audio_sf) 
            end_sample = start_sample + int(tmax * audio_sf)
            snippet = audio_wave_clean[start_sample - int(pre_audio * audio_sf):end_sample]
            if len(snippet) < int(tmax * audio_sf):
                padding_len = int(tmax * audio_sf) - len(snippet)
                snippet = np.pad(snippet, (0, padding_len), mode='constant')
            snippet = torchaudio.transforms.Resample(audio_sf, whisper_sr)(torch.tensor(snippet).float())
            inputs = processor(snippet.squeeze(0), sampling_rate=whisper_sr, return_tensors="pt")
            input_features = inputs['input_features'].to(device)
            with torch.no_grad():
                outputs = model.encoder(input_features=input_features)
                hidden_states = outputs.last_hidden_state[:,:int(2*50*(tmax+pre_audio))]
                hidden_states = hidden_states[:,::2]   # sort of downsampling
                audio_snip_whisper.append(hidden_states.squeeze(0).cpu())
        audio_snip_whisper = torch.stack(audio_snip_whisper, dim=0)
    else:
        audio_snip_whisper = torch.load(f"{base_path}/matteoc/podcast/audio_2_2_sec.pt")
    print(f"Audio snippets after processing have a shape of: {audio_snip_whisper.shape}")

    return epochs_snippet, audio_snip_whisper


brain_data, audio_data = get_stimuli_and_brain(file_path, audio_wave_clean, audio_sf, df, events, 
                                               model=model, processor=processor, pre_stimulus=pre_stimulus, pre_audio=pre_audio)

brain_timep = brain_data.shape[-1]
brain_channels = brain_data.shape[1]
audio_timep = audio_data.shape[1]

brain_np = brain_data.reshape(5130, -1) 
brain_scaler = StandardScaler()

brain_std = torch.tensor(brain_np, dtype=torch.float32, device=device).reshape(5130, brain_channels, brain_timep)
stimuli = audio_data.to(device)

train_size = int(0.8 * len(stimuli))
val_size = int(0.1 * len(stimuli))
test_size = len(stimuli) - train_size - val_size

dataset = TensorDataset(stimuli, brain_std)
train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

train_stimuli_ridge = torch.stack([train_ds[i][0] for i in range(len(train_ds))]).flatten(1, 2)
train_brain_ridge = torch.stack([train_ds[i][1] for i in range(len(train_ds))])

val_stimuli_ridge = torch.stack([val_ds[i][0] for i in range(len(val_ds))]).flatten(1, 2)
val_brain_ridge = torch.stack([val_ds[i][1] for i in range(len(val_ds))])

test_stimuli_ridge = torch.stack([test_ds[i][0] for i in range(len(test_ds))]).flatten(1, 2)
test_brain_ridge = torch.stack([test_ds[i][1] for i in range(len(test_ds))])

device_id = 6
backend = set_backend("torch_cuda")

X_train = train_stimuli_ridge.float().to(f"cuda:{device_id}")  # [N, D]
X_test = test_stimuli_ridge.float().to(f"cuda:{device_id}")    # [N, D]

n_channels = train_brain_ridge.shape[1]
n_timepoints = train_brain_ridge.shape[2]
n_samples_test = test_stimuli_ridge.shape[0]

test_preds = torch.zeros((n_samples_test, n_channels, n_timepoints), device=f"cuda:{device_id}")

for ch in tqdm.tqdm(range(n_channels)):
    
    Y_train = train_brain_ridge[:, ch, :].float().to(f"cuda:{device_id}")  # [N, T]
    
    model_base = RidgeCV(alphas=[0.1, 1, 10, 30, 50, 100, 1000])

    X_train_F = backend.asarray(X_train)
    Y_train_F = backend.asarray(Y_train)
    X_test_F = backend.asarray(X_test)
    
    model_base.fit(X_train_F, Y_train_F)
    
    Y_pred_F = model_base.predict(X_test_F)  # restituisce [N_test, T]
    Y_pred = backend.to_numpy(Y_pred_F)
    
    test_preds[:, ch, :] = torch.tensor(Y_pred, device=f"cuda:{device_id}")

torch.save(test_preds, base_path+'/matteoc/podcast/brain_data/sub_03_pred_ridge.pt')
