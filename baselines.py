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
import torchaudio
from transformers import WhisperProcessor, WhisperModel, AutoFeatureExtractor, AutoProcessor, WhisperForConditionalGeneration, WhisperTokenizer
from utils import preprocess_raw_audio, get_stimuli_and_brain, set_seed, contrastive_loss, get_stimuli_and_feature
from models import AttentiveStim2BrainNet, PositionalEncoding, LearnableTau, SoftMappingGRUSeq
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
import os
import h5py
from himalaya.backend import set_backend, get_backend
from himalaya.ridge import RidgeCV
from himalaya.scoring import correlation_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import fdrcorrection


device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
base_path = "/srv/nfs-data/sisko"
bids_root = base_path + "/storage/ECoG_podcast/ds005574-1.0.2"

func = partial(zscore)    # axis=1
ecog_sr = 512
ecog_sr_down = 32
whisper_sr = 16000
tmax = 2.0
pre_stimulus = 2.0
pre_audio = 0.0
post_audio = 0.5
n_permutations = 500

audio_path = f"{bids_root}/stimuli/podcast.wav"
audio_sf, audio_wave = wavfile.read(audio_path)
audio_wave_clean = preprocess_raw_audio(audio_wave, audio_sf)

transcript_path = f"{bids_root}/stimuli/podcast_transcript.csv"
df = pd.read_csv(transcript_path)
df.dropna(subset=['start'], inplace=True)
df.sort_values("start", inplace=True)
events = np.zeros((len(df), 3))
events[:, 0] = df.start

def batch_pearson_corr(x, y):
    """ x, y: tensors of shape (n_samples, n_features) """
    x_centered = x - x.mean(dim=1, keepdim=True)
    y_centered = y - y.mean(dim=1, keepdim=True)
    numerator = (x_centered * y_centered).sum(dim=1)
    denominator = torch.sqrt((x_centered**2).sum(dim=1) * (y_centered**2).sum(dim=1))
    return numerator / (denominator + 1e-8)

def train_encoding(X, Y):
    all_corrs = [] # empty array to store correlation results
    kfold = KFold(2, shuffle=False) # outer 2-fold cross-validation setup
    for train_index, test_index in kfold.split(X): # loop through folds

        # Split train and test datasets
        X1_train, X1_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        # Standardize Y
        scaler = StandardScaler()
        Y_train = scaler.fit_transform(Y_train)
        Y_test = scaler.transform(Y_test)

        model.fit(X1_train, Y_train) # Fit pipeline with transforms and ridge estimator
        Y_preds = model.predict(X1_test) # Predict on test set and reshape to epochs shape
        Y_preds = Y_preds.reshape(-1, epochs_shape[0], epochs_shape[1])

        corrs = np.zeros((epochs_shape[0], epochs_shape[1]))
        for ch in tqdm.tqdm(range(epochs_shape[0])):
            for t in range(epochs_shape[1]):
                corr, _ = pearsonr(Y_preds[:, ch, t], Y_test.reshape(-1, epochs_shape[0], epochs_shape[1])[:, ch, t])
                corrs[ch, t] = corr
        all_corrs.append(corrs)
        
        # corr = correlation_score(Y_test, Y_preds).reshape(epochs_shape) # Compute correlation score
        # if "torch" in get_backend().__name__: # if using gpu, transform tensor back to numpy
        #     corr = corr.numpy(force=True)
        # all_corrs.append(corr) # append fold correlation results to final results

    return np.stack(all_corrs), Y_preds, Y_test

# Loop sui 9 soggetti
subjects = [f"{i:02d}" for i in range(1, 10)]
layers = ['audio', 'text']
embeddin_matrix = None

alphas = np.logspace(1, 10, 10)
inner_cv = KFold(n_splits=5, shuffle=False) # inner 5-fold cross-validation setup
model = make_pipeline(
    StandardScaler(), RidgeCV(alphas, fit_intercept=True, cv=inner_cv) # pipeline
)

for layer in layers:
    print(f"\n{'='*50}")
    print(f"Elaborazione layer: {layer}")
    print(f"{'='*50}\n")
    if layer == 'text':
        embeddin_layer = torch.load(f"{base_path}/matteoc/podcast/text_embeds_gpt.pt")
    else: 
        embeddin_layer = torch.load(f"{base_path}/matteoc/podcast/audio_embeds_whis.pt")

    for subject in subjects:
        print(f"\n{'='*50}")
        print(f"Elaborazione soggetto: {subject}")
        print(f"{'='*50}\n")

        file_path = BIDSPath(root=bids_root+"/derivatives/ecogprep",
                            subject=subject,
                            task="podcast",
                            datatype="ieeg",
                            description="highgamma",
                            suffix="ieeg",
                            extension="fif")
        
        brain_data, feature_data = get_stimuli_and_feature(file_path, df, events, embeddin_layer, tmax=tmax, 
                                                         pre_stimulus=pre_stimulus, post_audio=post_audio, pre_audio=pre_audio,
                                                         ecog_sr_down=32, download_audio=False, baseline_flag=True)
        
        epochs_data = brain_data.reshape(len(brain_data), -1)
        X = feature_data
        Y = epochs_data
        if "torch" in get_backend().__name__:
            X = X.astype(np.float32)
            Y = Y.astype(np.float32)

        epochs_shape = brain_data.shape[1:]
        corrs_embedding, Y_preds, Y_test = train_encoding(X, Y)
        print(f"Encoding performance correlating matrix shape: {corrs_embedding.shape}")

        Y_preds = Y_preds.reshape(-1, epochs_shape[0], epochs_shape[1])[0:1024]
        Y_test = Y_test.reshape(-1, epochs_shape[0], epochs_shape[1])[0:1024]

        n_samples, n_channels, n_timepoints = Y_preds.shape
        null_distribution_tp = np.zeros((n_permutations, n_timepoints))
        null_distribution_ch = np.zeros((n_permutations, n_channels))

        for perm in tqdm.tqdm(range(n_permutations), desc="Permutation runs"):
            shuffled_idx = np.random.permutation(n_samples)
            y_true_perm = Y_test[shuffled_idx]

            for tp in range(n_timepoints):
                pred_tp = torch.tensor(Y_preds[:, :, tp]).to(device)  # (samples, voxels)
                true_tp = torch.tensor(y_true_perm[:, :, tp]).to(device)  
                corrs = batch_pearson_corr(pred_tp, true_tp)  # (samples,)
                null_distribution_tp[perm, tp] = corrs.mean().item()

            for ch in range(n_channels):
                pred_ch = torch.tensor(Y_preds[:, ch, :]).to(device)  
                true_ch = torch.tensor(y_true_perm[:, ch, :]).to(device)  
                corrs_ch = batch_pearson_corr(pred_ch, true_ch)  
                null_distribution_ch[perm, ch] = corrs_ch.mean().item()
        
        real_corr_tp = corrs_embedding.mean((0,1))
        real_corr_ch = corrs_embedding.mean((0,2))
        p_values_tp = np.mean(null_distribution_tp >= real_corr_tp[None, :], axis=0)  
        p_values_ch = np.mean(null_distribution_ch >= real_corr_ch[None, :], axis=0)  
        significant_tp, pvals_corrected_tp = fdrcorrection(p_values_tp, alpha=0.02)
        significant_ch, pvals_corrected_ch = fdrcorrection(p_values_ch, alpha=0.01)

        raw = mne.io.read_raw_fif(file_path, verbose=False)
        raw.load_data(verbose=False)
        raw = raw.apply_function(func, channel_wise=False, verbose=False)

        ch2loc = {ch['ch_name']: ch['loc'][:3] for ch in raw.info['chs']}
        coords = np.vstack([ch2loc[ch] for ch in raw.info['ch_names']])
        coords *= 1000  # nilearn likes to plot in meters, not mm
        print(f"Coordinate matrix shape for subject {subject}: {coords.shape}")

        to_save_path = f"{base_path}/matteoc/podcast/subj_data/sub_{subject}_baseline"
        os.makedirs(to_save_path, exist_ok=True)
        np.save(f"{to_save_path}/correlations_{layer}.npy", corrs_embedding)
        np.save(f"{to_save_path}/coords_ch.npy", coords)
        np.save(f"{to_save_path}/significant_tp_{layer}.npy", significant_tp)
        np.save(f"{to_save_path}/significant_ch_{layer}.npy", significant_ch)

    del embeddin_layer