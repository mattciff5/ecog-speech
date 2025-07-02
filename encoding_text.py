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
from transformers import GPT2TokenizerFast, GPT2Model, AutoTokenizer, AutoModelForCausalLM
from utils import preprocess_raw_audio, get_text_and_brain, set_seed, contrastive_loss
from models import AttentiveStim2BrainNet, PositionalEncoding, LearnableTau, SoftMappingGRUSeq, Audio2BrainCNN
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
import os



# Inizializzazione dei modelli e del dispositivo (fuori dal ciclo per evitare ricaricamenti)
model_name = "gpt2"
tokenizer_gpt = AutoTokenizer.from_pretrained(model_name)
tokenizer_gpt.pad_token = tokenizer_gpt.eos_token
model_gpt = AutoModelForCausalLM.from_pretrained(model_name)
model_gpt.eval()

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
base_path = "/srv/nfs-data/sisko"
bids_root = base_path + "/storage/ECoG_podcast/ds005574-1.0.2"

func = partial(zscore)    # axis=1
ecog_sr = 512
ecog_sr_down = 32
whisper_sr = 16000
tmax = 2.0
pre_stimulus = 2.0
pre_audio = 0.2

audio_path = f"{bids_root}/stimuli/podcast.wav"
audio_sf, audio_wave = wavfile.read(audio_path)
audio_wave_clean = preprocess_raw_audio(audio_wave, audio_sf)

transcript_path = f"{bids_root}/stimuli/podcast_transcript.csv"
df = pd.read_csv(transcript_path)
df.dropna(subset=['start'], inplace=True)
df.sort_values("start", inplace=True)
events = np.zeros((len(df), 3))
events[:, 0] = df.start

# Loop sui 9 soggetti
# Assumiamo che i soggetti siano numerati da '01' a '09'.
subjects = [f"{i:02d}" for i in range(1, 10)]
layers = ['six', 'seven', 'eight', 'nine', 'ten', 'eleven']

for layer in layers:
    print(f"\n{'='*50}")
    print(f"Elaborazione layer: {layer}")
    print(f"{'='*50}\n")

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

        brain_data, text_data, _ = get_text_and_brain(file_path, df, tmax=2.0, pre_audio=2.0, pre_stimulus=2.0,
                                                    model=model_gpt, tokenizer=tokenizer_gpt, ecog_sr_down=32,
                                                    device=device, download_text=False, base_path=base_path, layer=layer)

        brain_timep = brain_data.shape[-1]
        brain_channels = brain_data.shape[1]
        text_timep = text_data.shape[1]

        outer_cv = KFold(n_splits=4, shuffle=False)
        all_corrs, all_attn, all_pvals = [], [], []
        brain_np = brain_data.reshape(brain_data.shape[0], -1)

        for fold_idx, (train_idx, test_idx) in tqdm.tqdm(enumerate(outer_cv.split(brain_data))):
            print(f"\n--- Fold {fold_idx + 1} ---")

            scaler = StandardScaler()
            brain_train = scaler.fit_transform(brain_np[train_idx])
            brain_test = scaler.transform(brain_np[test_idx])

            brain_train = torch.tensor(brain_train, dtype=torch.float32).reshape(-1, brain_channels, brain_timep)
            brain_test = torch.tensor(brain_test, dtype=torch.float32).reshape(-1, brain_channels, brain_timep)

            stimuli_train = text_data[train_idx]
            stimuli_test = text_data[test_idx]

            train_dataset = TensorDataset(stimuli_train.to(device), brain_train.to(device))
            test_dataset = TensorDataset(stimuli_test.to(device), brain_test.to(device))

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            set_seed(42)
            model = AttentiveStim2BrainNet(input_dim=768, d_model=256, nhead=2, num_layers=2, time_in=text_timep, time_out=brain_timep,
                                           output_channels=brain_channels).to(device)
            tau_module = LearnableTau(init_tau=0.03).to(device)
            mse_loss = nn.MSELoss()
            mse_perc = 0.0
            cl_perc = 1.0

            optimizer = optim.AdamW(list(model.parameters()) + list(tau_module.parameters()), lr=1e-4, betas=(0.9, 0.99), weight_decay=1e-3)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

            best_loss = float('inf')
            for epoch in range(35):
                model.train()
                total_loss = 0
                for x, y in train_loader:
                    optimizer.zero_grad()
                    x, y = x.to(device), y.to(device)
                    y_pred, _ = model(x)
                    loss = mse_perc * mse_loss(y_pred, y) + cl_perc * contrastive_loss(y_pred, y, tau=tau_module())
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * x.size(0)
                total_loss /= len(train_loader.dataset)

                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for x, y in test_loader:
                        x, y = x.to(device), y.to(device)
                        y_pred, _ = model(x)
                        loss = mse_perc * mse_loss(y_pred, y) + cl_perc * contrastive_loss(y_pred, y, tau=tau_module())
                        val_loss += loss.item() * x.size(0)
                val_loss /= len(test_loader.dataset)
                scheduler.step(val_loss)

                print(f"Epoch {epoch+1} - Train Loss: {total_loss:.4f} | Test Loss: {val_loss:.4f}")

                if val_loss < best_loss:
                    best_loss = val_loss
                    # Salva il modello specifico per il soggetto e il fold
                    torch.save(model.state_dict(), f"/home/matteoc/ecog-speech/best_model_fold{fold_idx+1}.pt")

            model.load_state_dict(torch.load(f"/home/matteoc/ecog-speech/best_model_fold{fold_idx+1}.pt"))
            model.eval()
            preds, targets, attn_values = [], [], []
            with torch.no_grad():
                for x, y in test_loader:
                    y, x = y.to(device), x.to(device)
                    y_pred, attn_pred = model(x)
                    preds.append(y_pred.cpu().numpy())
                    targets.append(y.cpu().numpy())
                    attn_values.append(attn_pred.cpu().numpy())
            preds = np.concatenate(preds, axis=0)
            targets = np.concatenate(targets, axis=0)
            attn_values = np.concatenate(attn_values, axis=0)

            corrs = np.zeros((brain_channels, brain_timep))
            pvals = np.ones((brain_channels, brain_timep))
            for ch in range(brain_channels):
                for t in range(brain_timep):
                    corr, pval = pearsonr(preds[:, ch, t], targets[:, ch, t])
                    corrs[ch, t] = corr
                    pvals[ch, t] = pval
            all_corrs.append(corrs)
            all_pvals.append(pvals)
            all_attn.append(attn_values)

        all_corrs = np.stack(all_corrs)
        all_pvals = np.stack(all_pvals)
        all_attn = np.concatenate(all_attn, axis=0)
        print(f"\nFinal mean correlation across folds for subject {subject}: {np.nanmean(all_corrs):.4f}")

        raw = mne.io.read_raw_fif(file_path, verbose=False)
        raw.load_data(verbose=False)
        raw = raw.apply_function(func, channel_wise=False, verbose=False)

        ch2loc = {ch['ch_name']: ch['loc'][:3] for ch in raw.info['chs']}
        coords = np.vstack([ch2loc[ch] for ch in raw.info['ch_names']])
        coords *= 1000  # nilearn likes to plot in meters, not mm
        print(f"Coordinate matrix shape for subject {subject}: {coords.shape}")

        to_save_path = f"{base_path}/matteoc/podcast/subj_data/sub_{subject}_text"
        os.makedirs(to_save_path, exist_ok=True)
        # np.save(f"{to_save_path}/att_layer_{layer}.npy", all_attn)
        np.save(f"{to_save_path}/pval_layer_{layer}.npy", all_pvals)
        np.save(f"{to_save_path}/corr_layer_{layer}.npy", all_corrs)
        np.save(f"{to_save_path}/coords_ch.npy", coords)

    print("\nProcesso completato per tutti i soggetti.")