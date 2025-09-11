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
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.nn.utils.rnn import pad_sequence



func = partial(zscore) 

def contrastive_loss(pred, target, tau=0.05):
        target = target.reshape(target.shape[0], -1)
        pred = pred.reshape(pred.shape[0], -1)
        pred = F.normalize(pred, dim=1)
        target = F.normalize(target, dim=1)
        sim_matrix = torch.mm(pred, target.T) / tau
        loss = -torch.log(torch.exp(torch.diag(sim_matrix)) / sim_matrix.exp().sum(dim=1))
        return loss.mean()


def preprocess_raw_audio(audio_wave, fs):
    if audio_wave.ndim > 1:
        audio_wave = audio_wave[:, 0]
    audio_wave = audio_wave.astype(np.float32)
    # audio_wave = audio_wave / np.max(np.abs(audio_wave))
    return audio_wave


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def get_text_and_brain(file_path, df, tmax=2.0, pre_audio=2.0, pre_stimulus=2.0,
                          model=None, tokenizer=None, ecog_sr_down=32,
                          device=None, download_text=False, base_path=None, layer='last'):
    
    model = model.to(device)

    raw = mne.io.read_raw_fif(file_path, verbose=False)
    raw.load_data()
    raw = raw.apply_function(func, channel_wise=True, verbose=False)

    # events = np.zeros((len(df), 3), dtype=int)
    # events[:, 0] = (df.start * raw.info['sfreq']).astype(int)
    df.dropna(subset=['start'], inplace=True)
    df.sort_values("start", inplace=True)
    events = np.zeros((len(df), 3))
    events[:, 0] = df.start

    epochs = mne.Epochs(
        raw,
        (events * raw.info['sfreq']).astype(int),
        tmin=-pre_stimulus,
        tmax=tmax,
        baseline=None,
        proj=False,
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

    attention_mask_list = []
    if download_text:
        text_decoder_embd = []

        for row_idx in tqdm.tqdm(good_idx):
            
            row = df.iloc[row_idx]
            word_list = df[
                (df["start"] >= row["start"] - pre_audio) & 
                (df["start"] <= row["start"])
            ]
            words_in_segment = word_list["word"].tolist()
            transcription = " ".join(words_in_segment)

            with torch.no_grad():    
                # -------- GPT
                inputs = tokenizer(
                    transcription, 
                    return_tensors="pt"
                )
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                last_hidden_dec = outputs.hidden_states[-1]
                last_hidden_dec = last_hidden_dec.squeeze(0)
                attention_mask = attention_mask.squeeze(0)
                text_decoder_embd.append(last_hidden_dec.cpu())
                attention_mask_list.append(attention_mask.cpu())
        
        text_decoder_embd = pad_sequence(text_decoder_embd, batch_first=True, padding_side='left')   
        # text_decoder_embd = torch.stack(text_decoder_embd, dim=0) 
        attention_mask_list = pad_sequence(attention_mask_list, batch_first=True, padding_side='left')  
        # torch.save(text_decoder_embd, f"{base_path}/matteoc/podcast/text_2_2_sec_last.pt")

    else:
        text_decoder_embd = torch.load(f"{base_path}/matteoc/podcast/text_2_2_sec_{layer}.pt")
        text_decoder_embd = text_decoder_embd[good_idx]

    print(f"Text snippets after processing have a shape of: {text_decoder_embd.shape}")

    return epochs_snippet, text_decoder_embd, attention_mask_list



def get_stimuli_and_brain(file_path, audio_wave_clean, audio_sf, df, events, 
                          tmax=2.0, post_audio=2.0, pre_audio=2.0, pre_stimulus=2.0,
                          model=None, processor=None, whisper_sr=16000,
                          device=None, ecog_sr_down=32, download_audio=False, 
                          base_path=None, layer='new'):
    
    model = model.to(device)

    raw = mne.io.read_raw_fif(file_path, verbose=False)
    raw.load_data()
    raw = raw.apply_function(func, channel_wise=True, verbose=False)

    epochs = mne.Epochs(
        raw,
        (events * raw.info['sfreq']).astype(int),
        tmin=-pre_stimulus,
        tmax=tmax,
        baseline=None,
        proj=False,
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
        first_layer_whisper = []
        second_layer_whisper = []
        third_layer_whisper = []
        fourth_layer_whisper = []
        fifth_layer_whisper = []
        sixth_layer_whisper = []
        for idx, row in tqdm.tqdm(enumerate(good_idx)):
            row = df.iloc[idx]
            start_sample = int((row['start']) * audio_sf) 
            end_sample = start_sample + int(post_audio * audio_sf)
            snippet = audio_wave_clean[start_sample - int(pre_audio * audio_sf):end_sample]
            if len(snippet) < int(post_audio * audio_sf):
                padding_len = int(post_audio * audio_sf) - len(snippet)
                snippet = np.pad(snippet, (0, padding_len), mode='constant')
            snippet = torchaudio.transforms.Resample(audio_sf, whisper_sr)(torch.tensor(snippet).float())
            inputs = processor(snippet.squeeze(0), sampling_rate=whisper_sr, return_tensors="pt")
            input_features = inputs['input_features'].to(device)
            with torch.no_grad():

                # WHISPER ENCODER
                outputs = model.encoder(input_features=input_features, output_hidden_states=True)
                all_hidden_states = outputs.hidden_states
                first_layer = all_hidden_states[0][:,:int(50*(post_audio+pre_audio))]
                second_layer = all_hidden_states[1][:,:int(50*(post_audio+pre_audio))]
                third_layer = all_hidden_states[2][:,:int(50*(post_audio+pre_audio))]
                fourth_layer = all_hidden_states[3][:,:int(50*(post_audio+pre_audio))]
                fifth_layer = all_hidden_states[4][:,:int(50*(post_audio+pre_audio))]
                sixth_layer = all_hidden_states[5][:,:int(50*(post_audio+pre_audio))]
                last_hidden_layer = outputs.last_hidden_state[:,:int(50*(post_audio+pre_audio))]
                # hidden_states = hidden_states[:,::2]   # sort of downsampling

                audio_snip_whisper.append(last_hidden_layer.squeeze(0).cpu())
                first_layer_whisper.append(first_layer.squeeze(0).cpu())
                second_layer_whisper.append(second_layer.squeeze(0).cpu())
                third_layer_whisper.append(third_layer.squeeze(0).cpu())
                fourth_layer_whisper.append(fourth_layer.squeeze(0).cpu())
                fifth_layer_whisper.append(fifth_layer.squeeze(0).cpu())
                sixth_layer_whisper.append(sixth_layer.squeeze(0).cpu())
        
        audio_snip_whisper = torch.stack(audio_snip_whisper, dim=0)     
        first_layer_whisper = torch.stack(first_layer_whisper, dim=0)
        second_layer_whisper = torch.stack(second_layer_whisper, dim=0)
        third_layer_whisper = torch.stack(third_layer_whisper, dim=0)
        fourth_layer_whisper = torch.stack(fourth_layer_whisper, dim=0)
        fifth_layer_whisper = torch.stack(fifth_layer_whisper, dim=0)
        sixth_layer_whisper = torch.stack(sixth_layer_whisper, dim=0)

    else:
        audio_snip_whisper = torch.load(f"{base_path}/matteoc/podcast/audio_2_2_sec_{layer}.pt")

    print(f"Audio snippets after processing have a shape of: {audio_snip_whisper.shape}")

    return epochs_snippet, audio_snip_whisper



def get_stimuli_and_feature(file_path, df, events, embedd_file, tmax=2.0, post_audio=2.0, pre_audio=2.0, pre_stimulus=2.0,
                          ecog_sr_down=32, download_audio=False, baseline_flag=False):

    raw = mne.io.read_raw_fif(file_path, verbose=False)
    # raw = raw.pick("ecog", exclude="bads")
    raw.load_data()
    raw = raw.apply_function(func, channel_wise=True, verbose=False)

    epochs = mne.Epochs(
        raw,
        (events * raw.info['sfreq']).astype(int),
        tmin=-pre_stimulus,
        tmax=tmax,
        baseline=None,
        proj=False,
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
    layer_embedding = None

    if download_audio:
        layer_embedding = embedd_file[:, 100-int(pre_audio*50): 100+int(post_audio*50), :]
        if baseline_flag:
            layer_embedding = layer_embedding.mean(dim=1)
    else:
        layer_embedding = embedd_file[good_idx]

    print(f"Audio snippets after processing have a shape of: {layer_embedding.shape}")

    return epochs_snippet, layer_embedding