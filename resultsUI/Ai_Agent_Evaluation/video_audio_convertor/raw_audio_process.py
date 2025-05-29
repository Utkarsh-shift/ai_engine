import os
import librosa
import opensmile
import numpy as np
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import glob
 
 
class RawAudioProcessor():
 
    def __init__(self, mode, aud_dir, save_to):
        self.mode = mode
        self.saved_file = save_to
        os.makedirs(save_to, exist_ok=True)
        if mode == "opensmile":
            self.smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.ComParE_2016,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
        self.aud_file_ls = list(glob.glob(f"{aud_dir}/*.wav"))
 
    def __getitem__(self, idx):
        wav_file_path = self.aud_file_ls[idx]
        video_name = os.path.basename(wav_file_path)  
        if self.mode == "librosa":
            self.librosa_extract(wav_file_path, video_name)
        elif self.mode == "logfbank":
            self.logfbank_extract(wav_file_path, video_name)
        elif self.mode == "opensmile":
            self.opensmile_extract(wav_file_path, video_name)
 
    def __len__(self):
        return len(self.aud_file_ls)
 
    def librosa_extract(self, wav_file_path, video_name):
        try:
            print(f"Processing {wav_file_path}...")
            wav_ft = librosa.load(wav_file_path, sr=16000)[0][None, None, :]  
            np.save(f"{self.saved_file}/{video_name}.npy", wav_ft)
        except Exception as e:
            print(f"Error processing {wav_file_path}: {e}")
 
    def logfbank_extract(self, wav_file_path, video_name):
        try:
            print(f"Processing {wav_file_path}...")
            rate, sig = wav.read(wav_file_path)
            fbank_feat = logfbank(sig, rate)  
            a = fbank_feat.flatten()
            single_vec_feat = a.reshape(1, -1) 
            np.save(f"{self.saved_file}/{video_name}.npy", single_vec_feat)
        except Exception as e:
            print(f"Error processing {wav_file_path}: {e}")
 
    def opensmile_extract(self, wav_file_path, video_name):
        try:
            print(f"Processing {wav_file_path}...")
            out = self.smile.process_file(wav_file_path)
            arr = np.array(out)
            np.save(f"{self.saved_file}/{video_name}.npy", arr)
        except Exception as e:
            print(f"Error processing {wav_file_path}: {e}")
 
    @staticmethod
    def processed_files(save_to):
        processed_file = os.listdir(save_to)
        file_name = [item.replace(".npy", "") for item in processed_file]
        return file_name
 
 
def audio_process(mode, aud_dir, saved_dir):
    from tqdm import tqdm
 
    processor = RawAudioProcessor(mode, aud_dir, saved_dir)
    for idx in tqdm(range(len(processor))):
        processor[idx]
 

 
 
 
 