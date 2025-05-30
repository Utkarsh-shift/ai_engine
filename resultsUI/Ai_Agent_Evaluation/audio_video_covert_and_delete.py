import os,shutil,cv2,subprocess
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import os
import librosa
import opensmile
import numpy as np
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import glob

def delete_all_files_in_folder(folder_path):
    try:
        # Check if the folder exists
        if not os.path.exists(folder_path):
            return
        
        # Iterate over all files in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            # Check if the path is a file and not a directory
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted file: {filename}")
        
        print(f"All files deleted successfully from folder: {folder_path}")
    except OSError as e:
        print(f"Error deleting files from folder: {folder_path} - {e}")
 
 
def delete_all_folders_in_folder(folder_path):
    # Get a list of all subdirectories in the folder
    folders = [os.path.join(folder_path, d) for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
 
    for folder in folders:
        try:
            shutil.rmtree(folder)  # Remove the folder and all its contents
            print(f"Deleted folder: {folder}")
        except Exception as e:
            print(f"Error deleting folder {folder}: {e}")
 
 


def frame_extract(video_path, save_dir, resize=(456, 256), transform=None):

    skip_rate = max(1, int(round(original_fps / 15)))
 
    # Extract the base filename from the video path
    file_name = Path(video_path).stem
 
    # Construct the directory path where the frames will be saved
    save_path = Path(save_dir).joinpath(file_name)
    os.makedirs(save_path, exist_ok=True)
 
    frame_count = 0  # Counter for frames processed
    saved_frame_count = 0  # Counter for frames saved
 
    while True:
        ret, frame = cap.read()
 
        # Break the loop if no frame is read
        if not ret:
            break
 
        # Only save every skip_rate'th frame
        if frame_count % skip_rate == 0:
            if transform:
                frame = transform(frame)
 
            # Resize the frame
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_CUBIC)
 
            # Construct the filename for the saved frame
            frame_filename = f"{save_path}/frame_{saved_frame_count + 1}.jpg"
 
            # Save the frame
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
            
 
        frame_count += 1
 
 
    # Release the video capture object and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    print(f"Extraction completed, {saved_frame_count} frames saved.")


def crop_to_square(img):
    h, w, _ = img.shape
    c_x, c_y = int(w / 2), int(h / 2)
    img = img[:, c_x - c_y: c_x + c_y]
    return img
 
 
def long_time_task(video, parent_dir):
        print(f"execute {video} ...")
        return frame_extract(video_path=video, save_dir=parent_dir, resize=(256, 256), transform=crop_to_square)
 
 
def convert_videos_to_frames(video_dir, output_dir):
    p = Pool(8)
    path = Path(video_dir)
    i = 0
    video_pts = list(path.rglob("*.mp4"))
    print("Making frames ")
    for video in tqdm(video_pts):
        i += 1
        video_path = str(video)
        if output_dir is not None:
            saved_dir = output_dir
        else:
            saved_dir = output_dir
        p.apply_async(long_time_task, args=(video_path, saved_dir))
        # frame_extract(video_path=video_path, save_dir=saved_dir, resize=(256, 256), transform=crop_to_square)
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
    print(f"processed {i} videos")
    
    
    
    
    
    
################################ extract audio ##########################
# from pydub import AudioSegment
# from pydub.silence import split_on_silence
# def clean_audio(file_path):
#     audio = AudioSegment.from_file(file_path, format="wav")
#     chunks = split_on_silence(
#         audio, 
#         min_silence_len=500, 
#         silence_thresh=audio.dBFS-14,  
#         keep_silence=200  
#     )

#     combined = AudioSegment.empty()
#     for chunk in chunks:
#         combined += chunk
#     combined.export(file_path, format="wav")



def audio_extract(dir_path, output_dir=None):
    path = Path(dir_path)
    format_str = "./*.mp4"
    mp4_ls = path.rglob(format_str)
    for mp4 in mp4_ls:
        name = mp4.stem
        if output_dir is None:
            parent_dir = mp4.parent,
        else:
            os.makedirs(output_dir, exist_ok=True)
            parent_dir = output_dir
        cmd = f"ffmpeg -i {mp4} -ab 320k -ac 2 -ar 44100 -vn {parent_dir}/{name}.wav"
        subprocess.call(cmd, shell=True)    
        
        
 
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
      
from pydub import AudioSegment
from pydub.silence import split_on_silence
import subprocess

def extract_audio(mp4_file_path, output_audio_path):
    # Extract audio from MP4 file
    command = [
        "ffmpeg", "-i", mp4_file_path, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", output_audio_path
    ]
    subprocess.run(command, check=True)

def clean_audio(file_path):
    # Extract audio from MP4 if needed
    if not file_path.endswith(".wav"):
        wav_file_path = file_path.replace(".mp4", ".wav")
        extract_audio(file_path, wav_file_path)
        file_path = wav_file_path
    

    if  os.path.exists(file_path):   
    
        audio = AudioSegment.from_file(file_path, format="wav")
        chunks = split_on_silence(
            audio, 
            min_silence_len=500, 
            silence_thresh=audio.dBFS-14,  
            keep_silence=200  
        )

        combined = AudioSegment.empty()
        for chunk in chunks:
            combined += chunk
        combined.export(file_path, format="wav")
    else : 
        return 
