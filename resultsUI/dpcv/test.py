# import torch,torchaudio
# from transformers import WhisperForConditionalGeneration, WhisperProcessor

# # Load the fine-tuned model and processor
# model_path = "/media/almabay/StorageDisk/For_Varun/whisper-small-eng-15000"
# from transformers import WhisperForConditionalGeneration, WhisperProcessor

# # Load the fine-tuned model and processor
# # model_path = "/path/to/your/model_directory"
# model = WhisperForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
# processor = WhisperProcessor.from_pretrained(model_path, local_files_only=True)
# checkpoint_path = "/media/almabay/StorageDisk/For_Varun/whisper-small-eng-15000/checkpoint-5000"
# model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path, local_files_only=True)

# # Load the audio file (you can use any audio file path)
# audio_path = "/media/almabay/StorageDisk/Interviewer_22/Interviewer/datasets/ChaLearn/voice_data/voice_raw/test_data/watchright.wav"
# waveform, sample_rate = torchaudio.load(audio_path)

# # Resample the audio if it's not at 16000Hz
# if sample_rate != 16000:
#     waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

# # Ensure the waveform is mono (single channel)
# if waveform.shape[0] > 1:
#     waveform = waveform.mean(dim=0, keepdim=True)

# # Convert waveform to float32
# waveform = waveform.float()

# # Process the audio input for the Whisper model
# input_features = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_features

# # Generate transcription
# generated_ids = model.generate(input_features)
# transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


# # Print the transcription
# print("Transcription:", transcription)


# import os
# import glob
# output_dir = "datasets/ChaLearn/test_data"
# import os
# import shutil

# def delete_all_folders_in_folder(folder_path):
#     # Get a list of all subdirectories in the folder
#     folders = [os.path.join(folder_path, d) for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

#     for folder in folders:
#         try:
#             shutil.rmtree(folder)  # Remove the folder and all its contents
#             print(f"Deleted folder: {folder}")
#         except Exception as e:
#             print(f"Error deleting folder {folder}: {e}")

# # Specify the folder path
# folder_path = "/path/to/your/folder"

# # Call the function to delete all folders inside the specified folder
# delete_all_folders_in_folder(output_dir)
# from dotenv import load_dotenv
# import openai,os
# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")

# def check_grammar(text):
#     response = openai.ChatCompletion.create(
#         model="gpt-4",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant that provides grammar and style corrections."},
#             {"role": "user", "content": f"Please check the grammar and provide corrections for the following text:\n\n{text}"}
#         ],
#         max_tokens=1000,
#         temperature=0
#     )
#     return response.choices[0].message['content'].strip()


# text = "MANY OTHER IMPORTANT REFERENCES BETWEEN WHAT WE CONSIDER EARLIER WHOM WE WILL CALL MULTI LAYER FEEDFOWER NETWORK WHEN I WILL SAY MULTI FEEDFOWER NETWORK."
# grammar_feedback = check_grammar(text)
# print("Grammar Feedback:", grammar_feedback)
# import whisper,os

# model = whisper.load_model("medium",download_root=os.path.join(os.getcwd(), "whisper"))
# testdir = "./datasets/ChaLearn/voice_data/voice_raw/test_data"
# nltk.download('punkt')
# nltk.download('stopwords')

# for aud in os.listdir(testdir):
#     if not ".praat" in aud:
#         audio_file = os.path.join(testdir, aud)
#         audio = whisper.load_audio(audio_file)
#            # load audio and pad/trim it to fit 30 seconds
#         # audio = whisper.load_audio(audio_path)
#         audio = whisper.pad_or_trim(audio)

#         # make log-Mel spectrogram and move to the same device as the model
#         mel = whisper.log_mel_spectrogram(audio).to(model.device)

#         # detect the spoken language
#         _, probs = model.detect_language(mel)
#         print(f"Detected language: {max(probs, key=probs.get)}")
#         lang = max(probs, key=probs.get)
#         # decode the audio
#         options = whisper.DecodingOptions(temperature=1.0, best_of=5, fp16=False, sample_len=150)
#         result = whisper.decode(model, mel, options)

#     # print the recognized text
#     print(result.text)


# whisper.load_model("medium",download_root=os.path.join(os.getcwd(), "whisper")).gpu()


# audio_file="./datasets/ChaLearn/voice_data/voice_raw/test_data/good_score.wav"
# audio = whisper.load_audio(audio_file)
# #            # load audio and pad/trim it to fit 30 seconds
# # audio = whisper.load_audio(audio_path)
# audio = whisper.pad_or_trim(audio)

# # make log-Mel spectrogram and move to the same device as the model
# mel = whisper.log_mel_spectrogram(audio).to(model.device)

# # detect the spoken language
# _, probs = model.detect_language(mel)
# print(f"Detected language: {max(probs, key=probs.get)}")
# lang = max(probs, key=probs.get)
# # decode the audio
# options = whisper.DecodingOptions(temperature=1.0, best_of=10, fp16=False, sample_len=500)
# result = whisper.decode(model, mel, options)

# # print the recognized text
# print(result.text)



# import whisper
# from pydub import AudioSegment
# import os

# def split_audio(audio_path, chunk_length_ms=30000):
#     audio = AudioSegment.from_file(audio_path)
#     chunks = []
#     for i in range(0, len(audio), chunk_length_ms):
#         chunk = audio[i:i + chunk_length_ms]
#         chunk_path = f"chunk_{i // chunk_length_ms}.wav"
#         chunk.export(chunk_path, format="wav")
#         chunks.append(chunk_path)
#     return chunks

# def transcribe_audio_chunks(chunk_paths, model):
#     transcriptions = []
#     for chunk_path in chunk_paths:
#         audio = whisper.load_audio(chunk_path)
        
#         result = model.transcribe(audio)
#         transcriptions.append(result['text'])
#         os.remove(chunk_path)  # Clean up the chunk files
#     return " ".join(transcriptions)

# def main(audio_path):
#     model = whisper.load_model("medium",download_root=os.path.join(os.getcwd(), "whisper"))  # Load the Whisper model
#     chunks = split_audio(audio_path)
#     transcription = transcribe_audio_chunks(chunks, model)
#     print("Transcription:")
#     print(transcription)


# # import whisper
# # from pydub import AudioSegment
# # import os
 
# # def split_audio(audio_path, chunk_length_ms=30000):
# #     audio = AudioSegment.from_file(audio_path)
# #     chunks = []
# #     for i in range(0, len(audio), chunk_length_ms):
# #         chunk = audio[i:i + chunk_length_ms]
# #         chunk_path = f"chunk_{i // chunk_length_ms}.wav"
# #         chunk.export(chunk_path, format="wav")
# #         chunks.append(chunk_path)
# #     return chunks
 
# # def transcribe_audio_chunks(chunk_paths, model):
# #     transcriptions = []
# #     for chunk_path in chunk_paths:
# #         audio = whisper.load_audio(chunk_path)
# #         audio = whisper.pad_or_trim(audio)
 
# # # make log-Mel spectrogram and move to the same device as the model
# #         mel = whisper.log_mel_spectrogram(audio).to(model.device)
 
# #         # detect the spoken language
# #         _, probs = model.detect_language(mel)
# #         print(f"Detected language: {max(probs, key=probs.get)}")
# #         lang = max(probs, key=probs.get)
# #         # decode the audio
# #         options = whisper.DecodingOptions(temperature=1.0, best_of=5, fp16=False, sample_len=150)
# #         result = whisper.decode(model, mel, options)
# #         # result = model.transcribe(audio)
# #         transcriptions.append(result.text)
# #         os.remove(chunk_path)  # Clean up the chunk files
# #     return " ".join(transcriptions)
 
# # def main(audio_path):
# #     model = whisper.load_model("medium",download_root=os.path.join(os.getcwd(), "whisper"))  # Load the Whisper model
# #     chunks = split_audio(audio_path)
# #     transcription = transcribe_audio_chunks(chunks, model)
# #     print("Transcription:")
# #     print(transcription)



# if __name__ == "__main__":
#     # audio_path = "./datasets/ChaLearn/voice_data/voice_raw/test_data/good_score.wav"
#     audio_path="/home/almabay/Downloads/PM-Narendra-Modi_s-Address-To-Nation-_-India-Today.wav"

#     main(audio_path)


# import whisper
# from pydub import AudioSegment
# import os
# import numpy as np
# import concurrent.futures

# def split_audio(audio_path, chunk_length_ms=30000):
#     audio = AudioSegment.from_file(audio_path)
#     chunks = []
#     for i in range(0, len(audio), chunk_length_ms):
#         chunk = audio[i:i + chunk_length_ms]
#         if len(chunk) < chunk_length_ms:
#             # Pad short chunks with silence
#             silence = AudioSegment.silent(duration=(chunk_length_ms - len(chunk)))
#             chunk = chunk + silence
#         chunk_path = f"chunk_{i // chunk_length_ms}.wav"
#         chunk.export(chunk_path, format="wav")
#         chunks.append(chunk_path)
#     return chunks

# def audio_to_array(audio_path):
#     audio = AudioSegment.from_wav(audio_path)
#     audio_array = np.array(audio.get_array_of_samples())
#     return audio_array.astype(np.float32) / (2**15)  # Normalize to [-1, 1]

# def transcribe_chunk(chunk_path, model):
#     audio_array = audio_to_array(chunk_path)
#     # Whisper expects a tensor with a specific shape. Ensure your audio array has the correct shape.
#     # Adjust if needed based on Whisper’s requirements.
#     audio_tensor = whisper.pad_or_trim(audio_array)  # Replace with appropriate function if available
#     result = model.transcribe(audio_tensor)
#     os.remove(chunk_path)  # Clean up the chunk file
#     return result['text']

# def transcribe_audio_chunks(chunk_paths, model):
#     transcriptions = []
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         # Map the transcribe_chunk function to each chunk path
#         future_to_chunk = {executor.submit(transcribe_chunk, chunk_path, model): chunk_path for chunk_path in chunk_paths}
        
#         for future in concurrent.futures.as_completed(future_to_chunk):
#             chunk_path = future_to_chunk[future]
#             try:
#                 text = future.result()
#                 transcriptions.append(text)
#             except Exception as exc:
#                 print(f'Chunk {chunk_path} generated an exception: {exc}')
    
#     return " ".join(transcriptions)

# def main(audio_path):
#     model = whisper.load_model("medium",download_root=os.path.join(os.getcwd(), "whisper"))  # Load the Whisper model
#     chunks = split_audio(audio_path)
#     transcription = transcribe_audio_chunks(chunks, model)
#     print("Transcription:")
#     print(transcription)



# if __name__ == "__main__":
#     # audio_path = "./datasets/ChaLearn/voice_data/voice_raw/test_data/good_score.wav"
#     audio_path="/home/almabay/Downloads/PM-Narendra-Modi_s-Address-To-Nation-_-India-Today.wav"

#     main(audio_path)


# import whisper
# from pydub import AudioSegment
# import os
# from multiprocessing import Pool, cpu_count, set_start_method

# def split_audio(audio_path, chunk_length_ms=30000):
#     audio = AudioSegment.from_file(audio_path)
#     chunks = []
#     for i in range(0, len(audio), chunk_length_ms):
#         chunk = audio[i:i + chunk_length_ms]
#         chunk_path = f"chunk_{i // chunk_length_ms}.wav"
#         chunk.export(chunk_path, format="wav")
#         chunks.append(chunk_path)
#     return chunks

# def transcribe_chunk(chunk_path, model_name):
#     # Load the model within the subprocess
#     model = whisper.load_model(model_name, download_root=os.path.join(os.getcwd(), "whisper"))
#     audio = whisper.load_audio(chunk_path)
#     result = model.transcribe(audio)
#     os.remove(chunk_path)  # Clean up the chunk file
#     return result['text']

# def transcribe_audio_chunks(chunk_paths, model_name):
#     # Create a Pool of workers equal to the number of CPU cores
#     with Pool(processes=cpu_count()) as pool:
#         # Map the transcribe_chunk function to each chunk path
#         results = pool.starmap(transcribe_chunk, [(chunk_path, model_name) for chunk_path in chunk_paths])
#     return " ".join(results)




# def main(audio_path):
#     model = whisper.load_model("medium", download_root=os.path.join(os.getcwd(), "whisper"))  # Load the Whisper model
#     chunks = split_audio(audio_path)
#     transcription = transcribe_audio_chunks(chunks, model)
#     print("Transcription:")
#     print(transcription)

# if __name__ == "__main__":
#     # Make sure to call main only when the script is executed directly
#     set_start_method('spawn')
#     main("/home/almabay/Downloads/PM-Narendra-Modi_s-Address-To-Nation-_-India-Today.wav")







# import whisper,tqdm
# from pydub import AudioSegment
# import os,gc
# from multiprocessing import Pool,set_start_method
 
# def split_audio(audio_path, chunk_length_ms=30000):
#     audio = AudioSegment.from_file(audio_path)
#     chunks = []
#     for i in range(0, len(audio), chunk_length_ms):
#         chunk = audio[i:i + chunk_length_ms]
#         chunk_path = f"chunk_{i // chunk_length_ms}.wav"
#         chunk.export(chunk_path, format="wav")
#         chunks.append(chunk_path)
#     return chunks
 
# def transcribe_audio_chunks(chunk_paths, model):
    
#     for chunk_path in chunk_paths:
#         audio = whisper.load_audio(chunk_path)
        
#         result = model.transcribe(audio)
#         transcription=result["text"]
#         os.remove(chunk_path)  # Clean up the chunk files
#     return transcription
 

# transcriptions = []

# def mycallback(transcription):
#         transcriptions.append(transcription)
#            # Ensure the count is valid
#         return "".join(transcriptions)
        
# def long_time_task(chunks,model):
    
#     transcription = transcribe_audio_chunks(chunks, model)
#     # print("Transcription:")
#     # print(transcription)
#     return transcription
    
# def process_in_batches(audio_path):
#     cnt = None
#     model = whisper.load_model("medium",download_root=os.path.join(os.getcwd(), "whisper"))
#     chunks = split_audio(audio_path)
    
#     with Pool(8) as q:  
        
#         cnt = q.apply_async(long_time_task, args=(chunks, model), callback=mycallback)
#         q.close()
#         q.join()  
#     gc.collect()  
#     print(f"Processed batch 1")
   # print( "The get ( output ud d )", counts)
    # return cnt



# if __name__ == "__main__":
#     # audio_path = "./datasets/ChaLearn/voice_data/voice_raw/test_data/good_score.wav"
#     audio_path="/home/almabay/Downloads/PM-Narendra-Modi_s-Address-To-Nation-_-India-Today.wav"
#     set_start_method("spawn")
#     d=process_in_batches(audio_path)
#     print(d)
    
# import whisper
# from pydub import AudioSegment
# import os, gc
# from multiprocessing import Pool, set_start_method, Manager

# def split_audio(audio_path, chunk_length_ms=30000):
#     audio = AudioSegment.from_file(audio_path)
#     chunks = []
#     for i in range(0, len(audio), chunk_length_ms):
#         chunk = audio[i:i + chunk_length_ms]
#         chunk_path = f"chunk_{i // chunk_length_ms}.wav"
#         chunk.export(chunk_path, format="wav")
#         chunks.append(chunk_path)
#     return chunks

# def transcribe_chunk(chunk_path, model):
#     audio = whisper.load_audio(chunk_path)
#     result = model.transcribe(audio)
#     transcription = result["text"]
#     os.remove(chunk_path)  # Clean up the chunk files
#     return transcription

# def process_in_batches(audio_path):
#     # Load model once
#     model = whisper.load_model("medium", download_root=os.path.join(os.getcwd(), "whisper"))

#     # Split audio into chunks
#     chunks = split_audio(audio_path)

#     # Using multiprocessing.Manager to manage shared list of transcriptions
#     with Manager() as manager:
#         transcriptions = manager.list()

#         # Multiprocessing Pool
#         with Pool(8) as pool:
#             results = [pool.apply_async(transcribe_chunk, args=(chunk_path, model), callback=transcriptions.append) for chunk_path in chunks]
#             for r in results:
#                 r.wait()  # Ensure all processes complete

#         # Combine transcriptions
#         full_transcription = "".join(transcriptions)
#         print("Processed batch 1")
#         return full_transcription

# if __name__ == "__main__":
#     set_start_method("spawn") 

#     audio_path = "/home/almabay/Downloads/PM-Narendra-Modi_s-Address-To-Nation-_-India-Today.wav"
#     transcription = process_in_batches(audio_path)
#     print("Final Transcription:", transcription)
#     gc.collect()

# uper wala sahi hai

# import whisper
# from pydub import AudioSegment
# import os
# import gc
# from multiprocessing import Pool, set_start_method, Manager

# transcriptions=[]

# def split_audio(audio_path, chunk_length_ms=30000):
#     """Splits the audio file into chunks and keeps them on disk."""
#     audio = AudioSegment.from_file(audio_path)
#     chunks = []
#     for i in range(0, len(audio), chunk_length_ms):
#         chunk = audio[i:i + chunk_length_ms]
#         chunk_path = f"chunk_{i // chunk_length_ms}.wav"
#         chunk.export(chunk_path, format="wav")
#         chunks.append(chunk_path)
#     return chunks

# def transcribe_chunk_batch(chunk_path):
#     """Transcribes a batch of audio chunks using the Whisper model."""
#     model = whisper.load_model("medium", download_root=os.path.join(os.getcwd(), "whisper"))
#     print(chunk_path)
#     audio = whisper.load_audio(chunk_path)
#     audio = whisper.pad_or_trim(audio)
#     mel = whisper.log_mel_spectrogram(audio).to(model.device)
    
#     # Decode the audio
#     options = whisper.DecodingOptions()
#     result = whisper.decode(model, mel, options)
    
#     os.remove(chunk_path)  # Clean up the chunk files
#     print("transcribe function",result)
#     return result
 


# def callback(transcription):
#     transcriptions.append(transcription)

# def long_time_task(audio_file):
#     print(audio_file)
#     return transcribe_chunk_batch(chunk_path=audio_file)


# from tqdm import tqdm
# def process_in_batches(audio_path, batch_size=8):
#     """Processes the audio file in batches, transcribing 8 chunks at a time."""
#     # Split the audio file into manageable chunks
#     chunks = split_audio(audio_path)
#    # print(chunks)

#     # Group chunks into batches of specified size
#    # chunk_batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]

#     # Using multiprocessing.Manager to manage a shared list of transcriptions
#     # with Manager() as manager:
#     #     transcriptions = manager.list()
#     for i in range(0,len(chunks) , batch_size) :
#         batch = chunks[i:i+batch_size]
#        # print(batch)
#         # Multiprocessing Pool for parallel processing of batches
#         with Pool(8) as pool:
#             for aud_file in tqdm(batch):
#                # print("The batch size is given as ",aud_file)
#                 results = pool.apply_async(long_time_task, args=(aud_file), callback=callback)
#             pool.close()
#             pool.join() 
#              # Ensure all processes complete
#         gc.collect()

#         # Combine all transcriptions into a single output
#         full_transcription = "".join(transcriptions)
#         print("Processed all batches")
        
#         return full_transcription

# if __name__ == "__main__":
#     set_start_method("spawn", force=True)  # Use "spawn" to avoid issues with fork and CUDA

#     audio_path = "/home/almabay/Downloads/PM-Narendra-Modi_s-Address-To-Nation-_-India-Today.wav"
#     transcription = process_in_batches(audio_path, batch_size=8)
#     print("Final Transcription:", transcription)
#     gc.collect()





import whisper
from pydub import AudioSegment
import os
import gc
from multiprocessing import Pool, set_start_method
from tqdm import tqdm
from decouple import config
from openai import OpenAI
# Global list to collect transcriptions
transcriptions = []

def split_audio(audio_path, chunk_length_ms=30000):
    """Splits the audio file into chunks and keeps them on disk."""
    audio = AudioSegment.from_file(audio_path)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunk_path = f"chunk_{i // chunk_length_ms}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    
    del audio  # Free the audio object after splitting
    gc.collect()  # Trigger garbage collection
    return chunks

import re

def extract_chunk_number(s):
    """Extracts the chunk number from a string like '$#@chunk_1.wav$#@'."""
    match = re.search(r'\$#@chunk_(\d+)\.wav\$#@', s)
    if match:
        return int(match.group(1))
    return -1  # Return a default value if no match is found

def remove_chunk_tag(s):
    """Removes the chunk tag from the string."""
    return re.sub(r'\$#@chunk_\d+\.wav\$#@', '', s)

def sort_and_clean_chunks(lst):
    """Sorts the list of strings based on the chunk number and removes the tags."""
    # Sort the list using the extracted chunk number as the key
    sorted_list = sorted(lst, key=extract_chunk_number)
    
    # Remove the tags from each sorted item
    cleaned_list = [remove_chunk_tag(item) for item in sorted_list]
    
    return cleaned_list



#def transcribe_chunk_batch(chunk_path , mod):
    
    """Transcribes a single audio chunk using the Whisper model."""
  #  audio = whisper.load_audio(chunk_path)
  #  audio = whisper.pad_or_trim(audio)
#    client = OpenAI(api_key = config("OPENAI_API_KEY"))
#    audio_file = opne(chunk_path, "rb")
#    transcription = client.audio.transcriptions.create(
#    model="whisper-1", 
#    file=audio_file, 
#    temperature=0.0,
#    response_format="text"
#    )

#    full_transcription=transcription
   # print(full_transcription) 
    
    #result = mod.transcribe(audio)
    #print( "This result is for the wav file ", chunk_path ,"and The generated results are" ,result["text"])
#    res = "$#@"+  chunk_path + "$#@" + full_transcription
    
#    os.remove(chunk_path)  # Clean up the chunk files

    # Free up memory used by variables
#    del audio
#    gc.collect()  # Trigger garbage collection 
#    return res  # Return the transcribed text











def transcribe_chunk_batch(chunk_path, mod):
    client = OpenAI(api_key=config("OPENAI_API_KEY"))
    try:
        with open(chunk_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                temperature=0.0,
                response_format="text"
            )
        
        full_transcription = transcription

        
        res = "$#@" + chunk_path + "$#@" + full_transcription

    finally:
       
        if os.path.exists(chunk_path):
            os.remove(chunk_path)
       
        client.close()

       
        del transcription, full_transcription, client , chunk_path
        gc.collect()  

   
    return res



def callback(transcription):
    """Callback function to collect transcription results."""
    
    transcriptions.append(transcription)

def long_time_task(audio_file ):
    """Processes an individual audio file."""
   # print("The audio_file in long_time task is ", audio_file)
    return transcribe_chunk_batch(chunk_path=audio_file )

def process_in_batches(audio_path, batch_size=8):
    """Processes the audio file in batches, transcribing chunks in parallel."""
    # Split the audio file into manageable chunks
#    model = whisper.load_model("medium", download_root=os.path.join(os.getcwd(), "whisper"))
    chunks = split_audio(audio_path)

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        # Multiprocessing Pool for parallel processing of batches
        with Pool(8) as pool:
            for aud_file in tqdm(batch):
                # Apply async for each file in the batch
                pool.apply_async(long_time_task, args=(aud_file), callback=callback)
            pool.close()
            pool.join()  # Ensure all processes complete

        # Free up memory used by the batch
        del batch
        gc.collect()

    # Combine all transcriptions into a single output
    cleaned_list = sort_and_clean_chunks(transcriptions)
    full_transcription = "".join(cleaned_list)
    print("Processed all batches")
    
    # Clean up the transcriptions list
    del transcriptions[:]
    gc.collect()
    
    return full_transcription

if __name__ == "__main__":
    set_start_method("spawn", force=True)  # Use "spawn" to avoid issues with fork and CUDA
    audio_path = "/home/almabay/Downloads/PM-Narendra-Modi_s-Address-To-Nation-_-India-Today.wav"
    transcription = process_in_batches(audio_path, batch_size=8)
    
    print("Final Transcription:", transcription)
    gc.collect()
