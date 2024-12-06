import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import os
from pydub import AudioSegment
import urllib
import whisper
import openai
import argparse
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import torch,torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import whisper
from pydub import AudioSegment
import os
import gc
from multiprocessing import Pool, set_start_method
from tqdm import tqdm
from openai import OpenAI
import os ,json
from dotenv import load_dotenv
 
class TextProcessor:
    def __init__(self, badWordsFileName):
        badWordsFileURL = "https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/badwordslist/badwords.txt"
        if not os.path.exists(badWordsFileName):
            urllib.request.urlretrieve(badWordsFileURL, badWordsFileName)
        # Read bad words from file
        with open(badWordsFileName, 'r', encoding='utf-8') as file:
            self.badwords = file.read().splitlines()
 
    @staticmethod
    def remove_internet_chars(text):
        text = re.sub(r'\s+@\s+', ' ', text)  
        text = re.sub(r'@', ' ', text)        
        text = re.sub(r'#', ' ', text)        
        text = re.sub(r'//', ' ', text)       
        return text
 
    @staticmethod
    def remove_symbols(text):
        text = re.sub(r"[’‘`´]", "'", text)     
        # text = re.sub(r"[^a-zA-Z']", ' ', text)
        text = re.sub(r"'{2,}", ' ', text)      
        text = re.sub(r"'(\s+|$)", ' ', text)   
        text = re.sub(r"^\s+'|'\s+$", '', text)
        text = text.strip()                     
        return text
 
    def remove_badwords(self, text):
        if self.badwords:
            pattern = r'\b(?:{})\b'.format('|'.join(map(re.escape, self.badwords)))
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        return text
 
 
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
    
    del audio
    gc.collect()
    return chunks
 
import re
 
def extract_chunk_number(s):
    """Extracts the chunk number from a string like '$#@chunk_1.wav$#@'."""
    match = re.search(r'\$#@chunk_(\d+)\.wav\$#@', s)
    if match:
        return int(match.group(1))
    return -1  
 
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
 
 
 
def transcribe_chunk_batch(chunk_path , mod):
    
    """Transcribes a single audio chunk using the Whisper model."""
    audio = whisper.load_audio(chunk_path)
    audio = whisper.pad_or_trim(audio)
    
    
    result = mod.transcribe(audio)
    #print( "This result is for the wav file ", chunk_path ,"and The generated results are" ,result["text"])
    res = "$#@"+  chunk_path + "$#@" + result["text"]
    
    os.remove(chunk_path)  # Clean up the chunk files
 
    # Free up memory used by variables
    del audio
    gc.collect()  # Trigger garbage collection
    
    return res  # Return the transcribed text
 
def callback(transcription):
    """Callback function to collect transcription results."""
    
    transcriptions.append(transcription)
 
def long_time_task(audio_file , model):
    """Processes an individual audio file."""
   # print("The audio_file in long_time task is ", audio_file)
    return transcribe_chunk_batch(chunk_path=audio_file , mod= model)
 
def process_in_batches(audio_path, batch_size=8):
    """Processes the audio file in batches, transcribing chunks in parallel."""
    # Split the audio file into manageable chunks
    model = whisper.load_model("medium", download_root=os.path.join(os.getcwd(), "whisper"))
    chunks = split_audio(audio_path)
 
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print("The batch for ASR is processed")
        # Multiprocessing Pool for parallel processing of batches
        with Pool(16) as pool:
            for aud_file in tqdm(batch):
                # Apply async for each file in the batch
                pool.apply_async(long_time_task, args=(aud_file,model), callback=callback)
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
 
 
def hf(text):
    load_dotenv()
    client = OpenAI(api_key ="sk-proj-5J6AGNVXQEJ6Ji9NATgCT3BlbkFJblejOzq7DI9TgfagRxd7")
    message={"role": "system",
            "content": f"""If the given text is not in English, translate it to English only. If the text is already in English, return the exact same text without any modification. The text to be processed is: '{text}'"""}
    client = OpenAI(api_key =os.getenv("OPENAI_API_KEY"))
    chat_completion = client.chat.completions.create(
            model="gpt-4o",
            messages = [message]
            ,max_tokens=256
        )
        # print("|\n|\n|\n|\n|\n|\n|\n|\n|v",chat_completion)
    # finish_reason = chat_completion.choices[0].finish_reason
 
    
    newdata = chat_completion.choices[0].message.content
    print("newdta is ",newdata)
    return newdata
 

def transcribe_chunk_batch_new(audio_path):
    try:
        model = whisper.load_model("large-v2", download_root=os.path.join(os.getcwd(), "whisper"))
        chunks = split_audio(audio_path)
        chunks.sort()
        print(chunks)
        for i in chunks:

            audio = whisper.load_audio(i)
            audio = whisper.pad_or_trim(audio)

            result = model.transcribe(audio)
            #print( "This result is for the wav file ", chunk_path ,"and The generated results are" ,result["text"])
            res = result["text"]
            print("transcription is:",res)
            transcriptions.append(res)
            os.remove(i)
        # cleaned_list=sort_and_clean_chunks(transcriptions)
        return "".join(transcriptions)
    except Exception as e:
        print("exception occurs:",e)

def cal_uni_bi(audio_file):
  
   
    set_start_method("spawn", force=True)  # Use "spawn" to avoid issues with fork and CUDA
    
    # model = whisper.load_model("large-v2")ownload('punkt')
    # nltk.download('stopwords')    
    
    # transcription = process_in_batches(audio_file, batch_size=8)
    transcription=transcribe_chunk_batch_new(audio_file)
    torch.cuda.empty_cache()
    print("Final Transcription:", transcription)
    gc.collect()
    # text=transcription
    text=hf(transcription)
    print("The text is , ", text)
    load_dotenv()
    # test = "je m'appelle utkarsh, j'etudie avec varun et j'habite en panchkula"
    # message={"role": "system",
    #         "content": f"""calculate the grammer score for the text {text}, out of 100 consider vocabulary, grammer mistakes, sentence formation only give response in digit and no text, give minimum 10 score."""}
    client = OpenAI(api_key =os.getenv("OPENAI_API_KEY"))
    # chat_completion = client.chat.completions.create(
    #         model="gpt-3.5-turbo",
    #         messages = [message]
    #     )
    #     # print("|\n|\n|\n|\n|\n|\n|\n|\n|v",chat_completion)
    # finish_reason = chat_completion.choices[0].finish_reason
 
 
    # newdata = chat_completion.choices[0].message.content
    # double_digit = re.findall(r'\b\d{2}\b', newdata)
    # newdata = int(double_digit[0])
 
    example_json="""{"grammer_score":<<score>>,
            "grammer_comment":<<comment>>            
            }"""
    message={"role": "system",
            "content": f"""calculate the grammer score and also give comment for the text "{text}", out of 100 consider vocabulary, grammer mistakes, sentence formation.Give response in the json format as given below:
            {example_json}
            """}
    
    chat_completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            response_format={"type":"json_object"},
            messages = [message]
        )
        # print("|\n|\n|\n|\n|\n|\n|\n|\n|v",chat_completion)
    finish_reason = chat_completion.choices[0].finish_reason
 
 
    newdata = chat_completion.choices[0].message.content
    # newdata = chat_completion.choices[0].message.content
    json_data=json.loads(newdata)
    print("newdta is ",json_data)
    print("grammer_score is ",json_data["grammer_score"])
    print("grammer_comment",json_data["grammer_comment"])
    grammer_score=json_data["grammer_score"]
    grammer_comment=json_data["grammer_comment"]
    # newdata = json.loads(data)
 
 
    # text = TextProcessor.remove_internet_chars(text)
    # text = TextProcessor.remove_symbols(text)
    text = text.lower()
 
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    if scores['compound'] >= 0.05:
        sentiment='Positive'
    elif scores['compound'] <= -0.05:
        sentiment='Negative'
    else:
        sentiment='Neutral'
 
    if scores['compound'] >= 0.05:
 
        sentiment_sc=(100*scores["pos"])*10
        if sentiment_sc>100:
            sentiment_sc=100
 
    elif scores['compound'] <= -0.05:
 
        sentiment_sc=33
 
    else:
 
        sentiment_sc=50
 
    
    print("text without tokenization is --------------------------------------------------",text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokenized_corpus = [word for word in tokens if word.lower() not in stop_words]
    
    # word_freq = Counter(tokenized_corpus)
    
    #wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=50).generate_from_frequencies(word_freq)
    
    unigrams = list(ngrams(tokenized_corpus, 1))
    bigrams = list(ngrams(tokenized_corpus, 2))
#   trigrams = list(ngrams(tokenized_corpus, 3))
 
#   print("The unigrams are ",unigrams)
#   print("The bigrams are ",bigrams)
#    print("The trigrams are " , trigrams)
 
    unigram_freq = Counter(unigrams)
    bigram_freq = Counter(bigrams)
#    trigram_freq = Counter(trigrams)
    
    df_unigrams = pd.DataFrame(list(unigram_freq.items()), columns=['Word', 'Frequency'])
    df_bigrams = pd.DataFrame(list(bigram_freq.items()), columns=['Word', 'Frequency'])
#   df_trigrams = pd.DataFrame(list(trigram_freq.items()), columns=['Word', 'Frequency'])
    #print(df_unigrams['Frequency'])
    
    df_unigrams = df_unigrams.sort_values(by='Frequency', ascending=False).reset_index(drop=True)
    df_bigrams = df_bigrams.sort_values(by='Frequency', ascending=False).reset_index(drop=True)
#  df_trigrams = df_trigrams.sort_values(by='Frequency', ascending=False).reset_index(drop=True)
 
    #print(f"Uni-gram freq sum is :{sum(df_unigrams['Frequency'][:10])}")
    #print(f"Bi-gram freq sum is :{sum(df_bigrams['Frequency'][:10])}")
# print(f"Tri-gram freq sum is :{sum(df_trigrams['Frequency'][:10])}")
    # print("text  tokenization is --------------------------------------------------",text)
    torch.cuda.empty_cache()
    return df_unigrams , df_bigrams ,sentiment,sentiment_sc,text,grammer_score,grammer_comment
 
if __name__ == "__main__":
    cal_uni_bi()