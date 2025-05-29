
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter

from pydub import AudioSegment
import whisper

import math


from multiprocessing import Pool, set_start_method

import json
from dotenv import load_dotenv


import torch
import re 


import gc
import os


from openai import OpenAI
import nltk






def calculate_avg_logprob(transcription_segments):   
    avg_logprobs = [segment['avg_logprob'] for segment in transcription_segments]
    avg_logprob_mean = sum(avg_logprobs) / len(avg_logprobs)
    return avg_logprob_mean

def split_audio(audio_path, chunk_length_ms=15000):
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









 
def scale_avg_logprob_to_score(avg_logprob, ideal_min=-0.1, ideal_max=0.0, batch_size=0.02, batch_penalty=1):

    if ideal_max <= avg_logprob <= ideal_min:
        return 100
    if avg_logprob < ideal_min:
        distance = ideal_min - avg_logprob
    else:
        distance = avg_logprob - ideal_max

    quotient = int(distance // batch_size)
    remainder = distance % batch_size

    penalty = sum([batch_penalty * (2 ** i) for i in range(quotient)]) + remainder

    score = max(0, 100 - penalty)
 
    if score == 0:
        score = 40
 
    return score

def scale_wpm_to_score(wpm, min_wpm=140, max_wpm=160, batch_size=6, batch_penalty=2):
    if wpm <= 0:
        return 0  # No score, max penalty applied
   
    if min_wpm <= wpm <= max_wpm:
        return 100
   
    if wpm < min_wpm:
        distance = min_wpm - wpm
    else:
        distance = wpm - max_wpm
 
    quotient = distance // batch_size
    remainder = distance % batch_size
   
    penalty = sum([batch_penalty * (2 ** i) for i in range(int(quotient))]) + remainder
   
    score = max(0, 100 - penalty)
    if score == 0:
      score = 40    
    return score


def transcribe_chunk_batch_new(chunks, model):
    transcriptions = []
    total_duration = 0
    total_words=0
    avg_logprobs = []
    try:
        for i in chunks:
            audio = whisper.load_audio(i)
            audio = whisper.pad_or_trim(audio)
            transcription = model.transcribe(audio, language="en")
            try :
                final_transcription = transcription['text']
            except :
                final_transcription = transcription.get('text','')
            words = final_transcription.split()
            transcriptions.append(final_transcription)
            num_words = len(words)
            total_words += num_words
            for segment in transcription['segments']:
                segment_duration = segment['end'] - segment['start']
                total_duration += segment_duration
                avg_logprobs.append(segment['avg_logprob'])
            os.remove(i)
        print("*******************************" , transcriptions)    
        full_transcription = "".join(transcriptions)
        print( "Total words in transcription:", total_words,"__________________________________________")
        print("Total duration in seconds:", total_duration,"############################################")
        duration_in_minutes = total_duration/60
        if duration_in_minutes > 0:
            wpm = total_words / duration_in_minutes
        else:
            wpm = 0
 
        print(f"Total Words: {total_words}, Duration: {total_duration} seconds")
        print(f"Words per minute (WPM): {wpm:.2f}")
        try : 
        #  wpm_score = scale_wpm_to_score(wpm , ideal_min=120, ideal_max=180)
           wpm_score = scale_wpm_to_score(wpm )
        except Exception as e :
            print(e)
        print(f"wpm Score : {wpm_score}")
 
 
        pace_prompt = f"""If the person is speaking at an average rate of {wpm} words per minute and is given a score of {wpm_score}, then comment on the pace of the person's speech in one line. Also, mention the speed of the person in speech, and note that the ideal speed is between 140-160 words per minute. Comment only in one line."""

        
        pace_comment = get_comment(pace_prompt)
        print("pace comment is :",pace_comment)
        
        try : 
            avg_logprob_mean = sum(avg_logprobs) / len(avg_logprobs)
        except : 
            avg_logprob_mean = -0.9
        print("The average_logprob is calculates as" , avg_logprob_mean)
        
        avg_logprob_score = scale_avg_logprob_to_score(avg_logprob_mean )
        
        print("The arrticualtion score is given as " , avg_logprob_score)


        articulation_prompt = f"""If the person's pronunciation is given a score of {avg_logprob_score}, and the average log probability of the pronunciation is {avg_logprob_mean}, with the ideal range being -0.1 to -0.2, then comment on the person's pronunciation in one line only. Do not mention the average log probability or ideal range in the comment."""

        articulation_prompt_comment = get_comment(articulation_prompt)
        print("The WPM_score is given as" , wpm_score)
 
        print("**************************************************************************************************************",articulation_prompt)
 
        print(full_transcription , wpm_score , avg_logprob_score , pace_comment , articulation_prompt_comment)
 
        print("**************************************************************************************************************")
 
        return full_transcription, wpm_score, avg_logprob_score , pace_comment , articulation_prompt_comment
    
    
    except Exception as e:
        print("Exception occurred:", e)   
 
 



from decouple import config


client = OpenAI(api_key=config("OPENAI_API_KEY"))


def get_comment(prompt):
    message={"role": "system",
            "content": prompt}
    # client = OpenAI(api_key =os.getenv("OPENAI_API_KEY"))
    chat_completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages = [message],
            temperature=0.2
        )
        # print("|\n|\n|\n|\n|\n|\n|\n|\n|v",chat_completion)
    finish_reason = chat_completion.choices[0].finish_reason
 
 
    newdata = chat_completion.choices[0].message.content
    # print("new dta in text mining is :",newdata)
    # double_digit = re.findall(r'\b\d{2}\b', newdata)
    # newdata = int(double_digit[0])
    
    return newdata     

def get_score_transcipt_file(prompt):
    message={"role": "system",
            "content": prompt}
    # client = OpenAI(api_key =os.getenv("OPENAI_API_KEY"))
    chat_completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages = [message],
            temperature=0.2
        )
    text = chat_completion.choices[0].message.content
    score_match = re.search(r'\bscore\s*[\w]*\s*([+-]?\d*\.\d+|\d+)', text)
    
    if score_match:
        score = float(score_match.group(1))  # Extract the number as a float
        rounded_score = round(score, 2)      # Round to 2 decimal places
        print("Extracted and rounded score:", rounded_score)
        return rounded_score
    else:
        print("No score found in the text.")
        return 0

def cal_uni_bi(audio_file):
    chunks = split_audio(audio_file)
    chunks.sort()
    
    model = whisper.load_model("large-v3", download_root=os.path.join(os.getcwd(), "whisper"))
    final_transcription , pace_score , articulation_score ,pace_comment , articulation_comment = transcribe_chunk_batch_new(chunks,model)
 
    torch.cuda.empty_cache()
    
 
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
 
 
    message_score = {
        "role": "system",
        "content": f""""calculate the grammar score for the text: "{final_transcription}", out of 100 consider vocabulary, grammar mistakes, sentence formation only give response in digit and no text, give minimum 10 score."""
 
    }
    message_comment = {
        "role": "system",
        "content": f""""comment on the grammar issues for the text: "{final_transcription}", consider vocabulary, grammar mistakes, sentence formation. ALSO SUGGEST IMPROVEMENTS WITH POWER PHRASES AND VOCABULARY.comment only in one line."""
 
    }
 
    try:
        chat_completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[message_score]
        )
        newdata = chat_completion.choices[0].message.content
        print(newdata)
        if int(newdata) < 29 : 
            newdata = 30
        double_digit = re.findall(r'\b\d{2}\b', newdata)
        grammer_score = int(double_digit[0])
        chat_completion_comment = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[message_comment]
        )
        newdata = chat_completion_comment.choices[0].message.content
        grammer_comment = newdata
        grammer_comment=grammer_comment.replace('\"',"")
 
 
        print("Grammar Score:", grammer_score)
        print("Grammar Comment:", grammer_comment)
 
         
        pronunciation_score = articulation_score
        pronunciation_comment = articulation_comment


        print("Pronunciation Comment:", pronunciation_comment)
        print("Pronunciation Score:", pronunciation_score)

    except Exception as e:
        print("Error while evaluating pronunciation:", e)
        grammer_Score = 30 
        grammer_comment = "The grammer is not good and needs improvement"   
        pronunciation_score = 30
        pronunciation_comment = "The pronunciation is not good and needs improvement"
        
 
    example_json_structure = """{
    "sentiment_score":<<score>>,
    "sentiment_comment": <<comment>>,    
    }"""
 
    sentiment_analysis_message = {
        "role": "system",
       "content": f"""Calculate the sentiment score and provide a comment for the text "{final_transcription}" out of 100. Consider factors such as the emotional tone, positivity/negativity, overall sentiment, and mood conveyed in the text. Provide the response in the following JSON format:
        {example_json_structure}
        """
    }
 
    try :
        sentiment_analysis_response = client.chat.completions.create(model="gpt-3.5-turbo", messages=[sentiment_analysis_message])
        sentiment_choice = sentiment_analysis_response.choices[0].message.content
        sentiment_choice_data = json.loads(sentiment_choice)
        sentiment_score_value = sentiment_choice_data["sentiment_score"]
        sentiment_comment_value = sentiment_choice_data["sentiment_comment"]
 
    except Exception as error:
        print(f"Error occurred: {error}")
        sentiment_score_value, sentiment_comment_value = None, None
    gc.collect()
    
    tokens = word_tokenize(final_transcription)
    stop_words = set(stopwords.words('english'))
    tokenized_corpus = [word for word in tokens if word.lower() not in stop_words]
    unigrams = list(ngrams(tokenized_corpus, 1))
    
    unigram_freq = Counter(unigrams)
 
    df_unigrams = pd.DataFrame(list(unigram_freq.items()), columns=['Word', 'Frequency'])
    df_unigrams = df_unigrams.sort_values(by='Frequency', ascending=False).reset_index(drop=True)
    torch.cuda.empty_cache()
 
    return df_unigrams ,sentiment_score_value , sentiment_comment_value , final_transcription, grammer_score , grammer_comment , pace_score ,pace_comment ,pronunciation_score , pronunciation_comment







# if __name__ == "__main__":
#     audio_file1 = "/disk/new_AVIPA/60a54673aeec23e42964965c8dbb92672e6e717a217b67b71b03b97398df8e56.mp3"

#     audio_file = "/disk/new_AVIPA/f088d879927623ebdba85b3fa13f8bb2e8ea850d23275f4da8d517202e767622.mp3"


#     audio_file3 = "/disk/Examples of good speaking pace during a presentation.mp3"
#     cal_uni_bi(audio_file)
