import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter
import os
from pydub import AudioSegment
import whisper
import torch
import whisper
from pydub import AudioSegment
import os
import gc
from multiprocessing import Pool, set_start_method
from openai import OpenAI
import os ,json
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter
import pandas as pd
import torch
     

 
import re 
 
def hf(text):
    load_dotenv()
    client = OpenAI(api_key =config("OPENAI_API_KEY"))
    message={"role": "system",
            "content": f"""If the given text is not in English, translate it to English only. If the text is already in English, return the exact same text without any modification. The text to be processed is: '{text}'"""}
    client = OpenAI(api_key =os.getenv("OPENAI_API_KEY"))
    chat_completion = client.chat.completions.create(
            model="gpt-4o",
            messages = [message]
            ,max_tokens=256
        )
    
    
    newdata = chat_completion.choices[0].message.content
    print("newdta is ",newdata)
    return newdata
 
import whisper
import torch
import gc
import os
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

def calculate_avg_logprob(transcription_segments):
   
    avg_logprobs = [segment['avg_logprob'] for segment in transcription_segments]
    avg_logprob_mean = sum(avg_logprobs) / len(avg_logprobs)
    return avg_logprob_mean

def split_audio(audio_path, chunk_length_ms=15000):
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

def scale_wpm_to_score(wpm, min_wpm=50, max_wpm=250):
    """Scale WPM to a score between 0 and 100."""
    if wpm < min_wpm:
        return 0
    elif wpm > max_wpm:
        return 100
    else:
        return ((wpm - min_wpm) / (max_wpm - min_wpm)) * 100


def scale_avg_logprob_to_score(avg_logprob, min_logprob=-1.0, max_logprob=-0.1):
    """Scale avg_logprob to a score between 0 and 100."""
    if avg_logprob < min_logprob:
        return 0
    elif avg_logprob > max_logprob:
        return 100
    else:
        return ((avg_logprob - min_logprob) / (max_logprob - min_logprob)) * 100

def translate_text(text):
    message = [
    {"role": "system", "content": "You are a translation assistant. Translate all user-provided text into English. Respond ONLY with the translated text, without adding extra explanations or comments."},
    {"role": "user", "content": text }
    ]
    
    client = OpenAI(api_key =config("OPENAI_API_KEY"))
    chat_completion = client.chat.completions.create(
            model="gpt-4o",
            messages = message
            ,max_tokens=256
        )
    newdata = chat_completion.choices[0].message.content
    print(newdata)
    return newdata

def transcribe_chunk_batch_new(chunks, model):
    transcriptions = []
    total_duration = 0
    total_words=0
    avg_logprobs = []
     
    try:
        for i in chunks:
            audio = whisper.load_audio(i)
            audio = whisper.pad_or_trim(audio)
            transcription = model.transcribe(audio)
            final_transcription = transcription['text']
            words = final_transcription.split()
            transcriptions.append(final_transcription)
            num_words = len(words)
            total_words += num_words
            for segment in transcription['segments']:
                segment_duration = segment['end'] - segment['start']
                total_duration += segment_duration
                avg_logprobs.append(segment['avg_logprob'])
            
            os.remove(i)
        full_transcription = "".join(transcriptions)
        full_transcription=translate_text(full_transcription)
        duration_in_minutes = total_duration/60
        if duration_in_minutes > 0:
            wpm = total_words / duration_in_minutes
        else:
            wpm = 0

        print(f"Total Words: {total_words}, Duration: {total_duration} seconds")
        print(f"Words per minute (WPM): {wpm:.2f}")
        wpm_score = scale_wpm_to_score(wpm , min_wpm=50, max_wpm=130)
        print(f"wpm Score : {wpm_score}")
        pace_prompt =  f"""If the peron is speaking at a average word per minute rate of {wpm}, and is given a score of {wpm_score} , then comment about the pace of the person in speech """
        pace_comment = get_comment(pace_prompt)
        print("pace comment is :",pace_comment)
        avg_logprob_mean = sum(avg_logprobs) / len(avg_logprobs)
        print("The average_logprob is calculates as" , avg_logprob_mean)
        avg_logprob_score = scale_avg_logprob_to_score(avg_logprob_mean)
        print("The arrticualtion score is given as " , avg_logprob_score)
        articulation_prompt = f""" If the person is average log probability mean of articulation is {avg_logprob_mean} , and the score is {avg_logprob_score} , then comment about the articulation of the person in speech """

        articulation_prompt_comment = get_comment(articulation_prompt)
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
    print(newdata)
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
    print(chunks)
    model = whisper.load_model("large-v3", download_root=os.path.join(os.getcwd(), "whisper"))
    final_transcription , pace_score , articulation_score ,pace_comment , articulation_comment = transcribe_chunk_batch_new(chunks,model)

    torch.cuda.empty_cache()
    print("Final Transcription:", final_transcription)

    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    example_json = """{
        "grammer_score": <score>,
        "grammer_comment": <comment>            
    }"""

    message_score = {
        "role": "system",
        "content": f""""calculate the grammer score for the text: "{final_transcription}", out of 100 consider vocabulary, grammer mistakes, sentence formation only give response in digit and no text, give minimum 10 score."""

    }
    message_comment = {
        "role": "system",
        "content": f""""comment on the grammer issues for the text: "{final_transcription}", consider vocabulary, grammer mistakes, sentence formation."""

    }

    try:
        chat_completion = client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=[message_score]
        )
        newdata = chat_completion.choices[0].message.content
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

    except Exception as e:
        print(f"Error occurred: {e}")
        grammer_score, grammer_comment = None, None

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
        print("sentiment Score:", sentiment_score_value)
        print("sentiment Comment:", sentiment_comment_value)
    except Exception as error:
        print(f"Error occurred: {error}")
        sentiment_score_value, sentiment_comment_value = None, None
    gc.collect()

    nltk.download('punkt')
    nltk.download('stopwords')

    tokens = word_tokenize(final_transcription)

    stop_words = set(stopwords.words('english'))

    tokenized_corpus = [word for word in tokens if word.lower() not in stop_words]

    unigrams = list(ngrams(tokenized_corpus, 1))
    bigrams = list(ngrams(tokenized_corpus, 2))

    unigram_freq = Counter(unigrams)
    bigram_freq = Counter(bigrams)

    df_unigrams = pd.DataFrame(list(unigram_freq.items()), columns=['Word', 'Frequency'])
    

    df_unigrams = df_unigrams.sort_values(by='Frequency', ascending=False).reset_index(drop=True)

    # print(df_unigrams)
    


    torch.cuda.empty_cache()

 
    return df_unigrams ,sentiment_score_value , sentiment_comment_value , final_transcription, grammer_score , grammer_comment , pace_score , articulation_score,pace_comment , articulation_comment
 






# if __name__ == "__main__":
#     audio_file1 = "/disk/new_AVIPA/60a54673aeec23e42964965c8dbb92672e6e717a217b67b71b03b97398df8e56.mp3"

#     audio_file = "/disk/new_AVIPA/f088d879927623ebdba85b3fa13f8bb2e8ea850d23275f4da8d517202e767622.mp3"


#     audio_file3 = "/disk/Examples of good speaking pace during a presentation.mp3"
#     cal_uni_bi(audio_file)
