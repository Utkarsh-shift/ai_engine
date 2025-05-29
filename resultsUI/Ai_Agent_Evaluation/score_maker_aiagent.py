import numpy as np
from openai import OpenAI
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from resultsUI.Ai_Agent_Evaluation.dictionary_used_in_agent import *
from decouple import config
client = OpenAI(api_key=config("OPENAI_API_KEY"))
interview_words_set = INTERVIEW_WORD_SET


Power_phrases_set = POWER_PHRASES_SET


cognitive_words_set = COGNITIVE_WORDS_SET

social_words_set = SOCIAL_WORDS_SET
i_words_set = I_WORDS_SET

def ComputingValues(data):
 
    print(data)
    df_unigram = data['df_unigrams']
    try:
        top_two = df_unigram
    
    except:
        top_two = df_unigram[:2]  
  
    language_score = 0
    cognitive_score = 0
    social_score = 0
    i_score = 0
    power_score=0
    
    for i in top_two['Word']:
        j = i[0].lower()
        if j in interview_words_set:
            language_score += 0.384615385
        if j in cognitive_words_set:
            cognitive_score += 0.25
        if j in social_words_set:
            social_score += 0.1
        if j in i_words_set:
            i_score += 0.25
        if j in Power_phrases_set:
            power_score += 0.35
 
    if i_score>1:
        i_score=1
    if social_score>1:
        social_score=1
    if language_score>1:
        language_score=1
    if cognitive_score>1:
        cognitive_score=1
    if power_score>1:
        power_score=1
        
    language_score = language_score*100
    cognitive_score = cognitive_score*100
    social_score = social_score*100
    i_score = i_score*100
    power_score = power_score*100
    final_score = 0.5* language_score + 0.1 *i_score + 0.1*social_score + 0.1*cognitive_score + 0.1*power_score 
    
    return final_score*10
 
 
def articute_score_maker(power_sen, data, transcription):
      
    df_unigram = data
    try:
        top_two = df_unigram
    except:
        top_two = df_unigram[:2]  
    print(top_two)
    language_score=0
    language_count = 0
    cognitive_count = 0
    social_count = 0
    i_count = 0
    power_count=0
    
    for i in top_two['Word']:
        j = i[0].lower() 
        if j in interview_words_set:
            language_count += 1
        if j in cognitive_words_set:
            cognitive_count += 1
        if j in social_words_set:
           social_count += 1
        if j in i_words_set:
           i_count += 1
        if j in Power_phrases_set:
           power_count += 1
 
    power_count = power_count+power_sen
    if power_count == 0:
        score = 40
    elif power_count == 1:
        score = 60
    elif 2 <= power_count <= 4:
        score = 80
    elif power_count >= 5:
        score = 100

    
    message_comment = {
        "role": "system",
        "content": f""""comment on the articulate, vocabulary, grammer issues for the text: "{transcription}", consider vocabulary, grammar mistakes, sentence formation. ALSO SUGGEST IMPROVEMENTS WITH POWER PHRASES AND VOCABULARY FOR INTERVIEW SETTING"""

    }
    chat_completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[message_comment]
    )
    newdata = chat_completion.choices[0].message.content
    return score , newdata
 
 