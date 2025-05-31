

import cv2  
import base64
import time
from openai import OpenAI
import os,re
import requests
import tiktoken
from decouple import config
from dpcv.experiment.dictionary_used import INTERVIEW_GOALS_LIST
import traceback

MAX_TOKENS = 10000
OUTPUT_TOKENS = 300
goal_text = ", ".join(INTERVIEW_GOALS_LIST) 
def extract_score_from_text(text):
    text = text.lower().replace("*" ,"")
    score_pattern = re.compile(r'\b(?:score|grade|rating|evaluation|assessment)\s*[:=\-]?\s*(\d{1,3})\b', re.IGNORECASE)
    match = score_pattern.search(text)
    if match:
        score = int(match.group(1))
        return min(max(score, 0), 100) 
    return None


def count_tokens(text):
    encoder = tiktoken.get_encoding("cl100k_base")  
    tokens = encoder.encode(text)
    return len(tokens)

def truncate_text_to_fit(text, max_input_tokens):
    while count_tokens(text) + OUTPUT_TOKENS > max_input_tokens:
        text = " ".join(text.split()[:-5])  
    return text
 
def convert_images_to_base64(image_folder_path):
    image_folder=os.listdir(image_folder_path)
    image_folder.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    base64Frames = []
    for filename in image_folder:
        file_path = os.path.join(image_folder_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = cv2.imread(file_path)
            _, buffer = cv2.imencode('.jpg', image)
            base64_str = base64.b64encode(buffer).decode('utf-8')
            base64Frames.append(base64_str)
    return base64Frames



client = OpenAI(api_key=config("OPENAI_API_KEY"))




def estimate_image_token_count(base64Frames):
    return len(base64Frames) * 50





def analyze_body_language_with_vision(base64Frames , typeo):
    messages = [
        {   
            "role": "system",
            "content": f"You are an Interview assessment bot. Your job is to assess the candidate's visual behavior on the aspect: {typeo}. "
                        f"Provide a detailed and comprehensive analysis, and justify your comments with specific feedback on {typeo}. Scrutinize every aspect of {typeo} "
                        f"Be professional and objective in your assessment."
                        f"Additionally, provide a numerical score between 0 to 100 based on the quality of the candidate's {typeo}. Stick to the convention of GIVING THE SCORE AS 'SCORE : ' "
                        f"Do not mention anything about frames."
                        "Do NOT mention overall or conclusion statement."
                        "ALWAYS MENTION SCORE"
                        "MAKE SURE THE EVALUATION IS UNDER 150 WORDS.\n"
                        "Do Not write greeting message such as Certainly!, etc"
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Analyze the candidate's {typeo} based on the Frames of video."},
                *map(lambda x: {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{x}", "detail": "high"}}, base64Frames),
            ],
        }
    ]
 
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=600,
        temperature = 0.4
    )
 
    body_language_analysis = response.choices[0].message.content
    score = extract_score_from_text(body_language_analysis)
 
    print(f"The analysis for the part {typeo}" ,body_language_analysis )
    print("SCORE is", score)
 
 
    return score, body_language_analysis



def get_comments_for_gpt(base64Frames,prompt,transcript,typeo):
    base64Frames = base64Frames[0::15]
    prompt_tokens = count_tokens(prompt  + f"And the transcript is given as {transcript}")
    image_token = estimate_image_token_count(base64Frames)
    total_input_tokens = prompt_tokens + image_token
    available_tokens = MAX_TOKENS - OUTPUT_TOKENS
    if total_input_tokens > available_tokens:
        max_input_tokens = available_tokens - OUTPUT_TOKENS
        prompt = truncate_text_to_fit(prompt , max_input_tokens)  
        base64Frames = base64Frames[:max_input_tokens // 50]  
    score , body_language_summary = analyze_body_language_with_vision(base64Frames , typeo)
    print("Score and body summary is " , score , body_language_summary)
    return score , body_language_summary



def finalcomment(final_commentList , small = False):
    
    if small : 
        numbered_comments = [f"Q{i+1}:{comment}" for i, comment in enumerate(final_commentList)]
        compiled_comment = "|".join(numbered_comments)
        input_tokens = count_tokens(compiled_comment)
        if input_tokens + OUTPUT_TOKENS > MAX_TOKENS:
            max_input_tokens = MAX_TOKENS - OUTPUT_TOKENS
            compiled_comment = truncate_text_to_fit(compiled_comment, max_input_tokens)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI Interviewer giving a report of assessments. The assessments are of a single candidate for multiple questions. "
                    "Compile all individual assessments into one structured, professional summary while keeping important insights. "
                    "Do not mention question numbers. "
                    "Make sure to retain the numbered format for clarity.But NO POINTS\n\n"
                    "Do Not give scores only comment"
                    "Do Not write greeting message such as Certainly!, etc"
                    "GIVE ONE LINE COMMENT ONLY "
                    "MAKE SURE THE EVALUATION IS UNDER 150 WORDS.\n"
                )
            },
            {
                "role": "user",
                "content": (
                    "Analyze the individual assessments of all the questions and write a report of the analysis. "
                    f"Here is the structured analysis:\n\n{compiled_comment}"
                )
            }
        ]
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens = 128,
            temperature = 0.4
        )
        final_comment = response.choices[0].message.content
        comment= final_comment.replace("*" , "")
        return comment
    
    
    else : 
        
        
        numbered_comments = [f"Q{i+1}:{comment}" for i, comment in enumerate(final_commentList)]
        compiled_comment = "|".join(numbered_comments)
        input_tokens = count_tokens(compiled_comment)
        if input_tokens + OUTPUT_TOKENS > MAX_TOKENS:
            max_input_tokens = MAX_TOKENS - OUTPUT_TOKENS
            compiled_comment = truncate_text_to_fit(compiled_comment, max_input_tokens)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI Interviewer giving a report of assessments. The assessments are of a single candidate for multiple questions. "
                    "Compile all individual assessments into one structured, professional summary while keeping important insights. "
                    "Do not mention question numbers. "
                    "Make sure to retain the numbered format for clarity.But NO POINTS\n\n"
                    "Do Not give scores only comment"
                    "Do Not write greeting message such as Certainly!, etc"
                    "MAKE SURE THE EVALUATION IS UNDER 150 WORDS.\n"
                )
            },
            {
                "role": "user",
                "content": (
                    "Analyze the individual assessments of all the questions and write a report of the analysis. "
                    f"Here is the structured analysis:\n\n{compiled_comment}"
                )
            }
        ]
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens = 300,
            temperature = 0.4
        )
        final_comment = response.choices[0].message.content
        comment= final_comment.replace("*" , "")
        return comment
 
def finalcomment_forselfawreness(final_commentList, small=False):
    print("=======================================================")
    if all(comment == "No Comment" for comment in final_commentList):
        return "No Question was asked for Self Awareness"
    final_commentList = [comment for comment in final_commentList if comment != "No Comment"]
    
    if small: 
        numbered_comments = [f"Q{i+1}:{comment}" for i, comment in enumerate(final_commentList)]
        compiled_comment = "|".join(numbered_comments)
        input_tokens = count_tokens(compiled_comment)
        if input_tokens + OUTPUT_TOKENS > MAX_TOKENS:
            max_input_tokens = MAX_TOKENS - OUTPUT_TOKENS
            compiled_comment = truncate_text_to_fit(compiled_comment, max_input_tokens)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI Interviewer giving a report of assessments. The assessments are of a single candidate for multiple questions. "
                    "Compile all individual assessments into one structured, professional summary while keeping important insights. "
                    "Do not mention question numbers. "
                    "Make sure to retain the numbered format for clarity. But NO POINTS\n\n"
                    "Do Not give scores only comment. "
                    "Do Not write greeting message such as Certainly!, etc."
                    "GIVE ONE LINE COMMENT ONLY."
                )
            },
            {
                "role": "user",
                "content": (
                    "Analyze the individual assessments of all the questions and write a report of the analysis. "
                    f"Here is the structured analysis:\n\n{compiled_comment}"
                )
            }
        ]
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=64,
            temperature = 0.4
        )
        final_comment = response.choices[0].message.content
        comment = final_comment.replace("*", "")
        return comment
    
    else: 
        numbered_comments = [f"Q{i+1}:{comment}" for i, comment in enumerate(final_commentList)]
        compiled_comment = "|".join(numbered_comments)
        input_tokens = count_tokens(compiled_comment)
        if input_tokens + OUTPUT_TOKENS > MAX_TOKENS:
            max_input_tokens = MAX_TOKENS - OUTPUT_TOKENS
            compiled_comment = truncate_text_to_fit(compiled_comment, max_input_tokens)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI Interviewer giving a report of assessments. The assessments are of a single candidate for multiple questions. "
                    "Compile all individual assessments into one structured, professional summary while keeping important insights. "
                    "Do not mention question numbers. "
                    "Make sure to retain the numbered format for clarity. But NO POINTS\n\n"
                    "Do Not give scores only comment. "
                    "Do Not write greeting message such as Certainly!, etc."
                    "MAKE SURE THE EVALUATION IS UNDER 150 WORDS.\n"
                )
            },
            {
                "role": "user",
                "content": (
                    "Analyze the individual assessments of all the questions and write a report of the analysis. "
                    f"Here is the structured analysis:\n\n{compiled_comment}"
                )
            }
        ]
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=400,
            temperature = 0.4
        )
        final_comment = response.choices[0].message.content
        comment = final_comment.replace("*", "")
        return comment


def getcomment_communication(pace_comment,articulation_comment,grammar_comment):
    comm_1 = pace_comment
    comm_2 =    articulation_comment
    comm_3 = grammar_comment
    messages=[
    {   "role": "system",
        "content": "The communication is computed with the performance of candidate in pace, aticutaion and grammer "
                
                "What do you think how the person is in communication  ?"
                "Do Not write greeting message such as Certainly!, etc"
        } ,
        
    {
        "role": "user",
        "content": "Write about the communication skills of the candidate  These analysis of each aspect is given as " + comm_1 +  comm_2 + comm_3 + "explicitly write about grammar comment  DO NOT MISS ANY DETAILS  "
    }
        
        ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=400,
        temperature = 0.4
    )
    final_comment = response.choices[0].message.content
    comment= final_comment.replace("*" , "")
    return comment
 
def getcomment_sociability(energy_comment,sentiment_comment,emotion_comment):
   
    messages=[
        {   "role": "system",
            "content": "The sociability is computed with the performance of candidate in sentiment and emotion "
            "What do you think how the person is in sociability  ?"
            "Do Not write greeting message such as Certainly!, etc"
            "MAKE SURE THE EVALUATION IS UNDER 150 WORDS.\n"
            } ,
            
        {
            "role": "user",
            f"content": "Write about on the sociability skills of the candidate ,  These analysis of each aspect is given as {energy_comment} , {sentiment_comment} , {emotion_comment}"
        }
            
            ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=400,
        temperature = 0.4
    )
    final_comment = response.choices[0].message.content
    comment= final_comment.replace("*" , "")
    return comment
 
 
 
 
def getcomment_etiquette(base64Frames,prompt,transcript,typeo):
  
    base64Frames = base64Frames[0::15]
    prompt_tokens = count_tokens(prompt  + f"And the transcript is given as {transcript}")
    image_token = estimate_image_token_count(base64Frames)
    total_input_tokens = prompt_tokens + image_token
    available_tokens = MAX_TOKENS - OUTPUT_TOKENS
    if total_input_tokens > available_tokens:
        max_input_tokens = available_tokens - OUTPUT_TOKENS
        prompt = truncate_text_to_fit(prompt , max_input_tokens)  
        base64Frames = base64Frames[:max_input_tokens // 50]  
 
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI Interviewer assessing candidate etiquette based on visual behavior. "
                "Etiquette includes posture, grooming, dress, attentiveness, facial expressions, and professional demeanor. "
                "You must observe these frames and write a concise evaluation focused on their etiquette. "
                "Avoid mentioning the word 'image' or 'frames'. "
                "Do NOT include greetings, conclusions, or suggestions. "
                "Always provide a SCORE as 'Score: <number>' between 0 and 100. "
                "Be strictly professional in tone."
            )
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Evaluate the candidate's etiquette based on these visuals."},
                *map(lambda x: {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{x}", "detail": "high"}}, base64Frames),
            ],
        }
    ]
 
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=400,
        temperature = 0.4
    )
 
    content = response.choices[0].message.content
    etiquette_score = extract_score_from_text(content)
 
    print("\n Etiquette Analysis:\n", content)
    print(" Etiquette Score:", etiquette_score)
 
    return etiquette_score, content
 
 
 
def getcomment_positive_attitude(energy_comment):
    messages=[
            {   "role": "system",
                "content": "The positive attitide is computed with the performance of candidate in energy "
                "What do you think how the person is in positive attitide ?"
                "Do Not write greeting message such as Certainly!, etc"
                "MAKE SURE THE EVALUATION IS UNDER 150 WORDS.\n"
                } ,
                
            {
                "role": "user",
                f"content": "Write about on the  positive attitide skills of the candidate ,  These analysis of {energy_comment}"
            }
                
                ]
 
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=400,
        temperature = 0.4
    )
    final_comment = response.choices[0].message.content
    comment= final_comment.replace("*" , "")
    return comment


 
def evaluate_self_awareness(question, transcripts):
    total_score = 0
    question.replace("$#@True$#@","")
    print("The question is <>>>>>>>><>>>>>>>>>>>>>>>>>>>>>>>>" , question)
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI Interview Assistant specialized in evaluating self-awareness in interview responses.\n"
                "In this task, you will be given an interview question and its corresponding answer (transcript).\n"
                "Your job is to evaluate the candidate's self-awareness in the answer. Self-awareness refers to the candidate's ability to recognize and reflect upon their own thoughts, actions, and emotions. "
                "This includes showing understanding of their strengths, weaknesses, personal experiences, and behaviors.\n\n"
                "For each answer, evaluate the following aspects:\n"
                "1. Self-reflection: Does the candidate reflect on their personal experiences or actions?\n"
                "2. Emotional insight: Does the candidate show an understanding of their own feelings, motivations, and behaviors?\n"
                "3. Awareness of strengths and weaknesses: Does the candidate acknowledge areas of growth or personal strengths?\n"
                "4. Accountability: Does the candidate take responsibility for their actions and outcomes?\n"
                "5. Growth mindset: Does the candidate express a willingness to learn or improve based on past experiences?\n\n"
                f"Also, check if the candidate articulates clear personal or professional goals from this list: {goal_text}.\n"
                "Look for specific examples related to strengths, purpose, health, personal needs, or relationships.\n\n"
                f"ALSO IF TRANCRIPT HAS REALTION WITH ANY OF THE GOALS LIST {goal_text} THEN GIVE HIGHER SCORE. MORE THE RELATION MEANS MORE THE SCORE \n"
                "THE SCORE VALUE SHOULD OUT OF 100, GIVE MINIMUM OF 30"
                "Format your output as:\n"
                "\"score\": [score_value],\n"
                "\"self_awareness\": [evaluation]\n\n"
            )
        },
        {
            "role": "user",
            "content": f" Now, provide an evaluation based on the following question and answer:\n Question: {question}\nAnswer: {transcripts}"
        }
    ]

    try:
     
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=400,
            temperature = 0.4
        )

        evaluation = response.choices[0].message.content.strip()
        print("The evaluation response is given as for self awarness ********* " ,evaluation )
        score_start = evaluation.find("\"score\":") + len("\"score\":")
        score_end = evaluation.find(",", score_start)
        score = int(evaluation[score_start:score_end].strip())
        scaled_score = score
        if score < 30 : 
            score  = 30 
        self_awareness_start = evaluation.find("\"self_awareness\":") + len("\"self_awareness\":")
        comment = evaluation[self_awareness_start:].strip().strip('"')
        total_score = scaled_score
      
       

    except Exception as e:
        print(f"Exception arrives at {str(e)}")
        messages = [
        {
            "role": "system",
            "content": (
                "You are an AI Interview Assistant specialized in evaluating self-awareness in interview responses.\n"
                "In this task, you will be given an interview question and its corresponding answer (transcript).\n"
                "Your job is to evaluate the candidate's self-awareness in the answer. Self-awareness refers to the candidate's ability to recognize and reflect upon their own thoughts, actions, and emotions. "
                "This includes showing understanding of their strengths, weaknesses, personal experiences, and behaviors.\n\n"
                "For each answer, evaluate the following aspects:\n"
                "1. Self-reflection: Does the candidate reflect on their personal experiences or actions?\n"
                "2. Emotional insight: Does the candidate show an understanding of their own feelings, motivations, and behaviors?\n"
                "3. Awareness of strengths and weaknesses: Does the candidate acknowledge areas of growth or personal strengths?\n"
                "4. Accountability: Does the candidate take responsibility for their actions and outcomes?\n"
                "5. Growth mindset: Does the candidate express a willingness to learn or improve based on past experiences?\n\n"
                f"Also, check if the candidate articulates clear personal or professional goals from this list: {goal_text}.\n"
                "Look for specific examples related to strengths, purpose, health, personal needs, or relationships.\n\n"
                f"ALSO IF TRANCRIPT HAS REALTION WITH ANY OF THE GOALS LIST {goal_text} THEN GIVE HIGHER SCORE. MORE THE RELATION MEANS MORE THE SCORE \n"
                "THE SCORE VALUE SHOULD OUT OF 100, GIVE MINIMUM OF 30"
                "DO NOT GIVE ANY EMOJI"
                "GIVE THE ANSWER AND SCORE IN PLAIN ENGLISH AND NUMBER ONLY "
                "STRICTLY FOLLOW THE FORMAT PROVIDED"
                
                "Format your output as:\n"
                "\"score\": [score_value],\n"
                "\"self_awareness\": [evaluation]\n\n"
            )
        },
        {
            "role": "user",
            "content": f" Now, provide an evaluation based on the following question and answer:\n Question: {question}\nAnswer: {transcripts}"
        }
            ]

        try:
        
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=400,
                temperature = 0.4
            )

            evaluation = response.choices[0].message.content.strip()
            print("The evaluation response is given as for self awarness ********* " ,evaluation )
            score_start = evaluation.find("\"score\":") + len("\"score\":")
            score_end = evaluation.find(",", score_start)
            score = int(evaluation[score_start:score_end].strip())
            scaled_score = score
            if score < 30 : 
                score  = 30 
            self_awareness_start = evaluation.find("\"self_awareness\":") + len("\"self_awareness\":")
            comment = evaluation[self_awareness_start:].strip().strip('"')
            total_score = scaled_score
        
       

        except Exception as e:
            print(f"Exception arrives at {str(e)}")
            total_score = None
            comment = None

    return {
        "score": total_score,
        "self_awareness": comment
    }