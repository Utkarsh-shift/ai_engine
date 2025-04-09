

import cv2  
import base64
import time
from openai import OpenAI
import os,re
import requests
import tiktoken
from decouple import config
MAX_TOKENS = 10000
OUTPUT_TOKENS = 300
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
    return score , body_language_summary


# def get_score_if(comment) : 
#     print("????????????????????????????????????????????",comment,"~~~~~~~~~~~~~~~~~~~~~~~~")
#     messages = [
#         {
#             "role": "system",
#             "content": (
#                 "You are an AI Interviewer giving a score of assessments. "
#                 "Compile all individual assessments into one structured, professional summary while keeping important insights. "
#                 "provide a numerical score between 0 to 100 based on the quality of the candidate's by comment given."
#                 "Do not give justification reson for the score only give me the two digit score"
#                 "Do Not give scores only comment"
#             )
#         },
#         {
#             "role": "user",
#             "content": (
#                 f"Analyze the {comment}  and give a score in two ditis"
#             )
#         }
#     ]
#     response = client.chat.completions.create(
#         model="gpt-4o",
#         messages=messages,
#         max_tokens=256,
#     )
#     final_comment = response.choices[0].message.content
#     score = final_comment.replace("*" , "")
#     return score

def finalcomment(final_commentList):
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
        max_tokens=300,
    )
    final_comment = response.choices[0].message.content
    comment= final_comment.replace("*" , "")
    return comment


def getcomment_communication(pace_comment,articulation_comment,energy_comment):
       
    messages=[
                    {   "role": "system",
                        "content": "The communication is computed with the performance of candidate in pace, aticutaion and energy "
                               
                                "What do you think how the person is in communication  ?"
                                "Do Not write greeting message such as Certainly!, etc"
                       } ,
                       
                    {
                        "role": "user",
                        f"content": "Write about the communication skills of the candidate  These analysis of each aspect is given as {pace_comment} , {articulation_comment} , {energy_comment}"
                    }
                       
                        ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=300,
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
                       } ,
                       
                    {
                        "role": "user",
                        f"content": "Write about on the sociability skills of the candidate ,  These analysis of each aspect is given as {energy_comment} , {sentiment_comment} , {emotion_comment}"
                    }
                       
                        ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=300,
    )
    final_comment = response.choices[0].message.content
    comment= final_comment.replace("*" , "")
    return comment

 
def getcomment_positive_attitude(energy_comment):
    messages=[
                    {   "role": "system",
                        "content": "The positive attitide is computed with the performance of candidate in energy "
                        "What do you think how the person is in positive attitide ?"
                        "Do Not write greeting message such as Certainly!, etc"
                       } ,
                       
                    {
                        "role": "user",
                        f"content": "Write about on the  positive attitide skills of the candidate ,  These analysis of {energy_comment}"
                    }
                       
                        ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=300,
    )
    final_comment = response.choices[0].message.content
    comment= final_comment.replace("*" , "")
    return comment
