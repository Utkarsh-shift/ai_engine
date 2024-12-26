

import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
import time
from openai import OpenAI
import os,re
import requests
import tiktoken
from decouple import config

MAX_TOKENS = 10000
OUTPUT_TOKENS = 256

def count_tokens(text):
    encoder = tiktoken.get_encoding("cl100k_base")  
    tokens = encoder.encode(text)
    return len(tokens)



def truncate_text_to_fit(text, max_input_tokens):
    """Truncate the input text to fit within the available token limit."""
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
    # Base64 encoded images typically consume around 50 tokens per image (this is an estimate)
    return len(base64Frames) * 50


def get_comments_for_gpt(base64Frames,prompt,transcript):
    base64Frames = base64Frames[0::15]
    prompt_tokens = count_tokens(prompt  + f"And the transcript is given as {transcript}")

    image_token = estimate_image_token_count(base64Frames)

    total_input_tokens = prompt_tokens + image_token

    available_tokens = MAX_TOKENS - OUTPUT_TOKENS
    
    if total_input_tokens > available_tokens:
        # If input is too large, reduce the input size dynamically
        max_input_tokens = available_tokens - OUTPUT_TOKENS
        prompt = truncate_text_to_fit(prompt , max_input_tokens)  # Truncate the prompt if it's too large
        base64Frames = base64Frames[:max_input_tokens // 50]  # Reduce the number of images accordingly


    PROMPT_MESSAGES = [
   
        {   
            "role": "user",
            "content": [
                prompt + f"And the transcript is given as {transcript}",          # prototype only try and see 
                *map(lambda x: {"image": x, "resize": 768}, base64Frames),
            ],
        },
    ]
    params = {
        "model": "gpt-4o",
        "messages": PROMPT_MESSAGES,
        "max_tokens": 1000,
        "temperature": 0.2,
    }

    result = client.chat.completions.create(**params)
    print(result.choices[0].message.content)
    tokens_used = result.usage
    # print(f"Tokens used: {tokens_used}")
    return result.choices[0].message.content

def finalcomment(final_commentList):
    commenttext = " ".join(final_commentList)


    input_tokens = count_tokens(commenttext)

    if input_tokens + OUTPUT_TOKENS > MAX_TOKENS:
        max_input_token = MAX_TOKENS - OUTPUT_TOKENS
        commenttext = truncate_text_to_fit(commenttext , max_input_token)
    
    message={"role": "system",
            "content": "Summarize the text lines and keep all details."+f"The Given text is : '{commenttext}'"}
    # client = OpenAI(api_key =os.getenv("OPENAI_API_KEY"))
    chat_completion = client.chat.completions.create(
            model="gpt-4o",
            messages = [message]
            ,max_tokens=256
        )
        # print("|\n|\n|\n|\n|\n|\n|\n|\n|v",chat_completion)
    # finish_reason = chat_completion.choices[0].finish_reason
 
    
    final_comment = chat_completion.choices[0].message.content
    
    return final_comment

def getcomment_communication(pace_comment,articulation_comment,energy_comment):
       
    message={"role": "system",
            "content": f"The communication score is computed with pace , aticutaion and energy of the performance of person in these field is stated in english as {pace_comment} , {articulation_comment} , {energy_comment}, what do you think how the person is behaving ?"}
    # client = OpenAI(api_key =os.getenv("OPENAI_API_KEY"))
    chat_completion = client.chat.completions.create(
            model="gpt-4o",
            messages = [message]
            ,max_tokens=256
        )
        # print("|\n|\n|\n|\n|\n|\n|\n|\n|v",chat_completion)
    # finish_reason = chat_completion.choices[0].finish_reason
 
    
    final_comment = chat_completion.choices[0].message.content
    
    return final_comment


def getcomment_sociability(energy_score,sentiment_score,emotion_score):
       
    message={"role": "system",
            "content": f"The sociability score is computed with energy , sentiment and emotion of the performance of person in these field is stated in english as {energy_score} , {sentiment_score} , {emotion_score}, what do you think how the person is behaving ? ,Explain in about 2-3 lines"}
    # client = OpenAI(api_key =os.getenv("OPENAI_API_KEY"))
    chat_completion = client.chat.completions.create(
            model="gpt-4o",
            messages = [message]
            ,max_tokens=256
        )
        # print("|\n|\n|\n|\n|\n|\n|\n|\n|v",chat_completion)
    # finish_reason = chat_completion.choices[0].finish_reason
 
    
    final_comment = chat_completion.choices[0].message.content
    
    return final_comment

def getcomment_positive_attitude(energy_score):
       
    message={"role": "system",
            "content": f"The positive attitude score is computed with energy ,the performance of person in this field is stated in english as {energy_score} , what do you think how the person is behaving ?"}
    # client = OpenAI(api_key =os.getenv("OPENAI_API_KEY"))
    chat_completion = client.chat.completions.create(
            model="gpt-4o",
            messages = [message]
            ,max_tokens=256
        )
        # print("|\n|\n|\n|\n|\n|\n|\n|\n|v",chat_completion)
    # finish_reason = chat_completion.choices[0].finish_reason
 
    
    final_comment = chat_completion.choices[0].message.content
    
    return final_comment

def get_score(prompt,text):
    
    
    
    message={"role": "system",
            "content": prompt+f"""{text}, out of 100 only give response in digit and no text, give minimum 10 score."""}
    # client = OpenAI(api_key =os.getenv("OPENAI_API_KEY"))
    chat_completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages = [message],
            temperature=0.2
        )
        # print("|\n|\n|\n|\n|\n|\n|\n|\n|v",chat_completion)
    finish_reason = chat_completion.choices[0].finish_reason
 
 
    newdata = chat_completion.choices[0].message.content
    try :
        double_digit = re.findall(r'\b\d{2}\b', newdata)
        newdata = int(double_digit[0])
    except : 
        newdata = 10 
    print("The score is given as ",newdata )
    return newdata

# if __name__ == "__main__":

#    ff=finalcomment(["The candidate appears to be in a professional setting, likely an office environment, which suggests a level of professionalism. Their attire is casual but appropriate for a video interview. The candidate maintains focus and seems engaged, indicating attentiveness. The lighting and background are suitable, contributing to a professional presentation. Overall, the candidate demonstrates professionalism through their setting and demeanor.The candidate appears to be in a professional setting, likely an office environment, suggesting a level of professionalism. Their attire is casual but appropriate for a video interview. The candidate maintains focus and seems engaged, indicating attentiveness. The lighting and background are suitable, contributing to a professional presentation. Overall, the candidate demonstrates professionalism through their setting and demeanor."])
#    print(ff)