from IPython.display import display, Image, Audio

import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
import time
from openai import OpenAI
import os,re
import requests
from decouple import config


 
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

def get_comments_for_gpt(base64Frames,prompt,transcript):  
    
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                prompt + f"And the transcript is given as {transcript}",          # prototype only try and see 
                *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::15]),
            ],
        },
    ]
    params = {
        "model": "gpt-4o",
        "messages": PROMPT_MESSAGES,
        "max_tokens": 200,
        "temperature": 0.2,
    }

    result = client.chat.completions.create(**params)
    print(result.choices[0].message.content)
    tokens_used = result.usage
    # print(f"Tokens used: {tokens_used}")
    return result.choices[0].message.content

def finalcomment(final_commentList):
    commenttext = " ".join(final_commentList)
    
    message={"role": "system",
            "content": "Summarize the text  in 2-3 lines and keep all details."+f"The Given text is : '{commenttext}'"}
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
    double_digit = re.findall(r'\b\d{2}\b', newdata)
    newdata = int(double_digit[0])
    print(newdata)
    return newdata

# if __name__ == "__main__":

#    ff=finalcomment(["The candidate appears to be in a professional setting, likely an office environment, which suggests a level of professionalism. Their attire is casual but appropriate for a video interview. The candidate maintains focus and seems engaged, indicating attentiveness. The lighting and background are suitable, contributing to a professional presentation. Overall, the candidate demonstrates professionalism through their setting and demeanor.The candidate appears to be in a professional setting, likely an office environment, suggesting a level of professionalism. Their attire is casual but appropriate for a video interview. The candidate maintains focus and seems engaged, indicating attentiveness. The lighting and background are suitable, contributing to a professional presentation. Overall, the candidate demonstrates professionalism through their setting and demeanor."])
#    print(ff)