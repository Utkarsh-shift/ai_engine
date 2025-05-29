
from flask import Flask
import mysql.connector
import logging,json
import os
import openai
from Downlord_video_s3 import generate_presigned_url, download_file 
from decouple import config
from video_audio_evaluator import show_results
from extract_questions_answer import fetch_session_data,get_proctoring_metrics
logging.basicConfig(filename='evaluation_api.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import openai
import mysql.connector
import logging
import openai
import json


OPENAI_API_KEY = config("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)
 
host = config("HOST")
user = config("DB_USER")
password = config("DB_PASS")
database = config("DB_NAME")

app = Flask(__name__)
DB_CONFIG = {
    "host":host,
    "user": user,
    "password": password,
    "database": database
}


@shared_task
def start_evaluation(session_id,):
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    logging.info(f"Triggering evaluation for session: {session_id}")
    print("Pending session: ", session_id)
    query = f"SELECT Camera_uploads FROM interview_evaluations WHERE session_id = %s"
    cursor.execute(query, (session_id,))
    S3_fileurl = cursor.fetchall()
    print("==============================")
    print("S3_fileurl",S3_fileurl)
    if S3_fileurl and 'Camera_uploads' in S3_fileurl[0]:
        file_url = S3_fileurl[0]['Camera_uploads']
        print("S3_fileurl:", file_url)
    filename = "/".join(file_url.split("/")[3:])
    print(filename)
    bucket_name = config('bucket_name')
    secret_key = config('secret_key')
    access_key = config('access_key')
    object_key = filename
    url = generate_presigned_url(bucket_name, object_key, access_key=access_key, secret_key=secret_key)
    if url:
        print(f"Presigned URL: {url}")
        print(session_id,"++++++")
        path = os.getcwd()
        # path="/disk/AVI_PA-mainold/"
        print("Current Working Directory is",path)
        
        path = path + "/resultsUI/"
        print(f"Saving file to: C:\\Users\\ADMIN\\Desktop\\ai agent evaluation\\datasets\\ChaLearn\\test\\{session_id}.mp4")
        video_path=path+config('ai_agent_video_path')
        download_file(url,video_path )

    data=fetch_session_data(session_id) 
    questions = []
    answers = []
    for entry in data:
        question = entry['question']
        target_phrase = ("moving on to the next question","move on to the next part","continue with the next question")
        for phrase in target_phrase:
            if phrase in question.lower():
                result = question.lower().split(phrase, 1)[1].strip()
                questions.append(result)
                break
        else:
            questions.append(question)
        combined_answers = ' '.join([ans['text'].strip() for ans in entry['answers']])
        answers.append(combined_answers)
    print(questions)
    metrics = get_proctoring_metrics(session_id)
    
    skills=["react js", "restful api","sql"]     # change and get skills from database 
    focus_skills = ["react js"]   # change and get focus_skills from database 
    
    report=show_results(questions,answers,metrics,skills,focus_skills)
    if report:
        logging.info(f"Evaluation request sent successfully for session: {session_id}")
        update_query = "UPDATE interview_evaluations SET status = 'EVALUATED' WHERE session_id = %s"
        json_data = json.dumps(report)
        cursor.execute("UPDATE interview_evaluations SET evaluation_text = %s WHERE session_id = %s", (json_data, session_id))
        cursor.execute(update_query, (session_id,))
        conn.commit()

        video_path= path+config('ai_agent_video_path')
        if os.path.exists(video_path):
            os.remove(video_path)
            print("Video deleted successfully.")
        else:
            print("Video file not found.")
        logging.info(f"Session {session_id} marked as evaluated.")
    else:
        print("Failed to generate report.")

# start_evaluation("sess_BQTlyW4wGq5PcJK7DqNws")