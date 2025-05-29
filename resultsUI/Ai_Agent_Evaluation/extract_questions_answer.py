







# from flask import Flask, request, jsonify
# import mysql.connector
# import logging
# import threading

# import time
# import re
# import openai
# import json
# import traceback
# from decouple import config

# OPENAI_API_KEY = config("OPENAI_API_KEY")
# client = openai.OpenAI(api_key=OPENAI_API_KEY)
 
# DB_CONFIG = {
#     "host": config("HOST"),
#     "user": config("DB_USER"),
#     "password": config("DB_PASS"),
#     "database": config("DB_NAME2")
# }
# def get_db_connection():
#     return mysql.connector.connect(**DB_CONFIG)
 
# def fetch_session_data(session_id):
#     conn = get_db_connection()
#     if not conn:
#         print("Database connection failed. Cannot fetch data.")
#         return []
 
#     try:
#         cursor = conn.cursor(dictionary=True)
#         query = """
#             SELECT role, title, Vision_Analysis, createdAtMs
#             FROM interview_transcripts
#             WHERE session_id = %s AND type = 'MESSAGE'
#             ORDER BY createdAtMs
#         """
#         cursor.execute(query, (session_id,))
#         rows = cursor.fetchall()
 
#         if not rows:
#             logging.warning(f"No data found for session: {session_id}")
#             return []
 
#         qa_pairs = []
#         current_question = None
#         current_answers = []
        
#         for row in rows:
#             if row["role"] == "assistant":
                
#                 if current_question:
#                     qa_pairs.append({
#                         "question": current_question,
#                         "answers": current_answers,
#                         "vision_analysis": current_answers[-1]["vision_analysis"] if current_answers else None
#                     })
                
#                 current_question = row["title"]
#                 current_answers = []
#             elif row["role"] == "user" and current_question:
#                 current_answers.append({
#                     "text": row["title"],
#                     "vision_analysis": row.get("Vision_Analysis")
#                 })
        
#         if current_question:
#             qa_pairs.append({
#                 "question": current_question,
#                 "answers": current_answers,
#                 "vision_analysis": current_answers[-1]["vision_analysis"] if current_answers else None
#             })
#         # print(qa_pairs)
#         return qa_pairs
 
#     except Exception as e:
#         logging.error(f"Error fetching session data: {e}")
#         return []
#     finally:
#         cursor.close()
        
#         conn.close()
# # data=fetch_session_data("sess_BQBuD9lHLxiP7ANiH5ahm") 
# # questions = []
# # answers = []
# # for entry in data:
# #     question = entry['question']
# #     target_phrase = ("moving on to the next question","move on to the next part","continue with the next question")
# #     for phrase in target_phrase:
# #         if phrase in question.lower():
# #             result = question.lower().split(phrase, 1)[1].strip()
# #             questions.append(result)
# #             break
# #     else:
# #         questions.append(question)
# #     combined_answers = ' '.join([ans['text'].strip() for ans in entry['answers']])
# #     answers.append(combined_answers)
# # from identify_selfwareness_question import split_questions_by_type
# # sa_qs, sa_as, normal_qs, normal_as = split_questions_by_type(questions, answers)
# # print("Self-Awareness Questions:", sa_qs)
# # print("Self-Awareness Answers:", sa_as)
# # print("Normal Questions:", normal_qs)
# # print("Normal Answers:", normal_as)


# def get_db_connection():
#     return mysql.connector.connect(**DB_CONFIG)

# def get_proctoring_metrics(session_id):
#     metrics = {
#         'multi_person_count': 0,
#         'cell_phone_count': 0,
#         'tab_switch_count': 0,
#         'exited_full_screen_count': 0
#     }
    
#     try:
#         # Get data from detected_images table
#         connection = get_db_connection()
#         cursor = connection.cursor(dictionary=True)
        
#         # Query for person_count and cell_phone_detected
#         query_images = """
#         SELECT 
#             COUNT(CASE WHEN person_count > 1 THEN 1 END) as multi_person_count,
#             COUNT(CASE WHEN cell_phone_detected = 1 THEN 1 END) as cell_phone_count
#         FROM detected_images 
#         WHERE openai_session_id = %s
#         """
        
#         print(f"\n[DEBUG] Executing query on detected_images: {query_images % (session_id,)}")
#         cursor.execute(query_images, (session_id,))
#         image_result = cursor.fetchone()
#         print(f"[DEBUG] Result from detected_images: {image_result}")
        
#         if image_result:
#             metrics.update({
#                 'multi_person_count': image_result['multi_person_count'] or 0,
#                 'cell_phone_count': image_result['cell_phone_count'] or 0
#             })
        
#         cursor.close()
        
#         # Get data from tabswitch_data table
#         cursor = connection.cursor(dictionary=True)
        
#         # Query to get the latest tabswitch and fullscreen exit counts
#         query_tabswitch = """
#         SELECT 
#             tabswitch_count,
#             fullscreen_exit_count
#         FROM tabswitch_data
#         WHERE session_id = %s
#         ORDER BY id DESC
#         LIMIT 1
#         """
        
#         print(f"\n[DEBUG] Executing query on tabswitch_data: {query_tabswitch % (session_id,)}")
#         cursor.execute(query_tabswitch, (session_id,))
#         tabswitch_result = cursor.fetchone()
#         print(f"[DEBUG] Result from tabswitch_data: {tabswitch_result}")
        
#         if tabswitch_result:
#             metrics.update({
#                 'tab_switch_count': tabswitch_result['tabswitch_count'] or 0,
#                 'exited_full_screen_count': tabswitch_result['exited_fullscreen'] or 0
#             })
            
#     except:
#         print(f"[ERROR] Database error: ")
#     finally:
#         if 'connection' in locals() and connection.is_connected():
#             cursor.close()
#             connection.close()
    
#     print(f"\n[DEBUG] Final metrics: {metrics}")
#     return metrics


from flask import Flask, request, jsonify
import mysql.connector
import logging
import threading
import traceback
import time
import re
import openai
import json
import traceback
from decouple import config

OPENAI_API_KEY = config("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)
from flask import Flask, request, jsonify
import mysql.connector
import logging
import threading
import time
import re, os, whisper
import openai
import json
import traceback
from decouple import config
import logging
import traceback
OPENAI_API_KEY = config("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)
 
DB_CONFIG = {
    "host": config("HOST_AGENT"),
    "user": config("DB_USER_AGENT"),
    "password": config("DB_NAME_AGENT"),
    "database": config("DB_PASS_AGENT")
}

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)
 

host1 = config("HOST_AGENT")
user1 = config("DB_USER_AGENT")
password1 = config("DB_PASS_AGENT")
database1 = config("DB_NAME_AGENT")


 
conn = mysql.connector.connect(
    host=host1,
    user=user1,
    password=password1,
    database=database1,
    auth_plugin='mysql_native_password',
    connection_timeout=5,
    use_pure=True
)

 
import mysql.connector
import traceback
import logging
from decouple import config

def fetch_session_data(session_id):
    print("============================================================================")
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(
            host=config("HOST_AGENT"),
            user=config("DB_USER_AGENT"),
            password=config("DB_PASS_AGENT"),
            database=config("DB_NAME_AGENT"),
            port=config("DB_PORT", cast=int),
            connection_timeout=5,
            use_pure=True
        )
        cursor = conn.cursor(dictionary=True)

        query = """
            SELECT role, title, timestamp
            FROM interview_transcripts
            WHERE session_id = %s AND type = 'MESSAGE'
            ORDER BY timestamp
        """
        cursor.execute(query, (session_id,))
        rows = cursor.fetchall()

        if not rows:
            logging.warning(f"No data found for session: {session_id}")
            return []

        qa_pairs = []
        current_question = None
        answer_collected = False
        first_user_skipped = False

        for row in rows:
            role = row["role"]
            text = row["title"].strip()

            if role == "assistant":
                if current_question is not None and answer_collected:
                    qa_pairs.append({
                        "question": current_question,
                        "answers": current_answer
                    })
                current_question = text
                current_answer = ""
                answer_collected = False

            elif role == "user":
                if not first_user_skipped:
                    first_user_skipped = True
                    continue

                if text.lower() in ("[transcribing...]", "[inaudible]"):
                    text = "user did not respond clearly or did not speak."

                if current_question and not answer_collected:
                    current_answer = text
                    answer_collected = True

        if current_question and answer_collected:
            qa_pairs.append({
                "question": current_question,
                "answers": current_answer
            })

        return qa_pairs

    except Exception as e:
        traceback.print_exc()
        logging.error(f"Error fetching session data: {e}")
        return []
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()



def get_proctoring_metrics(session_id):
    metrics = {
        'multi_person_count': 0,
        'cell_phone_count': 0,
        'tab_switch_count': 0,
        'exited_full_screen_count': 0
    }
    
    try:
        # Get data from detected_images table
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Query for person_count and cell_phone_detected
        query_images = """
        SELECT 
            COUNT(CASE WHEN person_count > 1 THEN 1 END) as multi_person_count,
            COUNT(CASE WHEN cell_phone_detected = 1 THEN 1 END) as cell_phone_count
        FROM detected_images 
        WHERE openai_session_id = %s
        """
        
        print(f"\n[DEBUG] Executing query on detected_images: {query_images % (session_id,)}")
        cursor.execute(query_images, (session_id,))
        image_result = cursor.fetchone()
        print(f"[DEBUG] Result from detected_images: {image_result}")
        
        if image_result:
            metrics.update({
                'multi_person_count': image_result['multi_person_count'] or 0,
                'cell_phone_count': image_result['cell_phone_count'] or 0
            })
        
        cursor.close()
        
        # Get data from tabswitch_data table
        cursor = connection.cursor(dictionary=True)
        
        # Query to get the latest tabswitch and fullscreen exit counts
        query_tabswitch = """
        SELECT 
            tabswitch_count,
            fullscreen_exit_count
        FROM tabswitch_data
        WHERE session_id = %s
        ORDER BY id DESC
        LIMIT 1
        """
        
        print(f"\n[DEBUG] Executing query on tabswitch_data: {query_tabswitch % (session_id,)}")
        cursor.execute(query_tabswitch, (session_id,))
        tabswitch_result = cursor.fetchone()
        print(f"[DEBUG] Result from tabswitch_data: {tabswitch_result}")
        
        if tabswitch_result:
            metrics.update({
                'tab_switch_count': tabswitch_result['tabswitch_count'] or 0,
                'exited_full_screen_count': tabswitch_result['exited_fullscreen'] or 0
            })
            
    except:
        print(f"[ERROR] Database error: ")
    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()
    
    print(f"\n[DEBUG] Final metrics: {metrics}")
    return metrics


