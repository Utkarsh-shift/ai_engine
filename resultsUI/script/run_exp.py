import sys
import os
import cv2
import os
import zipfile
from pathlib import Path
import shutil
import os,json
from typing import List, Dict, Generator, Optional, Tuple
from openai import OpenAI
from dotenv import load_dotenv
 
current_path = os.path.dirname(os.path.abspath(__file__))
work_path = os.path.join(current_path, "../")
sys.path.append(work_path)

from dpcv.tools.common import parse_args
from dpcv.config.default_config_opt import cfg, cfg_from_file, cfg_from_list
# from torch.utils.tensorboard import SummaryWriter
from dpcv.experiment.new_exp import ExpRunner
from dpcv.data.utils.video_to_image import convert_videos_to_frames
from dpcv.data.utils.video_to_wave import audio_extract
from dpcv.data.utils.raw_audio_process import audio_process
from datetime import datetime
from decouple import config
 
 
 
 
def copy_files(source_dir, dest_dir):
    # Check if the source directory exists
    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' does not exist.")
        return
    
    # Create the destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Iterate over all files in the source directory
    for filename in os.listdir(source_dir):
        source_file = os.path.join(source_dir, filename)
        dest_file = os.path.join(dest_dir, filename)
        
        # Copy the file from source to destination
        if os.path.isfile(source_file):
            shutil.copy2(source_file, dest_file)
            print(f"Copied '{filename}' to '{dest_dir}'")
 
def create_folder_with_datetime(parent_dir):
    try:
        # Get current date and time
        now = datetime.now()
        # Format the datetime as desired
        folder_name = now.strftime("%Y-%m-%d_%H-%M-%S")
        # Construct the full path for the new folder
        folder_path = os.path.join(parent_dir, folder_name)
        
        # Create the new directory
        os.makedirs(folder_path)
        print(f"Folder created successfully at: {folder_path}")
    except OSError as e:
        print(f"Failed to create folder at: {folder_path} - {e}")
    return folder_path

def delete_all_files_in_folder(folder_path):
    try:
        # Check if the folder exists
        if not os.path.exists(folder_path):
            print(f"Folder '{folder_path}' does not exist.")
            return
        
        # Iterate over all files in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            # Check if the path is a file and not a directory
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted file: {filename}")
        
        print(f"All files deleted successfully from folder: {folder_path}")
    except OSError as e:
        print(f"Error deleting files from folder: {folder_path} - {e}")
 
 
def delete_all_folders_in_folder(folder_path):
    # Get a list of all subdirectories in the folder
    folders = [os.path.join(folder_path, d) for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
 
    for folder in folders:
        try:
            shutil.rmtree(folder)  # Remove the folder and all its contents
            print(f"Deleted folder: {folder}")
        except Exception as e:
            print(f"Error deleting folder {folder}: {e}")
 
 
 
def setup():
    args = parse_args()
 
    print(args)
 
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
 
    # if args.resume:
    #     cfg.TRAIN.RESUME = args.resume
    # if args.max_epoch:
    #     cfg.TRAIN.MAX_EPOCH = args.max_epoch
    # if args.lr:
    #     cfg.SOLVER.RESET_LR = True
    #     cfg.SOLVER.LR_INIT = args.lr
    if True:
        cfg.TEST.TEST_ONLY = True
 
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    return args
 
def dpmain(count):
 
    count += 1
    print("The cnt is **********************************************" , count)
    path = os.getcwd()
    # path="/disk/AVI_PA-mainold/"
    print("Current Working Directory is",path)
    
    path = path + "/resultsUI/"
    if path.endswith("/resultsUI/resultsUI/"):
        path=path.replace("/resultsUI/resultsUI/","/resultsUI/")
    if os.path.exists(path):
        os.chdir(path)
    print("The root directory is given as *****************************************************************************************" , path)
    cfg_from_file(path+"config/demo/bimodal_resnet18.yaml")
    #args = setup()
    video_dir = path+"datasets/ChaLearn/test"
    output_dir = path+"datasets/ChaLearn/test_data"
    outforwav = path+"datasets/ChaLearn/voice_data/voice_raw/test_data"
    outforlibrosa = path+"datasets/ChaLearn/voice_data/voice_librosa/test_data"
    source_directory = path+"datasets/ChaLearn/test"
    parent_directory = path+"datasets/ChaLearn"
 
 
 
    contents = os.listdir(source_directory)
    if not contents:
        print("Conitnue with the code")
    else:
        # tempfolder = create_folder_with_datetime(parent_directory)
        # copy_files(source_directory, tempfolder)
        pass
 
 
 
  #  delete_all_files_in_folder(video_dir)
    delete_all_files_in_folder(outforlibrosa)
    delete_all_files_in_folder(outforwav)
    delete_all_folders_in_folder(output_dir)
 
   # video_to_image()
    convert_videos_to_frames(video_dir,output_dir)
    audio_extract(video_dir,outforwav)
    audio_process(mode= 'librosa' ,aud_dir=outforwav , saved_dir=outforlibrosa)
 
    runner = ExpRunner(cfg)
    
    res = runner.test()



    return res
 
 
# def support( testdic )-> List[Dict[str, str]]:
#     """
#     Prepare messages to end the interview and generate feedback.
#     """
#     #transcript = [f"{message['role'].capitalize()}: {message['content']}" for message in chat_history[1:]]
#     system_prompt = testdic
#     # print("- - -  - - - - - - -",system_prompt)
#     return [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": "Grade the interview based on the transcript provided and give feedback."},
#     ]
 
 
# def test12(testdic ) -> Generator[str, None, None]:
#     """
#     End the interview and get feedback from the LLM.
#     """
    
#     message = support(testdic)
    
#     load_dotenv()
#     client = OpenAI(api_key =os.getenv("OPENAI_API_KEY"))
#     # message1={'role': 'system', 'content':message[0]['content']["audio_prompt"]}
#     # message2={'role': 'system', 'content':message[0]['content']["video_prompt"]}
#     message3={'role': 'system' ,'content':message[0]['content']['language_prompt']}
#     # print("\n|\n|\n|\n|\n|\n|\n|v",message)  # prompt returned from newprompt file
#     # print("message3 is ---------------------------------------",message3)
 
 
#     message=[
#     {
#         "role": "system",
#         "content": """
# You are an AI system designed to provide interview feedback in JSON format with clear sections and detailed comments. Ensure that the feedback is constructive, supportive, and aimed at helping the candidate enhance their performance.
# {
#   "feedback": {
#     "overall_score": {
#       "title": "Overall Score",
#       "comment": "<comment>",  
#       "score": "<value>"
#     },
#     "scores": {
#       "communication_score": {
#         "title": "Communication Score",
#         "comment": "<comment>",
#         "score": "<value>",
#         "subparts": {
#             "articulation_score": {
#               "title": "Articulation Score",
#               "comment": "<comment>",
#               "score": "<value>"
#             },
#             "pace_and_clarity_score": {
#               "title": "Pace and Clarity Score",
#               "comment": "<comment>",
#               "score": "<value>"
#             },
#             "grammar_score": {
#               "title": "Grammar Score",
#               "comment": "<comment>",
#               "score": "<value>"
#             }
          
#       }
#       },
#       "sociability_score": {
#         "title": "Sociability Score",
#         "comment": "<comment>",
#         "score": "<value>",
#         "subparts": 
#           {
#             "energy_score": {
#               "title": "Energy Score",
#               "comment": "<comment>",
#               "score": "<value>"
#             },
#             "sentiment_score": {
#               "title": "Sentiment Score",
#               "comment": "<comment>",
#               "score": "<value>"
#             },
#             "emotion_score": {
#               "title": "Emotion Score",
#               "comment": "<comment>",
#               "score": "<value>"
#             }
#           }
        
#       },
#       "positive_attitude_score": {
#         "title": "Positive Attitude Score",
#         "comment": "<comment>",
#         "score": "<value>",
#         "subparts": 
#           {
#             "energy_score": {
#               "title": "Energy Score",
#               "comment": "<comment>",
#               "score": "<value>"
#             }
#           }
        
#       },
#       "overall_professional_score": {
#         "title": "Overall Professional Score",
#         "comment": "<comment>",
#         "score": "<value>",
#         "subparts": 
#           {
#             "presentability_score": {
#               "title": "Presentability Score",
#               "comment": "<comment>",
#               "score": "<value>"
#             },
#             "body_language_score": {
#               "title": "Body Language Score",
#               "comment": "<comment>",
#               "score": "<value>"
#             },
#             "dressing_score": {
#               "title": "Dressing Score",
#               "comment": "<comment>",
#               "score": "<value>"
#             }
#           }
        
#       }
#     },
#     "suggestions_for_improvement": {
#       "title": "Suggestions for improvement",
#       "suggestions": [
#         "<suggestion1>",
#         "<suggestion2>",
#         "<suggestion3>",
#         "<suggestion4>"
#       ]
#     },
#     "areas_for_improvement": {
#       "title": "Areas for improvement",
#       "improvements": [
#         "<improvement1>",
#         "<improvement2>"
#       ]
#     },
#     "ocean_values_analysis": {
#       "title": "Ocean Values Analysis",
#       "ocean_values": [
#         "<value1>",
#         "<value2>",
#         "<value3>",
#         "<value4>",
#         "<value5>"
#       ],
#       "comment": "<comment>"
#     },
#     "strengths": {
#       "title": "Strengths",
#       "strengths": [
#         "<strength1>",
#         "<strength2>",
#         "<strength3>",
#         "<strength4>"
#       ]
#     },
#     "weakness": {
#       "title": "Weakness",
#       "weakness": [
#         "<weakness1>",
#         "<weakness2>",
#         "<weakness3>",
#         "<weakness4>"
#       ]
#     }
#   }
# }

 
# """  
#     },

#     message3,
#     # message2,
 
# ]
#     print("-------------------------------------------------------------------------------------------")
#     # model="gpt-4o-2024-08-06",
#     #     response_format={"type":"json_schema","json_schema":example_json},
#     chat_completion = client.chat.completions.create(
#         model="gpt-4o-2024-08-06",
#         response_format={"type":"json_object"},
#         messages = message
#     )
 
#     finish_reason = chat_completion.choices[0].finish_reason
 
#     # if(finish_reason == "stop"):
#     data = chat_completion.choices[0].message.content
#     newdata = json.loads(data)
#     print(newdata)
#     return newdata
 
 
 
 
def evaluate_student_answer(question, student_answer):
 
    prompt = { "role": "system",
            "content":f"""
    You are tasked with evaluating a candidate's answer to a question. Follow these instructions to provide a thorough and constructive evaluation:
 
    Understand the Question: First, ensure you understand the question fully. Identify the main components or steps required to reach a correct answer.
 
    Review the Candidate's Answer: Carefully read the Candidate's response. Identify the approach they took, noting any key points or methods they used.
 
    Compare with an Ideal Solution:
    - Break down the ideal solution step-by-step and see if the candidate's answer aligns with each step.
    - Note any parts where the candidate deviated from the ideal solution, missed steps, or made incorrect assumptions.
 
    Check for Accuracy:
    - Verify each calculation, reasoning step, or logical point for correctness.
    - Ensure the final answer is in the correct form (e.g., simplified fraction, correct units).
 
    Assess Clarity and Completeness:
    - Determine if the answer is easy to follow and if the candidate clearly explains their thought process.
    - Check if the answer addresses all parts of the question, including any specific conditions or assumptions.
 
    Provide Constructive Feedback:
    - If the answer is correct, highlight what the candidate did well, such as clarity, accurate calculations, or thorough explanations.
    - If there are errors, gently explain each mistake and suggest improvements or correct methods.
 
    Assign a Score or Rating (if applicable):
    - Based on accuracy, clarity, and completeness, assign a score or rating according to the provided grading criteria.
 
    
    Here is the question and the candidate's answer:
 
    **Question**: {question}
    **Candidate's Answer**: {student_answer}

    Summarize the Evaluation:
    Conclude with a brief summary in 3-4 lines 
 
    """}
 
 
    client = OpenAI(api_key =config("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  
        messages=[prompt],
        max_tokens=500,
        temperature=0.5
    )
 
 
    evaluation = response.choices[0].message.content
    evaluation=evaluation.replace("\n","")
    return evaluation
 

def format_feedback(input_json):
    
    formatted_json = {
        "feedback": {
            "overall_score": {
                "title": "Overall Score",
                "comment": "Overall performance summary will go here.",
                "score": input_json["overall_score"]
            },
            "scores": {
                "communication_score": {
                    "title": "Communication Score",
                    "comment": input_json["communication_comment"],
                    "score": input_json["communication_score"],
                    "subparts": 
                        {
                            "articulation_score": {
                                "title": "Articulation Score",
                                "comment": input_json["articulation_comment"],
                                "score": input_json["articulation_score"]
                            },
                            "pace_and_clarity_score": {
                                "title": "Pace and Clarity Score",
                                "comment": input_json["pace_score_comment"],
                                "score": input_json["pace_score"]
                            },
                            "grammar_score": {
                                "title": "Grammar Score",
                                "comment": input_json["grammar_comment"],
                                "score": input_json["grammar_score"]
                            }
                        }
                    
                },
                "sociability_score": {
                    "title": "Sociability Score",
                    "comment": input_json["sociability_comment"],
                    "score": input_json["sociability_score"],
                    "subparts": 
                        {
                            "energy_score": {
                                "title": "Energy Score",
                                "comment": input_json["energy_comment"],
                                "score": input_json["emotion_score"]
                            },
                            "sentiment_score": {
                                "title": "Sentiment Score",
                                "comment": input_json["sentiment_comment"],
                                "score": input_json["sentiment_score"]
                            },
                            "emotion_score": {
                                "title": "Emotion Score",
                                "comment": input_json["emotion_comment"],
                                "score": input_json["emotion_score"]
                            }
                        }
                    
                },
                "positive_attitude_score": {
                    "title": "Positive Attitude Score",
                    "comment": input_json["positive_attitude_comment"],  # Fixed closing bracket error here
                    "score": input_json["positive_attitude_score"],
                    "subparts": 
                        {
                            "energy_score": {
                                "title": "Energy Score",
                                "comment": input_json["energy_comment"],
                                "score": input_json["energy_score"]
                            }
                        }
                    
                },
                "professional_score": {
                    "title": "Professional Score",
                    "comment": input_json["professional_comment"],
                    "score": input_json["professional_score"],
                    "subparts": 
                        {
                            "presentability_score": {
                                "title": "Presentability Score",
                                "comment": input_json["presentability_comment"],
                                "score": input_json["presentability_score"]
                            },
                            "body_language_score": {
                                "title": "Body Language Score",
                                "comment": input_json["body_language_comment"],
                                "score": input_json["bodylang_score"]
                            },
                            "dressing_score": {
                                "title": "Dressing Score",
                                "comment": input_json["dressing_comment"],
                                "score": input_json["dressing_score"]
                            }
                        }
                    
                }
            },
            "transcription": input_json["transcription"]
        }
    }

    return formatted_json

def get_ocean_comment(ocean_list):
    message={"role": "system",
            "content": f"These are the ocean values: {ocean_list[0]}, {ocean_list[1]},{ocean_list[2]},{ocean_list[3]},{ocean_list[4]} representing openness to experience, conscientiousness, extraversion, agreeableness and neuroticism. Give comment in 2-3 lines based on these values"}
    client = OpenAI(api_key =os.getenv("OPENAI_API_KEY"))
    chat_completion = client.chat.completions.create(
            model="gpt-4o",
            messages = [message]
            ,max_tokens=256
        )

    final_comment = chat_completion.choices[0].message.content
    final_comment=final_comment.replace("\n"," ")
    return final_comment


import time
def get_sw(prompt,json_schema):
    retries = 0

    max_retries = 3

    while retries < max_retries :
      try :       
        message={"role": "system",
                "content": prompt}
        client = OpenAI(api_key =os.getenv("OPENAI_API_KEY"))
        chat_completion = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            response_format={"type":"json_schema","json_schema":json_schema},
                messages = [message]
                ,max_tokens=256
            )
            # print("|\n|\n|\n|\n|\n|\n|\n|\n|v",chat_completion)
        # finish_reason = chat_completion.choices[0].finish_reason
    
        
        final = chat_completion.choices[0].message.content
        print("strength and weakness json is ---------",final)
        final=final.replace("'", '"')
        final_res=json.loads(final)    
        return final_res
      except (json.JSONDecodeError , Exception) as e :
          retries += 1
          print(f"Error occurred: {e}. Retrying {retries}/{max_retries}.")
          time.sleep(10)
    print("Max retries exceeded. Failed to generate valid JSON.")

    return None 

    
import json

def show_results(Questions):
    try:
      counts = 0  
      final_dict= dpmain(counts)


      print(f"Questions | {Questions}")
      
    
      if isinstance(final_dict, str):
          results1 = json.loads(final_dict)
      else:
          results1 = final_dict


      


      print(":::::::::::::::::::::::::::::::::::::::::::::::::::::")

      print(results1)

      print(":::::::::::::::::::::::::::::::::::::::::::::::::::::")


      ids = [entry['id'] for entry in results1['transcription']]
      answers = [entry['transcript'] for entry in results1['transcription']]
      ans_feedback=[]
      for i,j in zip(Questions,answers):

          ans=evaluate_student_answer(i,j)
          ans_feedback.append(ans)
      ques_feedback = [{"id": id_, "answer_evaluation": transcript} for id_, transcript in zip(ids,ans_feedback)]
      ques_feedback={"answer_feedback":ques_feedback}
      if isinstance(ques_feedback, dict):
          results1.update(ques_feedback)
      else:
          print("Final score is not a dictionary, unable to update results.")
      display_data = results1
      for transcript in display_data['transcription']:
          evaluation = next((feedback['answer_evaluation'] for feedback in display_data['answer_feedback'] if feedback['id'] == transcript['id']), None)
          if evaluation:
              transcript['answer_evaluation'] = evaluation

      # Remove the 'answer_feedback' field entirely
      del display_data['answer_feedback']

      # Print the modified JSON
      # print(json.dumps(display_data, indent=2))   
      # 
      ocean_values=display_data["ocean_values"]
      ocean_comment=get_ocean_comment(ocean_values)
      formatted_dta=format_feedback(display_data)
      
      sociability_score = formatted_dta["feedback"]["scores"]["sociability_score"]["score"]
      communication_score = formatted_dta["feedback"]["scores"]["communication_score"]["score"]
      positive_attitude_score = formatted_dta["feedback"]["scores"]["positive_attitude_score"]["score"]
      overall_score = formatted_dta["feedback"]["scores"]["professional_score"]["score"]

      sociability_comment = formatted_dta["feedback"]["scores"]["sociability_score"]["comment"]
      communication_comment = formatted_dta["feedback"]["scores"]["communication_score"]["comment"]
      positive_attitude_comment = formatted_dta["feedback"]["scores"]["positive_attitude_score"]["comment"]
      formatted_dta["feedback"]["ocean_values_analysis"]={"values":ocean_values,"title":"Ocean values analysis","comment":ocean_comment}

      
      example_json="""{
          "strengths": {
              "title": "Strengths",
              "strengths": [
                  "<strength1>",
                  "<strength2>",
                  "<strength3>",
                  "<strength4>"
              ]
          },
          "weakness": {
              "title": "Weakness",
              "weakness": [
                  "<weakness1>",
                  "<weakness2>",
                  "<weakness3>",
                  "<weakness4>"
              ]
          }
      }"""

      json_schema={
  "description": "A schema representing an individual's strengths and weaknesses, with titles and lists of strengths/weaknesses.",
  "name": "strengths_and_weakness",
    "strict": True,
    "schema":{
  "type": "object",
  "properties": {
    "strengths": {
      "type": "object",
      "properties": {
        "title": {
          "type": "string",
          "enum": ["Strengths"]
        },
        "strengths": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "A list of strengths. Can contain up to four strength entries."
        }
      },
      "required": ["title", "strengths"],
      "additionalProperties": False
    },
    "weakness": {
      "type": "object",
      "properties": {
        "title": {
          "type": "string",
          "enum": ["Weakness"]
        },
        "weakness": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "A list of weaknesses. Can contain up to four weakness entries."
        }
      },
      "required": ["title", "weakness"],
      "additionalProperties": False
    }
  },
  "required": ["strengths", "weakness"],
  "additionalProperties": False
}}

      prompt = f"""
      The following feedback was provided for a student interview:

      - Sociability Score: {sociability_score}, Comment: {sociability_comment}.
      - Communication Score: {communication_score}, Comment:{communication_comment}.
      - Positive Attitude Score: {positive_attitude_score}, Comment:{positive_attitude_comment}.
      - Overall Professional Score: {overall_score}

      Please provide a summary of the student's strengths and weaknesses in the following format:

      {example_json}

      Make sure the strengths and weaknesses are based on the comments for each of the scores provided.
      """

      dict_result=get_sw(prompt,json_schema)

      formatted_dta["feedback"]["strengths"]=dict_result["strengths"]
      formatted_dta["feedback"]["weakness"]=dict_result["weakness"]
      video_dir = "datasets/ChaLearn/test"
      delete_all_files_in_folder(video_dir)
      print("Files are deleted in Test Folder.")
      import torch
      torch.cuda.empty_cache()
      return formatted_dta

    except Exception as e:
        print(e)
        video_dir = "datasets/ChaLearn/test"
        batch_dir=os.getcwd()
        batch_dir=batch_dir.replace("/resultsUI","/videos")
        delete_all_files_in_folder(video_dir)
        delete_all_folders_in_folder(batch_dir)
        print("Files are deleted in Test Folder.")
 
 
 
 
 
if __name__ == "__main__":
    count = 0
    test = show_results()
    print(test)
 