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
from dpcv.experiment.exp_runner import ExpRunner
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
    
    return runner.test()
 
 
def support( testdic )-> List[Dict[str, str]]:
    """
    Prepare messages to end the interview and generate feedback.
    """
    #transcript = [f"{message['role'].capitalize()}: {message['content']}" for message in chat_history[1:]]
    system_prompt = testdic
    # print("- - -  - - - - - - -",system_prompt)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Grade the interview based on the transcript provided and give feedback."},
    ]
 
 
def test12(testdic ) -> Generator[str, None, None]:
    """
    End the interview and get feedback from the LLM.
    """
    
    message = support(testdic)
    
    load_dotenv()
    client = OpenAI(api_key =os.getenv("OPENAI_API_KEY"))
    # message1={'role': 'system', 'content':message[0]['content']["audio_prompt"]}
    # message2={'role': 'system', 'content':message[0]['content']["video_prompt"]}
    message3={'role': 'system' ,'content':message[0]['content']['language_prompt']}
    # print("\n|\n|\n|\n|\n|\n|\n|v",message)  # prompt returned from newprompt file
    # print("message3 is ---------------------------------------",message3)
 
 
    message=[
    {
        "role": "system",
        "content": """
You are an AI system designed to provide interview feedback in JSON format with clear sections and detailed comments. Ensure that the feedback is constructive, supportive, and aimed at helping the candidate enhance their performance.
{
  "feedback": {
    "overall_score": {
      "title": "Overall Score",
      "comment": "<comment>",
      "score": "<value>"
    },
    "scores": {
      "communication_score": {
        "title": "Communication Score",
        "comment": "<comment>",
        "score": "<value>",
        "subparts": {
            "articulation_score": {
              "title": "Articulation Score",
              "comment": "<comment>",
              "score": "<value>"
            },
            "pace_and_clarity_score": {
              "title": "Pace and Clarity Score",
              "comment": "<comment>",
              "score": "<value>"
            },
            "grammar_score": {
              "title": "Grammar Score",
              "comment": "<comment>",
              "score": "<value>"
            }
          
      }
      },
      "sociability_score": {
        "title": "Sociability Score",
        "comment": "<comment>",
        "score": "<value>",
        "subparts": 
          {
            "energy_score": {
              "title": "Energy Score",
              "comment": "<comment>",
              "score": "<value>"
            },
            "sentiment_score": {
              "title": "Sentiment Score",
              "comment": "<comment>",
              "score": "<value>"
            },
            "emotion_score": {
              "title": "Emotion Score",
              "comment": "<comment>",
              "score": "<value>"
            }
          }
        
      },
      "positive_attitude_score": {
        "title": "Positive Attitude Score",
        "comment": "<comment>",
        "score": "<value>",
        "subparts": 
          {
            "energy_score": {
              "title": "Energy Score",
              "comment": "<comment>",
              "score": "<value>"
            }
          }
        
      },
      "overall_professional_score": {
        "title": "Overall Professional Score",
        "comment": "<comment>",
        "score": "<value>",
        "subparts": 
          {
            "presentability_score": {
              "title": "Presentability Score",
              "comment": "<comment>",
              "score": "<value>"
            },
            "body_language_score": {
              "title": "Body Language Score",
              "comment": "<comment>",
              "score": "<value>"
            },
            "dressing_score": {
              "title": "Dressing Score",
              "comment": "<comment>",
              "score": "<value>"
            }
          }
        
      }
    },
    "suggestions_for_improvement": {
      "title": "Suggestions for improvement",
      "suggestions": [
        "<suggestion1>",
        "<suggestion2>",
        "<suggestion3>",
        "<suggestion4>"
      ]
    },
    "areas_for_improvement": {
      "title": "Areas for improvement",
      "improvements": [
        "<improvement1>",
        "<improvement2>"
      ]
    },
    "ocean_values_analysis": {
      "title": "Ocean Values Analysis",
      "ocean_values": [
        "<value1>",
        "<value2>",
        "<value3>",
        "<value4>",
        "<value5>"
      ],
      "comment": "<comment>"
    },
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
  }
}

 
"""  
    },

    message3,
    # message2,
 
]
    print("-------------------------------------------------------------------------------------------")
    # model="gpt-4o-2024-08-06",
    #     response_format={"type":"json_schema","json_schema":example_json},
    chat_completion = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        response_format={"type":"json_object"},
        messages = message
    )
 
    finish_reason = chat_completion.choices[0].finish_reason
 
    # if(finish_reason == "stop"):
    data = chat_completion.choices[0].message.content
    newdata = json.loads(data)
    print(newdata)
    return newdata
 
 
 
 
def evaluate_student_answer(question, student_answer):
 
    prompt = { "role": "system",
            "content":f"""
    You are tasked with evaluating a student's answer to a question. Follow these instructions to provide a thorough and constructive evaluation:
 
    1. Understand the Question: First, ensure you understand the question fully. Identify the main components or steps required to reach a correct answer.
 
    2. Review the Student's Answer: Carefully read the student's response. Identify the approach they took, noting any key points or methods they used.
 
    3. Compare with an Ideal Solution:
       - Break down the ideal solution step-by-step and see if the student's answer aligns with each step.
       - Note any parts where the student deviated from the ideal solution, missed steps, or made incorrect assumptions.
 
    4. Check for Accuracy:
       - Verify each calculation, reasoning step, or logical point for correctness.
       - Ensure the final answer is in the correct form (e.g., simplified fraction, correct units).
 
    5. Assess Clarity and Completeness:
       - Determine if the answer is easy to follow and if the student clearly explains their thought process.
       - Check if the answer addresses all parts of the question, including any specific conditions or assumptions.
 
    6. Provide Constructive Feedback:
       - If the answer is correct, highlight what the student did well, such as clarity, accurate calculations, or thorough explanations.
       - If there are errors, gently explain each mistake and suggest improvements or correct methods.
 
    7. Assign a Score or Rating (if applicable):
       - Based on accuracy, clarity, and completeness, assign a score or rating according to the provided grading criteria.
 
    8. Summarize the Evaluation:
       - Conclude with a brief summary, noting both strengths and areas for improvement.
 
    Here is the question and the student's answer:
 
    **Question**: {question}
    **Student's Answer**: {student_answer}
    """}
 
 
    client = OpenAI(api_key ="sk-proj-5J6AGNVXQEJ6Ji9NATgCT3BlbkFJblejOzq7DI9TgfagRxd7")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  
        messages=[prompt],
        max_tokens=500,
        temperature=0.5
    )
 
 
    evaluation = response.choices[0].message.content
    
    return evaluation
 
 
    
import json
# from sagemaker.remote_function import remote
# @remote(instance_type="ml.p3.2xlarge")
def show_results(Questions):
    try:
        counts = 0  
        allprompts, final_score,comment_dict= dpmain(counts)
        print(Questions)
        results = test12(allprompts)
        print("********************##########################", results)
        print("-" * 50, final_score)

        # Check if 'results' is a JSON string and convert to a dictionary
        if isinstance(results, str):
            # Convert string to dict
            results1 = json.loads(results)
        else:
            # If it's already a dict or similar object, use it as is
            results1 = results

        # Ensure final_score is a dict and update the results
        if isinstance(final_score, dict):
            print("Final score is ------------------------------------------------------------------------",final_score)
            results1.update(final_score)
            

        else:
            print("Final score is not a dictionary, unable to update results.")
        # answers=results1['transcription']
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
        # Convert the updated dictionar back to a JSON string
        promt="Summarize the given text and also maintain all detail"
        gpt_comm=" ".join(comment_dict['gpt_grammer_comment'])
        client = OpenAI(api_key =config('OPENAI_API_KEY'))
        message={"role": "system",
                "content": promt+f"The Given text is : '{gpt_comm}'"}
        # client = OpenAI(api_key =os.getenv("OPENAI_API_KEY"))
        chat_completion = client.chat.completions.create(
                model="gpt-4o",
                messages = [message]
                ,max_tokens=256
            )
            # print("|\n|\n|\n|\n|\n|\n|\n|\n|v",chat_completion)
        # finish_reason = chat_completion.choices[0].finish_reason


        final_grammer_comment = chat_completion.choices[0].message.content

        if isinstance(results1,dict):
            results1["feedback"]["scores"]["overall_professional_score"]["comment"]=comment_dict["professional_comment"]
            results1["feedback"]["scores"]["overall_professional_score"]["subparts"]["body_language_score"]["comment"]=comment_dict["bodylang_comment"]
            results1["feedback"]["scores"]["sociability_score"]["subparts"]["emotion_score"]["comment"]=comment_dict["emotioncomment"]
            results1["feedback"]["scores"]["overall_professional_score"]["subparts"]["presentability_score"]["comment"]=comment_dict["grommingcomment"]+comment_dict["dressingcomment"]
            results1["feedback"]["scores"]["communication_score"]["subparts"]["grammar_score"]["comment"]=final_grammer_comment
            results1["feedback"]["scores"]["overall_professional_score"]["subparts"]["dressing_score"]["comment"]=comment_dict["dressingcomment"]
            

        display_data = json.dumps(results1, indent=4)

        video_dir = "datasets/ChaLearn/test"
        delete_all_files_in_folder(video_dir)
        print("Files are deleted in Test Folder.")
        import torch
        torch.cuda.empty_cache()
        return display_data

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
 