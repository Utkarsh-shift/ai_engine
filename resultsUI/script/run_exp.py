import sys
import os
# import cv2
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
from dpcv.experiment.exp_runner_new import ExpRunner
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
    if True:
        cfg.TEST.TEST_ONLY = True
 
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    return args
 


 
def dpmain(count,Questions):
 
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
 
 
 
 
    delete_all_files_in_folder(outforlibrosa)
    delete_all_files_in_folder(outforwav)
    delete_all_folders_in_folder(output_dir)
 
 
    convert_videos_to_frames(video_dir,output_dir)
    audio_extract(video_dir,outforwav)
    audio_process(mode= 'librosa' ,aud_dir=outforwav , saved_dir=outforlibrosa)
    runner = ExpRunner(cfg)
    res = runner.test(Questions)


    return res
 
def is_self_awareness_question(q):
    keywords = [
        
        "strength", "strengths", "greatest strength", "core strength", "key strength",
        "weakness", "weaknesses", "biggest weakness", "overcome weakness", "handle weakness",
        
        "self-aware", "self-awareness", "describe yourself", "personal qualities",
        "how do you view yourself", "how do others describe you", "know about yourself",
        "perceive yourself", "self-perception", "how would you describe yourself",
        "what have you learned about yourself", "reflect on yourself", "your personality",
        
        "what motivates you", "why do you do", "passionate about", "what drives you",
        "what inspires you", "personal values", "your goals", "career goals",
        "life goals", "future plans", "ambition", "personal mission",
        
        "biggest failure", "deal with failure", "learned from failure",
        "handle failure", "fail at", "mistake you made", "major mistake", "overcome challenge",
        
        "personal growth", "how have you grown", "develop yourself", "improve yourself",
        "learning experience", "continuous improvement", "developed over time",
        "professional development", "resilience", "bounce back", "cope with setbacks",
        
        "emotional intelligence", "how do you handle pressure", "stay calm", "manage stress",
        "how do you react", "emotional response", "control emotions", "stay focused",
        "handle criticism", "accept feedback", "give feedback", "receive feedback",
        
        "what do you value", "what matters to you", "ethical dilemma", "morals",
        "what do you believe in", "work ethic", "integrity", "trust", "honesty",
    
        "make decisions", "hardest decision", "difficult choice", "regret", "hindsight",
        
        "work with others", "team conflict", "resolve conflict", "how do you communicate",
        "communication style", "listen", "express yourself",
        
        "time management", "organize yourself", "procrastinate", "stay productive",
        "how do you plan", "daily routine", "structure your day"
    ]
 
    return any(k.lower() in q.lower() for k in keywords)
 
def mark_self_awareness_questions(questions):
    count = 0
    max_self_awareness = 3
    modified = []
 
    for q in questions:
        if count < max_self_awareness and is_self_awareness_question(q):
            modified.append(q + "?$#@True$#@")
            count += 1
        else:
            modified.append(q)
    return modified



def  show_results(skills,focus_skills,Questions,proctoring_data) :
    Questions = mark_self_awareness_questions(Questions)
    try:
        counts = 0  
        final_dict= dpmain(counts,Questions)

        print(final_dict,"[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]")
        print(f"Questions 111111111111111111111| {Questions}")
        if isinstance(final_dict, str):
            results1 = json.loads(final_dict)
        else:
            results1 = final_dict
        print(":::::::::::::::::::::::::::::::::::::::::::::::::::::")

        print(results1)

        print(":::::::::::::::::::::::::::::::::::::::::::::::::::::")


        ids = [entry['id'] for entry in results1['transcription']]
        answers = [entry['transcript'] for entry in results1['transcription']]
        articulation_score=final_dict['articulation_score']
        articulation_comment=final_dict['articulation_comment']
        body_language_comment=final_dict['bodylang_comment']
        bodylang_score=final_dict['bodylang_score']
        transcription=final_dict['transcription']
        communication_comment=final_dict['communication_comment']
        etiquette_score=final_dict['etiquette_score']
        etiquette_comment=final_dict['etiquette_comment']
        grammer_score=final_dict['grammar_score']
        grammer_comment=final_dict['grammar_comment']
        pace_score=final_dict['pace_score']
        pace_comment=final_dict['pace_comment']
        pronunciation_score=final_dict['pronounciation_score']
        pronunciation_comment=final_dict['pronounciation_comment']
        self_awareness_comment=final_dict['self_awareness_comment']
        self_awareness_score=final_dict['self_awareness_score']
        print("answers",answers)
        print("::::::::::::::::::::::::::::::::::::::::::::::::::::",transcription)

        from openai import OpenAI
        import re,requests
        
        client = OpenAI(api_key=config("OPENAI_API_KEY"))
        
               

        
        skills =  [skill['skill_title'] for skill in skills]
        print("skills--------------------------",skills)
  
        
        skills_score = {}
        overall_scores = {}    
        focus_skills_score = {}
        focus_skill_comment = {}
        
        focus_skills = [skill['skill_title'] for skill in focus_skills]
        
        question_answer_dict = {}
        
        for i, question in enumerate(Questions):
            question_answer_dict[question] = answers[i]
            print("Question-Answer Dictionary:Question-Answer Dictionary:Question-Answer Dictionary:", question_answer_dict)
            
        
            
        for skill in skills: 
            for skill in skills: 
                if skill not in focus_skills:
                    print("Skill not in focus skills, skipping:", skill)
                    prompt = f"""
                    You are an expert evaluator who will assess the following answer to the interview question for the skill '{skill}'. 
                    The person's response to the interview question is:
                    {question_answer_dict}

                    Please evaluate the answer and:
                    1. Provide a score from 1 to 100 based on the relevance and quality of the answer.

                    Your response should only include the score.
                    """
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are an experienced evaluator who rates interview answers based only on relevance and quality."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=64 ,
                        temperature = 0.4
                    )
                    response_text = response.choices[0].message.content.strip()
                    match = re.search(r'\d+', response_text)
                    if match:
                        score = int(match.group(0))
                        focus_skills_score[skill] = score
                        print(f"Score for non-focus skill '{skill}': {score}")
                        skills_score[skill] = score
                    else:
                        print(f"Warning: No score found in response for non-focus skill '{skill}'. Response: {response_text}")
                        skills_score[skill] = 50
                else : 
                    prompt = f"""
                    You are an expert evaluator assessing the following answer to an interview question for the skill '{skill}'. 
                    The person's response to the interview question is:
                    {question_answer_dict}

                    Please evaluate the answer based on the skill '{skill}' only, and:
                    1. Provide a score from 1 to 100 based on the relevance and quality of the answer specific to the skill '{skill}'.
                    2. Write a single, concise comment that covers all relevant aspects of the answer related to the skill '{skill}', including strengths, weaknesses, and key takeaways. The comment should be focused solely on the skill '{skill}' and should not include any feedback or comments related to other skills (e.g., React.js comments should not include feedback on PHP or vice versa).

                    Your response should include both the score (1-100) and a one-line comment specific to the skill '{skill}'.
                    """

                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[ 
                            {"role": "system", "content": "You are an experienced evaluator who rates answers to interview questions. Your task is to rate answers based on relevance and quality, provide a score from 1 to 100, and give a one-line comment specific to the skill."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=64,
                        temperature = 0.4
                    )

                    response_text = response.choices[0].message.content.strip()
                    comment = re.sub(r"Score: \d+\n+Comment: ", "", response_text).strip()
                    match = re.search(r'\d+', response_text)

                    focus_skill_comment[skill] = comment  #   # Always save the comment
                    if match:
                        focus_skills_score[skill] = int(match.group(0))
                    else:
                        inferred_prompt = f"""
                        Based on the following feedback you provided, rate the answer on a scale of 1 to 100:
                        Feedback: {response_text}

                        Please provide a score (1 to 100) that reflects the feedback.
                        """
                        inferred_response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[ 
                                {"role": "system", "content": "You are an expert evaluator who rates interview answers based on feedback given."},
                                {"role": "user", "content": inferred_prompt}
                            ],
                            max_tokens=64 ,
                            temperature = 0.4 
                        )

                        inferred_score_text = inferred_response.choices[0].message.content.strip()
                        inferred_match = re.search(r'\d+', inferred_score_text)
                        if inferred_match:
                            inferred_score = int(inferred_match.group(0))
                            focus_skills_score[skill] = inferred_score
                        else:
                            focus_skills_score[skill] = 50  # If no inferred score, default to 50

            # Generate the final output for focus skills with both score and comment
            output = []
            for skill in focus_skills:
                skill_data = {
                    "skill_name": skill,
                    "score": focus_skills_score.get(skill, "No score available"),
                    "comment": focus_skill_comment.get(skill, "No comment available")
                }
                output.append(skill_data)

            # Print or return final output
            print("Final Output:", output)
                        

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an AI Interview Summary Generator.\n"
                        "You will receive detailed evaluation comments for a candidate across multiple skills.\n"
                        "Your task is to read all the individual skill-based comments and generate one comprehensive, insightful overall evaluation paragraph summarizing the candidate’s performance.\n\n"
                        "Your response should be long and detailed, written in a clear, professional tone.\n"
                        "Do not repeat the skill names or list them again. Focus on synthesizing all the insights into one flowing summary that highlights strengths, communication ability, technical understanding, and overall competence."
                    )
                },
                {
                    "role": "user",
                    "content": f"Here are the detailed skill-wise score is :\n\n{skill}\n\n and the score and comment for the focus skills is :\n\n{focus_skills_score}\n\n and the comment is \n\n{focus_skill_comment}\n\nGenerate a single overall evaluation paragraph based on these."
                }
            ]

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=400,
                temperature = 0.4
            )

            technical_comment = response.choices[0].message.content.strip()
            print(technical_comment)







################################################# evoluation , suggestions ################################################# 
        def evaluate_student_answer(question, student_answer):
            print(f"Evaluating Answer for Question: {question}")
            prompt = {
                "role": "system",
                "content": f"""
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
                """
            }

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  
                messages=[prompt],
                max_tokens=400,
                temperature = 0.4
            )

            evaluation = response.choices[0].message.content
            evaluation = evaluation.replace("\n", " ")
            return evaluation




        def generate_feedback_for_multiple_responses(questions, input_data):

            if len(questions) != len(input_data):
                raise ValueError("Number of questions must match number of transcript entries")
            
            evaluations = []

            for i, question in enumerate(questions):
                answer = input_data[i]["transcript"]

                # Get the suggestion based on the transcript
                suggestion_prompt = {
                    "role": "system",
                    "content": (
                        "You are an AI Interview Assistant.\n"
                        "You will be given a set of interview questions along with their respective answers.\n"
                        "Your task is to analyze the answers and provide constructive suggestions for improvements.\n\n"
                        "For each answer, provide suggestions for improving the response, including but not limited to:\n"
                        "1. Clarity: How can the candidate make the answer clearer?\n"
                        "2. Depth: Are there details missing that would strengthen the answer?\n"
                        "3. Accuracy: Are there any factual inaccuracies or better ways to express ideas?\n"
                        "4. Relevance: Does the answer fully address the question, or does it stray off-topic?\n"
                        "5. Professionalism: Is the language and tone appropriate for a professional setting?\n\n"
                        "Format your output like this:\n"
                        "{\n"
                        "   [Your detailed suggestion here]\"\n"
                        "}\n\n"
                        "Do this for each question-answer pair provided."
                    )
                }
                
                suggestion_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        suggestion_prompt,
                        {
                            "role": "user",
                            "content": f"Question: {question}\nAnswer: {answer}\n\nPlease provide suggestions for this answer."
                        }
                    ],
                    max_tokens=400,
                    temperature = 0.4
                )

                suggestion = suggestion_response.choices[0].message.content.strip()

                # Get the evaluation based on the question and answer
                evaluation = evaluate_student_answer(question, answer)

                evaluations.append({
                    "id": input_data[i]["id"],
                    "transcript": answer,
                    "suggestions": suggestion,
                    "evaluation": evaluation
                })
            
            return evaluations

       
       
        transcription = generate_feedback_for_multiple_responses(Questions, transcription)
        ################################################# proctoring_report #################################################

        multi_face_count = 0
        face_not_visible_count = 0
        tab_switched = 0
        
        Exited_Full_Screen = 0

        for item in proctoring_data : 
            title = item['proctering_title']
            count = item['proctering_count']
            print(title , count)   
            if title == "Multiple Faces Detected": 
                multi_face_count = count
            if title == "No Face Detected":
                face_not_visible_count = count 
            if title == "Tab Switched":
                tab_switched = count 
            if title == "Exited Full Screen" : 
                Exited_Full_Screen = count

        print(face_not_visible_count,multi_face_count,tab_switched,Exited_Full_Screen,"____________$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        def calculate_proctoring_score(face_not_visible, multi_face, tab_switched, Exited_Full_Screen):
            score = (
                face_not_visible * 5 +
                multi_face * 10 +
                tab_switched * 20 +
                Exited_Full_Screen * 10
            )
            final = 100 - score
            return max(final, 0 )  
 
        proctoring_score = calculate_proctoring_score(
            face_not_visible_count,
            multi_face_count,
            tab_switched,
            Exited_Full_Screen
        )
        message_proctoring_comment = {
            "role": "system",
            "content": f"""
        You are a proctoring analysis assistant. Generate a professional and concise summary based on the following metrics from an online test session:
 
        - Candidate Face not visible count: {face_not_visible_count}
        - Multiple faces detected count: {multi_face_count}
        - Screen violation count (tab switches or fullscreen exits): {Exited_Full_Screen + tab_switched}
 
        Evaluate the user's behavior based on these numbers. Highlight any suspicious behavior or concerns.
        Do not include the numbers in the final comment—only describe the situation based on them using natural, human-like phrasing.
        """
        }
 
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[message_proctoring_comment],
                max_tokens=200,
                temperature = 0.4
            )
 
            # Extract and clean comment
            proctoring_comment = response.choices[0].message.content.strip().replace('"', '')
 
            print(" Proctoring Report Comment:")
            print(proctoring_comment)
            print(" Proctoring Suspicion Score :", proctoring_score)
 
        except Exception as e:
            print(" Error generating proctoring report:", e)
 
        overall_scores = skills_score.copy()  # Make a copy of skills_score to avoid modifying it
        overall_scores.update(focus_skills_score)  # Update with focus_skills_score
 
        print(overall_scores)
        formatted_data =  format_feedback(overall_scores,articulation_comment,articulation_score,body_language_comment,bodylang_score,communication_comment,transcription,etiquette_score,etiquette_comment,technical_comment,grammer_comment,grammer_score,pace_comment,pace_score,output,pronunciation_score,pronunciation_comment,proctoring_comment,proctoring_score,self_awareness_score,self_awareness_comment,face_not_visible_count,multi_face_count,tab_switched,Exited_Full_Screen)
        return formatted_data
   
    except Exception as e:
        print(f" Error computing final node summary: {e}")
        import traceback
        traceback.print_exc()
 
def format_feedback(overall_scores, articulation_comment, articulation_score, body_language_comment, bodylang_score, communication_comment,transcription,etiquette_score,etiquette_comment,technical_comment,grammer_comment,grammer_score,pace_comment,pace_score,output,pronunciation_score,pronunciation_comment,proctoring_comment,proctoring_score,self_awareness_score,self_awareness_comment,face_not_visible_count,multi_face_count,tab_switched,Exited_Full_Screen):
    formatted_json = {
        "feedback": {
            "skill_analysis": {
                "skills_score": overall_scores
            },
            "scores": {
                "subjective_analysis": {
                    "technical_skills": {
                        "title": "operational_technical_skills",
                        "comment": technical_comment
                    },
                    "behaviour_analysis": {
                        "self_awareness": {
                            "title": "self_awareness",
                            "score": self_awareness_score,
                            "comment": self_awareness_comment
                        },
                        "etiquette": {
                            "title": "etiquette",
                            "score": etiquette_score,
                            "comment": etiquette_comment
                        }
                    },
                    "focus_skills": {  # Corrected typo here
                        "skills": output
                    },
                    "communication": {
                        "grammer_score": grammer_score,
                        "grammer_comment": grammer_comment,
                        "pronounciation_score": pronunciation_score,
                        "pronounciation_comment": pronunciation_comment,
                        "pace_score": pace_score,
                        "pace_comment": pace_comment,
                        "body_language_score": bodylang_score,  # Corrected typo here
                        "body_language_comment": body_language_comment,
                        "articulation_score": articulation_score,
                        "articulation_comment": articulation_comment
                    }
                },
                "is_new":"1",
                "proctoring_report": {
                    "title": "proctoring_report",
                    "score": proctoring_score,
                    "comment": proctoring_comment,
                    "face_not_visible_count":face_not_visible_count,
                    "multi_face_count":multi_face_count,
                    "tab_switched_count":tab_switched,
                    "exited_Full_Screen_count":Exited_Full_Screen


                }
            },
            "transcription": transcription
        }
    }

    print(formatted_json, "=============================================")
    return formatted_json

 
# if __name__ == "__main__":
#     count = 0
#     test = show_results()
#     print(test)
 