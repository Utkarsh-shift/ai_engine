import json,os,torch,gc,nltk,re,whisper,math
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from resultsUI.Ai_Agent_Evaluation.audio_video_covert_and_delete import delete_all_files_in_folder,delete_all_folders_in_folder
from resultsUI.Ai_Agent_Evaluation.audio_video_covert_and_delete import convert_videos_to_frames,audio_extract,audio_process,clean_audio
from pathlib import Path
from pydub import AudioSegment
import pandas as pd
from dotenv import load_dotenv
from collections import Counter
from openai import OpenAI
 
from resultsUI.Ai_Agent_Evaluation.identify_selfwareness_question import split_questions_by_type
from decouple import config
from sentence_transformers import SentenceTransformer, util
from resultsUI.Ai_Agent_Evaluation.dictionary_used_in_agent import POWER_SET_SENTENCES
from resultsUI.Ai_Agent_Evaluation.score_maker_aiagent import articute_score_maker
from resultsUI.Ai_Agent_Evaluation.comment_fetcher_used_in_aiagent import convert_images_to_base64,get_comments_for_gpt,getcomment_etiquette,evaluate_self_awareness_combined
from resultsUI.Ai_Agent_Evaluation. report_send import format_feedback

 
client = OpenAI(api_key=config("OPENAI_API_KEY"))
 
 
 
 
################################ Suggest Certfications #############################################################
 
 
 
def get_certification_names(
    skills, focus_skills,
    articulation_comment, articulation_scores,
    bodylang_comment, body_lang_score,
    transcription, etiquette_score, etiquette_comment,
    technical_comment, grammer_score, grammer_comment,
    pace_comment, pace_score, output,
    proctoring_score, proctoring_comment,
    pronounciation_comment, Pronounciation_score,
    self_awareness_score, self_awareness_comment
):
    try:
        prompt = f"""
You are an AI career advisor.
 
The user completed a job simulation task and received the following profile:
 
**General Skills:** {', '.join(skills)}
**Focus Skills:** {', '.join(focus_skills)}
 
**Soft Skills Assessment:**
- Articulation: {articulation_scores}/10 — {articulation_comment}
- Body Language: {body_lang_score}/10 — {bodylang_comment}
- Etiquette: {etiquette_score}/10 — {etiquette_comment}
- Grammar: {grammer_score}/10 — {grammer_comment}
- Pronunciation: {Pronounciation_score}/10 — {pronounciation_comment}
- Pace: {pace_score}/10 — {pace_comment}
- Self Awareness: {self_awareness_score}/10 — {self_awareness_comment}
- Proctoring Score: {proctoring_score}/10 — {proctoring_comment}
 
**Technical Feedback:** {technical_comment}
**Transcription Output:** {transcription}
**Final Output Summary:** {output}
 "MAKE SURE THE EVALUATION IS UNDER 150 WORDS.\n"
Based on this profile, suggest **at least 2 and at most 5** relevant certifications (professional, technical, or communication-related) that will improve the user's employability.
 
 Return only the certification names as a **numbered list**. No descriptions or organizations.
"""
 
        messages = [
            {"role": "system", "content": "You are a professional career advisor AI."},
            {"role": "user", "content": prompt}
        ]
 
        print("Sending messages to OpenAI:", messages)
 
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=256,
            temperature = 0.4
        )
 
        raw_output = response.choices[0].message.content.strip()
        print("Raw OpenAI response:", raw_output)
 
        # Extract just the certification names from numbered list
        cert_names = re.findall(r'^\d+\.\s*(.+)', raw_output, re.MULTILINE)
        cert_names = cert_names[:5]
 
        if len(cert_names) < 2:
            raise ValueError("GPT returned fewer than 2 certifications.")
 
        return cert_names
 
    except Exception as e:
        print(f"Error fetching certification names: {e}")
        return []
   
 
 
 
 
 ############################ Evaluation , suggestion ####################################################################
def generate_feedback_for_multiple_responses(questions, input_data):
    print("length of questions ====================================",len(questions))
    print("length of answers =======================================",len(input_data))
    if len(questions) != len(input_data):
        raise ValueError("Number of questions must match number of transcript entries")
   
    evaluations = []
    for i, question in enumerate(questions):
        answer = input_data[i]
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
        evaluation = evaluate_student_answer(question, answer)
 
        evaluations.append({
            "question_number": i + 1,  
            "question": question,  
            "transcript": answer,
            "suggestions": suggestion,
            "evaluation": evaluation
        })
   
    return evaluations
 
def generate_overall_suggestion(evaluations):
    print("evaluations=========",evaluations)
    """
    Takes in a list of evaluations and generates a single, concise overall suggestion
    based on the individual suggestions for each response.
    """
    try:
        all_suggestions_text = "\n\n".join(
            f"Question {item['question_number']}: {item['suggestions']}"
            for item in evaluations
        )
        print(all_suggestions_text,">>>>>>>>>>>>>>>>>>>>>>>>>here is all suggestions text ++++++++++++++++++++++++++++++++")
        prompt = {
            "role": "system",
            "content": (
                "You are a professional interview coach.\n"
                "You will receive suggestions generated from multiple interview question answers.\n"
                "Your task is to summarize these into a **single overall suggestion** for the candidate.\n\n"
                "MAKE SURE THE EVALUATION IS UNDER 150 WORDS.\n"
                "Focus on:\n"
                "1. Recurring weaknesses (e.g. clarity, articulation, depth).\n"
                "2. Overall strengths.\n"
                "3. Areas for improvement across all answers.\n\n"
                "Return a clear and professional paragraph (4–6 sentences) of general feedback."
            )
        }
 
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                prompt,
                {
                    "role": "user",
                    "content": f"Here are the suggestions for individual questions:\n\n{all_suggestions_text}"
                }
            ],
            max_tokens=400,
            temperature = 0.4
        )
 
        overall_feedback = response.choices[0].message.content.strip()
        return overall_feedback
 
    except Exception as e:
        print(f"Error generating overall suggestion: {e}")
        return "Could not generate overall suggestion due to an error."
 
 
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
        "MAKE SURE THE EVALUATION IS UNDER 150 WORDS.\n"
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
 
def show_results_agent(Questions,answer,proctoring_data,skills_,focus_skills_,s3_url):
    print("Her is shs h" ,skills_ , "asfasfa" ,focus_skills_ )

    if skills_ != ['none'] : 
        skills = [*skills_[0]]
       
    else : 
        skills = skills_

    if focus_skills_ !=  ['none'] : 
        focus_skills = [*focus_skills_[0]]
    else : 
        focus_skills = focus_skills_
        
    print(skills)
    print(focus_skills)
    print(Questions)
    print(answer)
    transcription = generate_feedback_for_multiple_responses(Questions, answer)

    overall_suggesstions=generate_overall_suggestion(transcription)
 
    print(f"<><><><><><><><><><><><><><><><><><><><><><><><><><>{overall_suggesstions}<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")

 
    try:
        counts = 0  
        grammer_score,grammer_comment,pace_score,pace_comment,Pronounciation_score,pronounciation_comment,articulation_scores,articulation_comment,etiquette_score,etiquette_comment,body_lang_score,bodylang_comment = delete_and_covert_frames_and_chunks(counts)
        print("pronunciation_score",Pronounciation_score)
        print("pronunciation_comment",pronounciation_comment)
        print("articulation_score",articulation_scores)
        print("articulation_comment",articulation_comment)
        print("etiquette_score",etiquette_score)
        print("etiquette_comment",etiquette_comment)
        print("body_lang_score",body_lang_score)
        print("bodylang_comment",bodylang_comment)
        print("grammer_score",grammer_score)
        print("grammer_comment",grammer_comment)
        print(20*"*")
       
       
        print(f"Questions | {Questions}","=+++++++++++++++++++++++++++++++++++++++++++++++++",)
        print("answer are :----------------",answer)
       
        sa_qs, sa_as, normal_qs, normal_as = split_questions_by_type(Questions, answer)
        print("Self-Awareness Questions:", sa_qs)
        print("Self-Awareness Answers:", sa_as)
        print("Normal Questions:", normal_qs)
        print("Normal Answers:", normal_as)
       
        self_awareness_questions=sa_qs
        self_awareness_answer=sa_as
       
        technical_questions=normal_qs
        technical_answer=normal_as
       
        if self_awareness_questions:
            self_awreness=evaluate_self_awareness_combined(self_awareness_questions, self_awareness_answer)
            self_awareness_score = self_awreness["score"]
            self_awareness_comment = self_awreness["self_awareness"]
            print("self_awareness_score----------",self_awareness_score)
            print("self_awareness_comment----------",self_awareness_comment)
        else:
            self_awareness_score=0
            self_awareness_comment="No Question has asked for self awareness"
       
       
        skills_score = {}
        overall_scores = {}    
        focus_skills_score = {}
        focus_skill_comment = {}
       
        question_answer_dict = {}
       
        for i, question in enumerate(technical_questions):
            question_answer_dict[question] = technical_answer[i]
            print("Question-Answer Dictionary:Question-Answer Dictionary:Question-Answer Dictionary:", question_answer_dict)
           
        questions = []
        answers = []    
        for question, answer in question_answer_dict.items():
            questions.append(question)
            answers.append(answer)

        print(skills,"here are all the skills and focus skills",focus_skills,"9999999999999999999999999999999999999999999999999")
        if skills in ("none",["none"]) and focus_skills == ("none",["none"]):
            skills = ["there is no skills present"]
            focus_skills = ["there is no skills present"]
            technical_comment = "There is no skills present"
            print("Skills:", skills)
            print("Focus Skills:", focus_skills)
            print("Technical Comment:", technical_comment)
            return
    
        if skills in ("none",["none"]):
            skills = []
        if focus_skills in ("none",["none"]):
            focus_skills = []
        
        skills = list(set(skills + focus_skills))
    
        print("Questions:", Questions)
        print("Answers:", answer)
        print("Skills after merge:", skills)
    
        skills_score = {}
        overall_scores = {}    
        focus_skills_score = {}
        focus_skill_comment = {}
    
        question_answer_dict = {Questions[i]: answer[i] for i in range(len(Questions))}
        print("Question-Answer Dictionary:", question_answer_dict)
    
        for skill in skills:
            for question, ans in question_answer_dict.items():
                if skill not in focus_skills:
                    prompt = f"""You are an expert evaluator who will assess the following answer to the interview question for the skill '{skill}'.
                    The question asked was: '{question}'
                    The person's response to the interview question is: '{ans}'
    
                    Provide a score from 20 to 100."""
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are an experienced evaluator."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=64,
                        temperature = 0.4
                    )
                    response_text = response.choices[0].message.content.strip()
                    match = re.search(r'\d+', response_text)
                    score = int(match.group(0)) if match else 20
                    skills_score[skill] = score
                
                else:
                    prompt = f"""You are evaluating the answer to the question for the skill '{skill}': {question_answer_dict}
                    Provide a score (1–100) and a one-line comment."""
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are an experienced evaluator."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=64,
                        temperature = 0.4
                    )
                    response_text = response.choices[0].message.content.strip()
                    match = re.search(r'\d+', response_text)
                    comment = re.sub(r"Score: \d+\n+Comment: ", "", response_text).strip()
                    score = int(match.group(0)) if match else 50
                    focus_skills_score[skill] = score
                    focus_skill_comment[skill] = comment
                
            print(f"Score for '{skill}' (non-focus):", score)
        overall_scores = skills_score.copy()
        overall_scores.update(focus_skills_score)
    
        # If skills originally were 'none', use keys from overall_scores
        if not skills:
            skills = list(overall_scores.keys())
    
        # Final output list
        output = []
        if focus_skills:  # Only populate output if focus_skills is not empty
            for skill in focus_skills:
                skill_data = {
                    "skill_name": skill,
                    "score": focus_skills_score.get(skill, "No score available"),
                    "comment": focus_skill_comment.get(skill, "No comment available")
                }
                output.append(skill_data)
        print("Final Output:", output)
    
        # Technical comment generation
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI Interview Summary Generator. Generate an insightful summary paragraph "
                    "based on detailed evaluation comments, excluding skill names, and focused on overall strengths and areas of improvement."
                )
            },
            {
                "role": "user",
                "content": f"Skills: {skills}\nFocus Scores: {focus_skills_score}\nComments: {focus_skill_comment}"
            }
        ]
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=400,
            temperature = 0.4
        )
        technical_comment = response.choices[0].message.content.strip()
        print("Technical Comment:", technical_comment)
        print("Overall Scores:", overall_scores)















       
    ################################################# proctoring_report #################################################
       
 
 
        multi_face_count = proctoring_data[2]["proctering_count"]
        tab_switched = proctoring_data[0]["proctering_count"]
        Exited_Full_Screen = proctoring_data[1]["proctering_count"]
        cell_phone_detected = proctoring_data[3]["proctering_count"]
        multi_monitor_detected=proctoring_data[4]["proctering_count"]
        no_face_detected=proctoring_data[5]["proctering_count"]
        def calculate_proctoring_score(cell_phone_detected, multi_face, tab_switched, Exited_Full_Screen):
            score = (
                cell_phone_detected * 15 +
                multi_face * 10 +
                tab_switched * 20 +
                Exited_Full_Screen * 10
            )
            final = 100 - score
            return max(final, 0 )  
 
        proctoring_score = calculate_proctoring_score(
            cell_phone_detected,
            multi_face_count,
            tab_switched,
            Exited_Full_Screen
        )
        message_proctoring_comment = {
            "role": "system",
            "content": f"""
        You are a proctoring analysis assistant. Generate a professional and concise summary based on the following metrics from an online test session:
 
        - cell phone detected count: {cell_phone_detected}
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
            proctoring_comment = response.choices[0].message.content.strip().replace('"', '')
            print(" Proctoring Report Comment:")
            print(proctoring_comment)
            print(" Proctoring Suspicion Score :", proctoring_score)
        except Exception as e:
            print(" Error generating proctoring report:", e)
        # overall_scores = skills_score.copy()
        # overall_scores.update(focus_skills_score)
 
        # print(overall_scores)
        ################################################# evoluation , suggestions #################################################
        certifications = get_certification_names(skills, focus_skills,articulation_comment,articulation_scores,bodylang_comment,body_lang_score,transcription,etiquette_score,etiquette_comment,technical_comment,grammer_score,grammer_comment,pace_comment,pace_score,output,proctoring_score,proctoring_comment,pronounciation_comment,Pronounciation_score,self_awareness_score,self_awareness_comment)
        print(certifications)
 
 
       
        # transcription = generate_feedback_for_multiple_responses(Questions, answer)
 
        formatted_data =  format_feedback(overall_scores,articulation_comment,articulation_scores,bodylang_comment,body_lang_score,transcription,etiquette_score,etiquette_comment,technical_comment,grammer_score,grammer_comment,pace_comment,pace_score,output,proctoring_score,proctoring_comment,pronounciation_comment,Pronounciation_score,self_awareness_score,self_awareness_comment,cell_phone_detected,Exited_Full_Screen,tab_switched,multi_face_count,multi_monitor_detected,no_face_detected,s3_url,certifications,overall_suggesstions)
        return formatted_data
   
    except Exception as e:
        print(f" Error computing final node summary: {e}")
        import traceback
        traceback.print_exc()          
           
def delete_and_covert_frames_and_chunks(count):
 
    count += 1
    home = os.getcwd() 
    print("The cnt is **********************************************" , count , home)

    video_dir = home +'/resultsUI/Ai_Agent_Evaluation/ai_agent_database/ChaLearn/test'
    output_dir = home + '/resultsUI/Ai_Agent_Evaluation/ai_agent_database/ChaLearn/test_data'
    outforwav = home + '/resultsUI/Ai_Agent_Evaluation/ai_agent_database/ChaLearn/voice_data/voice_raw/test_data'
    outforlibrosa = home + '/resultsUI/Ai_Agent_Evaluation/ai_agent_database/ChaLearn/voice_data/voice_librosa/test_data'
    source_directory = home + '/resultsUI/Ai_Agent_Evaluation/ai_agent_database/ChaLearn/test'
 
    contents = os.listdir(source_directory)
    if not contents:
        print("Conitnue with the code")
    else:
        pass

    delete_all_files_in_folder(outforlibrosa)
    delete_all_files_in_folder(outforwav)
    delete_all_folders_in_folder(output_dir)
 
    convert_videos_to_frames(video_dir,output_dir)
    audio_extract(video_dir,outforwav)
    audio_process(mode= 'librosa' ,aud_dir=outforwav , saved_dir=outforlibrosa)
    audio_path = home + '/resultsUI/Ai_Agent_Evaluation/ai_agent_database/ChaLearn/voice_data/voice_raw/test_data/final_video.wav'
    clean_audio(audio_path)
    result=test()
    print("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee`",result)
    return result   
 
def test():
        torch.cuda.empty_cache()
        home = os.getcwd()
        
        folderpath = home + "/resultsUI/Ai_Agent_Evaluation/ai_agent_database/ChaLearn/voice_data/voice_raw/test_data"
        audio_list = os.listdir(folderpath)
        videoPath  = r"resultsUI/Ai_Agent_Evaluation/ai_agent_database/ChaLearn/test"
       
        for idx, audio in enumerate(audio_list):
            file_path = os.path.join(folderpath, audio)
            print(file_path)
            if os.path.isfile(file_path) and ".praat" not in file_path:
                videoPath  = r"resultsUI/Ai_Agent_Evaluation/ai_agent_database/ChaLearn/test"
                df_unigrams ,sentiment_score_value , sentiment_comment_value , grammer_score , grammer_comment , pace_score  ,pace_comment ,Pronounciation_score,pronounciation_comment,articulation_scores,articulation_comment,final_transcription= evaluate_data_from_audio(file_path)
                
               
                ####################### body language #############################################
                frames_dir_path = r"resultsUI/Ai_Agent_Evaluation/ai_agent_database/ChaLearn/test_data"
                folderaud = os.path.basename(file_path).split('/')[-1]
                folderaud = folderaud.replace(".wav","")
                frames_folder = frames_dir_path
             

                print("The frame forder is ****************************" , frames_folder)
                home = os.getcwd()
                video_dir= home + "/resultsUI/Ai_Agent_Evaluation/ai_agent_database/ChaLearn/test"
                video_file = os.path.join(video_dir, folderaud + ".mp4")      
                print("video file path is " , video_file)
             
                base_encoder_frames = convert_images_to_base64(frames_folder)
                # print(base_encoder_frames,"=++++++++++++++++++++++++++++++++++")
                body_langauage_prompt = "You are an interviewer conducting a candidate interview. Based on the video, provide a summary of the person's body language in the interview situation. Do not offer any suggestions or advice; just describe the person's body language observed during the interview.comment only in one line without header."
                body_lang_score,bodylang_comment=get_comments_for_gpt(base64Frames=base_encoder_frames,prompt=body_langauage_prompt,transcript=final_transcription, typeo = "body_langauage")
                print("body language score is ",body_lang_score)
                print("body langauge comment is ",bodylang_comment)
               
               
                ########################## etiquette #################################################
                etiquette_prompt = "You are an interviewer conducting a candidate interview. Based on the video, provide a summary of the person's Etiquette in the interview situation. Do not offer any suggestions or advice; just describe the person's Etiquette observed during the interview."
                etiquette_score,etiquette_comment = getcomment_etiquette(base64Frames=base_encoder_frames,prompt=etiquette_prompt,transcript=final_transcription , typeo = "Etiquette")
                print("here is etiquette score",etiquette_score)
                print("here is etiquette comment",etiquette_comment)
                print(grammer_score,grammer_comment,pace_score,pace_comment,Pronounciation_score,pronounciation_comment,articulation_scores,articulation_comment)
               
        return grammer_score,grammer_comment,pace_score,pace_comment,Pronounciation_score,pronounciation_comment,articulation_scores,articulation_comment,etiquette_score,etiquette_comment,body_lang_score,bodylang_comment
       
 
def split_audio(audio_path, chunk_length_ms=15000):
    audio = AudioSegment.from_file(audio_path)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunk_path = f"chunk_{i // chunk_length_ms}.mp4"
        chunk.export(chunk_path, format="mp4")
        chunks.append(chunk_path)
    del audio
    gc.collect()
    return chunks  
 
 
def scale_wpm_to_score(wpm, min_wpm=120, max_wpm=180, k=0.01):  # Lower k
    ideal_wpm = (min_wpm + max_wpm) / 2
 
    if min_wpm <= wpm <= max_wpm:
        return 100
 
    distance = abs(wpm - ideal_wpm)
    penalty = math.exp(-k * distance)
    score = 100 * penalty
 
    return round(score, 2)
   
def get_comment(prompt):
    message={"role": "system",
            "content": prompt}
    chat_completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages = [message],
            max_tokens=128,
            temperature = 0.4
        )
    finish_reason = chat_completion.choices[0].finish_reason
    newdata = chat_completion.choices[0].message.content
    return newdata
 
 
def scale_avg_logprob_to_score(avg_logprob, ideal_min=-0.1, ideal_max=0.0, batch_size=0.02, batch_penalty=1):
    if ideal_max <= avg_logprob <= ideal_min:
        return 100
    if avg_logprob < ideal_min:
        distance = ideal_min - avg_logprob
    else:
        distance = avg_logprob - ideal_max
    quotient = int(distance // batch_size)
    remainder = distance % batch_size
    penalty = sum([batch_penalty * (2 ** i) for i in range(quotient)]) + remainder
    score = max(0, 100 - penalty)
    if score == 0:
        score = 40
    return score
 

def transcribe_chunk_batch_new(chunks, model):
    import os
    transcriptions = []
    total_duration = 0
    total_words = 0
    avg_logprobs = []
    print("[DEBUG] Starting transcription of audio chunks...")

    try:
        for idx, i in enumerate(chunks):
            print(f"[DEBUG] Processing chunk {idx + 1}/{len(chunks)}: {i}")
            audio = whisper.load_audio(i)
            audio = whisper.pad_or_trim(audio)

            print("[DEBUG] Transcribing audio using model...")
            transcription = model.transcribe(audio, language="en")

            try:
                final_transcription = transcription['text']
                print(f"[DEBUG] Transcription text: {final_transcription}")
            except KeyError:
                final_transcription = transcription.get('text', '')
                print("[WARNING] KeyError accessing 'text'; used .get fallback")

            words = final_transcription.split()
            num_words = len(words)
            total_words += num_words
            transcriptions.append(final_transcription)

            print(f"[DEBUG] Number of words in chunk: {num_words}")

            for segment in transcription['segments']:
                segment_duration = segment['end'] - segment['start']
                total_duration += segment_duration
                avg_logprobs.append(segment['avg_logprob'])

            try:
                os.remove(i)
                print(f"[DEBUG] Deleted processed file: {i}")
            except Exception as file_error:
                print(f"[WARNING] Could not delete file {i}: {file_error}")

        full_transcription = "".join(transcriptions)
        print("\n[INFO] Final full transcription generated.")
        print("The full transcript is " , full_transcription)
        print("[INFO] Total words:", total_words)
        print("[INFO] Total duration (seconds):", total_duration)

        duration_in_minutes = total_duration / 60
        wpm = total_words / duration_in_minutes if duration_in_minutes > 0 else 0
        print(f"[INFO] Words per minute (WPM): {wpm:.2f}")

        try:
            wpm_score = scale_wpm_to_score(wpm, min_wpm=50, max_wpm=130)
            print(f"[DEBUG] WPM Score: {wpm_score}")
        except Exception as e:
            print(f"[ERROR] Error scaling WPM to score: {e}")
            wpm_score = 0  # Default in case of error

        pace_prompt = (
            f"If the peron is speaking at a average word per minute rate of {wpm}, and is given a score of {wpm_score}, "
            f"then comment about the pace of the person in speech in one line only. "
            f"Also mention the speed of the person in speech. The ideal speed is 140-160. Comment only in one line."
        )
        pace_comment = get_comment(pace_prompt)
        print("[INFO] Pace comment:", pace_comment)

        print(f"[DEBUG] avg_logprobs list length: {len(avg_logprobs)}")
        avg_logprob_mean = sum(avg_logprobs) / len(avg_logprobs) if avg_logprobs else -1
        print("[INFO] Average log probability:", avg_logprob_mean)

        pronunciation_score = scale_avg_logprob_to_score(avg_logprob_mean)
        print("[INFO] Pronunciation score:", pronunciation_score)

        pronunciation_prompt = (
            f"If the person's pronunciation is given by the score {pronunciation_score}, and the average log probability "
            f"of the pronunciation is {avg_logprob_mean}, the ideal range is kept as -0.1 to -0.2 then comment about the "
            f"pronunciation of the person in one line only. Do not mention average log probability or ideal range in the comment."
        )
        pronunciation_comment = get_comment(pronunciation_prompt)
        print("[INFO] Pronunciation comment:", pronunciation_comment)

        return full_transcription, wpm_score, pronunciation_score, pace_comment, pronunciation_comment

    except Exception as e:
        print("[ERROR] Exception occurred in transcribe_chunk_batch_new:", e)
        return "", 0, 0, "Error in pace comment", "Error in pronunciation comment"




# def transcribe_chunk_batch_new(chunks, model):
#     transcriptions = []
#     total_duration = 0
#     total_words=0
#     avg_logprobs = []
#     try:
#         for i in chunks:
#             audio = whisper.load_audio(i)
#             audio = whisper.pad_or_trim(audio)
#             transcription = model.transcribe(audio, language="en")
#             try :
#                 final_transcription = transcription['text']
#             except :
#                 final_transcription = transcription.get('text','')
#             words = final_transcription.split()
#             transcriptions.append(final_transcription)
#             num_words = len(words)
#             total_words += num_words
#             for segment in transcription['segments']:
#                 segment_duration = segment['end'] - segment['start']
#                 total_duration += segment_duration
#                 avg_logprobs.append(segment['avg_logprob'])
#             os.remove(i)
#         full_transcription = "".join(transcriptions)
#         print( "Total words in transcription:", total_words,"__________________________________________", full_transcription)
#         print("Total duration in seconds:", total_duration,"############################################")
#         duration_in_minutes = total_duration/60
#         if duration_in_minutes > 0:
#             wpm = total_words / duration_in_minutes
#         else:
#             wpm = 0
 
#         print(f"Total Words: {total_words}, Duration: {total_duration} seconds")
#         print(f"Words per minute (WPM): {wpm:.2f}")
#         try :
#            wpm_score = scale_wpm_to_score(wpm , min_wpm=50, max_wpm=130)
#         except Exception as e :
#             print(e)
#         print(f"wpm Score : {wpm_score}")
 
 
#         pace_prompt =  f"""If the peron is speaking at a average word per minute rate of {wpm}, and is given a score of {wpm_score} ,
#         then comment about the pace of the person in speech in one line only.also mention the speed of the person in speech. and the idel speed is 140-160.comment only in one line."""
       
#         pace_comment = get_comment(pace_prompt)
#         print("pace comment is :",pace_comment)
       
       
#         print(avg_logprobs,"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",len(avg_logprobs))
#         avg_logprob_mean = sum(avg_logprobs) / len(avg_logprobs)
#         print("The average_logprob is calculates as" , avg_logprob_mean)
#         pronunciation_score = scale_avg_logprob_to_score(avg_logprob_mean )
#         print("The pronunciation_score is given as " , pronunciation_score)
#         pronunciation_prompt=f"If the person's pronunciation is given by the score {pronunciation_score}, and the average log probability of the pronunciation is {avg_logprob_mean}, the ideal range is kept as -0.1 to -0.2 then comment about the pronunciation of the person in one line only. Do not mention avgerage log probability or ideal range in the comment "
#         pronunciation_comment = get_comment(pronunciation_prompt)
#         print("The WPM_score is given as" , wpm_score)
 
#         return full_transcription, wpm_score, pronunciation_score , pace_comment , pronunciation_comment
#     except Exception as e:
#         print("Exception occurred:", e)  
 
 
def evaluate_data_from_audio(audio_file):
    chunks = split_audio(audio_file)
    chunks.sort()
   
    model = whisper.load_model("large-v3", download_root=os.path.join(os.getcwd(), "whisper"))
    final_transcription , pace_score , pronunciation_score ,pace_comment , pronunciation_comment = transcribe_chunk_batch_new(chunks,model)
 
    torch.cuda.empty_cache()
    load_dotenv()
   
   
   
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    message_score = {
        "role": "system",
        "content": f""""calculate the grammar score for the text: "{final_transcription}", out of 100 consider vocabulary, grammar mistakes, sentence formation only give response in digit and no text, give minimum 10 score."""
 
    }
    message_comment = {
        "role": "system",
        "content": f""""comment on the grammar issues for the text: "{final_transcription}", consider vocabulary, grammar mistakes, sentence formation. ALSO SUGGEST IMPROVEMENTS WITH POWER PHRASES AND VOCABULARY.comment only in one line."""
 
    }
    try:
        chat_completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[message_score],
            temperature = 0.4
        )
        newdata = chat_completion.choices[0].message.content
        print(newdata)
        double_digit = re.findall(r'\b\d{2}\b', newdata)
        grammer_score = int(double_digit[0])
        chat_completion_comment = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[message_comment],
            temperature = 0.4
        )
        newdata = chat_completion_comment.choices[0].message.content
        grammer_comment = newdata
        grammer_comment=grammer_comment.replace('\"',"")
 
        print("Grammar Score:", grammer_score)
        print("Grammar Comment:", grammer_comment)
 
        ########################## Pronunciation SCORE /COMMENT  ################################################################################################
 
        print("Pronunciation Comment:", pronunciation_comment)
        print("Pronunciation Score:", pronunciation_score)
 
    except Exception as e:
        print("Error while evaluating pronunciation:", e)
 
    example_json_structure = """{
    "sentiment_score":<<score>>,
    "sentiment_comment": <<comment>>,    
    }"""
 
        ################################################ sentiment score / comment ##################################################################################
 
    sentiment_analysis_message = {
        "role": "system",
       "content": f"""Calculate the sentiment score and provide a comment for the text "{final_transcription}" out of 100. Consider factors such as the emotional tone, positivity/negativity, overall sentiment, and mood conveyed in the text. Provide the response in the following JSON format:
        {example_json_structure}
        """
    }
 
    try :
        sentiment_analysis_response = client.chat.completions.create(model="gpt-3.5-turbo", messages=[sentiment_analysis_message], temperature = 0.4)
        sentiment_choice = sentiment_analysis_response.choices[0].message.content
        sentiment_choice_data = json.loads(sentiment_choice)
        sentiment_score_value = sentiment_choice_data["sentiment_score"]
        sentiment_comment_value = sentiment_choice_data["sentiment_comment"]
 
    except Exception as error:
        print(f"Error occurred: {error}")
        sentiment_score_value, sentiment_comment_value = None, None
    gc.collect()
    
    tokens = word_tokenize(final_transcription)
    stop_words = set(stopwords.words('english'))
    tokenized_corpus = [word for word in tokens if word.lower() not in stop_words]
    unigrams = list(ngrams(tokenized_corpus, 1))
   
    unigram_freq = Counter(unigrams)
 
    df_unigrams = pd.DataFrame(list(unigram_freq.items()), columns=['Word', 'Frequency'])
    df_unigrams = df_unigrams.sort_values(by='Frequency', ascending=False).reset_index(drop=True)
    torch.cuda.empty_cache()
 
    ################################# Articulation Score ###########################################################################
       
    model = SentenceTransformer('all-MiniLM-L6-v2')
    transcript_embed = model.encode(final_transcription, convert_to_tensor=True)
    power_embeds = model.encode(list(POWER_SET_SENTENCES), convert_to_tensor=True)
    cos_scores = util.cos_sim(transcript_embed, power_embeds)
    power_score = 0  
    for i, score in enumerate(cos_scores[0]):
        if score > 0.7:  
            matched_sentence = list(POWER_SET_SENTENCES)[i]
            print(f"Matched: {matched_sentence} (Score: {score:.2f})")
            power_score += 1
   
   
    articulation_scores , articulation_comment =articute_score_maker(power_score,df_unigrams, final_transcription)
    return df_unigrams ,sentiment_score_value , sentiment_comment_value , grammer_score , grammer_comment , pace_score ,pace_comment ,pronunciation_score , pronunciation_comment,articulation_scores,articulation_comment,final_transcription
 