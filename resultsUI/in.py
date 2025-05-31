import os
import cv2
import typing
from moviepy.editor import ImageSequenceClip
import numpy as np
import gradio as gr
from typing import List, Dict, Generator, Optional, Tuple
from api.audio import STTManager, TTSManager
from api.llm import LLMManager
from api.avllm import AVLLMManager
from utils.config import Config
from resources.prompts import prompts

from ui.coding import get_problem_solving_ui
# from ui.instructions import get_instructions_ui
from openai import OpenAI
import json
from utils.params import default_audio_params
import threading
from script.run_exp import dpmain
from dpcv.experiment.score_maker import ComputingValues



import cv2
import pyaudio
import wave
import threading
import time
import subprocess
import os
import gradio as gr


# def GPTCall( message):
    # client = OpenAI(api_key = 'sk-proj-5J6AGNVXQEJ6Ji9NATgCT3BlbkFJblejOzq7DI9TgfagRxd7')


    # chat_completion = client.chat.completions.create(
    #     model="gpt-40-mini",
    #     response_format={"type":"json_object"},
    #     messages=message
    # )

    # finish_reason = chat_completion.choices[0].finish_reason

    # if(finish_reason == "stop"):
    #     data = chat_completion.choices[0].message.content
    #     newdata = json.loads(data)
    #     print(type(newdata))
    #     return newdata
    
class VideoRecorder():
    
    # Video class based on openCV 
    def __init__(self):
        self.open = False  # Flag to check if recording is ongoing
        self.device_index = 0
        self.fps = 25              # fps should be the minimum constant rate at which the camera can
        self.fourcc = "XVID"       # capture images (with no decrease in speed over time; testing is required)
        self.frameSize = (640,480) # video formats and sizes also depend and vary according to the camera used
        self.video_filename = "./videodata/temp_video.mp4"
        self.video_cap = None
        self.video_writer = None
        self.frame_counts = 1
        self.start_time = None
    
    # Video starts being recorded 
    def record(self):
        self.open = True
        self.video_cap = cv2.VideoCapture(self.device_index)
        self.video_writer = cv2.VideoWriter(self.video_filename, cv2.VideoWriter_fourcc(*self.fourcc), self.fps, self.frameSize)
        self.start_time = time.time()
        
        while self.open:
            ret, video_frame = self.video_cap.read()
            if ret:
                self.video_writer.write(video_frame)
                self.frame_counts += 1

    # Finishes the video recording therefore the thread too
    def stop(self):
        self.open = False
        if self.video_writer:
            self.video_writer.release()
        if self.video_cap:
            self.video_cap.release()
        cv2.destroyAllWindows()

    # Checks if recording is ongoing
    def is_recording(self):
        return self.open


class AudioRecorder():

    # Audio class based on pyAudio and Wave
    def __init__(self):
        self.open = False  # Flag to check if recording is ongoing
        self.rate = 44100
        self.frames_per_buffer = 1024
        self.channels = 2
        self.format = pyaudio.paInt16
        self.audio_filename = "./videodata/temp_audio.wav"
        self.audio = None
        self.stream = None
        self.audio_frames = []

    # Audio starts being recorded
    def record(self):
        self.open = True
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=self.format,
                                      channels=self.channels,
                                      rate=self.rate,
                                      input=True,
                                      frames_per_buffer=self.frames_per_buffer)
        
        while self.open:
            data = self.stream.read(self.frames_per_buffer)
            self.audio_frames.append(data)

    # Finishes the audio recording therefore the thread too    
    def stop(self):
        self.open = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate()

            waveFile = wave.open(self.audio_filename, 'wb')
            waveFile.setnchannels(self.channels)
            waveFile.setsampwidth(self.audio.get_sample_size(self.format))
            waveFile.setframerate(self.rate)
            waveFile.writeframes(b''.join(self.audio_frames))
            waveFile.close()

    # Checks if recording is ongoing
    def is_recording(self):
        return self.open


def start_AVrecording(filename):
    global video_thread
    global audio_thread
    
    video_thread = VideoRecorder()
    audio_thread = AudioRecorder()

    audio_thread_record = threading.Thread(target=audio_thread.record)
    video_thread_record = threading.Thread(target=video_thread.record)

    audio_thread_record.start()
    video_thread_record.start()

    return filename

def convert_avi_to_mp4(avi_file_path, output_name):
    os.popen("ffmpeg -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 '{output}.mp4'".format(input = avi_file_path, output = output_name))
    return True 


import random
import string

def generate_random_string():
    # Choose from all lowercase letters, uppercase letters, digits, and punctuation
    characters = string.ascii_letters + string.digits 
    # Generate a random string
    random_string = ''.join(random.choice(characters) for _ in range(8))
    return random_string

def stop_AVrecording(filename):
    
    global video_thread
    global audio_thread

    audio_thread.stop() 
    while audio_thread.is_recording():
        time.sleep(0.1)  # Wait until audio recording stops
    
    video_thread.stop() 
    while video_thread.is_recording():
        time.sleep(0.1)  # Wait until video recording stops

    frame_counts = video_thread.frame_counts
    elapsed_time = time.time() - video_thread.start_time
    recorded_fps = frame_counts / elapsed_time
    print( "total frames " + str(frame_counts))
    print( "elapsed time " + str(elapsed_time))
    print( "recorded fps " + str(recorded_fps))
    filename = "./datasets/ChaLearn/test/" + filename
    # Merging audio and video signal
    
    if abs(recorded_fps - 6) >= 0.01:    # If the fps rate was higher/lower than expected, re-encode it to the expected
        print( "Re-encoding")
        cmd = "ffmpeg -r " + str(recorded_fps) + " -i ./videodata/temp_video.mp4 -pix_fmt yuv420p -r 6 ./videodata/temp_video2.mp4"
        subprocess.call(cmd, shell=True)
    
        print( "Muxing")
        cmd = "ffmpeg -ac 2 -channel_layout stereo -i ./videodata/temp_audio.wav -i ./videodata/temp_video2.mp4 -pix_fmt yuv420p " + filename + ".mp4"
        subprocess.call(cmd, shell=True)

        # print("Converting to mp4")
        # f_name=generate_random_string()
        # f_name = "/home/almabay/Documents/AVI-PA/DeepPersonality/Interviewer/datasets/ChaLearn/test/" +f_name 
        # convert_avi_to_mp4("./videodata/Default_user.avi", f_name)
        
        # filename=filename +".avi"

        # cmd="ffmpeg -i '{filename}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 '{output}.mp4'"

    else:
        print( "Normal recording\nMuxing")
        cmd = "ffmpeg -ac 2 -channel_layout stereo -i ./videodata/temp_audio.wav -i ./videodata/temp_video.avi -pix_fmt yuv420p " + filename + ".avi"
        subprocess.call(cmd, shell=True)
    
    return "Recorded successfully "
# Example usage



# Required and wanted processing of final files
def file_manager(filename):
    local_path = "/home/almabay/Documents/Interviewer/videodata"

    if os.path.exists(str(local_path) + "/temp_audio.wav"):
        os.remove(str(local_path) + "/temp_audio.wav")
    
    if os.path.exists(str(local_path) + "/temp_video.avi"):
        os.remove(str(local_path) + "/temp_video.avi")

    if os.path.exists(str(local_path) + "/temp_video2.avi"):
        os.remove(str(local_path) + "/temp_video2.avi")

    if os.path.exists(str(local_path) + "/" + filename + ".avi"):
        os.remove(str(local_path) + "/" + filename + ".avi")
    
    # if os.path.exists(str(local_path) + "/" + filename + ".avi"):
        # os.remove(str(local_path) + "/" + filename + ".avi")
    # f_name=generate_random_string()
    # f_name = "/home/almabay/Documents/AVI-PA/DeepPersonality/Interviewer/datasets/ChaLearn/test" +f_name + ".mp4"
    # convert_avi_to_mp4("./videodata/Default_user.avi", f_name)
    # print("-------------------------------------------------u9jhyhybhybybybb")
    # else:
    #     print("errobdbhbdsbhd--------------------")


radio = gr.Radio(choices=["Start Recording", "Stop Recording"], label="Recording Control")  # Start/Stop action

# Gradio Interface
def gradio_interface():
    filename = generate_random_string()
    # file_manager(filename)
    
    def toggle_recording(choice):
        if choice == "Start Recording":
            start_AVrecording(filename)
            return "Recording started"
        elif choice == "Stop Recording":
            stop_AVrecording(filename)
            return "Recorded Successfully"


    return gr.Interface(
        fn=toggle_recording,
        inputs=radio,
        outputs=gr.Textbox(label="Recording Info"),
        live=True,
        title="Audio and Video Recorder",
        description="Select 'Start Recording' to begin recording audio and video. Select 'Stop Recording' to finish recording.",
        
    )



def support( testdic )-> List[Dict[str, str]]:
    """
    Prepare messages to end the interview and generate feedback.
    """
    #transcript = [f"{message['role'].capitalize()}: {message['content']}" for message in chat_history[1:]]
    system_prompt = testdic
    print("- - -  - - - - - - -",system_prompt)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Grade the interview based on the transcript provided and give feedback."},
    ]


def test12(testdic ) -> Generator[str, None, None]:
    """
    End the interview and get feedback from the LLM.
    """
    message = support(testdic)
    from dotenv import load_dotenv
    load_dotenv()

    client = OpenAI(api_key =os.getenv("OPENAI_API_KEY"))
    message1={'role': 'system', 'content':message[0]['content']["audio_prompt"]}
    message2={'role': 'system', 'content':message[0]['content']["video_prompt"]}
    message3={'role': 'system' ,'content':message[0]['content']['language_prompt']}
    # print("\n|\n|\n|\n|\n|\n|\n|v",message)  # prompt returned from newprompt file 

    message=[
    {
        "role": "system",
        "content": """
You are an AI system designed to provide interview feedback in JSON format. Please generate a JSON response following the structure below.

{
    "feedback": {
        "speaking_performance": {
            "confidence_and_stability": {
                "f0_semitone_mean": <value>,
                "comment": "<comment>"
            },
            "nervousness": {
                "jitter_local_mean": <value>,
                "comment": "<comment>"
            },
            "voice_clarity": {
                "harmonics_to_noise_ratio": <value>,
                "comment": "<comment>"
            },
            "pronunciation": {
                "f1_frequency_mean": <value>,
                "comment": "<comment>"
            },
            "voice_quality_and_emotion": {
                "shimmer_local_db_mean": <value>,
                "comment": "<comment>"
            },
            "communication_effectiveness": {
                "loudness_mean": <value>,
                "comment": "<comment>"
            },
            "psychological_state": {
                "alpha_ratio_mean": <value>,
                "comment": "<comment>"
            },
            "voice_brilliance": {
                "hammarberg_index_mean": <value>,
                "comment": "<comment>"
            },
            "energy_distribution": {
                "slope_v0_500_mean": <value>,
                "comment": "<comment>"
            },
            "speech_fluency": {
                "voiced_segments_per_sec": <value>,
                "comment": "<comment>"
            },
            "sustained_speech": {
                "mean_voiced_segment_length_sec": <value>,
                "comment": "<comment>"
            },
            "hesitations": {
                "mean_unvoiced_segment_length": <value>,
                "comment": "<comment>"
            }
        },
        "visual_analysis": {
        "engagement":{
            "inner_brow_raiser_mean": <<value>>,
            "engagement_comment": <<comment>>,
        },
        "surprise_curiosity":{
            "outer_brow_raiser_mean": <<value>>,
            "surprise_curiosity_comment":<<comment>>,
        },
        "negative_emotions":{
            "brow_lowerer_mean": <<value>>,
            "negative_emotion_comment": <<comment>>,
        },
        "alertness":{
            "upper_lid_raiser_mean": <<value>>,
            "alertness_comment": <<comment>>,
        },
        "positive_expression":{
            "cheek_raiser_mean": <<value>>,
            "positive_expression_comment": <<comment>>,
        },
        "focus_concentrator":{
            "lid_tightener_mean": <<value>>,
            "focus_concentration_comment": <<comment>>,
        },
        "disgust":{
            "nose_wrinkler_mean": <<value>>,
            "disgust_comment": <<comment>>,
        },
        "disgust_contempt":{
            "upper_lip_raiser_mean": <<value>>,
            "disgust_contempt_comment": <<comment>>,
        },
        "negative_emotion_2":{
            "nasolabial_deepener_mean": <<value>>,
            "negative_emotion_comment_2": <<comment>>,
        },
        "positive_emotion_2":{
            "lip_corner_puller_mean": <<value>>,
            "positive_emotion_comment_2": <<comment>>,
        },
        "emotion_comment":{
            "sharp_lip_puller_mean": <<value>>,
            "emotion_comment": <<comment>>,
        },
        "mixed_emotion":{
            "dimpler_mean": <<value>>,
            "mixed_emotion_comment": <<comment>>,
        },
        "negative_emotion_3":{
            "lip_corner_depressor_mean": <<value>>,
            "negative_emotion_comment_3":<<comment>>,
        },
        "sadness_regret_emotion":{
            "lower_lip_depressor_mean": <<value>>,
            "sadness_regret_comment": <<comment>>,
        },
        "mixed_negative_emotion":{
            "chin_raiser_mean": <<value>>,
            "mixed_negative_emotion_comment": <<comment>>,
        },
        "contemplation_expression":{
            "lip_pucker_mean": <<value>>,
            "contemplation_comment": <<comment>>,
        },
        "playful_teasing_emotion":{
            "tongue_show_mean": <<value>>,
            "playful_teasing_comment": <<comment>>,
        },
        "tension_anxiety_emotion":{
            "lip_stretcher_mean": <<value>>,
            "tension_anxiety_comment": <<comment>>,
        },
        "determination_focus_expression":{
            "lip_funneler_mean": <<value>>,
            "determination_focus_comment": <<comment>>,
        },
        "negative_emotion_4":{
            "lip_tightener_mean": <<value>>,
            "negative_emotion_comment_4": <<comment>>,
        },
        "stress_determination_emotion":{
            "lip_pressor_mean": <<value>>,
            "stress_determination_comment": <<comment>>,
        },
        "interest_readiness_expression":{
            "lips_part_mean": <<value>>,
            "interest_readiness_comment": <<comment>>,
        },
        "shock_expression":{
            "jaw_drop_mean": <<value>>,
            "shock_amazement_comment": <<comment>>,
        },
        "shock_emotion":{
            "mouth_stretched_mean": <<value>>,
            "shock_effort_comment": <<comment>>,
        },
        "anxiety_expression":{
            "lip_bite_mean": <<value>>,
            "anxiety_nervousness_comment": <<comment>>,
        },
        "anger_expression":{
            "nostril_dilator_mean": <<value>>,
            "anger_excitemen_fear_comment": <<comment>>,
        },
        "disgust_anger":{
            "nostril_compressor_mean": <<value>>,
            "disgust_disdain_anger_comment": <<comment>>,
        },
        "left_engagement":{
            "left_inner_brow_raiser_mean": <<value>>,
            "left_engagement_comment": <<comment>>,
        },
        "right_engagement":{
            "right_inner_brow_raiser_mean": <<value>>,
            "right_engagement_comment": <<comment>>,
        },
        "left_surprise_emotion":{
            "left_outer_brow_raiser_mean": <<value>>,
            "left_surprise_curiosity_comment": <<comment>>,
        },
        "right_surpurise_emotion":{
            "right_outer_brow_raiser_mean": <<value>>,
            "right_surprise_curiosity_comment": <<comment>>,
        },
        "left_negative_emotion":{
            "left_brow_lowerer_mean": <<value>>,
            "left_negative_emotion_comment": <<comment>>,
        },
        "right_negative_emotion":{
            "right_brow_lowerer_mean": <<value>>,
            "right_negative_emotion_comment": <<comment>>,
        },
        "left_positive_expression":{
            "left_cheek_raiser_mean": <<value>>,
            "left_positive_expression_comment": <<comment>>,
        },
        "right_positive_expression":{
            "right_cheek_raiser_mean": <<value>>,
            "right_positive_expression_comment": <<comment>>,
        },
        "left_disgust":{
            "left_upper_lip_raiser_mean": <<value>>,
            "left_disgust_contempt_comment": <<comment>>,
        },
        "right_disgust":{
            "right_upper_lip_raiser_mean": <<value>>,
            "right_disgust_contempt_comment": <<comment>>,
        },
        "left_negative_emotion_2":{  
            "left_nasolabial_deepener_mean": <<value>>,
            "left_negative_emotion_comment_2": <<comment>>,
        },
        "right_negative_emotion_2":{
            "right_nasolabial_deepener_mean": <<value>>,
            "right_negative_emotion_comment_2": <<comment>>,
        },
        "left_mixed_emotion":{
            "left_dimpler_mean": <<value>>,
            "left_mixed_emotion_comment": <<comment>>,
        },
        "right_mixed_emotion":{
            "right_dimpler_mean": <<value>>,
            "right_mixed_emotion_comment": <<comment>>,
        } },
        "Focus_analysis":  {
            "headpose_direction": {
                "Up": <<value>>,
                "Down": <<value>>,
                "Left": <<value>>,
                "Right": <<value>>,
                "Straight": <<value>>,
                "headpose_comment": "<<comment>>"
            },

            "eyegaze_direction": {
                "blink": <<value>>,
                "Looking center": <<value>>,
                "Looking right": <<value>>,
                "Looking left": <<value>>
            },
            "eyegaze_comment": "<<comment>>"
            },
               



        "suggestions_for_improvement": [
            "<suggestion1>",
            "<suggestion2>",
            "<suggestion3>",
            "<suggestion4>"
        ]
    },
    "overall_assessment": {
        "suitability": "<comment>",
        "strengths": [
            "<strength1>",
            "<strength2>",
            "<strength3>"
        ],
        "areas_for_improvement": [
            "<improvement1>",
            "<improvement2>"
        ]
    },
    "ocean_values_analysis": {
        "ocean_values": [<value1>, <value2>, <value3>, <value4>, <value5>],
        "comment": "<comment>"
    },
    "vocabulary_analysis": {
        "bigram_frequency": {
            "bigram1": {"words": "<word1>, <word2>", "frequency": <value>},
            "bigram2": {"words": "<word1>, <word2>", "frequency": <value>}
        },
        "unigram_frequency": {
            "unigram1": {"word": "<word>", "frequency": <value>},
            "unigram2": {"word": "<word>", "frequency": <value>}
        },
        "audio_sentiment_analysis":{
        "sentiment_comment":<<comment>>
        },
        "comment": "<comment>"
    },
    "Strengths":[
    <<strenght1>>
    <<strenght2>>
    <<strenght3>>
    <<strenght4>>
    ],
    "Weakness":[
    <<weakness1>>
    <<weakness2>>
    <<weakness3>>
    <<weakness4>>
    
    
    ]
}
"""

   
    },
    message1,
    message3,
    message2,

]
    print("-------------------------------------------------------------------------------------------")
    chat_completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_format={"type":"json_object"},
        messages = message,
        temperature = 0.4
    )
    print("|\n|\n|\n|\n|\n|\n|\n|\n|v",chat_completion)
    finish_reason = chat_completion.choices[0].finish_reason

    if(finish_reason == "stop"):
        data = chat_completion.choices[0].message.content
        newdata = json.loads(data)
        print(type(newdata))
        return newdata
    
def show_results():
    allprompts , final_score=dpmain()
    results = test12(allprompts)
    
    print("Type of the response -------------------------------------------------------------",type(results))
    suggestions_for_improvement = results['feedback']['suggestions_for_improvement']
    overall_assessment = results['overall_assessment']

    # Create a combined dictionary for display
    # display_data = {
    #     "suggestions_for_improvement": suggestions_for_improvement,
    #     "overall_assessment": overall_assessment
    # }
    display_data=results
    # Pretty print the JSON
    pretty_json = json.dumps(display_data, indent=4)

    return pretty_json

# Function to create the UI and display results
def avi_score():
    def display_results():
        # Call the show_results function and get the results
        result_json = show_results()
        result_dict = json.loads(result_json)
        
        # Extract parts for display
        suggestions = result_dict['feedback']['suggestions_for_improvement']
        overall_assessment = result_dict["overall_assessment"]
        all_result=result_dict
        ocean_value_analysis=result_dict["ocean_values_analysis"]
        vocab_analysis=result_dict["vocabulary_analysis"]
        visual_analysis=result_dict['feedback']["visual_analysis"]
        speking_p=result_dict["feedback"]["speaking_performance"]
        focus_analysis=result_dict["feedback"]["Focus_analysis"]
        suggestions_str = json.dumps(suggestions, indent=4)
        overall_assessment_str = json.dumps(overall_assessment, indent=4)
        all_resjson=json.dumps(all_result,indent=4)
        ocean_value_analysis=json.dumps(ocean_value_analysis,indent=4)
        vocab_analysis=json.dumps(vocab_analysis,indent=4)
        visual_analysis=json.dumps(visual_analysis,indent=4)
        speaking_p=json.dumps(speking_p,indent=4)
        focus_analysis=json.dumps(focus_analysis,indent=4)
        
        return suggestions_str, overall_assessment_str,ocean_value_analysis,vocab_analysis,visual_analysis,speaking_p,focus_analysis,all_resjson

    with gr.Blocks() as demo:
        with gr.Column():
            button = gr.Button("Show Audio-Visual Results")
            with gr.Accordion("Suggestions",open=False):
                suggestions_textbox = gr.Textbox(label="Suggestions for Improvement", interactive=False, lines=10)
            with gr.Accordion("Assessment",open=False):
                assessment_textbox = gr.Textbox(label="Overall Assessment", interactive=False, lines=10)
            
            with gr.Accordion("Ocean_value_analysis",open=False):
                ocean_textbox=gr.Textbox(label="Ocean_values_results",interactive=False,lines=10)
            
            with gr.Accordion("vocab_analysis",open=False):
                vocab_textbox=gr.Textbox(label="vocab_results",interactive=False,lines=10)
            with gr.Accordion("visual_analysis",open=False):
                visual_textbox=gr.Textbox(label="visual_results",interactive=False,lines=10)
            with gr.Accordion("Speaking_performance",open=False):
                speaking_textbox=gr.Textbox(label="speaking_results",interactive=False,lines=10)
            with gr.Accordion("Focus Analysis",open=False):
                focus_textbox=gr.Textbox(label="Focus_results",interactive=False,lines=10)
            with gr.Accordion("Full Result",open=False):
                res_textbox=gr.Textbox(label="Result",interactive=False,lines=15)
            # Connect the button click to the display_results function
            button.click(fn=display_results, outputs=[suggestions_textbox, assessment_textbox,ocean_textbox,vocab_textbox,visual_textbox,speaking_textbox,focus_textbox,res_textbox])
    
    return demo
    


def initialize_services():
    """Initialize configuration, LLM, TTS, and STT services."""
    config = Config()
    llm = LLMManager(config,  prompts)
    tts = TTSManager(config)
    stt = STTManager(config)
    avllm = AVLLMManager(config , prompts)
    default_audio_params["streaming"] = stt.streaming
    if os.getenv("SILENT", False):
        tts.read_last_message = lambda x: None
    return config, llm, tts, stt , avllm

def video_record():
    gr.Markdown("### Webcam Video Recorder")
    gr.Markdown("Click 'Start' to begin recording from your webcam. Click 'Stop' to end the recording and save the video file.")
    radio = gr.Radio(choices=["Start Recording", "Stop Recording"], label="Recording Control")  # Start/Stop action
    # fps = gr.Slider(minimum=10, maximum=60, value=20, step=1, label="FPS")  # FPS control
    video_file = gr.Text(label="Recorded Video")  # Output component
    # radio.change(gradio_interface, inputs=[radio],outputs=[video_file])
    

def create_interface(llm, tts, stt, audio_params , avllm):
    """Create and configure the Gradio interface."""
    with gr.Blocks(title="AI Interviewer", theme=gr.themes.Default()) as demo:
        audio_output = gr.Audio(label="Play audio", autoplay=True, visible=os.environ.get("DEBUG", False), streaming=tts.streaming)
        
        with gr.Tab("AI Interviewer"):
            get_problem_solving_ui(llm, tts, stt, audio_params, audio_output).render()
            # get_instructions_ui(llm, tts, stt, audio_params).render()
            gradio_interface()
        with gr.Tab("Hr results"):
            avi_score()

    return demo

#! /usr/bin/env python
import sys
import os
import cv2
import os
import zipfile
from pathlib import Path
import shutil
import os


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

def setup():
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.resume:
        cfg.TRAIN.RESUME = args.resume
    if args.max_epoch:
        cfg.TRAIN.MAX_EPOCH = args.max_epoch
    if args.lr:
        cfg.SOLVER.RESET_LR = True
        cfg.SOLVER.LR_INIT = args.lr
    if args.test_only:
        cfg.TEST.TEST_ONLY = True
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    return args



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


def dpmain():
    args = setup()
    video_dir = "datasets/ChaLearn/test"
    output_dir = "datasets/ChaLearn/test_data"
    outforwav = "datasets/ChaLearn/voice_data/voice_raw/test_data"
    outforlibrosa = "datasets/ChaLearn/voice_data/voice_librosa/test_data"
    source_directory = "datasets/ChaLearn/test"
    parent_directory = "datasets/ChaLearn"



    contents = os.listdir(source_directory)
    if not contents:
        print("Conitnue with the code")
    else:
        tempfolder = create_folder_with_datetime(parent_directory)
        copy_files(source_directory, tempfolder)



    delete_all_files_in_folder(outforlibrosa)
    delete_all_files_in_folder(outforwav)
    delete_all_files_in_folder(output_dir)
  #  delete_all_files_in_folder(video_dir)


   # video_to_image()
    convert_videos_to_frames(video_dir,output_dir)
    audio_extract(video_dir,outforwav)
    audio_process(mode= 'librosa' ,aud_dir=outforwav , saved_dir=outforlibrosa)

    runner = ExpRunner(cfg)
    
    # if args.test_only:
    return runner.test()
    # runner.run()
    

def main():
    # dpmain()
    """Main function to initialize services and launch the Gradio interface."""
    config, llm, tts, stt ,avllm= initialize_services()
    demo = create_interface(llm, tts, stt, default_audio_params , avllm)
    demo.config["dependencies"][0]["show_progress"] = "hidden"
    demo.launch(show_api=False, share=False)  # Launch with sharing enabled for public link 
    


if __name__ == "__main__":

    main()
