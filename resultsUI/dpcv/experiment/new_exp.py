import pandas as pd
import os
import json
import numpy as np
import torch
from dpcv.faceTracker.face import process_in_batches
from datetime import datetime
from dpcv.data.datasets.build import build_dataloader
from dpcv.modeling.networks.build import build_model
from dpcv.modeling.loss.build import build_loss_func
from dpcv.modeling.solver.build import build_solver, build_scheduler
from dpcv.engine.build import build_trainer
from dpcv.evaluation.summary import TrainSummary
from dpcv.checkpoint.save import save_model, resume_training, load_model
from dpcv.tools.logger import make_logger
import pandas as pd
from dpcv.text_mining_using_python import cal_uni_bi
from OpenGraphAU.demo1 import CallMethod
from dpcv.score_predict_sound import sound_score
from .FinalNode import FullNode
from .score_maker import ComputingValues 
from .comment_fetcher import get_comments_for_gpt,convert_images_to_base64,finalcomment,getcomment_communication,getcomment_positive_attitude,getcomment_sociability
# from .comment_fetcher import get_score,finalcomment,getcomment_communication,getcomment_sociability,getcomment_positive_attitude
import traceback

import cv2  
import base64
import time
from openai import OpenAI
import os,re
import requests
import tiktoken
from decouple import config
client = OpenAI(api_key=config("OPENAI_API_KEY"))


MAX_TOKENS = 10000
OUTPUT_TOKENS = 300
class FinalNode:
    def __init__(self , file_path):

        self.file_path = file_path
        self.link_id = None

        self.ocean_values = None

        self.sentiment_comment = None
        self.transcript = None
        self.grammar_score = None
        self.grammar_comment = None
        self.pace_score = None
        self.pace_score_comment = None
        self.articulation_score = None
        self.articulation_comment = None
        self.sentiment_score = None
        self.pace_comment = None
        self.df_unigrams = None

       
        self.presentability_score = None
        self.dressing_score = None
        self.bodylang_score = None
        self.professional_score = None
        self.emotion_score = None
        self.energy_score = None

        
        self.professional_comment = None
        self.bodylang_comment = None
        self.emotion_comment = None
        self.presentability_comment = None
        self.dressing_comment = None
        self.energy_comment = None
        self.positive_attitude=None
        self.communication_score=None
        self.sociability_score=None
        self.transcription=None
        self.communication_comment=None
        self.positive_attitude_comment=None
        self.sociability_comment=None
        self.overall_score=None
        

    def get_summary(self):
        return {
    "link_id": self.link_id or "Unknown",
    "ocean_values": self.ocean_values if self.ocean_values is not None else [],
    "sentiment_score": self.sentiment_score if self.sentiment_score is not None else 0.0,
    "sentiment_comment": self.sentiment_comment or "No Comment",
    "transcript": self.transcript or "No Transcript",
    "grammar_score": self.grammar_score if self.grammar_score is not None else 0.0,
    "grammar_comment": self.grammar_comment or "No Comment",
    "pace_score": self.pace_score if self.pace_score is not None else 0.0,
    "pace_score_comment": self.pace_score_comment or "No Comment",
    "articulation_score": self.articulation_score if self.articulation_score is not None else 0.0,
    "articulation_comment": self.articulation_comment or "No Comment",
    "pace_comment": self.pace_comment or "No Comment",
    "presentability_score": self.presentability_score if self.presentability_score is not None else 0.0,
    "dressing_score": self.dressing_score if self.dressing_score is not None else 0.0,
    "bodylang_score": self.bodylang_score if self.bodylang_score is not None else 0.0,
    "professional_score": self.professional_score if self.professional_score is not None else 0.0,
    "emotion_score": self.emotion_score if self.emotion_score is not None else 0.0,
    "professional_comment": self.professional_comment or "No Comment",
    "emotion_comment": self.emotion_comment or "No Comment",
    "presentability_comment": self.presentability_comment or "No Comment",
    "dressing_comment": self.dressing_comment or "No Comment",
    "energy_comment": self.energy_comment or "No Comment",
    "positive_attitude_score": self.positive_attitude if self.positive_attitude is not None else 0.0,
    "communication_score": self.communication_score if self.communication_score is not None else 0.0,
    "sociability_score": self.sociability_score if self.sociability_score is not None else 0.0,
    "transcription": self.transcription or [],
    "communication_comment": self.communication_comment or "No Comment",
    "positive_attitude_comment": self.positive_attitude_comment or "No Comment",
    "sociability_comment": self.sociability_comment or "No Comment",
    "energy_score": self.energy_score if self.energy_score is not None else 0.0,
    "body_language_comment": self.bodylang_comment or "No Comment",
    "overall_score": self.overall_score if self.overall_score is not None else 0.0
}


class AudioFileNode:
    def __init__(self, file_path):

        self.file_path = file_path
        self.link_id = None

        self.ocean_values = None
        
        self.sentiment_comment = None
        self.transcript = None
        self.grammar_score = None
        self.grammar_comment = None
        self.pace_score = None
        self.pace_score_comment = None
        self.articulation_score = None
        self.articulation_comment = None
        self.sentiment_score = None
        self.pace_comment = None
        self.df_unigrams = None
        
        self.positive_attitude=None
        self.presentability_score = None
        self.dressing_score = None
        self.bodylang_score = None
        self.professional_score = None
        self.emotion_score = None
        self.energy_score = None
        self.communication_score=None
        self.professional_comment = None
        self.bodylang_comment = None
        self.emotion_comment = None
        self.presentability_comment = None
        self.dressing_comment = None
        self.energy_comment = None
        self.sociability_score=None
        

    def get_summary(self):
        return {
            "file_path": self.file_path,
            "link_id": self.link_id,
            "ocean_values": self.ocean_values,
            "sentiment_comment": self.sentiment_comment,
            "transcript": self.transcript,
            "grammar_score": self.grammar_score,
            "grammar_comment": self.grammar_comment,
            "pace_score": self.pace_score,
            "pace_score_comment": self.pace_score_comment,
            "articulation_score": self.articulation_score,
            "articulation_comment": self.articulation_comment,
            "sentiment_score": self.sentiment_score,
            "pace_comment": self.pace_comment,
            "df_unigrams" : self.df_unigrams,
            "ocean_values":self.ocean_values,
            "presentability_score": self.presentability_score,
            "dressing_score": self.dressing_score,
            "bodylang_score": self.bodylang_score,
            "professional_score": self.professional_score,
            "emotion_score": self.emotion_score,
            "energy_score":self.energy_score,

            # Include the newly added comment attributes

            "presentability_comment": self.presentability_comment,
            "dressing_comment": self.dressing_comment,
            "bodylang_comment": self.bodylang_comment,
            "professional_comment": self.professional_comment,
            "emotion_comment": self.emotion_comment,          
        }

class ExpRunner:

    def __init__(self, cfg, feature_extract=None):
        # print("The configuration used is:" , cfg)
        self.cfg = cfg
        self.logger, self.log_dir = make_logger(cfg.TRAIN.OUTPUT_DIR)
        self.log_cfg_info()
        if not feature_extract:
            self.data_loader = self.build_dataloader()

        self.model = self.build_model()
        self.loss_f = self.build_loss_function()

        self.optimizer = self.build_solver()
        self.scheduler = self.build_scheduler()

        self.collector = TrainSummary()
        self.trainer = self.build_trainer()

        self.audio_nodes = []

    def build_dataloader(self):
        return build_dataloader(self.cfg)

    def build_model(self):
        return build_model(self.cfg)

    def build_loss_function(self):
        return build_loss_func(self.cfg)

    def build_solver(self):
        return build_solver(self.cfg, self.model)

    def build_scheduler(self):
        return build_scheduler(self.cfg, self.optimizer)

    def build_trainer(self):
        return build_trainer(self.cfg, self.collector, self.logger)

    def before_train(self, cfg):
        # cfg = self.cfg.TRAIN
        if cfg.RESUME:
            self.model, self.optimizer, epoch = resume_training(cfg.RESUME, self.model, self.optimizer)
            cfg.START_EPOCH = epoch
            self.logger.info(f"resume training from {cfg.RESUME}")
        if self.cfg.SOLVER.RESET_LR:
            self.logger.info("change learning rate form [{}] to [{}]".format(
                self.optimizer.param_groups[0]["lr"],
                self.cfg.SOLVER.LR_INIT,
            ))
            self.optimizer.param_groups[0]["lr"] = self.cfg.SOLVER.LR_INIT

    def test(self, weight=None):
        self.logger.info("Test only mode and clearing the GPU")
        torch.cuda.empty_cache()
        
        cfg = self.cfg.TEST
        cfg.WEIGHT = weight if weight else cfg.WEIGHT

        if cfg.WEIGHT:
            self.model = load_model(self.model, cfg.WEIGHT)
        else:
            weight_file = "./checkpoint.pkl"
            self.logger.info(f"Test with model {weight_file}")
            self.model = load_model(self.model, weight_file)

        if not self.cfg.TEST.FULL_TEST:
            ocean_acc_avg, ocean_acc, dataset_output, dataset_label, mse = self.trainer.test(
                self.data_loader["test"], self.model
            )
            self.logger.info("MSE: {} Mean: {}".format(mse[0], mse[1]))
        else:
            ocean_acc_avg, ocean_acc, dataset_output, dataset_label = self.trainer.full_test(
                self.data_loader["full_test"], self.model
            )

        folderpath = "./datasets/ChaLearn/voice_data/voice_raw/test_data/"
        audio_list = os.listdir(folderpath)
        # audio_list = list(filter(lambda x: ".praat" not in x, audio_list))   
        videopath  = "./datasets/ChaLearn/test/"

        for idx, audio in enumerate(audio_list):
            file_path = os.path.join(folderpath, audio)
            if os.path.isfile(file_path) and ".praat" not in file_path:
                videoPath  = "./datasets/ChaLearn/test/"

                node = AudioFileNode(file_path)
                
                df_unigrams ,sentiment_score_value , sentiment_comment_value , final_transcription, grammer_score , grammer_comment , pace_score , articulation_score ,pace_comment ,articulation_comment= cal_uni_bi(file_path)
                print("******************************** Final Transcription:***********************************", pace_score)
                
                
                node.pace_score = pace_score or 0.0
                link_id=str(node.file_path).split(".")[1]
                gg=link_id.split("/")[-1]

                node.link_id = gg or 0.0
                
                node.sentiment_score = sentiment_score_value or 0.0
                node.sentiment_comment = sentiment_comment_value or 0.0
                
                node.transcript=final_transcription or 0.0

                node.grammar_score = grammer_score or 0.0
                node.grammar_comment=grammer_comment or 0.0
       
                
                node.pace_score_comment=pace_comment or 0.0


                node.articulation_score = articulation_score or 0.0
                node.articulation_comment=articulation_comment or 0.0

                node.df_unigrams = df_unigrams 
                
                grammer_score_dictionarymatching = ComputingValues(node.get_summary())   # Dictionary updation is required 
                if grammer_score_dictionarymatching is not None and grammer_score is not None:
                    grammer_final_score = (0.1 * grammer_score_dictionarymatching + 0.9 * grammer_score)
                elif grammer_score_dictionarymatching is not None:
                    grammer_final_score = grammer_score_dictionarymatching
                elif grammer_score is not None:
                    grammer_final_score = grammer_score
                else:
                    grammer_final_score = 0  # Default value if both are None

                print(f"Grammar score is @@@@####: {grammer_final_score}")


                node.grammar_score = grammer_final_score or 0.0
                
                if idx < len(dataset_output):
                    node.ocean_values = dataset_output[idx] 
                else:
                    print(f"Warning: dataset_output index {idx} is out of bounds")
                    node.ocean_values = [0] * 5  



                frames_dir_path = "datasets/ChaLearn/test_data"
                folderaud = os.path.basename(file_path).split('/')[-1]
                folderaud = folderaud.replace(".wav","")
                frames_folder = os.path.join(frames_dir_path , folderaud)
                video_dir="datasets/ChaLearn/test"
                video_file = os.path.join(video_dir, folderaud + ".mp4")      
                print("video file path is " , video_file)




                presentability_prompt="You are an interviewer conducting a candidate interview. Based on the video, provide a summary of the person's presentability in the interview situation. Do not offer any suggestions or advice; just describe the person's presentability observed during the interview."
                body_langauage_prompt="You are an interviewer conducting a candidate interview. Based on the video, provide a summary of the person's body language in the interview situation. Do not offer any suggestions or advice; just describe the person's body language observed during the interview."
                dressing_prompt="You are an interviewer conducting a candidate interview. Based on the video, provide a summary of the person's dressing and gromming in the interview situation. Do not offer any suggestions or advice; just describe the person's dressing and gromming observed during the interview."
                professional_prompt="You are an interviewer conducting a candidate interview. Based on the video, provide a summary of the person's professionalism in the interview situation. Do not offer any suggestions or advice; just describe the professionalism observed during the interview."
                emotion_prompt="You are an interviewer conducting a candidate interview. Based on the video, provide a summary of the person's emotion's in the interview situation. Do not offer any suggestions or advice; just describe the emotion's observed during the interview."
                energy_prompt = "You are an interviewer conducting a candidate interview. Based on the video, provide a summary of the person's enerygy in the interview situation. Do not offer any suggestions or advice; just describe the energy observed during the interview."
                base_encoder_frames=convert_images_to_base64(frames_folder)

                print(len(base_encoder_frames),"()()()()()()()()()()()")




                presentability_score,presentabilitycomment=get_comments_for_gpt(base64Frames=base_encoder_frames,prompt=presentability_prompt,transcript=node.transcript,typeo = "presentability")

                if presentability_score == None : 
                    print("the 2nd try is working in enerygu")
                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are an AI Interviewer given as analysis of candidates presentability. "
                                "Give a score on the bases on analysis out of 100"
                                "Do not mention any reasoning or expaination of the score "
                                "Only five value sin integer format"
                            )
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Analyze the presentability assessments {presentabilitycomment} "
                                "Give score out of 100 "
                            )
                        }
                    ]
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        max_tokens=300,
                    )
                    final_comment = response.choices[0].message.content
                    presentability_score = final_comment

                    print(presentability_score)


                dressing_score,dressingcomment=get_comments_for_gpt(base64Frames=base_encoder_frames,prompt=dressing_prompt,transcript=node.transcript , typeo = "dressing")

                if dressing_score == None : 
                  
                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are an AI Interviewer given as analysis of candidates dressing. "
                                "Give a score on the bases on analysis out of 100"
                                "Do not mention any reasoning or expaination of the score "
                                "Only five value sin integer format"
                            )
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Analyze the dressing assessments {dressingcomment} "
                                "Give score out of 100 "
                            )
                        }
                    ]
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        max_tokens=300,
                    )
                    final_comment = response.choices[0].message.content
                    dressing_score = final_comment

                    print(dressing_score)



                professional_score,professionalcomment=get_comments_for_gpt(base64Frames=base_encoder_frames,prompt=professional_prompt,transcript=node.transcript ,typeo = "professionalism")

                if professional_score == None : 
                   
                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are an AI Interviewer given as analysis of candidates professionalism. "
                                "Give a score on the bases on analysis out of 100"
                                "Do not mention any reasoning or expaination of the score "
                                "Only five value sin integer format"
                            )
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Analyze the dressing assessments {professionalcomment} "
                                "Give score out of 100 "
                            )
                        }
                    ]
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        max_tokens=300,
                    )
                    final_comment = response.choices[0].message.content
                    professional_score = final_comment

                    print(professional_score)
                



                body_lang_score,bodylang_comment=get_comments_for_gpt(base64Frames=base_encoder_frames,prompt=body_langauage_prompt,transcript=node.transcript , typeo = "body_langauage")
          
                if body_lang_score == None : 
          
                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are an AI Interviewer given as analysis of candidates body language. "
                                "Give a score on the bases on analysis out of 100"
                                "Do not mention any reasoning or expaination of the score "
                                "Only five value sin integer format"
                            )
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Analyze the dressing assessments {bodylang_comment} "
                                "Give score out of 100 "
                            )
                        }
                    ]
           
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        max_tokens=300,
                    )
                    final_comment = response.choices[0].message.content
                    body_lang_score = final_comment

                    print(body_lang_score)



                emotion_score,emotioncomment=get_comments_for_gpt(base64Frames=base_encoder_frames,prompt=emotion_prompt,transcript=node.transcript , typeo = "emotion")

                if emotion_score == None : 

                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are an AI Interviewer given as analysis of candidates emotion. "
                                "Give a score on the bases on analysis out of 100"
                                "Do not mention any reasoning or expaination of the score "
                                "Only five value sin integer format"
                            )
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Analyze the dressing assessments {emotioncomment} "
                                "Give score out of 100 "
                            )
                        }
                    ]
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        max_tokens=300,
                    )
                    final_comment = response.choices[0].message.content
                    emotion_score = final_comment

                    print(emotion_score)



                energy_score,energycomment=get_comments_for_gpt(base64Frames=base_encoder_frames,prompt=energy_prompt,transcript=node.transcript , typeo = "energy")
                
                if energy_score == None : 
            
                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are an AI Interviewer given as analysis of candidates energy. "
                                "Give a score on the bases on analysis out of 100"
                                "Do not mention any reasoning or expaination of the score "
                                "Only five value sin integer format"
                            )
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Analyze the dressing assessments {energycomment} "
                                "Give score out of 100 "
                            )
                        }
                    ]
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        max_tokens=300,
                    )
                    final_comment = response.choices[0].message.content
                    energy_score = final_comment

                    print(energy_score)


                node.professional_comment = professionalcomment or 0.0

                node.bodylang_comment = bodylang_comment or 0.0
                node.emotion_comment = emotioncomment or 0.0 
                node.presentability_comment = presentabilitycomment or 0.0
                node.dressing_comment = dressingcomment or 0.0
                node.energy_comment = energycomment or 0.0
                
                print("presentability_score",presentability_score)
                print("dressing_score",dressing_score)
                print("body_lang_score",body_lang_score)
                if body_lang_score == None : 
                    body_lang_score = 10

                print("The energy_score" ,energy_score)
                print("emotion_score" ,emotion_score)



                if emotion_score == None : 
                    emotion_score = 10
            

                try:
                    # ✅ Safe conversion function: Only replace non-numeric strings
                    def safe_float(value):
                        if isinstance(value, (int, float)):  # Keep numbers unchanged
                            return value
                        try:
                            return float(value)  # Convert valid numeric strings
                        except (ValueError, TypeError):  # Replace invalid strings
                            return 0.0

                    # ✅ Ensure all scores are correctly formatted
                    presentability_score = safe_float(presentability_score)
                    dressing_score = safe_float(dressing_score)
                    body_lang_score = safe_float(body_lang_score)  # FIXED: Prevents invalid string conversion
                    emotion_score = safe_float(emotion_score)
                    energy_score = safe_float(energy_score)
                    sentiment_score = safe_float(getattr(node, 'sentiment_score', 0.0))
                    articulation_score = safe_float(getattr(node, 'articulation_score', 0.0))
                    pace_score = safe_float(getattr(node, 'pace_score', 0.0))
                    grammar_score = safe_float(getattr(node, 'grammar_score', 0.0))

                    # ✅ Compute scores
                    professional_score = (presentability_score + dressing_score + body_lang_score) / 3
                    node.professional_score = round(professional_score, 2)

                    node.presentability_score = presentability_score
                    node.dressing_score = dressing_score
                    node.bodylang_score = body_lang_score
                    node.emotion_score = emotion_score
                    node.energy_score = energy_score

                    # ✅ Compute positive attitude safely
                    node.positive_attitude = (energy_score + sentiment_score) / 2

                    # ✅ Compute communication score safely
                    communication_score = (articulation_score + pace_score + grammar_score) / 3
                    node.communication_score = round(communication_score, 2)

                    # ✅ Compute sociability score safely
                    sociability_score = (emotion_score + energy_score + sentiment_score) / 3
                    node.sociability_score = round(sociability_score, 2)

                    self.audio_nodes.append(node)

                except Exception as e:
                    print(f"❌ Error computing final node summary: {e}")
                    import traceback
                    traceback.print_exc()



        Finalnode = FinalNode(videopath)
        list_link_id = []
        list_ocean_values = []
        list_sentiment_comment = []
        list_transcript = []
        list_grammar_score = []
        list_grammar_comment = []
        list_pace_score = []
        list_articulation_score = []
        list_articulation_comment = []
        list_sentiment_score = []
        list_pace_comment = []
        list_positive_attitude=[]
        list_presentability_score = []
        list_dressing_score = []
        list_bodylang_score = []
        list_professional_score = []
        list_emotion_score = []
        list_energy_score = []
        list_professional_comment = []
        list_bodylang_comment = []
        list_emotion_comment = []
        list_presentability_comment = []
        list_dressing_comment = []
        list_energy_comment = []
        list_unigrams=[]
        list_sociability_score=[]
        list_communication_score=[]
        
        for node in self.audio_nodes:
          try :  
            list_link_id.append(node.link_id or "Unknown")
            summary = node.get_summary()

            list_grammar_score.append(node.grammar_score if node.grammar_score is not None else 0.0)
            list_grammar_comment.append(node.grammar_comment or "No Comment")
            list_pace_score.append(node.pace_score if node.pace_score is not None else 0.0)
            list_pace_comment.append(node.pace_score_comment or "No Comment")
            list_articulation_score.append(node.articulation_score if node.articulation_score is not None else 0.0)
            list_articulation_comment.append(node.articulation_comment or "No Comment")
            list_sentiment_comment.append(node.sentiment_comment or "No Comment")
            list_sentiment_score.append(node.sentiment_score if node.sentiment_score is not None else 0.0)
            list_transcript.append(summary.get("transcript", "No Transcript"))
            list_ocean_values.append(summary.get("ocean_values", [0] * 5))  # Default empty vector
            list_presentability_score.append(node.presentability_score if node.presentability_score is not None else 0.0)
            list_presentability_comment.append(node.presentability_comment or "No Comment")
            list_dressing_score.append(node.dressing_score if node.dressing_score is not None else 0.0)
            list_dressing_comment.append(node.dressing_comment or "No Comment")
            list_bodylang_score.append(node.bodylang_score if node.bodylang_score is not None else 0.0)
            list_bodylang_comment.append(node.bodylang_comment or "No Comment")
            list_emotion_score.append(node.emotion_score if node.emotion_score is not None else 0.0)
            list_emotion_comment.append(node.emotion_comment or "No Comment")
            list_energy_score.append(node.energy_score if node.energy_score is not None else 0.0)
            list_energy_comment.append(node.energy_comment or "No Comment")
            list_professional_score.append(node.professional_score if node.professional_score is not None else 0.0)
            list_professional_comment.append(node.professional_comment or "No Comment")
            list_unigrams.append(node.df_unigrams if node.df_unigrams is not None else [])
            list_positive_attitude.append(node.positive_attitude if node.positive_attitude is not None else 0.0)
            list_sociability_score.append(node.sociability_score if node.sociability_score is not None else 0.0)
            list_communication_score.append(node.communication_score if node.communication_score is not None else 0.0)
            print("<<<<<<<<<<<<<<<<<<<<<<<<<><><><><><><><><><>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print("The transcript is " , list_transcript)
          except Exception as e:
            print(f"Error processing node: {e}")     
        from collections import Counter



        print(f"Link ID: {list_link_id}")
        print(f"Grammar Score: {list_grammar_score} | Comment: {list_grammar_comment}")
        print(f"Pace Score: {list_pace_score} | Comment: {list_pace_comment}")
        print(f"Articulation Score: {list_articulation_score} | Comment: {list_articulation_comment}")
        print(f"Sentiment Score: {list_sentiment_score} | Comment: {list_sentiment_comment}")
        print(f"Transcript: {list_transcript}")
        print(f"OCEAN Values: {list_ocean_values}")
        print(f"Presentability Score: {list_presentability_score} | Comment: {list_presentability_comment}")
        print(f"Dressing Score: {list_dressing_score} | Comment: {list_dressing_comment}")
        print(f"Body Language Score: {list_bodylang_score} | Comment: {list_bodylang_comment}")
        print(f"Emotion Score: {list_emotion_score} | Comment: {list_emotion_comment}")
        print(f"Energy Score: {list_energy_score} | Comment: {list_energy_comment}")
        print(f"Professional Score: {list_professional_score} | Comment: {list_professional_comment}")
        print(f"Positive Attitude Score: {list_positive_attitude}")
        print(f"Sociability Score: {list_sociability_score}")
        print(f"Communication Score: {list_communication_score}")
        print(f"Unigrams: {list_unigrams}")


        try:
            # Get final comments safely
            professional_comm = finalcomment(list_professional_comment)
            body_comm = finalcomment(list_bodylang_comment)
            emotion_comm = finalcomment(list_emotion_comment)
            presentability_comm = finalcomment(list_presentability_comment)
            dressing_comm = finalcomment(list_dressing_comment)
            grammar_comm = finalcomment(list_grammar_comment).replace("\"", "")
            sentiment_comm = finalcomment(list_sentiment_comment)
            pace_comm = finalcomment(list_pace_comment)
            articulation_comm = finalcomment(list_articulation_comment)
            energy_comm = finalcomment(list_energy_comment)

            # Compute safe means
            Finalnode.emotion_score = round(np.mean([x for x in list_emotion_score if x is not None]), 2)
            Finalnode.articulation_score = round(np.mean([x for x in list_articulation_score if x is not None]), 2)
            Finalnode.bodylang_score = round(np.mean([x for x in list_bodylang_score if x is not None]), 2)
            Finalnode.dressing_score = round(np.mean([x for x in list_dressing_score if x is not None]), 2)
            Finalnode.energy_score = round(np.mean([x for x in list_energy_score if x is not None]), 2)
            Finalnode.ocean_values = np.mean([x for x in list_ocean_values if x is not None], axis=0).tolist()
            Finalnode.grammar_score = round(np.mean([x for x in list_grammar_score if x is not None]), 2)
            Finalnode.pace_score = round(np.mean([x for x in list_pace_score if x is not None]), 2)
            Finalnode.sentiment_score = round(np.mean([x for x in list_sentiment_score if x is not None]), 2)
            Finalnode.professional_score = round(np.mean([x for x in list_professional_score if x is not None]), 2)
            Finalnode.presentability_score = round(np.mean([x for x in list_presentability_score if x is not None]), 2)
            Finalnode.transcript = list_transcript

            # Assign comments
            Finalnode.sentiment_comment = sentiment_comm
            Finalnode.grammar_comment = grammar_comm
            Finalnode.pace_score_comment = pace_comm
            Finalnode.articulation_comment = articulation_comm
            Finalnode.professional_comment = professional_comm
            Finalnode.bodylang_comment = body_comm
            Finalnode.emotion_comment = emotion_comm
            Finalnode.presentability_comment = presentability_comm
            Finalnode.dressing_comment = dressing_comm
            Finalnode.energy_comment = energy_comm
            Finalnode.link_id = list_link_id
            Finalnode.df_unigrams = list_unigrams

            # Compute final scores
            Finalnode.positive_attitude = round(np.mean([x for x in list_positive_attitude if x is not None]), 2)
            Finalnode.sociability_score = round(np.mean([x for x in list_sociability_score if x is not None]), 2)
            Finalnode.communication_score = round(np.mean([x for x in list_communication_score if x is not None]), 2)

            # Generate final comments
            Finalnode.communication_comment = getcomment_communication(
                pace_comment=Finalnode.pace_score_comment,
                articulation_comment=Finalnode.articulation_comment,
                energy_comment=Finalnode.energy_comment
            )


            
            Finalnode.sociability_comment = getcomment_sociability(
                energy_comment=Finalnode.energy_comment,
                sentiment_comment=Finalnode.sentiment_comment,
                emotion_comment=Finalnode.emotion_comment
            )
            Finalnode.positive_attitude_comment = getcomment_positive_attitude(energy_comment=Finalnode.energy_comment)

            Finalnode.transcription = [
                {"id": id_, "transcript": transcript}
                for id_, transcript in zip(Finalnode.link_id, Finalnode.transcript)
            ]

            Finalnode.overall_score = round(
                (Finalnode.sociability_score + Finalnode.professional_score + Finalnode.communication_score + Finalnode.positive_attitude) / 4,
                2
            )

        except Exception as e:
                print(f"Error computing final hfghfg node summary: {e}")
                traceback.print_exc()  # Prints the full error stack trace

        print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
        print(Finalnode.get_summary())
        return Finalnode.get_summary()


    def run(self):
        self.train()
        self.test()

    def log_cfg_info(self):

        string = json.dumps(self.cfg, sort_keys=True, indent=4, separators=(',', ':'))
        self.logger.info(string)

    def data_extract(self, dataloader, output_dir):

        return self.trainer.data_extract(self.model, dataloader, output_dir)

