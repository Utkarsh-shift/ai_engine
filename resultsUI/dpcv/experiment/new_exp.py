import pandas as pd
import os
import json
import numpy as np
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
import opensmile,threading
import pandas as pd
from dpcv.text_mining_using_python import cal_uni_bi
from OpenGraphAU.demo1 import CallMethod
from dpcv.score_predict_sound import sound_score
# from VideoLLaVA.videollava.serve.cli import VLmain
# from LLaVA.llava.serve.cli import lvmain
#from .AudioNode import AudioFileNode
from .FinalNode import FullNode
from .score_maker import ComputingValues 
from .comment_fetcher import get_comments_for_gpt,get_score,finalcomment,convert_images_to_base64,getcomment_communication,getcomment_sociability,getcomment_positive_attitude


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
            
            "link_id": self.link_id,
            "ocean_values": self.ocean_values,
            "sentiment_score":self.sentiment_score,
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
            "ocean_values":self.ocean_values,
            "presentability_score": self.presentability_score,
            "dressing_score": self.dressing_score,
            "bodylang_score": self.bodylang_score,
            "professional_score": self.professional_score,
            "emotion_score": self.emotion_score,
            "professional_comment":self.professional_comment,
            "emotion_comment":self.emotion_comment,
            "presentability_comment":self.presentability_comment,
            "dressing_comment":self.dressing_comment,
            "energy_comment":self.energy_comment,
            "positive_attitude_score":self.positive_attitude,
            "communication_score":self.communication_score,
            "sociability_score":self.sociability_score,
            "transcription":self.transcription,
            "communication_comment":self.communication_comment,
            "positive_attitude_comment":self.positive_attitude_comment,
            "sociability_comment":self.sociability_comment,
            "energy_score":self.energy_score,
            "body_language_comment":self.bodylang_comment,
            "overall_score":self.overall_score
            
        
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
        print("The configuration used is:" , cfg)
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
        self.logger.info("Test only mode")
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
        audio_list = list(filter(lambda x: ".praat" not in x, audio_list))   
        videopath  = "./datasets/ChaLearn/test/"

        for idx, audio in enumerate(audio_list):
            file_path = os.path.join(folderpath, audio)
            if os.path.isfile(file_path) and ".praat" not in file_path:
                videoPath  = "./datasets/ChaLearn/test/"

                node = AudioFileNode(file_path)
                
                df_unigrams ,sentiment_score_value , sentiment_comment_value , final_transcription, grammer_score , grammer_comment , pace_score , articulation_score ,pace_comment ,articulation_comment= cal_uni_bi(file_path)
             
                link_id=str(node.file_path).split(".")[1]
                gg=link_id.split("/")[-1]

                node.link_id = gg
                
                node.sentiment_score = sentiment_score_value
                node.sentiment_comment = sentiment_comment_value

                node.transcript=final_transcription

                node.grammar_score = grammer_score
                node.grammar_comment=grammer_comment
                
                node.pace_score = pace_score
                node.pace_score_comment=pace_comment


                node.articulation_score = articulation_score
                node.articulation_comment=articulation_comment

                node.df_unigrams = df_unigrams
                


                grammer_score_dictionarymatching = ComputingValues(node.get_summary())   # Dictionary updation is required 
                if grammer_score_dictionarymatching!=None and grammer_score!=None:
                    grammer_final_score = (0.1*grammer_score_dictionarymatching + 0.9*grammer_score)
                    print("grammer _score is :",grammer_final_score)
                node.grammar_score = grammer_final_score
                node.ocean_values = dataset_output[idx]  # check This code again ? what is index ? why is this code here ?

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
                presentabilitycomment=get_comments_for_gpt(base64Frames=base_encoder_frames,prompt=presentability_prompt,transcript=node.transcript)
                dressingcomment=get_comments_for_gpt(base64Frames=base_encoder_frames,prompt=dressing_prompt,transcript=node.transcript)
                professionalcomment=get_comments_for_gpt(base64Frames=base_encoder_frames,prompt=professional_prompt,transcript=node.transcript)
                bodylang=get_comments_for_gpt(base64Frames=base_encoder_frames,prompt=body_langauage_prompt,transcript=node.transcript)
                emotioncomment=get_comments_for_gpt(base64Frames=base_encoder_frames,prompt=emotion_prompt,transcript=node.transcript)
                energycomment=get_comments_for_gpt(base64Frames=base_encoder_frames,prompt=energy_prompt,transcript=node.transcript)


                node.professional_comment = professionalcomment
                node.bodylang_comment = bodylang
                node.emotion_comment = emotioncomment
                node.presentability_comment = presentabilitycomment
                node.dressing_comment = dressingcomment
                node.energy_comment = energycomment



                presentability_score_prompt="calculate the gromming score using the text"
                dressing_score_prompt="calculate the dressing and grooming score using the text"
                
                body_lang_prompt="calculate the body language score using the text"
                emotion_score_prompt="calculate the emotion score using the text"
                energy_score_prompt = "calculate the energy score using the text"


                
                presentability_score=get_score(presentability_score_prompt,presentabilitycomment)
                dressing_score=get_score(dressing_score_prompt,dressingcomment)
                body_lang_score=get_score(body_lang_prompt,bodylang)
                emotion_score=get_score(emotion_score_prompt,emotioncomment)
                energy_score=get_score(energy_score_prompt , energycomment)
                
                professional_score=(presentability_score+dressing_score+body_lang_score)/3
                node.professional_score = round(professional_score, 2)
                node.presentability_score = presentability_score
                node.dressing_score = dressing_score
                node.bodylang_score = body_lang_score
                node.emotion_score = emotion_score
                node.energy_score = energy_score
                node.positive_attitude=energy_score  # need to change as sentiment is not included only energy
                communication_score=(node.articulation_score+node.pace_score+node.grammar_score)/3
                node.communication_score=round(communication_score,2)
                sociability_score=(node.emotion_score+node.energy_score+node.sentiment_score)/3
                node.sociability_score=round(sociability_score,2)

                self.audio_nodes.append(node)
                
        
        Finalnode = FinalNode(videopath)
        print(self.audio_nodes)
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
            list_link_id.append(node.link_id)
            summary = node.get_summary()
            list_grammar_score.append(node.grammar_score)
            list_grammar_comment.append(node.grammar_comment)
            list_pace_score.append(node.pace_score)
            if node.pace_score_comment!=None:
                list_pace_comment.append(node.pace_score_comment)
                
            list_articulation_score.append(node.articulation_score)
            list_articulation_comment.append(node.articulation_comment)
            list_sentiment_comment.append(node.sentiment_comment)
            list_sentiment_score.append(node.sentiment_score)
            list_transcript.append(summary["transcript"])
            list_ocean_values.append(summary["ocean_values"])
            list_presentability_score.append(node.presentability_score )
            list_presentability_comment.append(node.presentability_comment)
            list_dressing_score.append(node.dressing_score)
            list_dressing_comment.append(node.dressing_comment)
            list_bodylang_score.append(node.bodylang_score)
            list_bodylang_comment.append(node.bodylang_comment)
            list_emotion_score.append(node.emotion_score)
            list_emotion_comment.append(node.emotion_comment)
            list_energy_score.append(node.energy_score)
            list_energy_comment.append(node.energy_comment)
            list_professional_score.append(node.professional_score)
            list_professional_comment.append(node.professional_comment)
            list_unigrams.append(node.df_unigrams)
            list_positive_attitude.append(node.positive_attitude)
            list_sociability_score.append(node.sociability_score)
            list_communication_score.append(node.communication_score)
            
        from collections import Counter


        professional_comm=finalcomment(list_professional_comment)
        body_comm=finalcomment(list_bodylang_comment)
        emotion_comm=finalcomment(list_emotion_comment)
        presentability_comm=finalcomment(list_presentability_comment)
        dressing_comm=finalcomment(list_dressing_comment)
        grammer_comm=finalcomment(list_grammar_comment)
        grammer_comm=grammer_comm.replace("\"","")
        sentiment_comm=finalcomment(list_sentiment_comment)
        pace_comm=finalcomment(list_pace_comment)
        articulation_comm=finalcomment(list_articulation_comment)
        energy_comm=finalcomment(list_energy_comment)
        emotion_sc=np.mean(list_emotion_score)
        Finalnode.emotion_score=round(emotion_sc)
        articulation_sc=np.mean(list_articulation_score)
        Finalnode.articulation_score = round(articulation_sc,2)
        body_lang_sc= np.mean(list_bodylang_score)
        Finalnode.bodylang_score = round(body_lang_sc,2)
        dressing_sc=np.mean(list_dressing_score)
        Finalnode.dressing_score = round(dressing_sc,2)
        energy_sc=np.mean(list_energy_score)
        Finalnode.energy_score = round(energy_sc,2)
        Finalnode.ocean_values = (np.mean(list_ocean_values, axis = 0)).tolist()
        grammer_sc=np.mean(list_grammar_score)
        Finalnode.grammar_score=round(grammer_sc,2)
        pace_sc=np.mean(list_pace_score)
        Finalnode.pace_score=round(pace_sc,2)
        sentiment_sc=np.mean(list_sentiment_score)
        Finalnode.sentiment_score=round(sentiment_sc,2)
        professional_sc=np.mean(list_professional_score)
        Finalnode.professional_score=round(professional_sc,2)
        presentability_sc=np.mean(list_presentability_score)
        Finalnode.presentability_score=round(presentability_sc,2)
        Finalnode.transcript=list_transcript
        Finalnode.sentiment_comment=sentiment_comm
        Finalnode.grammar_comment=grammer_comm
        Finalnode.pace_score_comment=pace_comm
        Finalnode.articulation_comment=articulation_comm
        Finalnode.professional_comment=professional_comm
        Finalnode.bodylang_comment=body_comm
        Finalnode.emotion_comment=emotion_comm
        Finalnode.presentability_comment=presentability_comm
        Finalnode.dressing_comment=dressing_comm
        Finalnode.energy_comment=energy_comm
        Finalnode.link_id=list_link_id
        Finalnode.df_unigrams=list_unigrams
        positive_attitude_sc=np.mean(list_positive_attitude)
        Finalnode.positive_attitude=round(positive_attitude_sc,2) 
        sociability_sc=np.mean(list_sociability_score)
        Finalnode.sociability_score=round(sociability_sc,2)
        communication_sc=np.mean(list_communication_score) 
        Finalnode.communication_score=round(communication_sc,2)   
        communication_comm=getcomment_communication(pace_comment=Finalnode.pace_comment,articulation_comment=Finalnode.articulation_comment,energy_comment=Finalnode.energy_comment)
        Finalnode.communication_comment=communication_comm
        sociability_comm=getcomment_sociability(energy_score=Finalnode.energy_score,sentiment_score=Finalnode.sentiment_score,emotion_score=Finalnode.emotion_score)
        Finalnode.sociability_comment=sociability_comm
        positive_attitude_comm=getcomment_positive_attitude(energy_score=Finalnode.energy_score)
        Finalnode.positive_attitude_comment=positive_attitude_comm
        transcription = [{"id": id_, "transcript": transcript} for id_, transcript in zip(Finalnode.link_id,Finalnode.transcript)]
        Finalnode.transcription=transcription
        overall_score=(Finalnode.sociability_score+Finalnode.professional_score+Finalnode.communication_score+Finalnode.positive_attitude)/4
        Finalnode.overall_score=round(overall_score,2)
        return Finalnode.get_summary()




    def run(self):
        self.train()
        self.test()

    def log_cfg_info(self):
        """
        record training info for convenience of results analysis
        """
        string = json.dumps(self.cfg, sort_keys=True, indent=4, separators=(',', ':'))
        self.logger.info(string)

    def data_extract(self, dataloader, output_dir):

        return self.trainer.data_extract(self.model, dataloader, output_dir)

