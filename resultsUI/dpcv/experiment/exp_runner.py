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
from .comment_fetcher import get_comments_for_gpt,get_score,finalcomment
class AudioFileNode:
    def __init__(self, file_path):
        self.file_path = file_path
        self.link_id=None
        self.ocean_values = None
        self.df_unigram = None
        self.df_bigram = None
        self.sentiment=None
        self.f0_semitone_mean = None
        self.jitter_local_mean = None
        self.f1_frequency_mean = None
        self.shimmer_local_db_mean = None
        self.loudness_mean = None
        self.hnr_mean = None
        self.alpha_ratio_mean = None
        self.hammarberg_index_mean = None
        self.slope_v0_500_mean = None
        self.slope_v0_500_stddev_norm = None
        self.slope_v500_1500_mean = None
        self.slope_v500_1500_stddev_norm = None
        self.loudness_peaks_per_sec = None
        self.voiced_segments_per_sec = None
        self.mean_voiced_segment_length_sec = None
        self.mean_unvoiced_segment_length = None
        self.mean = None    
        self.std = None
        self.skew = None
        self.kurtosis = None
        self.right = None
        self.left = None
        self.centre = None
        self.eyeblink = 0
        self.data = None
        self.Interview_score = None
        self.shoulder=None
        self.pacescore = None
        self.clarityscore = None 
        self.energyscore = None
        self.fluencyscore=None
        self.articulationscore=None
        self.communicationscore=None
        self.paceandclarityscore=None
        self.sociablility_score=None
        self.face_confidence_score=None
        self.positive_attitude=None
        self.Professional_Score=None
        self.Overall_Score=None
        self.transcript=None
        self.grammer = None 
        self.hlwl_count=None
        self.hlwr_count=None
        self.hlwc_count=None
        self.hrwl_count=None
        self.hrwr_count=None
        self.hrwc_count=None
        self.hdwl_count=None
        self.hdwr_count=None
        self.hdwc_count=None
        self.huwl_count=None
        self.huwr_count=None
        self.huwc_count=None
        self.hswl_count=None
        self.hswr_count=None
        self.hswc_count=None
        self.GPTgrammer_score=None
        self.professional_sc=None
        self.body_lang_score=None
        self.emotion_score=None
        self.professionalcomment=None,
        self.bodylang_comment=None,
        self.emotioncomment=None,
        self.grommingcomment=None,
        self.dressingcomment=None,
        self.cheating_comment=None
        self.gpt_grammer_comment=None

        self.presentiablity_score=None
    def set_scores(self,pacescore,clarityscore,energyscore,fluencyscore,articulatioscore,communicationscore,paceandclarityscore,sociablility_score,face_confidence_score,positive_attitude,Professional_score,Overall_Score,Grammer_score,professional_sc,body_lang_score,emotion_score,presentiablity_score):
        self.pacescore = pacescore
        self.clarityscore=clarityscore
        self.energyscore=energyscore
        self.fluencyscore=fluencyscore
        self.articulationscore=articulatioscore
        self.communicationscore=communicationscore
        self.paceandclarityscore=paceandclarityscore
        self.sociablility_score=sociablility_score
        self.face_confidence_score=face_confidence_score
        self.positive_attitude=positive_attitude
        self.Professional_Score=Professional_score
        self.Overall_Score=Overall_Score
        self.grammer = Grammer_score
        self.professional_sc=professional_sc
        self.body_lang_score=body_lang_score
        self.emotion_score=emotion_score
        self.presentiablity_score=presentiablity_score

    def set_link_id(self,link_id):
        self.link_id=link_id

    def set_interview_score(self,score):
        self.Interview_score = score

    def set_data(self, data_value):
        self.data = data_value

    def set_shoulder(self, data_value):
        self.shoulder = data_value
  
    def set_headpose_and_eyegaze(self,hlwl_count,hlwr_count,hlwc_count,hrwl_count,hrwr_count,hrwc_count,hdwl_count,hdwr_count,
                                 hdwc_count,huwl_count,huwr_count,huwc_count,hswl_count,hswr_count,hswc_count,fnic_count
                                 
                                 ):
        self.hlwl_count=hlwl_count
        self.hlwr_count=hlwr_count
        self.hlwc_count=hlwc_count
        self.hrwl_count=hrwl_count
        self.hrwr_count=hrwr_count
        self.hrwc_count=hrwc_count
        self.hdwl_count=hdwl_count
        self.hdwr_count=hdwr_count
        self.hdwc_count=hdwc_count
        self.huwl_count=huwl_count
        self.huwr_count=huwr_count
        self.huwc_count=huwc_count
        self.hswl_count=hswl_count
        self.hswr_count=hswr_count
        self.hswc_count=hswc_count
        self.fn_count = fnic_count

    def set_kurtosis(self , kurtosis_values):
        self.kurtosis = kurtosis_values

    def set_skew(self, skew_values):
        self.skew = skew_values
        
        
    def set_std(self, std_values):
        self.std = std_values

    def set_mean(self,mean_values):
        self.mean = mean_values

    def set_ocean_values(self, ocean_values):
        self.ocean_values = ocean_values

    def set_df_unigram(self, df_unigram):
        self.df_unigram = df_unigram

    def set_df_bigram(self, df_bigram):
        self.df_bigram = df_bigram

    def set_senti(self,sentiment):
        self.sentiment=sentiment

    def set_senti_sc(self,sentiment_sc):
        self.sentiment_sc=sentiment_sc
    
    def set_transcript(self,transcript):
        self.transcript=transcript
    def set_filepath(self,filepath):
        self.file_path=filepath
    def set_grammer_score(self,grammer_Score):
        self.GPTgrammer_score=grammer_Score

    def set_grammer_comment(self,grammer_comment):
        self.gpt_grammer_comment=grammer_comment

    def set_comments(self,professionalcomment,bodylang,emotioncomment,grommingcomment,dressingcomment,final_cheating):
        self.professionalcomment=professionalcomment
        self.bodylang_comment=bodylang
        self.emotioncomment=emotioncomment
        self.grommingcomment=grommingcomment
        self.dressingcomment=dressingcomment
        self.cheating_comment=final_cheating
        

    def set_smile_features(self, smile_features):
        self.f0_semitone_mean = smile_features.get('F0semitoneFrom27.5Hz_sma3nz_amean')
        self.jitter_local_mean = smile_features.get('jitterLocal_sma3nz_amean')
        self.f1_frequency_mean = smile_features.get('F1frequency_sma3nz_amean')
        self.shimmer_local_db_mean = smile_features.get('shimmerLocaldB_sma3nz_amean')
        self.loudness_mean = smile_features.get('loudness_sma3_amean')
        self.hnr_mean = smile_features.get('HNRdBACF_sma3nz_amean')
        self.alpha_ratio_mean = smile_features.get('alphaRatioV_sma3nz_amean')
        self.hammarberg_index_mean = smile_features.get('hammarbergIndexV_sma3nz_amean')
        self.slope_v0_500_mean = smile_features.get('slopeV0-500_sma3nz_amean')
        self.slope_v0_500_stddev_norm = smile_features.get('slopeV0-500_sma3nz_stddevNorm')
        self.slope_v500_1500_mean = smile_features.get('slopeV500-1500_sma3nz_amean')
        self.slope_v500_1500_stddev_norm = smile_features.get('slopeV500-1500_sma3nz_stddevNorm')
        self.loudness_peaks_per_sec = smile_features.get('loudnessPeaksPerSec')
        self.voiced_segments_per_sec = smile_features.get('VoicedSegmentsPerSec')
        self.mean_voiced_segment_length_sec = smile_features.get('MeanVoicedSegmentLengthSec')
        self.mean_unvoiced_segment_length = smile_features.get('MeanUnvoicedSegmentLength')

    def get_summary(self):
        return {
            "file_path": self.file_path,
            "ocean_values": self.ocean_values,
            "df_unigram": self.df_unigram,
            "df_bigram": self.df_bigram,
            "sentiment":self.sentiment,
            "f0_semitone_mean": self.f0_semitone_mean,
            "jitter_local_mean": self.jitter_local_mean,
            "f1_frequency_mean": self.f1_frequency_mean,
            "shimmer_local_db_mean": self.shimmer_local_db_mean,
            "loudness_mean": self.loudness_mean,
            "hnr_mean": self.hnr_mean,
            "alpha_ratio_mean": self.alpha_ratio_mean,
            "hammarberg_index_mean": self.hammarberg_index_mean,
            "slope_v0_500_mean": self.slope_v0_500_mean,
            "slope_v0_500_stddev_norm": self.slope_v0_500_stddev_norm,
            "slope_v500_1500_mean": self.slope_v500_1500_mean,
            "slope_v500_1500_stddev_norm": self.slope_v500_1500_stddev_norm,
            "loudness_peaks_per_sec": self.loudness_peaks_per_sec,
            "voiced_segments_per_sec": self.voiced_segments_per_sec,
            "mean_voiced_segment_length_sec": self.mean_voiced_segment_length_sec,
            "mean_unvoiced_segment_length": self.mean_unvoiced_segment_length,
            "mean_Action_unit" : self.mean,
            "Std_Action_unit" : self.std,
            "Skewness_Action_unit":self.skew,
            "Kurtosis_Action_unit": self.kurtosis,
            "headleft_watchleft":self.hlwl_count,
            "headleft_watchright":self.hlwr_count,
            "headleft_watchcentre":self.hlwc_count,
            "headright_watchleft":self.hrwl_count,
            "headright_watchright":self.hrwr_count,
            "headright_watchcentre":self.hrwc_count,
            "headdown_watchleft":self.hdwl_count,
            "headdown_watchright":self.hdwr_count,
            "headdown_watchcentre":self.hdwc_count,
            "headup_watchleft":self.huwl_count,
            "headup_watchright":self.huwr_count,
            "headup_watchcentre":self.huwc_count,
            "headstraight_watchleft":self.hswl_count,
            "headstraight_watchright":self.hswr_count,
            "headstraight_watchcentre":self.hswc_count,
            "shoulderPose": self.shoulder,
            "sentiment_score" : self.sentiment_sc,
            "energy": self.energyscore,
            "pace" :self.pacescore,
            "clarity" :self.clarityscore,
            "fluency" : self.fluencyscore,
            "pace_and_clarity" :self.paceandclarityscore,
            "Articulation": self.articulationscore,
            "Communication":self.communicationscore,
            "sentiment" : self.sentiment_sc,
            "Sociability_score":self.sociablility_score,
            "Confidence_score":self.face_confidence_score,
            "Positive_attitude":self.positive_attitude,
            "Overall_Professional_Score":self.Professional_Score,
            "Overall_Score":self.Overall_Score,
            "Transcript":self.transcript,
            "Grammer_score" : self.grammer,
            "GPT_Grammer_score": self.GPTgrammer_score,
            "professional_score":self.professional_sc,
            "body_language_score":self.body_lang_score,
            "emotion_score":self.emotion_score,
            "presentability_score":self.presentiablity_score,
            "professional_comment":self.professionalcomment,
            "bodylang_comment":self.bodylang_comment,
            "emotioncomment":self.emotioncomment,
            "grommingcomment":self.grommingcomment,
            "dressingcomment":self.dressingcomment,
            "cheating_comment":self.cheating_comment
        }



class ExpRunner:
    def __init__(self, cfg, feature_extract=None):
        print("The configuration used is:", cfg)
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

        col_name = [
            'F0semitoneFrom27.5Hz_sma3nz_amean', 
            'jitterLocal_sma3nz_amean', 
            'F1frequency_sma3nz_amean', 
            'shimmerLocaldB_sma3nz_amean', 
            'loudness_sma3_amean', 
            'HNRdBACF_sma3nz_amean',
            'alphaRatioV_sma3nz_amean', 
            'hammarbergIndexV_sma3nz_amean',
            'slopeV0-500_sma3nz_amean',
            'slopeV0-500_sma3nz_stddevNorm',
            'slopeV500-1500_sma3nz_amean',
            'slopeV500-1500_sma3nz_stddevNorm',
            'loudnessPeaksPerSec',
            'VoicedSegmentsPerSec',
            'MeanVoicedSegmentLengthSec',
            'MeanUnvoicedSegmentLength'
        ]

        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        

        folderpath = "./datasets/ChaLearn/voice_data/voice_raw/test_data/"
        audio_list = os.listdir(folderpath)
        audio_list = list(filter(lambda x: ".praat" not in x, audio_list))   
        
        for idx, audio in enumerate(audio_list):
            file_path = os.path.join(folderpath, audio)
            id_link=file_path.split(".")[0]
            if os.path.isfile(file_path) and ".praat" not in file_path:
                
                videoPath  = "./datasets/ChaLearn/test/"
                node = AudioFileNode(file_path)
                print("*******************************************",type(smile), smile)
                y = smile.process_file(node.file_path)
                smile_features = y.filter(col_name).mean().to_dict()
                # Calculate unigrams and bigrams
                df_unigram, df_bigram,sentiment,sentiment_sc,transcription,gPTgrammer_Score,gpt_grammer_comment = cal_uni_bi(file_path)
                print("file_path is here 888888888888888888888888888888888888888888888888888888888888888888888888888888888",file_path)
                link_id=str(node.file_path).split(".")[1]
                gg=link_id.split("/")[-1]
                print("link id is here **********************************************************************************",gg,"gvvvgfvg")
                node.set_link_id(gg)
                # Set OpenSMILE features
                node.set_smile_features(smile_features)
                node.set_df_unigram(df_unigram)
                node.set_df_bigram(df_bigram)
                node.set_senti(sentiment)
                node.set_senti_sc(sentiment_sc)
                node.set_transcript(transcription)
                node.set_grammer_score(gPTgrammer_Score)
                node.set_grammer_comment(gpt_grammer_comment)


               # node.process_file1(smile, cal_uni_bi(), col_name , audio_file= file_path)
                grammer_final_score = ComputingValues(node.get_summary())

                grammer_final_score = (0.3*grammer_final_score + 0.7*gPTgrammer_Score)

                p_score,clarity_score,energy_score,fluency_score,articulation_score,communication_score,pace_a_clarity=sound_score(file_path,grammer_final_score)
                enrgy_fsc=(0.5*energy_score)+(0.5*node.sentiment_sc)
                
               
                node.set_ocean_values(dataset_output[idx])
                self.audio_nodes.append(node)
              #  print(f"Processed: {audio}")
                frames_dir_path = "datasets/ChaLearn/test_data"
              #  print("The folder path is five as" , frames_dir_path)
                folderaud = os.path.basename(file_path).split('/')[-1]
                folderaud = folderaud.replace(".wav","")
                frames_folder = os.path.join(frames_dir_path , folderaud)
                video_dir="datasets/ChaLearn/test"
                video_file = os.path.join(video_dir, folderaud + ".mp4")
                # video_path = os.path.join(video_dir , folderaud)
                print("video file path is -------------",video_file)

                gromming_prompt="You are an interviewer conducting a candidate interview. Based on the video, provide a summary of the person's gromming in the interview situation. Do not offer any suggestions or advice; just describe the person's gromming observed during the interview."
                body_langauage_prompt="You are an interviewer conducting a candidate interview. Based on the video, provide a summary of the person's body language in the interview situation. Do not offer any suggestions or advice; just describe the person's body language observed during the interview."
                dressing_prompt="You are an interviewer conducting a candidate interview. Based on the video, provide a summary of the person's dressing in the interview situation. Do not offer any suggestions or advice; just describe the person's dressing observed during the interview."
                professional_prompt="You are an interviewer conducting a candidate interview. Based on the video, provide a summary of the person's professionalism in the interview situation. Do not offer any suggestions or advice; just describe the professionalism observed during the interview."
                emotion_prompt="You are an interviewer conducting a candidate interview. Based on the video, provide a summary of the person's emotion's in the interview situation. Do not offer any suggestions or advice; just describe the emotion's observed during the interview."
                # combined_score,professionalcomment,bodylang,emotioncomment,grommingcomment,dressingcomment,final_cheating=  lvmain( frames_folder, max_new_tokens=512)
                # professional_score,professionalcomment=VLmain(file=video_file,prompt="""Give professionalism Score and a detailed comment about 3-4 lines , followed by this format : {"Score": [score],"Comment": [comment]}""")
                # body_lang_score,bodylang=VLmain(file=video_file,prompt="""Give Body Language Score and a detailed comment about 3-4 lines , followed by this format : {"Score": [score],"Comment": [comment]}""")
                # emotion_score,emotioncomment=VLmain(file=video_file,prompt="""Give Emotion Score and a detailed comment about 3-4 lines , followed by this format : {"Score": [score],"Comment": [comment]}""")
                # gromming_score,grommingcomment=VLmain(file=video_file,prompt="""Give Gromming Score and a detailed comment about 3-4 lines , followed by this format : {"Score": [score],"Comment": [comment]}""")
                # dresssing_score,dressingcomment=VLmain(file=video_file,prompt="""Give Dressing Score and a detailed comment about 3-4 lines , followed by this format : {"Score": [score],"Comment": [comment]}""")
                # final_cheating=""
                grommingcomment=get_comments_for_gpt(video_path=video_file,prompt=gromming_prompt)
                dressingcomment=get_comments_for_gpt(video_path=video_file,prompt=dressing_prompt)
                professionalcomment=get_comments_for_gpt(video_path=video_file,prompt=professional_prompt)
                bodylang=get_comments_for_gpt(video_path=video_file,prompt=body_langauage_prompt)
                emotioncomment=get_comments_for_gpt(video_path=video_file,prompt=emotion_prompt)
                final_cheating=""
                node.set_comments(professionalcomment,bodylang,emotioncomment,grommingcomment,dressingcomment,final_cheating)
                gromming_score_prompt="calculate the gromming score using the text"
                dressing_score_prompt="calculate the dressing score using the text"
                professional_score_prompt="calculate the professional score using the text"
                body_lang_prompt="calculate the body language score using the text"
                emotion_score_prompt="calculate the emotion score using the text"
                professional_score=get_score(professional_score_prompt,professionalcomment)
                gromming_score=get_score(gromming_score_prompt,grommingcomment)
                dressing_score=get_score(dressing_score_prompt,dressingcomment)
                body_lang_score=get_score(body_lang_prompt,bodylang)
                emotion_score=get_score(emotion_score_prompt,emotioncomment)

                print("The node comment is  3333333333333333333333333333333set",node.professionalcomment)
                
                LLava_professional_score=professional_score
                LLava_body_lang_score=body_lang_score
                LLava_emotion_score=emotion_score
                LLava_gromming_score=gromming_score
                LLava_dressing_score=dressing_score
                # LLava_professional_score=combined_score["Professionalism"]*10
                # LLava_body_lang_score=combined_score["Body Language"]*10
                # LLava_emotion_score=combined_score["Emotion"]*10
                # LLava_gromming_score=combined_score["Grooming"]*10
                # LLava_dressing_score=combined_score["Dressing"]*10


                mean, std , skew , kurtosis , emotion_lst,Brow_Lower,Lip_Corner_Puller,Upper_Lip_Raiser,Lips_Part, expression_counts ,non_verbal_counts=  CallMethod(frames_folder)

                face_smiles_per = expression_counts['Smile']

                face_Rapport_per = non_verbal_counts['Rapport']

                action_unit_positive_score = (face_smiles_per + face_Rapport_per) /2


                # print(emotion_lst)
                print("ACTION UNITS",Brow_Lower,Lip_Corner_Puller,Upper_Lip_Raiser,Lips_Part)
        
                if Brow_Lower>=21.08 and Brow_Lower<=83.6:
                    bl=25
                else :
                    bl=0
                if Lip_Corner_Puller>=15.65 and Lip_Corner_Puller<=28.56:
                    lcp=25
                else:
                    lcp=0
                if Upper_Lip_Raiser>=14.99 and Upper_Lip_Raiser<=23.23:
                    ulr=25
                else:
                    ulr=0
                if Lips_Part>=29.06 and Lips_Part<=65.19:
                    lp=25
                else:
                    lp=0
                Action_Score=(bl+lcp+ulr+lp)
                Action_Score = (Action_Score + action_unit_positive_score ) /2
                print("Action Score",Action_Score)
                
                emotion_counts = {
                    'angry': 0,
                    'disgust': 0,
                    'fear': 0,
                    'happy': 0,
                    'sad': 0,
                    'surprise': 0,
                    'neutral': 0
                }
                confidence=0
                # Iterate through the data to count dominant emotions
                for entry in emotion_lst:
                    for face in entry:
                        dominant_emotion = face['dominant_emotion']
                        confidence+=face['face_confidence']
                        if dominant_emotion in emotion_counts:
                            emotion_counts[dominant_emotion] += 1
                total_emotion_count=len(emotion_lst)
                # Print the counts
                # confidence_score=(confidence/total_emotion_count)

                videoPath = videoPath + folderaud+".mp4"
                
                list_dir = os.listdir(frames_folder)
                #f = open("temp.txt", "x")
              

                res =  process_in_batches(list_dir ,frames_folder ,batch_size = 64)
     
                print(res)
                Score=0
                # print(confidence_score)
                positive_emotion=(emotion_counts["happy"]/total_emotion_count)*100
                negative_emotion=((emotion_counts["angry"]+emotion_counts["fear"]+emotion_counts["sad"]+emotion_counts["disgust"])/total_emotion_count)*100
                netural_emotion=(emotion_counts["neutral"]/total_emotion_count)*100
                final_emotion={"positive":positive_emotion,"negative":negative_emotion,"neutral":netural_emotion}
                print("Emotion Counts",emotion_counts)
                print("positive Emotion",positive_emotion)
                print("negative Emotion",negative_emotion)
                print("netutal",netural_emotion)
                if final_emotion["positive"]>=final_emotion["negative"] and final_emotion["positive"]>=final_emotion["neutral"]:
                    f_emotion={"positive":final_emotion["positive"],"negative":0,"neutral":0}
                if final_emotion["negative"]>=final_emotion["positive"] and final_emotion["negative"]>=final_emotion["neutral"]:
                    f_emotion={"negative":final_emotion["negative"],"positive":0,"neutral":0} 
                else:
                    f_emotion={"neutral":final_emotion["neutral"],"positive":0,"negative":0}
                # f_emotion=max(final_emotion)
                if netural_emotion==50.0:
                    Score=50
                if netural_emotion>50.0:
                    Score=50+(netural_emotion/10)*3
                if positive_emotion>50.0:
                    Score=50+(positive_emotion/10)*5
                # if negative_emotion>50.0 and negative_emotion<60.0:
                #     Score=(negative_emotion)/1.4
                # if negative_emotion>60.0:
                #     Score=(negative_emotion)/1.4
                # negative_emotion=55
                
                if negative_emotion >50.0:
                   Score+=50
                   decrement=(negative_emotion-50)
                   Score-=decrement    
                print("score",Score)
                counts = {
                    11: 0, 12: 0, 13: 0,
                    21: 0, 22: 0, 23: 0,
                    31: 0, 32: 0, 33: 0,
                    41: 0, 42: 0, 43: 0,
                    51: 0, 52: 0, 53: 0,44:0
                }
                for i in res:
                    if i in counts:
                        counts[i] += 1
                hlwl_count = counts[11]
                hlwr_count = counts[12]
                hlwc_count = counts[13]
                hrwl_count = counts[21]
                hrwr_count = counts[22]
                hrwc_count = counts[23]
                hdwl_count = counts[31]
                hdwr_count = counts[32]
                hdwc_count = counts[33]
                huwl_count = counts[41]
                huwr_count = counts[42]
                huwc_count = counts[43]
                hswl_count = counts[51]
                hswr_count = counts[52]
                hswc_count = counts[53]
                fnic_count = counts[44]
                # Professional Score
                # good_pose=((hlwc_count+hrwc_count+huwc_count+hdwc_count)/len(res))*100
                # bad_pose=hlwl_count+hlwr_count+hrwl_count+hrwr_count+hdwl_count+hdwr_count+huwl_count+huwr_count+hswl_count+hswr_count
                # Professional_Score=good_pose
                Left_look=hlwl_count+hrwl_count+huwl_count+hdwl_count+hswl_count
                Right_look=hlwr_count+hrwr_count+huwr_count+hdwr_count+hswr_count
                Centre_look=hlwc_count+hrwc_count+huwc_count+hdwc_count+hswc_count
                head_left=hlwc_count+hlwl_count+hlwr_count
                head_upper=huwc_count+huwl_count+huwr_count
                head_straight=hswc_count+hswl_count+hswr_count
                head_down=hdwc_count+hdwl_count+hdwr_count
                head_right=hrwc_count+hrwl_count+hrwr_count
                # Total_Head_Movements = hlwl_count + hlwr_count + hlwc_count + hrwl_count + hrwr_count + hrwc_count + hdwl_count + hdwr_count + hdwc_count
                # Total_Eye_Movements = huwl_count + huwr_count + huwc_count + hswl_count + hswr_count + hswc_count
                # # Center_Focus_Ratio = (hlwc_count + hrwc_count + hdwc_count + huwc_count + hswc_count) / Total_Head_Movements
                # Center_Focus_Ratio=(Total_Eye_Movements/len(res))*100-(Total_Head_Movements/len(res))*100
                # # print("Professional Score",Professional_Score)
                # print("total head and eye movement",Total_Head_Movements,Total_Eye_Movements)
                # print("Center focus ratio",Center_Focus_Ratio)
                Left_Look=(Left_look/len(res))*100
                Right_Look=(Right_look/len(res))*100
                Centre_Look=(Centre_look/len(res))*100
                Head_Left=(head_left/len(res))*100
                Head_Upper=(head_upper/len(res))*100
                Head_Straight=(head_straight/len(res))*100
                Head_Down=(head_down/len(res))*100
                Head_Right=(head_right/len(res))*100
                face_not= (fnic_count/len(res))*100
                print("Left_look",Left_Look)
                print("Right_look",Right_Look)
                print("Centre_look",Centre_Look)
                print("Head_Left",Head_Left)
                print("Head_upper",Head_Upper)
                print("Head_straight",Head_Straight)
                print("Head_Down",Head_Down)
                print("Head_right",Head_Right)
                print("Face not in camera ", face_not)
                Focus_score=Centre_Look
                Head_Score=Head_Straight
                body_lang=(((Head_Score+Focus_score)/2)+LLava_body_lang_score)/2
                confidence_score=0.20*positive_emotion-0.20*negative_emotion+0.20*Focus_score+0.20*node.sentiment_sc+0.20*energy_score
                Professional_Score=(0.2*((Focus_score+Head_Score+Action_Score+confidence_score)/4))+(0.2*(LLava_professional_score))+(0.2*(LLava_gromming_score))+(0.2*(LLava_dressing_score))+(0.2*(LLava_body_lang_score))                                                                                            
                print("Professsional Score is ",Professional_Score)
                emotion_sc=(0.3*Score)+(0.7*LLava_emotion_score)
                sociablility_score=(emotion_sc+enrgy_fsc)/2
                # positive_attitude=0.2*enrgy_fsc+0.8*sociablility_score  ## Disscussion Required
                positive_attitude=energy_score
                Overall_Score=0.3*Professional_Score+0.2*positive_attitude+0.35*communication_score+0.15*sociablility_score
                Pesentability_score=(0.5*LLava_dressing_score)+(0.5*LLava_gromming_score)
                node.set_scores(p_score,clarity_score,energy_score,fluency_score,articulation_score,communication_score,pace_a_clarity,sociablility_score,confidence_score,positive_attitude,Professional_Score,Overall_Score,grammer_final_score,LLava_professional_score,body_lang,emotion_sc,Pesentability_score)
                node.set_mean(mean)
                node.set_std(std)
                node.set_skew(skew)
                node.set_kurtosis(kurtosis)
                node.set_headpose_and_eyegaze(hlwl_count,hlwr_count,hlwc_count,hrwl_count,hrwr_count,hrwc_count,hdwl_count,hdwr_count,
                                 hdwc_count,huwl_count,huwr_count,huwc_count,hswl_count,hswr_count,hswc_count,face_not)
                videoPath = None
                import torch
                torch.cuda.empty_cache()
        from resources.newprompt import AI_resultGenerator
        Finalnode = FullNode()
        print(self.audio_nodes)
        # ocean_values_mean_new=[]
        df_unigram_new = pd.DataFrame() 
        df_bigram_new = pd.DataFrame()
        sentimentt_ls = []
        f0_semitone_mean_ls = []
        jitter_local_mean_ls = [] 
        f1_frequency_mean_ls = []
        shimmer_local_db_mean_ls = []
        loudness_mean_ls  = []
        hnr_mean_ls = []
        alpha_ratio_mean_ls = []
        hammarberg_index_mean_ls = []
        slope_v0_500_mean_ls = []
        slope_v0_500_stddev_norm_ls = []
        slope_v500_1500_mean_ls = []
        slope_v500_1500_stddev_norm_ls = []
        loudness_peaks_per_sec_ls = []
        voiced_segments_per_sec_ls = []
        mean_voiced_segment_length_sec_ls = []
        mean_unvoiced_segment_length_ls= []
        meann_ls= []
        stdd_ls = []
        skeww = [] 
        kurtosiss = []
        hlwl_count1 = []
        hlwr_count1 = []
        hlwc_count1 = []
        hrwl_count1 =[]
        hrwr_count1 = []
        hrwc_count1 = []
        hdwl_count1 = []
        hdwr_count1 = []
        hdwc_count1 = []
        huwl_count1 =[] 
        huwr_count1= []
        huwc_count1  = []
        hswl_count1 = []
        hswr_count1 = []
        hswc_count1= []
        # shoulder = []
        sentiment_sc_ls = []
        energyscore_ls = []
        pacescore_ls = []
        clarityscore_ls = []
        fluencyscore_ls = []
        paceandclarityscore_ls = []
        articulationscore_ls = []
        communicationscore_ls = []
        sentiment_ls = []
        sociablility_score1_ls = []
        face_confidence_score_ls= []
        positive_attitude1_ls = []
        Professional_Score1_ls= []
        Overall_Score1_ls =[]
        ocean_values_ls=[]
        professional_sc_ls=[]  # not overall
        # mean_action_unit=[]
        transcriptss_ls=[]
        grammer_Score1_ls= []
        link_id_ls=[]
        body_lang_ls=[]
        emotion_score_ls=[]
        presentabilty_ls=[]
        professionalcomment_ls=[]
        bodylang_comment_ls=[]
        emotioncomment_ls=[]
        grommingcomment_ls=[]
        dressingcomment_ls=[]
        cheating_comment_ls=[]
        gpt_grammer_comment_ls=[]
        
        
        for node in self.audio_nodes:
        
            link_id_ls.append(node.link_id)
            summary = node.get_summary()
            print("rthe ode prodfessional comment after the node.sumary ",node.professionalcomment)

            final_grammer_score=0.1*node.grammer+0.9*node.GPTgrammer_score

            print("The GPT score is:",node.GPTgrammer_score )
            grammer_Score1_ls.append(final_grammer_score)
            
            # score_dict={"pace_score":p_score,"clarity_score":clarity_score,"energy_score":enrgy_fsc,"fluency_score":fluency_score,"articulation_score":articulation_score,"pace_and_clarity_score":
                        # pace_a_clarity,"communication_score":communication_score,"Socialbility_Score":sociablility_score,"Confidence_Score":confidence_score,"Professional_Score":Professional_Score,"Overall_Score":Overall_Score}
            ocean_values_ls.append(node.ocean_values)
            ocean_values_mean = np.vstack(ocean_values_ls)
            
            df_unigram_ls = pd.concat([df_unigram_new, summary['df_unigram']], ignore_index=True)
            transcriptss_ls.append(summary["Transcript"])
            # df_unigram = df_unigram.append(summary['df_unigram'], ignore_index=True)
            df_bigram_ls = pd.concat([df_bigram_new,summary['df_bigram']], ignore_index=True)
            sentimentt_ls.append(summary['sentiment'])
            f0_semitone_mean_ls.append(summary['f0_semitone_mean'])
            jitter_local_mean_ls.append(summary['jitter_local_mean'])
            f1_frequency_mean_ls.append(summary['f1_frequency_mean'])
            shimmer_local_db_mean_ls.append(summary['shimmer_local_db_mean'])
            loudness_mean_ls.append(summary['loudness_mean'])
            hnr_mean_ls.append(summary['hnr_mean'])
            alpha_ratio_mean_ls.append(summary['alpha_ratio_mean'])
            hammarberg_index_mean_ls.append(summary['hammarberg_index_mean'])
            slope_v0_500_mean_ls.append(summary['slope_v0_500_mean'])
            slope_v0_500_stddev_norm_ls.append(summary['slope_v0_500_stddev_norm'])
            slope_v500_1500_mean_ls.append(summary['slope_v500_1500_mean'])
            slope_v500_1500_stddev_norm_ls.append(summary['slope_v500_1500_stddev_norm'])
            loudness_peaks_per_sec_ls.append(summary['loudness_peaks_per_sec'])
            voiced_segments_per_sec_ls.append(summary['voiced_segments_per_sec'])
            mean_voiced_segment_length_sec_ls.append(summary['mean_voiced_segment_length_sec'])
            mean_unvoiced_segment_length_ls.append(summary['mean_unvoiced_segment_length'])
            meann_ls.append(summary['mean_Action_unit'])
            mean_action_unit = np.vstack(meann_ls)
            stdd_ls.append(summary['Std_Action_unit'])
            mean_std_unit = np.vstack(stdd_ls)
            skeww.append(summary['Skewness_Action_unit'])
            mean_skeweness_unit=np.vstack(skeww)
            kurtosiss.append(summary['Kurtosis_Action_unit'])
            hlwl_count1.append(summary['headleft_watchleft'])
            hlwr_count1.append(summary['headleft_watchright'])
            hlwc_count1.append(summary['headleft_watchcentre'])
            hrwl_count1.append(summary['headright_watchleft'])
            hrwr_count1.append(summary['headright_watchright'])
            hrwc_count1.append(summary['headright_watchcentre'])
            hdwl_count1.append(summary['headdown_watchleft'])
            hdwr_count1.append(summary['headdown_watchright'])
            hdwc_count1.append(summary['headdown_watchcentre'])
            huwl_count1.append(summary['headup_watchleft'])
            huwr_count1.append(summary['headup_watchright'])
            huwc_count1.append(summary['headup_watchcentre'])
            hswl_count1.append(summary['headstraight_watchleft'])
            hswr_count1.append(summary['headstraight_watchright'])
            hswc_count1.append(summary['headstraight_watchcentre'])
            # shoulder.append(summary['shoulderPose'])
            sentiment_sc_ls.append(summary['sentiment_score'])
            energyscore_ls.append(summary['energy'])
            pacescore_ls.append(summary['pace'])
            clarityscore_ls.append(summary['clarity'])
            fluencyscore_ls.append(summary['fluency'])
            paceandclarityscore_ls.append(summary['pace_and_clarity'])
            articulationscore_ls.append(summary['Articulation'])
            communicationscore_ls.append(summary['Communication'])
            sentiment_ls.append(summary['sentiment'])
            sociablility_score1_ls.append(summary['Sociability_score'])
            face_confidence_score_ls.append(summary['Confidence_score'])
            positive_attitude1_ls.append(summary['Positive_attitude'])
            Professional_Score1_ls.append(summary['Overall_Professional_Score'])
            professional_sc_ls.append(summary['professional_score'])
            Overall_Score1_ls.append(summary['Overall_Score'])
            body_lang_ls.append(summary['body_language_score'])
            emotion_score_ls.append(summary['emotion_score'])
            presentabilty_ls.append(summary['presentability_score'])
            professionalcomment_ls.append(node.professionalcomment)
            bodylang_comment_ls.append(node.bodylang_comment)
            emotioncomment_ls.append(node.emotioncomment)
            grommingcomment_ls.append(node.grommingcomment)
            dressingcomment_ls.append(node.dressingcomment)
            cheating_comment_ls.append(node.cheating_comment)
            gpt_grammer_comment_ls.append(node.gpt_grammer_comment)
            
# Compute the averages and set the values in Finalnode
        from collections import Counter
        print("In Final node block for mean calculation -----------------------------------------------------------------------")
        print("All lists are printed as follows:--")
        # print(Finalnode.get_summary())
        
        counts = Counter(sentimentt_ls)

        # Find the maximum occurrence
        max_occurrence = max(counts.values())

        most_common_strings = [key for key, value in counts.items() if value == max_occurrence]
        Finalnode.set_gpt_grammer_comment_ls(gpt_grammer_comment_ls)
        Finalnode.set_link_id_ls(link_id_ls)
        Finalnode.avg_oceanvalues(ocean_values_mean)
        Finalnode.avg_df_unigram(df_unigram_ls)
        Finalnode.avg_df_bigram(df_bigram_ls)
        Finalnode.avg_sentiment(most_common_strings)
        Finalnode.avg_professional_sc(professional_sc_ls)
        Finalnode.avg_f0_semitone_mean(f0_semitone_mean_ls)
        Finalnode.avg_jitter_local_mean(jitter_local_mean_ls)
        Finalnode.avg_f1_frequency_mean(f1_frequency_mean_ls)
        Finalnode.avg_shimmer_local_db_mean(shimmer_local_db_mean_ls)
        Finalnode.avg_loudness_mean(loudness_mean_ls)
        Finalnode.avg_hnr_mean(hnr_mean_ls)
        Finalnode.avg_alpha_ratio_mean(alpha_ratio_mean_ls)
        Finalnode.avg_hammarberg_index_mean(hammarberg_index_mean_ls)
        Finalnode.avg_slope_v0_500_mean(slope_v0_500_mean_ls)
        Finalnode.avg_slope_v0_500_stddev_norm(slope_v0_500_stddev_norm_ls)
        Finalnode.avg_slope_v500_1500_mean(slope_v500_1500_mean_ls)
        Finalnode.avg_slope_v500_1500_stddev_norm(slope_v500_1500_stddev_norm_ls)
        Finalnode.avg_loudness_peaks_per_sec(loudness_peaks_per_sec_ls)
        Finalnode.avg_voiced_segments_per_sec(voiced_segments_per_sec_ls)
        Finalnode.avg_mean_voiced_segment_length_sec(mean_voiced_segment_length_sec_ls)
        Finalnode.avg_mean_unvoiced_segment_length(mean_unvoiced_segment_length_ls)
        Finalnode.avg_mean(mean_action_unit)
        Finalnode.avg_std(mean_std_unit)
        Finalnode.avg_skew(mean_skeweness_unit)
        Finalnode.avg_kurtosis(kurtosiss)
        # Finalnode.avg_shoulder(shoulder)
        Finalnode.avg_sentiment_sc(sentiment_sc)
        Finalnode.avg_energyscore(energyscore_ls)
        Finalnode.avg_pacescore(pacescore_ls)
        Finalnode.avg_clarityscore(clarityscore_ls)
        Finalnode.avg_fluencyscore(fluencyscore_ls)
        Finalnode.avg_paceandclarityscore(paceandclarityscore_ls)
        Finalnode.avg_articulationscore(articulationscore_ls)
        Finalnode.avg_communicationscore(communicationscore_ls)
        Finalnode.avg_sociablility_score(sociablility_score1_ls)
        Finalnode.avg_face_confidence_score(face_confidence_score_ls)
        Finalnode.avg_presentablity_score(presentabilty_ls)
        Finalnode.avg_positive_attitude(positive_attitude1_ls)
        Finalnode.avg_Professional_Score(Professional_Score1_ls)
        Finalnode.avg_Overall_Score(Overall_Score1_ls)
        Finalnode.set_headpose_and_eyegaze(hlwl_count1,hlwr_count1,hlwc_count1,hrwl_count1,hrwr_count1,hrwc_count1,hdwl_count1,hdwr_count1,
                                 hdwc_count1,huwl_count1,huwr_count1,huwc_count1,hswl_count1,hswr_count1,hswc_count1
                                 
                                 )
        professional_comm=finalcomment(professionalcomment_ls)
        body_comm=finalcomment(bodylang_comment_ls)
        emotion_comm=finalcomment(emotioncomment_ls)
        gromming_comm=finalcomment(grommingcomment_ls)
        dressing_comm=finalcomment(dressingcomment_ls)
        Finalnode.comment_list_joiner(professional_comm,body_comm,emotion_comm,gromming_comm,dressing_comm)
        Finalnode.avg_grammer(grammer_Score1_ls)
        Finalnode.avg_emotion_score(emotion_score_ls)
        Finalnode.avg_body_language(body_lang_ls)
        Finalnode.set_transcript(transcriptss_ls)
        transcription = [{"id": id_, "transcript": transcript} for id_, transcript in zip(Finalnode.link_id_ls,Finalnode.transcripts)]
        Finalnode.set_new_trancript_ls_of_dict(transcription)

        # Generate the final results
        # print("articylation hjbhjbhhgjtbjk5btj---------------------------------------",Finalnode.articulationscore)
        promptare = AI_resultGenerator(Finalnode.get_summary())
        score_dict = {
            "transcription":Finalnode.new_transcript
            # "pace_score": Finalnode.pacescore,
            # "clarity_score": Finalnode.clarityscore,
            # "energy_score": Finalnode.energyscore,
            # "fluency_score": Finalnode.fluencyscore,
            # "articulation_score": Finalnode.articulationscore,
            # "pace_and_clarity_score": Finalnode.paceandclarityscore,
            # "communication_score": Finalnode.communicationscore,
            # "socialbility_score": Finalnode.sociablility_score,
            # "confidence_score": Finalnode.face_confidence_score,
            # "professional_score": Finalnode.Professional_Score,
            # "overall_score": Finalnode.Overall_Score,
            # "positive_attitude":Finalnode.positive_attitude,
            # "grammer_score":Finalnode.grammer_Score,
            # "sentiment_score":Finalnode.sentiment
        }
        comment_dict={
            "professional_comment":Finalnode.professionalcomment,
            "bodylang_comment":Finalnode.bodylang_comment,
            "emotioncomment":Finalnode.emotioncomment,
            "grommingcomment":Finalnode.grommingcomment,
            "dressingcomment":Finalnode.dressingcomment,
            
            "gpt_grammer_comment":Finalnode.gpt_grammer_comment_ls
        }

        print("RESULTS ARE :",Finalnode.get_summary())
        return promptare , score_dict ,comment_dict 

            

    def run(self):
        self.train()
        self.test()

    def log_cfg_info(self):
        string = json.dumps(self.cfg, sort_keys=True, indent=4, separators=(',', ':'))
        self.logger.info(string)

    def data_extract(self, dataloader, output_dir):
        return self.trainer.data_extract(self.model, dataloader, output_dir)