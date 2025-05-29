import pandas as pd
import traceback
import os
import json
import numpy as np
import torch
from dpcv.data.datasets.build import build_dataloader
from dpcv.modeling.networks.build import build_model
from dpcv.modeling.loss.build import build_loss_func
from dpcv.modeling.solver.build import build_solver, build_scheduler
from dpcv.engine.build import build_trainer
from dpcv.evaluation.summary import TrainSummary
from dpcv.checkpoint.save import  resume_training, load_model
from dpcv.tools.logger import make_logger
import pandas as pd
from dpcv.text_mining_using_python import cal_uni_bi

from .score_maker import articute_score_maker
from .comment_fetcher import evaluate_self_awareness,get_comments_for_gpt, getcomment_etiquette,convert_images_to_base64,finalcomment,getcomment_communication,finalcomment_forselfawreness
from openai import OpenAI
import os


from decouple import config
client = OpenAI(api_key=config("OPENAI_API_KEY"))


MAX_TOKENS = 10000
OUTPUT_TOKENS = 300

Power_sentence_set =  {
    "I don’t wait to be told — I anticipate needs and act.",
    "I take ownership from day one.",
    "I’m always looking for ways to optimize and improve workflows.",
    "I believe in taking the first step, even when the path isn't fully clear.",
    "I act fast, then refine based on feedback.",
    "If something needs to be done, I’m on it — even if it’s not in my job description.",
    "I don’t just identify problems — I bring solutions.",
    "When things get complex, I break them down and find clarity.",
    "I thrive in ambiguity and enjoy building structure where there is none.",
    "I question assumptions and test ideas before scaling them.",
    "I approach problems with curiosity, not panic.",
    "I enjoy challenging the status quo when it leads to better outcomes.",
    "I believe the best outcomes happen through clear, honest communication.",
    "I’m comfortable working across teams and bringing stakeholders together.",
    "I listen first — and speak with intention.",
    "I tailor my communication style to different audiences.",
    "I believe collaboration is a multiplier — not a compromise.",
    "I ask the right questions to bring alignment and clarity.",

    # Leadership & Growth Mindset
    "I lead by example and bring others along for the win.",
    "I see every challenge as a learning opportunity.",
    "I give and receive feedback as a tool for growth.",
    "I empower those around me to succeed.",
    "I stay calm under pressure and focus on solutions.",
    "I don't need a title to lead — I lead through action and influence.",

    # Execution & Delivery
    "I’m outcome-driven — I focus on impact, not just activity.",
    "I prioritize well and execute fast without compromising quality.",
    "Deadlines are non-negotiable for me — I deliver.",
    "I break big goals into small, actionable steps.",
    "I measure success through results, not just effort.",
    "I hold myself accountable for delivering top-tier work consistently.",

    # Strategic Thinking
    "I align my work with the bigger picture — always asking 'why' behind the 'what'.",
    "I think long-term but act short-term to drive momentum.",
    "I always consider the downstream impact of today's decisions.",
    "I balance execution with strategy — knowing when to zoom in and when to zoom out.",
    "I ask tough questions to ensure we’re solving the right problems.",
    "I focus on scalable, sustainable growth — not just quick wins.",

    # Culture Fit & Values
    "I bring positivity and resilience, even under pressure.",
    "I value empathy just as much as efficiency.",
    "I’m not just looking for a job — I’m looking to contribute to a mission.",
    "I treat people with respect, regardless of their title or background.",
    "I believe diverse teams make better decisions.",
    "I hold myself and others to a high standard of integrity.",

    # Adaptability
    "I pivot quickly when priorities change.",
    "I’m comfortable in fast-paced, ever-evolving environments.",
    "I see change as an opportunity to innovate, not a threat.",
    "I adapt my approach based on feedback and results.",
    "I can jump into unfamiliar territory and figure things out fast.",
    "I’ve learned to stay flexible while still staying focused.",

    # Results Orientation
    "I measure success by the value I deliver.",
    "I track progress and celebrate wins — big or small.",
    "I’m focused on creating impact, not just checking boxes.",
    "I don’t just set goals — I hit them.",
    "I make data-informed decisions that move the needle.",
    "I follow through — always.",
     
 }

class FinalNode:
    def __init__(self, file_path):
        self.file_path = file_path
        self.link_id = None
        self.transcript = None
        self.pronounciation_comment = None
        self.pronounciation_score = None
        self.articulation_score = None
        self.articulation_comment = None
        self.bodylang_comment = None
      
        self.overall_score = None

        self.pace_score = None 
        self.pace_score_comment = None

        self.sentiment_score = None
        self.sentiment_comment = None
        self.sentiment_comment = None
        self.self_awareness_score = None
        self.grammar_score = None 
        self.grammar_comment = None

        self.ocean_values = None

        self.communication_score = None
        self.communication_comment = None
  
        self.etiquette_score = None
        self.etiquette_comment = None



    def get_summary(self):
           return {
            "file_path": self.file_path,
            "link_id": self.link_id,
            "ocean_values": self.ocean_values,
            "sentiment_comment": self.sentiment_comment,
            "transcript": self.transcript,
            "grammar_score": self.grammar_score,
            "self_awareness_score": self.self_awareness_score,
            "self_awareness_comment": self.self_awareness_comment,
            "grammar_comment": self.grammar_comment,
            "pace_score": self.pace_score,
            "communication_comment": self.communication_comment or "No Comment",
            "pace_score_comment": self.pace_score_comment,
            "pronounciation_score": self.pronounciation_score,
            "pronounciation_comment" : self.pronounciation_comment or "No Comment",
            "articulation_score": self.articulation_score,
            "articulation_comment": self.articulation_comment,
            "sentiment_score": self.sentiment_score,
            "pace_comment": self.pace_score_comment,
            "transcription": self.transcription or [],
            "ocean_values":self.ocean_values, 
            "bodylang_score": self.bodylang_score,
            "bodylang_comment": self.bodylang_comment,

           
            "etiquette_score": self.etiquette_score if self.etiquette_score is not None else 0.0,
            "etiquette_comment": self.etiquette_comment or "No Comment",     

        }



class AudioFileNode:
    def __init__(self, file_path):
        self.file_path = file_path
        self.link_id = None
        self.transcript = None
        self.articulation_score = None
        self.pronounciation_score = None
        self.articulation_comment = None
        self.pronounciation_comment = None
        self.df_unigrams = None
        self.bodylang_score = None
        self.bodylang_comment = None
        self.overall_score = None
        self.pace_score = None
        self.pace_score_comment = None
        self.sentiment_score = None
        self.sentiment_comment = None
        self.grammar_score = None
        self.grammar_comment = None
        self.ocean_values = None
        self.communication_score = None
        self.self_awareness_score = None
        self.self_awareness_comment = None
        self.etiquette_score = None
        self.etiquette_comment = None

    def get_summary(self):
        return {
            "file_path": self.file_path or "Unknown",
            "link_id": self.link_id or "Unknown",
            "transcript": self.transcript or "No Transcript",
            "articulation_score": self.articulation_score if self.articulation_score is not None else 0.0,
            "pronounciation_score": self.pronounciation_score if self.pronounciation_score is not None else 0.0,
            "articulation_comment": self.articulation_comment or "No Comment",
            "pronounciation_comment": self.pronounciation_comment or "No Comment",
            "bodylang_score": self.bodylang_score if self.bodylang_score is not None else 0.0,
            "bodylang_comment": self.bodylang_comment or "No Comment",
            "overall_score": self.overall_score if self.overall_score is not None else 0.0,
            "df_unigrams" : self.df_unigrams,
            "pace_score": self.pace_score if self.pace_score is not None else 0.0,
            "pace_score_comment": self.pace_score_comment or "No Comment",
            "sentiment_score": self.sentiment_score if self.sentiment_score is not None else 0.0,
            "sentiment_comment": self.sentiment_comment or "No Comment",
            "grammar_score": self.grammar_score if self.grammar_score is not None else 0.0,
            "grammar_comment": self.grammar_comment or "No Comment",
            "ocean_values": self.ocean_values,
            "communication_score": self.communication_score if self.communication_score is not None else 0.0,
            "etiquette_score": self.etiquette_score if self.etiquette_score is not None else 0.0,
            "etiquette_comment": self.etiquette_comment or "No Comment",    
        }



class ExpRunner:

    def __init__(self, cfg, feature_extract=None):
     
        self.cfg = cfg
        # self.logger, self.log_dir = make_logger(cfg.TRAIN.OUTPUT_DIR)
        # self.log_cfg_info()
        # if not feature_extract:
        #     self.data_loader = self.build_dataloader()

        # self.model = self.build_model()
        # self.loss_f = self.build_loss_function()

        # self.optimizer = self.build_solver()
        # self.scheduler = self.build_scheduler()

        # self.collector = TrainSummary()
        # self.trainer = self.build_trainer()

        self.audio_nodes = []
    
    # def build_dataloader(self):
    #     return build_dataloader(self.cfg)

    # def build_model(self):
    #     return build_model(self.cfg)

    # def build_loss_function(self):
    #     return build_loss_func(self.cfg)

    # def build_solver(self):
    #     return build_solver(self.cfg, self.model)

    # def build_scheduler(self):
    #     return build_scheduler(self.cfg, self.optimizer)

    # def build_trainer(self):
    #     return build_trainer(self.cfg, self.collector, self.logger)
    



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


    def test(self,Questions):
        torch.cuda.empty_cache()
        folderpath = "./datasets/ChaLearn/voice_data/voice_raw/test_data/"
        audio_list = os.listdir(folderpath)
        videoPath  = "./datasets/ChaLearn/test/"
       
        for idx, audio in enumerate(audio_list):
            file_path = os.path.join(folderpath, audio)
            print(file_path)
            if os.path.isfile(file_path) and ".praat" not in file_path:
                videoPath  = "./datasets/ChaLearn/test/"
                node = AudioFileNode(file_path)
                df_unigrams ,sentiment_score_value , sentiment_comment_value , final_transcription, grammer_score , grammer_comment , pace_score  ,pace_comment ,Pronounciation_score,pronounciation_comment= cal_uni_bi(file_path)
                from sentence_transformers import SentenceTransformer, util
                model = SentenceTransformer('all-MiniLM-L6-v2')
                transcript_embed = model.encode(final_transcription, convert_to_tensor=True)
                power_embeds = model.encode(list(Power_sentence_set), convert_to_tensor=True)
                cos_scores = util.cos_sim(transcript_embed, power_embeds)
                power_score = 0  
                for i, score in enumerate(cos_scores[0]):
                    if score > 0.7:  
                        matched_sentence = list(Power_sentence_set)[i]
                        print(f"Matched: {matched_sentence} (Score: {score:.2f})")
                        power_score += 1
                
                print("The power socre calculated is " , power_score)
                print("******************************** Final Transcription:***********************************", grammer_score,idx)
               
                if idx < len(Questions) and "$#@True$#@" in Questions[idx]:    
                    print("____________________________")
                    print("&&&& inside the self awareness &&&&")
                    print(Questions[idx])
                    self_awreness = evaluate_self_awareness(Questions[idx], final_transcription)
                    node.self_awareness_score = self_awreness["score"]
                    node.self_awareness_comment = self_awreness["self_awareness"]
                    print("None is 1 ",node.self_awareness_score)
                    print("None is 2 ",node.self_awareness_comment)
                
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
                node.df_unigrams = df_unigrams
                articulation_scores , articulation_comment = articute_score_maker(power_score,node.get_summary() , final_transcription)
                node.articulation_score = articulation_scores or 0.0
                node.pronounciation_score = Pronounciation_score or 0.0
                node.articulation_comment=articulation_comment or 0.0
                node.pronounciation_comment=pronounciation_comment or 0.0
                node.df_unigrams = df_unigrams

                grammer_final_score =grammer_score

                node.grammar_score = grammer_final_score or 0.0

                frames_dir_path = "datasets/ChaLearn/test_data"
                folderaud = os.path.basename(file_path).split('/')[-1]
                folderaud = folderaud.replace(".wav","")
                frames_folder = os.path.join(frames_dir_path , folderaud)
                video_dir="datasets/ChaLearn/test"
                video_file = os.path.join(video_dir, folderaud + ".mp4")      
                print("video file path is " , video_file)

                base_encoder_frames = convert_images_to_base64(frames_folder)

                etiquette_prompt = "You are an interviewer conducting a candidate interview. Based on the video, provide a summary of the person's Etiquette in the interview situation. Do not offer any suggestions or advice; just describe the person's Etiquette observed during the interview."
                

                etiquette_score,etiquette_comment = getcomment_etiquette(base64Frames=base_encoder_frames,prompt=etiquette_prompt,transcript=node.transcript , typeo = "Etiquette")

                if etiquette_score == None:

                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are an AI Interviewer given as analysis of candidates etiquette . "
                                "Give a score on the bases on analysis out of 100"
                                "Do not mention any reasoning or expaination of the score "
                                "Only five value sin integer format"
                            )
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Analyze the dressing assessments {etiquette_comment} "
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
                    etiquette_score = final_comment

                node.etiquette_comment = etiquette_comment or 0.0
                
                if etiquette_score == None :
                    etiquette_score = 10


                node.etiquette_score = etiquette_score


                body_langauage_prompt = "You are an interviewer conducting a candidate interview. Based on the video, provide a summary of the person's body language in the interview situation. Do not offer any suggestions or advice; just describe the person's body language observed during the interview.comment only in one line."
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

                node.bodylang_comment = bodylang_comment or 0.0

                if body_lang_score == None :
                    body_lang_score = 10

                try:
        
                    def safe_float(value):
                        if isinstance(value, (int, float)):  
                            return value
                        try:
                            return float(value)  
                        except (ValueError, TypeError):  
                            return 0.0

                    etiquette_score = safe_float(etiquette_score)
                    body_lang_score = safe_float(body_lang_score)  
                    sentiment_score = safe_float(getattr(node, 'sentiment_score', 0.0))
                    articulation_score = safe_float(getattr(node, 'articulation_score', 0.0))
                    Pronounciation_score = safe_float(getattr(node, 'pronounciation_score', 0.0))
                    
                    print("The pace scoer is " , pace_score)
                    
                    pace_score = pace_score
                    grammar_score = safe_float(getattr(node, 'grammar_score', 0.0))

                    node.etiquette_score = etiquette_score
                    
                    node.bodylang_score = body_lang_score

                    
                    communication_score = (articulation_score + pace_score + grammar_score) / 3
                    node.communication_score = round(communication_score, 2)

                    print("[[[[[[[[[[[[[[[[[[[[[[[[[[[11122222222222222222]]]]]]]]]]]]]]]]]]]]]]]]]]]")
                    print(grammar_score,pace_score,articulation_score,sentiment_score,communication_score,node.self_awareness_score,node.self_awareness_comment)
                    
                    self.audio_nodes.append(node)
                    print(node.get_summary())
                    
                except Exception as e:
                    print(f"❌ Error computing final node summary: {e}")
                    traceback.print_exc()

 
        Finalnode  = FinalNode(videoPath)    
 
        list_link_id = []
        list_ocean_values = []
    
        list_transcript = []
        list_self_awareness_score = []
        list_self_awareness_comment = []
        list_grammar_score = []
        list_grammar_comment = []
 
        list_pace_score = []
        list_pace_comment = []
 
        list_articulation_score = []
        list_Pronounciation_score=[]
        list_articulation_comment = []
        list_pronounciation_comment = []
        list_sentiment_score = []
        list_sentiment_comment = []
        
 
        list_bodylang_score = []
        list_bodylang_comment = []
 
        list_unigrams=[]
 
        list_communication_score=[]
 
        list_etiquette_score = []
        list_etiquette_comment = []
 
   
 
        for node in self.audio_nodes :
            try :
                list_link_id.append(node.link_id or "Unknown")
                summary = node.get_summary()
                list_grammar_score.append(node.grammar_score if node.grammar_score is not None else 0.0)
                list_grammar_comment.append(node.grammar_comment or "No Comment")
                list_pace_score.append(node.pace_score if node.pace_score is not None else 0.0)
                list_pace_comment.append(node.pace_score_comment or "No Comment")
                list_articulation_score.append(node.articulation_score if node.articulation_score is not None else 0.0)
                list_Pronounciation_score.append(node.pronounciation_score if node.pronounciation_score is not None else 0.0)
                list_articulation_comment.append(node.articulation_comment or "No Comment")
                list_pronounciation_comment.append(node.pronounciation_comment or "No Comment")
                list_sentiment_comment.append(node.sentiment_comment or "No Comment")
                list_self_awareness_comment.append(node.self_awareness_comment or "No Comment")
                list_self_awareness_score.append(node.self_awareness_score if node.self_awareness_score is not None else 0.0)
                print("The list is <<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",list_self_awareness_score)
                
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<< ",list_self_awareness_comment)
                list_sentiment_score.append(node.sentiment_score if node.sentiment_score is not None else 0.0)
                list_transcript.append(summary.get("transcript", "No Transcript"))
                list_ocean_values.append(summary.get("ocean_values", [0] * 5))  # Default empty vector
 
                list_etiquette_score.append(node.etiquette_score if node.etiquette_score is not None else 0.0)
                list_etiquette_comment.append(node.etiquette_comment or "No Comment")
 
                list_bodylang_score.append(node.bodylang_score if node.bodylang_score is not None else 0.0)
                list_bodylang_comment.append(node.bodylang_comment or "No Comment")
 
 
                list_unigrams.append(node.df_unigrams if node.df_unigrams is not None else [])
 
                list_communication_score.append(node.communication_score if node.communication_score is not None else 0.0)
               
                print("<<<<<<<<<<<<<<<<<<<<<<<<<><><><><><><><><><>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                print("The transcript is " , list_transcript)
            
            
            except Exception as e:
                    print(f"Error computing final hfghfg node summary: {e}")
                    traceback.print_exc()   
 
        from collections import Counter
 
        print(f"Link ID*********: {list_link_id}")
        print(f"Grammar Score: {list_grammar_score} | Comment: {list_grammar_comment}")
        print(f"Pace Score: {list_pace_score} | Comment: {list_pace_comment}")
        print(f"Articulation Score: {list_articulation_score} | Comment: {list_articulation_comment}")
        
        print(f"Pronounciation Score: {list_Pronounciation_score} | Comment: {list_pronounciation_comment}")
        print(f"Sentiment Score: {list_sentiment_score} | Comment: {list_sentiment_comment}")
        print(f"Transcript: {list_transcript}")
        print(f"OCEAN Values: {list_ocean_values}")
        print(f"Communication Score: {list_communication_score}")
        print(f"Unigrams: {list_unigrams}")
        print(f"list_etiquette : {list_etiquette_score}")
        print(f"list_etiquette_comment:{list_etiquette_comment}")
 
        try :
            body_comm = finalcomment(list_bodylang_comment , small = True)
            grammar_comm = finalcomment(list_grammar_comment , small  = True).replace("\"", "")
            sentiment_comm = finalcomment(list_sentiment_comment , small  = True)
            print(list_self_awareness_comment,">>>>>!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("2222222222222222222")
            self_awareness_comm = finalcomment_forselfawreness(list_self_awareness_comment )
            print("------------------------------------------",self_awareness_comm)
            pace_comm = finalcomment(list_pace_comment , small  = True)
            articulation_comm = finalcomment(list_articulation_comment , small  = True)
            pronounciation_comm = finalcomment(list_pronounciation_comment , small  = True)
           
            etiquette_comm = finalcomment(list_etiquette_comment )
            print(" *****************" ,pronounciation_comm)
            
            # Compute safe means
            Finalnode.articulation_score = round(np.mean([x for x in list_articulation_score if x is not None]), 2)
            Finalnode.self_awareness_score = round(np.mean([x for x in list_self_awareness_score if x is not None]), 2)
            Finalnode.pronounciation_score = round(np.mean([x for x in list_Pronounciation_score if x is not None]), 2)
            Finalnode.bodylang_score = round(np.mean([x for x in list_bodylang_score if x is not None]), 2)
            Finalnode.ocean_values = np.mean([x for x in list_ocean_values if x is not None], axis=0).tolist()
            Finalnode.grammar_score = round(np.mean([x for x in list_grammar_score if x is not None]), 2)
            Finalnode.pace_score = round(np.mean([x for x in list_pace_score if x is not None]), 2)
            Finalnode.sentiment_score = round(np.mean([x for x in list_sentiment_score if x is not None]), 2)
            Finalnode.transcript = list_transcript
            Finalnode.etiquette_score = round(np.mean([x for x in list_etiquette_score if x is not None]), 2)
            
            # Assign comments
            Finalnode.self_awareness_comment = self_awareness_comm
            Finalnode.sentiment_comment = sentiment_comm
            Finalnode.grammar_comment = grammar_comm
            Finalnode.pace_score_comment = pace_comm
            Finalnode.articulation_comment = articulation_comm  
            Finalnode.pronounciation_comment = pronounciation_comm
            Finalnode.bodylang_comment = body_comm
            Finalnode.link_id = list_link_id
            Finalnode.df_unigrams = list_unigrams
            Finalnode.etiquette_comment = etiquette_comm

 
             # Generate final comments
            Finalnode.communication_comment = getcomment_communication(
                pace_comment=Finalnode.pace_score_comment,
                articulation_comment=Finalnode.articulation_comment,
                grammar_comment=Finalnode.grammar_comment,
 
            )
 
            Finalnode.communication_score = round(np.mean([x for x in list_communication_score if x is not None]), 2)
            Finalnode.transcription = [
                {"id": id_, "transcript": transcript}
                for id_, transcript in zip(Finalnode.link_id, Finalnode.transcript)
            ]
       
            print(Finalnode.transcription)
      
        except Exception as e:
            print(f"Error computing final hfghfg node summary: {e}")
            traceback.print_exc()  
  


        return Finalnode.get_summary()

    def run(self):
        self.train()
        self.test()    

    def log_cfg_info(self):
        string = json.dumps(self.cfg, sort_keys=True, indent=4, separators=(',', ':'))
        self.logger.info(string)

    def data_extract(self, dataloader, output_dir):

        return self.trainer.data_extract(self.model, dataloader, output_dir)

    