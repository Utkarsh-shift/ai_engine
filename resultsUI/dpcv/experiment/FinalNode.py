import numpy as np


class FullNode:
    def __init__(self, ):
        
        self.link_id_ls=None
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
        self.hnr_mean = None
        # self.shoulder=None
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
        self.transcripts=None
        self.grammer_Score = None
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
        self.new_transcript=None
        self.professional_sc=None
        self.body_lang_sc=None
        self.emotion_score=None
        self.presentablity_score=None
        self.professionalcomment=None,
        self.bodylang_comment=None,
        self.emotioncomment=None,
        self.grommingcomment=None,
        self.dressingcomment=None,
        self.cheating_comment=None
        self.gpt_grammer_comment_ls=None


    def avg_body_language(self, body_lang_sc):
        self.body_lang_sc = np.mean(body_lang_sc)
    def avg_emotion_score(self, emotion_score):
        self.emotion_score = np.mean(emotion_score)

    def set_new_trancript_ls_of_dict(self,ls_of_dict):
        self.new_transcript=ls_of_dict

    def set_link_id_ls(self, link_id_ls):
        self.link_id_ls = link_id_ls
    
    def set_gpt_grammer_comment_ls(self, gpt_grammer_comment_ls):
        self.gpt_grammer_comment_ls = gpt_grammer_comment_ls

    def avg_grammer(self, list_grammer):
        self.grammer_Score = np.mean(list_grammer)

    def avg_oceanvalues(self, list_of_list):
        self.ocean_values = np.mean(list_of_list, axis=0)
        
    def avg_df_unigram(self, list_of_dfs):
        self.df_unigram = list_of_dfs
        
    def avg_df_bigram(self, list_of_dfs):
        self.df_bigram = list_of_dfs

    def avg_sentiment(self, sentimentt):
        self.sentiment = sentimentt

    def avg_f0_semitone_mean(self, f0_semitone_mean):
        self.f0_semitone_mean = np.mean(f0_semitone_mean)

    def avg_jitter_local_mean(self, jitter_local_mean):
        self.jitter_local_mean = np.mean(jitter_local_mean)

    def avg_f1_frequency_mean(self, f1_frequency_mean):
        self.f1_frequency_mean = np.mean(f1_frequency_mean)

    def avg_shimmer_local_db_mean(self, shimmer_local_db_mean):
        self.shimmer_local_db_mean = np.mean(shimmer_local_db_mean)

    def avg_loudness_mean(self, loudness_mean):
        self.loudness_mean = np.mean(loudness_mean)

    def avg_hnr_mean(self, hnr_mean):
        self.hnr_mean = np.mean(hnr_mean)

    def avg_alpha_ratio_mean(self, alpha_ratio_mean):
        self.alpha_ratio_mean = np.mean(alpha_ratio_mean)

    def avg_hammarberg_index_mean(self, hammarberg_index_mean):
        self.hammarberg_index_mean = np.mean(hammarberg_index_mean)

    def avg_slope_v0_500_mean(self, slope_v0_500_mean):
        self.slope_v0_500_mean = np.mean(slope_v0_500_mean)

    def avg_slope_v0_500_stddev_norm(self, slope_v0_500_stddev_norm):
        self.slope_v0_500_stddev_norm = np.mean(slope_v0_500_stddev_norm)

    def avg_slope_v500_1500_mean(self, slope_v500_1500_mean):
        self.slope_v500_1500_mean = np.mean(slope_v500_1500_mean)

    def avg_slope_v500_1500_stddev_norm(self, slope_v500_1500_stddev_norm):
        self.slope_v500_1500_stddev_norm = np.mean(slope_v500_1500_stddev_norm)

    def avg_loudness_peaks_per_sec(self, loudness_peaks_per_sec):
        self.loudness_peaks_per_sec = np.mean(loudness_peaks_per_sec)

    def avg_voiced_segments_per_sec(self, voiced_segments_per_sec):
        self.voiced_segments_per_sec = np.mean(voiced_segments_per_sec)

    def avg_mean_voiced_segment_length_sec(self, mean_voiced_segment_length_sec):
        self.mean_voiced_segment_length_sec = np.mean(mean_voiced_segment_length_sec)

    def avg_mean_unvoiced_segment_length(self, mean_unvoiced_segment_length):
        self.mean_unvoiced_segment_length = np.mean(mean_unvoiced_segment_length)

    def avg_mean(self, meann):
        self.mean = np.mean(meann,axis=0)

    def avg_std(self, stdd):
        self.std = np.mean(stdd)

    def avg_skew(self, skeww):
        self.skew = np.mean(skeww)

    def avg_kurtosis(self, kurtosiss):
        self.kurtosis = np.mean(kurtosiss)

    # def avg_shoulder(self, shoulder):
    #     self.shoulder = np.mean(shoulder)

    def avg_sentiment_sc(self, sentiment_sc):
        self.sentiment = np.mean(sentiment_sc)

    def avg_energyscore(self, energyscore):
        self.energyscore = np.mean(energyscore)

    def avg_pacescore(self, pacescore):
        self.pacescore = np.mean(pacescore)

    def avg_clarityscore(self, clarityscore):
        self.clarityscore = np.mean(clarityscore)

    def avg_fluencyscore(self, fluencyscore):
        self.fluencyscore = np.mean(fluencyscore)

    def avg_paceandclarityscore(self, paceandclarityscore):

        self.paceandclarityscore = np.mean(paceandclarityscore)

    def avg_articulationscore(self, articulationscore):

        self.articulationscore = np.mean(articulationscore)

    def avg_communicationscore(self, communicationscore):

        self.communicationscore = np.mean(communicationscore)

    def avg_sociablility_score(self, sociablility_score1):

        self.sociablility_score = np.mean(sociablility_score1)

    def avg_face_confidence_score(self, face_confidence_score):

        self.face_confidence_score = np.mean(face_confidence_score)

    def avg_presentablity_score(self, presentablity_score):

        self.presentablity_score = np.mean(presentablity_score)

    def avg_positive_attitude(self, positive_attitude1):

        self.positive_attitude = np.mean(positive_attitude1)

    def avg_Professional_Score(self, Professional_Score1):

        self.Professional_Score = np.mean(Professional_Score1)

    def avg_Overall_Score(self, Overall_Score1):

        self.Overall_Score = np.mean(Overall_Score1)

    def comment_list_joiner(self, professionalcomment,bodylang_comment,emotioncomment,grommingcomment,dressingcomment):

        self.professionalcomment =professionalcomment
        self.bodylang_comment=bodylang_comment
        self.emotioncomment=emotioncomment
        self.grommingcomment=grommingcomment
        self.dressingcomment=dressingcomment

    def set_headpose_and_eyegaze(self,hlwl_count,hlwr_count,hlwc_count,hrwl_count,hrwr_count,hrwc_count,hdwl_count,hdwr_count,
                                hdwc_count,huwl_count,huwr_count,huwc_count,hswl_count,hswr_count,hswc_count
                                
                                ):
        




        self.hlwl_count=np.mean(hlwl_count)
        self.hlwr_count=np.mean(hlwr_count)
        self.hlwc_count=np.mean(hlwc_count)
        self.hrwl_count=np.mean(hrwl_count)
        self.hrwr_count=np.mean(hrwr_count)
        self.hrwc_count=np.mean(hrwc_count)
        self.hdwl_count=np.mean(hdwl_count)
        self.hdwr_count=np.mean(hdwr_count)
        self.hdwc_count=np.mean(hdwc_count)
        self.huwl_count=np.mean(huwl_count)
        self.huwr_count=np.mean(huwr_count)
        self.huwc_count=np.mean(huwc_count)
        self.hswl_count=np.mean(hswl_count)
        self.hswr_count=np.mean(hswr_count)
        self.hswc_count=np.mean(hswc_count)


    def set_transcript(self,transcripts):
        self.transcripts=transcripts

    def avg_professional_sc(self, professional_sc):

        self.professional_sc = np.mean(professional_sc)

    def get_summary(self):
        return {
            "ocean_values": self.ocean_values,
            "df_unigram": self.df_unigram,
            "df_bigram": self.df_bigram,
            # "sentiment": self.sentiment,
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
            # "shoulder": self.shoulder,
            "sentiment_score": self.sentiment,
            "energy_score": self.energyscore,
            "pace_score": self.pacescore,
            "clarity_score": self.clarityscore,
            "fluency_score": self.fluencyscore,
            "pace_and_clarity_score": self.paceandclarityscore,
            "articulation_score": self.articulationscore,
            "communication_score": self.communicationscore,
            "sociability_score": self.sociablility_score,
            "confidence_score": self.face_confidence_score,
            "positive_attitude": self.positive_attitude,
            "overall_professional_Score": self.Professional_Score,
            "professional_score":self.professional_sc,
            "overall_Score": self.Overall_Score,
            "transcription":self.new_transcript,
            "grammer" : self.grammer_Score,
            "body_language_score":self.body_lang_sc,
            "emotion_score":self.emotion_score,
            "presentabilty_score":self.presentablity_score,
            "professional_comment":self.professionalcomment,
            "bodylang_comment":self.bodylang_comment,
            "emotioncomment":self.emotioncomment,
            "grommingcomment":self.grommingcomment,
            "dressingcomment":self.dressingcomment,
            "cheating_comment":self.cheating_comment

        }






