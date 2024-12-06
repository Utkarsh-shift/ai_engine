class AudioFileNode:
    def __init__(self, file_path):
        self.file_path = file_path
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

    def set_scores(self,pacescore,clarityscore,energyscore,fluencyscore,articulatioscore,communicationscore,paceandclarityscore,sociablility_score,face_confidence_score,positive_attitude,Professional_score,Overall_Score,Grammer_score):
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


    def set_interview_score(self,score):
        self.Interview_score = score

    def set_data(self, data_value):
        self.data = data_value

    def set_shoulder(self, data_value):
        self.shoulder = data_value
    
    # def set_cntright(self , cntright_values):
    #     self.right = cntright_values

    # def set_cntblink(self , cntblink_values):
    #     self.eyeblink = cntblink_values

    # def set_cntleft(self , cntleft_values):
    #     self.left = cntleft_values

    # def set_cntcenter(self , cntcentre_values):
    #     self.centre= cntcentre_values
    def set_headpose_and_eyegaze(self,hlwl_count,hlwr_count,hlwc_count,hrwl_count,hrwr_count,hrwc_count,hdwl_count,hdwr_count,
                                 hdwc_count,huwl_count,huwr_count,huwc_count,hswl_count,hswr_count,hswc_count
                                 
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
    def set_grammer_score(self,grammer_Score):
        self.GPTgrammer_score=grammer_Score

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
            # "blink": self.eyeblink,
            # "lookcentre" : self.centre,
            # "lookright" : self.right,
            # "lookleft" : self.left,
            # "headPose": self.data,
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
            "sentiment" : self.sentiment,
            "Sociability_score":self.sociablility_score,
            "Confidence_score":self.face_confidence_score,
            "Positive_attitude":self.positive_attitude,
            "Professional_Score":self.Professional_Score,
            "Overall_Score":self.Overall_Score,
            "Transcript":self.transcript,
            "Grammer_score " : self.grammer,
            "GPT_Grammer_score": self.GPTgrammer_score
        }

    def process_file1(self, smile, cal_uni_bi, col_name):
        print("*******************************************",type(smile), smile)
        # Process the audio file with OpenSMILE
        if not ".praat" in self.file_path:
            y = smile.process_file(self.file_path)
            smile_features = y.filter(col_name).mean().to_dict()
            # Calculate unigrams and bigrams
            df_unigram, df_bigram,sentiment,sentiment_sc,transcription,grammer_Score = cal_uni_bi()
            # Set OpenSMILE features
            self.set_smile_features(smile_features)
            self.set_df_unigram(df_unigram)
            self.set_df_bigram(df_bigram)
            self.set_senti(sentiment)
            self.set_senti_sc(sentiment_sc)
            self.set_transcript(transcription)
            self.set_grammer_score(grammer_Score)