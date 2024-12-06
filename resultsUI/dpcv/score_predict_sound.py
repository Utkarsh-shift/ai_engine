import opensmile
import audiofile
import pandas as pd

import noisereduce as nr
import librosa
import soundfile as sf

import librosa
import numpy as np
 
import numpy as np

import scipy.io.wavfile as wav
def calculate_pace_score(p_loud, p_vsp, p_mvsl, p_musl):
    # Define score adjustments for p_loud
    if p_loud > 5:
        score_loud = -1 * p_loud  # More negative as p_loud increases above 5
    elif 2< p_loud <= 5:
        score_loud = 1 * (5 - p_loud) / 4  # Moderately positive, less positive as p_loud approaches 5
    else:
        score_loud = 1 + (1 - p_loud)  # Most positive when p_loud is below 1

    # Corrected conditions for p_vsp based on new ranges
    if p_vsp >= 4:
        score_vsp = 1 * (p_vsp - 4)  # Good as p_vsp increases above 4
    elif 1 <= p_vsp < 4:
        score_vsp = 1* (4 - p_vsp)  # Moderately poor, decreasing as p_vsp approaches 1
    else:
        score_vsp = -1 * (1 - p_vsp)  # Poor when p_vsp is below 1

    # Adjusted conditions for p_mvsl based on new ranges
    if p_mvsl > 0.7:
        score_mvsl = 3 * (p_mvsl - 0.7)  # Good as p_mvsl increases above 0.7
    elif 0.4 <= p_mvsl <= 0.7:
        score_mvsl = 1 * (0.7 - p_mvsl)  # Moderately poor, decreasing as p_mvsl approaches 0.4
    else:
        score_mvsl = -3 * (0.4 - p_mvsl)  # Poor when p_mvsl is below 0.4

    # Adjusted conditions for p_musl based on new ranges
    if p_musl > 0.6:
        score_musl = -3 * (p_musl - 0.6)  # Poor as p_musl increases above 0.6
    elif 0.3 <= p_musl <= 0.6:
        score_musl = 1 * (0.6 - p_musl)  # Moderately positive, decreasing as p_musl approaches 0.3
    else:
        score_musl = 3 *(0.3 - p_musl)  # Good when p_musl is below 0.3   # chnge 3+ to 3*

    print(f"P loud is {score_loud}")
    print(f"P Vsp is {score_vsp}")
    print(f"p mvsl is {score_mvsl}")
    print(f"p musl is {score_musl}")
    p_score = score_loud + score_vsp + score_mvsl + score_musl
    if p_score!=0:
        p_score= max(0, min(10, (p_score + 10) / 2))
        return p_score
    else:
        return 0


def calculate_positive_clarity_score(c_shi, c_jitter, c_hamind, c_alpha, c_loud, c_hnr):
    # Initialize scores
    c_shi_score = 0
    c_jitter_score = 0
    c_hamind_score = 0
    c_alpha_score = 0
    c_loud_score = 0
    c_hnr_score = 0

    # Calculate shimmer score (good if ≤ 1, bad if > 1)
    if c_shi <= 1:
        c_shi_score = 1 + (1 - c_shi)  # Better scores for smaller shimmer
    else:
        c_shi_score = 0  # Minimum score for bad shimmer

    # Calculate jitter score (good if < 0.5, bad if > 1)
    if c_jitter < 0.5:
        c_jitter_score = 1 + (0.5 - c_jitter)  # Better scores for smaller jitter
    elif c_jitter <= 1:
        c_jitter_score = 1 * (1 - (c_jitter - 0.5) / 0.5)  # Interpolated score
    else:
        c_jitter_score = 0  # Minimum score for bad jitter

    # Calculate Hammarberg Index score (good if > 0.5, moderate if 0.3 to 0.5)
    if c_hamind > 0.5:
        c_hamind_score = 1 + (c_hamind - 0.5)  # Better scores for higher Hammarberg Index
    elif c_hamind >= 0.3:
        c_hamind_score = 1 * (c_hamind - 0.3) / 0.2  # Interpolated score
    else:
        c_hamind_score = 0  # Minimum score for bad Hammarberg Index

    # Calculate Alpha Ratio score (good if > 0.6, bad if ≤ 0.6)
    if c_alpha > 0.6:
        c_alpha_score = 1 + (c_alpha - 0.6)  # Better scores for higher Alpha Ratio
    else:
        c_alpha_score = 0  # Minimum score for low Alpha Ratio

    # Calculate Loudness score (good if < 1, moderate if 1 to 5, bad if > 5)
    if c_loud < 1:
        c_loud_score = 1 + (1 - c_loud)  # Better scores for lower loudness
    elif c_loud <= 5:
        c_loud_score = 1* (5 - c_loud) / 4  # Interpolated score for moderate loudness
    else:
        c_loud_score = 0  # Minimum score for high loudness

    # Calculate HNR score (good if > 15, moderate if 10 to 15, bad if < 10)
    if c_hnr > 15:
        c_hnr_score = 1 + (c_hnr - 15)  # Better scores for higher HNR
    elif c_hnr >= 10:
        c_hnr_score = 1 * (c_hnr - 10) / 5  # Interpolated score for moderate HNR
    else:
        c_hnr_score = 0  # Minimum score for low HNR

    # Calculate overall clarity score
    clarity_score = (c_shi_score + c_jitter_score + c_hamind_score + c_alpha_score + c_loud_score + c_hnr_score) /6
    print(clarity_score)
    # Ensure the clarity score is within the 0 to 5 range
    if clarity_score > 5:
        clarity_score = 5
    elif clarity_score < 0:
        clarity_score = 0
    # clarity_score= max(0, min(10, (p_score + 10 / 2)))

    return clarity_score


def calculate_energy_score(loudness_mean, shimmerLocaldB_mean, f0semitone_mean, alphaRatio_mean, hnr_mean):
    # Initialize scores
    F0semitone_mean_score = 0
    Loudness_mean_score = 0
    alphaRatio_mean_score = 0
    HNR_mean_score = 0
    shimmerLocaldB_mean_score = 0

    # Scoring for Loudness
    if loudness_mean < 20:
        Loudness_mean_score = 0  # Bad
    elif 20 <= loudness_mean < 40:
        Loudness_mean_score = 0.5  # Moderate
    else:
        Loudness_mean_score = 1  # Good

    # Scoring for Shimmer
    if shimmerLocaldB_mean > 1:
        shimmerLocaldB_mean_score = 0  # Bad
    else:
        shimmerLocaldB_mean_score = 1  # Good

    # Scoring for F0semitone
    if f0semitone_mean < 16 or f0semitone_mean > 45:
        F0semitone_mean_score = 0  # Bad
    else:
        F0semitone_mean_score = 1  # Good

    # Scoring for Alpha Ratio
    if alphaRatio_mean < 10:
        alphaRatio_mean_score = 0  # Bad
    elif 10 <= alphaRatio_mean <= 20:
        alphaRatio_mean_score = 0.5  # Moderate
    else:
        alphaRatio_mean_score = 1  # Good

    # Scoring for HNR
    if hnr_mean < 10:
        HNR_mean_score = 0  # Bad
    elif 10 <= hnr_mean <= 20:
        HNR_mean_score = 0.5  # Moderate
    else:
        HNR_mean_score = 1  # Good

    # Calculate total energy score
    energy_score = F0semitone_mean_score + Loudness_mean_score + alphaRatio_mean_score + HNR_mean_score + shimmerLocaldB_mean_score

    # Normalize to percentage
    energy_score_percentage = (energy_score / 5) * 100

    return energy_score_percentage

# Example usage with provided values
# loudness_mean = 25  # example values
# shimmerLocaldB_mean = 0.8
# f0semitone_mean = 30
# alphaRatio_mean = 15
# hnr_mean = 18




# Example usage with provided values
# loudness_mean = 25  # example values
# shimmerLocaldB_mean = 0.8
# f0semitone_mean = 30
# alphaRatio_mean = 15
# hnr_mean = 18

def load_audio(file_path, sr=16000):
    audio, _ = librosa.load(file_path, sr=sr)
    return audio, sr

def detect_voice_segments(audio, sr, top_db=20):
    intervals = librosa.effects.split(audio, top_db=top_db)
    return intervals

def calculate_speech_rate(audio, intervals, sr):
    total_speech_duration = sum((end - start) for start, end in intervals) / sr
    total_words = (total_speech_duration / 60) * 160  # Average words per minute
    speech_rate = total_words / (len(audio) / sr / 60)
    return speech_rate

def calculate_pause_features(intervals, sr):
    pauses = [(intervals[i][0] - intervals[i-1][1]) / sr for i in range(1, len(intervals))]
    avg_pause_duration = np.mean(pauses) if pauses else 0
    num_pauses = len(pauses)
    return avg_pause_duration, num_pauses

def normalize_score(raw_score, min_score, max_score):
    normalized = 100 * (raw_score - min_score) / (max_score - min_score)
    return np.clip(normalized, 0, 100)

def analyze_fluency(file):
    audio, sr = load_audio(file)
    intervals = detect_voice_segments(audio, sr)
    
    speech_rate = calculate_speech_rate(audio, intervals, sr)
    avg_pause_duration, num_pauses = calculate_pause_features(intervals, sr)
    
    # Define min and max values for normalization
    min_speech_rate, max_speech_rate = 50, 200  # words per minute
    min_avg_pause_duration, max_avg_pause_duration = 0.1, 1.0  # seconds
    min_num_pauses, max_num_pauses = 0, 50
    
    # Normalize individual metrics
    norm_speech_rate = normalize_score(speech_rate, min_speech_rate, max_speech_rate)
    norm_avg_pause_duration = normalize_score(max_avg_pause_duration - avg_pause_duration, min_avg_pause_duration, max_avg_pause_duration)
    norm_num_pauses = normalize_score(max_num_pauses - num_pauses, min_num_pauses, max_num_pauses)
    
    # Combine normalized metrics into a single fluency score
    fluency_score = (norm_speech_rate * 0.4) + (norm_avg_pause_duration * 0.3) + (norm_num_pauses * 0.3)
    
    return {
        "speech_rate": speech_rate,
        "avg_pause_duration": avg_pause_duration,
        "num_pauses": num_pauses,
        "fluency_score": fluency_score
    }



 
def rms_to_db(rms_value):

    # Avoid log of zero by adding a small constant

    return 20 * np.log10(rms_value + 1e-10)
 
def db_to_percentage(db_value, min_db=-100, max_db=0):

    # Normalize dB value to a 0-100% scale

    percentage = (db_value - min_db) / (max_db - min_db) * 100

    return np.clip(percentage, 0, 100)
 
def calculate_energy_percentage(file_path):

    # Load the WAV file

    sample_rate, data = wav.read(file_path)
 
    # Normalize the data to range [-1, 1] if it's not already normalized

    if np.issubdtype(data.dtype, np.integer):

        data = data / np.max(np.abs(data), axis=0)
 
    # Compute the RMS (Root Mean Square) value for each channel

    rms = np.sqrt(np.mean(data**2, axis=0))

    # Convert RMS to dB

    db_value = rms_to_db(np.mean(rms))

    # Convert dB to percentage

    percentage = db_to_percentage(db_value)

    if np.isnan(percentage):

        percentage=0
    print(f"Final Energy Score is {percentage}")
    return percentage
 



 


 




def sound_score(filepath,grammer_Score):
    # Load the noisy audio file
    noisy_file = filepath
    y, sr = librosa.load(noisy_file, sr=None)

    # Identify a segment of the audio that contains only noise
    # Here, we assume that the first second of the audio is noise
    noise_sample = y[0:sr]

    # Apply noise reduction
    reduced_noise_audio = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample,stationary=True)

    # Save the cleaned audio file
    cleaned_file = './clean_audio/cleanedair12.wav'
    sf.write(cleaned_file, reduced_noise_audio, sr)

    print(f"Cleaned audio saved as {cleaned_file}")
    filepath=cleaned_file

    smile1 = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    signal, sampling_rate = audiofile.read(
        filepath,

        always_2d=True,
    )
    smile1.process_signal(
        signal,
        sampling_rate
    )
    y = smile1.process_file(filepath)


    mean_df=pd.DataFrame(y)
    pace_df=mean_df[["loudnessPeaksPerSec","VoicedSegmentsPerSec","MeanVoicedSegmentLengthSec","MeanUnvoicedSegmentLength"]]
    p_loud,p_vsp,p_mvsl,p_musl=pace_df.values[0][0],pace_df.values[0][1],pace_df.values[0][2],pace_df.values[0][3]



    p_score = calculate_pace_score(p_loud, p_vsp, p_mvsl, p_musl)
    print(f"Computed Pace Score: {p_score} Out of 10")

    clarity_df=mean_df[["shimmerLocaldB_sma3nz_amean",'jitterLocal_sma3nz_amean','hammarbergIndexUV_sma3nz_amean','alphaRatioUV_sma3nz_amean'
        ,"loudnessPeaksPerSec", "HNRdBACF_sma3nz_amean"]]
    c_shi,c_jitter,c_hamind,c_alpha,c_loud,c_hnr=clarity_df.values[0][0],clarity_df.values[0][1],clarity_df.values[0][2],clarity_df.values[0][3],clarity_df.values[0][4],clarity_df.values[0][5]


    score = calculate_positive_clarity_score(c_shi, c_jitter, c_hamind, c_alpha, c_loud, c_hnr)
    print(f"Clarity Score is {score:.2f} Out of 5")


    fluency_df=mean_df[["jitterLocal_sma3nz_amean","HNRdBACF_sma3nz_amean","shimmerLocaldB_sma3nz_amean","loudness_sma3_amean","VoicedSegmentsPerSec","MeanVoicedSegmentLengthSec","MeanUnvoicedSegmentLength"]]

    f_jit,f_hnr,f_shi,f_loud,f_vls,f_mvsl,f_musl=fluency_df.values[0][0],fluency_df.values[0][1],fluency_df.values[0][2],fluency_df.values[0][3],fluency_df.values[0][4],fluency_df.values[0][5],fluency_df.values[0][6]


    mean_df=mean_df[['F0semitoneFrom27.5Hz_sma3nz_amean', 
            
                'loudness_sma3_amean', 
            'alphaRatioV_sma3nz_amean',
                'HNRdBACF_sma3nz_amean',
                'shimmerLocaldB_sma3nz_amean', ]]


    F0semitone_mean,Loudness_mean,alphaRatio_mean,HNR_mean,shimmerLocaldB_mean=mean_df.values[0][0],mean_df.values[0][1],mean_df.values[0][2],mean_df.values[0][3],mean_df.values[0][4]

    # energy_score = calculate_energy_score(Loudness_mean, shimmerLocaldB_mean, F0semitone_mean, alphaRatio_mean, HNR_mean)
    # print(f"Energy Score is {energy_score}%")
    energy_score=calculate_energy_percentage(filepath)

    fluency_metrics = analyze_fluency(filepath)


    fluency_score=fluency_metrics['fluency_score']
    clarity_score=score*(100/5)
    articulation_score=np.round((fluency_score+clarity_score)/2)
    p_score=p_score*10
    pace_a_clarity=np.round((p_score+energy_score)/2)
    communication_score= 0.4*(articulation_score)+ 0.4*(pace_a_clarity)+ 0.2*(grammer_Score)
    print(communication_score)

    return p_score,clarity_score,energy_score,fluency_score,articulation_score,communication_score,pace_a_clarity


# if __name__ == "__main__" :
#      sound_score("/media/almabay/New Volume/interviewer1/Interviewer_1/Interviewer/datasets/ChaLearn/voice_data/voice_raw/test_data/vfwjalgclawjgcwjlavg.wav")