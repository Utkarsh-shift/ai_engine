import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import logging
from .dataset import pil_loader
from .utils import *
from .conf import set_env
import cv2,re
import pandas as pd
from easydict import EasyDict as Edict
import numpy as np
import cv2
import numpy as np
from scipy.interpolate import lagrange
from deepface import DeepFace
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def process_image(img_path, net, img_transform, conf, results):
    try:
        img = pil_loader(img_path)
    except PermissionError as e:
        logging.error(f"PermissionError: {e} for file {img_path}")
        return
    except Exception as e:
        logging.error(f"Unexpected error: {e} for file {img_path}")
        return

    img_ = img_transform(img).unsqueeze(0)

#    if torch.cuda.is_available():
#        net = net.cuda()
#        img_ = img_.cuda()

    with torch.no_grad():
        pred = net(img_)
        pred = pred.squeeze().cpu().numpy()

    infostr_probs, infostr_aus = hybrid_prediction_infolist(pred, 0.5)

    results.append({
        'image': img_path,
        'action_units': infostr_aus,
        'probabilities': infostr_probs
    })
    #print("/n|/n")
    #print(results)
    #print("/n|/n")
    if conf.draw_text:
        img = draw_text(img_path, list(infostr_aus), pred)




def interpolate_fps(dataList, original_fps, target_fps):
    data_array = np.array(dataList)
    original_frames = len(data_array)
    interpolation_factor = target_fps / original_fps
    interpolated_frames = int((original_frames - 1) * interpolation_factor) + 1
    interpolated_data = np.zeros((interpolated_frames, data_array.shape[1]))
    original_times = np.linspace(0, original_frames - 1, original_frames)
    interpolated_times = np.linspace(0, original_frames - 1, interpolated_frames)
    for feature_index in range(data_array.shape[1]):
        interp_func = interp1d(original_times, data_array[:, feature_index], kind='linear')
        interpolated_data[:, feature_index] = interp_func(interpolated_times)

    
    actionalUnits = pd.DataFrame(interpolated_data)
    return actionalUnits



def calculate_emotion_difference(current_emotion, next_emotion):
  
    return abs(current_emotion - next_emotion)
import re

def extract_number(s):
    numbers = re.findall(r'\d+', s)
    return int(numbers[0]) if numbers else 0

def mainExec(conf):
    if conf.stage == 1:
        from .model.ANFL import MEFARG
        net = MEFARG(num_main_classes=conf.num_main_classes, num_sub_classes=conf.num_sub_classes, backbone=conf.arc, neighbor_num=conf.neighbor_num, metric=conf.metric)
    else:
        from .model.MEFL import MEFARG
        net = MEFARG(num_main_classes=conf.num_main_classes, num_sub_classes=conf.num_sub_classes, backbone=conf.arc)
    if conf.resume != '':
        logging.info("Resume from | {} ]".format(conf.resume))
        net = load_state_dict(net, conf.resume)
    net.eval()
#    print("The network is given as ",net)
    img_transform = image_eval()
    # Check if input is a directory
    if os.path.isdir(conf.input):
        image_paths = [os.path.join(conf.input, img) for img in os.listdir(conf.input) if img.endswith(('.png', '.jpg', '.jpeg'))]
    else:
        image_paths = [conf.input]
    results = []
    base_gap = 3
    max_gap = 30
    current_gap = base_gap
    emotion_threshold = 5.0
    count = 0
    emotion_lst = []
    first_try = True
    previous_dominant_emotion = None
    image_paths1 = sorted(image_paths, key=extract_number)
    total_frames = len(image_paths1)
    print("The total images in folder is ," , total_frames)
    for img_path in image_paths1:
        process_image(img_path, net, img_transform, conf, results)
        # face_track = FaceTrack()
        # data , left , straight , right, blink, shoulder_pose =  face_track.process_video(img_path=img_path)
        if count % current_gap == 0:
            try:
           
                temp = DeepFace.analyze(img_path=img_path, actions=['emotion'])
          
                if first_try:
                    first_emo = temp[0]['emotion']
                    previous_dominant_emotion = temp[0]['dominant_emotion']
                  
                    emotion_lst.append(temp)
                    first_try = False
                else:
                    current_emo = emotion_lst[-1][0]['emotion']
                    next_emo = temp[0]['emotion']
                    current_dominant_emotion_value = current_emo[temp[0]['dominant_emotion']]               
                    next_dominant_emotion_value = next_emo[temp[0]['dominant_emotion']]
                    emotion_diff = calculate_emotion_difference(
                        current_dominant_emotion_value, next_dominant_emotion_value)
                    current_dominant_emotion = temp[0]['dominant_emotion']
                    if current_dominant_emotion != previous_dominant_emotion:
                        current_gap = base_gap
                    else:
                        if emotion_diff < emotion_threshold:
                            current_gap = min(max_gap, current_gap * 2)
                        else:
                            current_gap = base_gap
                    previous_dominant_emotion = current_dominant_emotion
                    emotion_lst.append(temp)
                count = 0  
            except Exception as e:
                print(f"Error analyzing frame: {e}")
        count += 1  
    headerList=[]
    dataList=[]
    # string=results[0]['probabilities']
    for j in results:
        string = j['probabilities']
        for i in string:
            data = re.findall(r'\b\d+\.\d+|\b\d+\b', i)
            data = [float(item) for item in data]
            dataList.append(data) 

    original_fps = 10
    target_fps = 30
    interpolated_df = interpolate_fps(dataList, original_fps, target_fps)
    #print(interpolated_df)
    actionalUnits=pd.DataFrame(interpolated_df)   
    print("******************************************************************************************")
    print(actionalUnits)
    print("___________________________________________________________________________________________")

    num_rows = actionalUnits.shape[0]


    print("The number of rows in action untis" , num_rows)


        # Map your AU columns to the dataframe indices (0-indexed)
    au_columns = {
        "AU1": 0, "AU2": 1, "AU4": 2, "AU5": 3, "AU6": 4, "AU7": 5, 
        "AU9": 6, "AU10": 7, "AU11": 8, "AU12": 9, "AU13": 10, "AU14": 11, 
        "AU15": 12, "AU16": 13, "AU17": 14, "AU18": 15, "AU19": 16, 
        "AU20": 17, "AU22": 18, "AU23": 19, "AU24": 20, "AU25": 21, 
        "AU26": 22, "AU27": 23, "AU32": 24, "AU38": 25, "AU39": 26,
        # Add any additional AUs here as needed
    }




    thresholds = {
        "Smile": {"AU6": 45, "AU12": 17},
        "Sadness": {"AU1": 39.45, "AU4": 57.75, "AU15": 15.12},
        "Surprise": {"AU1": 41.9, "AU2": 41.1, "AU5": 23.88, "AU26": 60.2}, 
        "Fear": {"AU1": 48.46, "AU2": 37.01, "AU4": 51.6, "AU5": 21.34, "AU7": 47.70, "AU20": 14.88},
        "Anger": {"AU4": 42.87, "AU5": 14.25, "AU7": 47.437, "AU23": 31.48},
        "Disgust": {"AU9": 23.27, "AU10": 52.225, "AU16": 13.71},
        "Contempt": {"AU12": 17, "AU14": 28},
        "Confusion": {"AU1": 31.5, "AU4": 40, "AU7": 28},
     #   "Boredom": {"AU43": 25, "AU64": 20},  # Assuming you have AU43 and AU64, adjust as needed
        "Skepticism": {"AU12": 17, "AU14": 28}
    }



    std_values = {
    "Smile": {"AU6": 5, "AU12": 3},
    "Sadness": {"AU1": 19.75, "AU4": 24.24, "AU15": 6.531}, #done
    "Surprise": {"AU1": 22.6, "AU2": 23.105, "AU5": 16.75, "AU26": 19.4}, #done
    "Fear": {"AU1": 18.78, "AU2": 17.24, "AU4": 22.50, "AU5": 12.23, "AU7": 16.13, "AU20": 7.15}, #done 
    "Anger": {"AU4": 22.002, "AU5": 5.21, "AU7": 16.16, "AU23": 12.03}, # Done 
    "Disgust": {"AU9": 19.34, "AU10": 20.47, "AU16": 3.80}, #done
    "Contempt": {"AU12": 1.7, "AU14": 2.8},
    "Confusion": {"AU1": 1.65, "AU4": 4.0, "AU7": 2.8},
    "Skepticism": {"AU12": 1.7, "AU14": 2.8},
}

    expression_counts = {expression: 0 for expression in thresholds}
    for expression, aus in thresholds.items():
        condition = True
        for au, threshold in aus.items():
            # Ensure the AU column exists in the dataframe
            if au_columns[au] in actionalUnits.columns:
                std = std_values[expression].get(au,0)
                #condition &= (actionalUnits[au_columns[au]] > threshold)
                condition &= ((actionalUnits[au_columns[au]] >= (threshold - std)) & 
                          (actionalUnits[au_columns[au]] <= (threshold + std)))
        expression_counts[expression] = condition.sum()

    print("The sum is given as")    

    # Print the counts of each expression
    for expression, count in expression_counts.items():
        expression_counts[expression] = (count/num_rows)*100
        print(f"{expression}: {count}")
        print(f"{expression}: {(count/num_rows)*100}")


    non_verbal_thresholds = {
        "Engagement": {"AU1": 31.5, "AU2": 16.0, "AU5": 24.5, "AU7": 28},
        "Concentration": {"AU4": 40, "AU7": 28, "AU17": 27},
        "Discomfort": {"AU4": 20, "AU5": 24.5, "AU7": 28, "AU14": 28, "AU15": 5.2},
        "Dominance": {"AU4": 30, "AU5": 11.5, "AU23": 13.35, "AU24": 5.7},
        "Submission": {"AU12": 17, "AU17": 27, "AU24": 5.7},
        "Interest": {"AU1": 31.5, "AU2": 16.0, "AU5": 24.5},
        "Rapport": {"AU12": 17, "AU6": 45, "AU2": 16.0},
        "Skepticism": {"AU12": 17, "AU14": 28},
    }


    std_values_nonverbal = {
        "Engagement": {"AU1": 31.5, "AU2": 16.0, "AU5": 24.5, "AU7": 28},
        "Concentration": {"AU4": 40, "AU7": 28, "AU17": 27},
        "Discomfort": {"AU4": 20, "AU5": 24.5, "AU7": 28, "AU14": 28, "AU15": 5.2},
        "Dominance": {"AU4": 30, "AU5": 11.5, "AU23": 13.35, "AU24": 5.7},
        "Submission": {"AU12": 17, "AU17": 27, "AU24": 5.7},
        "Interest": {"AU1": 31.5, "AU2": 16.0, "AU5": 24.5},
        "Rapport": {"AU12": 17, "AU6": 45, "AU2": 16.0},
        "Skepticism": {"AU12": 17, "AU14": 28},
}



    non_verbal_counts = {behavior: 0 for behavior in non_verbal_thresholds}

    for behavior, aus in non_verbal_thresholds.items():
        condition = True
        for au, threshold in aus.items():
            if au_columns[au] in actionalUnits.columns:
                std_nonverbal = std_values_nonverbal[behavior].get(au,0)
                condition &= ((actionalUnits[au_columns[au]] >= (threshold - std)) & 
                          (actionalUnits[au_columns[au]] <= (threshold + std)))

        non_verbal_counts[behavior] = condition.sum()

    for behavior, count in non_verbal_counts.items():
        non_verbal_counts[behavior] = (count/num_rows)*100
        print(f"{behavior}: {count}")

        print(f"{behavior}: {(count/num_rows)*100}")
  

    return emotion_lst,expression_counts, non_verbal_counts

# ---------------------------------------------------------------------------------

def CallMethod(input_path):
    # conf = {
    #         'dataset': 'hybrid', 'batch_size': 64, 'learning_rate': 1e-05, 'epochs': 20, 'num_workers': 4, 'weight_decay': 0.0005, 'optimizer_eps': 1e-08, 'crop_size': 224, 'evaluate': False, 'arc': 'resnet50', 'metric': 'dots', 'lam': 0.001, 'gpu_ids': '0', 'seed': 0, 'exp_name': 'demo', 'resume': 'checkpoints/OpenGprahAU-ResNet50_second_stage.pth', 'input': '', 
    #         'draw_text': True, 'stage': 2, 'dataset_path': 'data/Hybrid', 'num_main_classes': 27, 'num_sub_classes': 14, 'neighbor_num': 4
    #     }
    conf = {
            'epochs': 20, 'optimizer_eps': 1e-08, 'evaluate': False, 'arc': 'resnet50', 'gpu_ids': '0', 'seed': 0, 'resume': 'checkpoints/OpenGprahAU-ResNet50_second_stage.pth', 'input': '', 
            'draw_text': True, 'stage': 2, 'num_main_classes': 27, 'num_sub_classes': 14, 'neighbor_num': 4
        }
    conf['input']=input_path
    conf = Edict(conf)
    conf.evaluate = True
    set_env(conf)

    return  mainExec(conf)
