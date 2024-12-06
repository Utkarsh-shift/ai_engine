import cv2 as cv
import numpy as np
import mediapipe as mp
from scipy.interpolate import interp1d
import collections
import os
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import gc

class AngleBuffer:
    def __init__(self, size=40):
        self.buffer = collections.deque(maxlen=size)

    def add(self, angles):
        self.buffer.append(angles)

    def get_average(self):
        return np.mean(self.buffer, axis=0)

# Global variables
big_res = []
data_list = []
USER_FACE_WIDTH = 140  # [mm]
NOSE_TO_CAMERA_DISTANCE = 600  # [mm]
PRINT_DATA = True
DEFAULT_WEBCAM = 0
LOG_DATA = True
LOG_ALL_FEATURES = False
ENABLE_HEAD_POSE = True
LOG_FOLDER = "logs"
TOTAL_BLINKS = 0
EYES_BLINK_FRAME_COUNTER = 0
BLINK_THRESHOLD = 0.51
EYE_AR_CONSEC_FRAMES = 2
LEFT_EYE_IRIS = [474, 475, 476, 477]
RIGHT_EYE_IRIS = [469, 470, 471, 472]
LEFT_EYE_OUTER_CORNER = [33]
LEFT_EYE_INNER_CORNER = [133]
RIGHT_EYE_OUTER_CORNER = [362]
RIGHT_EYE_INNER_CORNER = [263]
RIGHT_EYE_POINTS = [33, 160, 159, 158, 133, 153, 145, 144]
LEFT_EYE_POINTS = [362, 385, 386, 387, 263, 373, 374, 380]
NOSE_TIP_INDEX = 4
CHIN_INDEX = 152
LEFT_EYE_LEFT_CORNER_INDEX = 33
RIGHT_EYE_RIGHT_CORNER_INDEX = 263
LEFT_MOUTH_CORNER_INDEX = 61
RIGHT_MOUTH_CORNER_INDEX = 291
MIN_DETECTION_CONFIDENCE = 0.8
MIN_TRACKING_CONFIDENCE = 0.8
MOVING_AVERAGE_WINDOW = 10
initial_pitch, initial_yaw, initial_roll = None, None, None
calibrated = False
SHOW_BLINK_COUNT_ON_SCREEN = False
IS_RECORDING = False

# Total counts list
counts = []

def compute_horizontal(left_eye, right_eye, left_iris, right_iris):
    pupil_left = left_iris[0] / (left_eye[0] * 2 - 10)
    pupil_right = right_iris[0] / (right_eye[0] * 2 - 10)
    return (pupil_left + pupil_right) / 2

def compute_vertical(left_eye, right_eye, left_iris, right_iris):
    pupil_left = left_iris[1] / (left_eye[1] * 2 - 10)
    pupil_right = right_iris[1] / (right_eye[1] * 2 - 10)
    return (pupil_left + pupil_right) / 2

def is_right(left_eye, right_eye, left_iris, right_iris):
    return compute_horizontal(left_eye, right_eye, left_iris, right_iris) < 0.0065

def is_left(left_eye, right_eye, left_iris, right_iris):
    return compute_horizontal(left_eye, right_eye, left_iris, right_iris) > 0.02

def is_center(left_eye, right_eye, left_iris, right_iris):
    return not (is_left(left_eye, right_eye, left_iris, right_iris) or is_right(left_eye, right_eye, left_iris, right_iris))

def compute_face_direction(pitch, roll, yaw):
    pitch_rad = np.radians(pitch)
    roll_rad = np.radians(roll)
    yaw_rad = np.radians(yaw)
    
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ])
    R_y = np.array([
        [np.cos(roll_rad), 0, np.sin(roll_rad)],
        [0, 1, 0],
        [-np.sin(roll_rad), 0, np.cos(roll_rad)]
    ])
    R_z = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad), np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])
    
    R = np.dot(R_z, np.dot(R_y, R_x))
    initial_vector = np.array([0, 0, 1])
    direction_vector = np.dot(R, initial_vector)
    return direction_vector

def interpolate_list(dataList, original_fps, target_fps):
    data_array = np.array(dataList)
    original_frames = len(data_array)
    interpolation_factor = target_fps / original_fps
    interpolated_frames = int((original_frames - 1) * interpolation_factor) + 1
    
    original_times = np.linspace(0, original_frames - 1, original_frames)
    interpolated_times = np.linspace(0, original_frames - 1, interpolated_frames)
    
    interp_func = interp1d(original_times, data_array, kind='linear')
    
    interpolated_data = interp_func(interpolated_times)
    
    return interpolated_data.tolist()

def multiply_values_by_3(d):
    for key in d:
        d[key] *= 3
    return d

def vector_position(point1, point2):
    x1, y1 = point1.ravel()
    x2, y2 = point2.ravel()
    return x2 - x1, y2 - y1

def euclidean_distance_3D(points):
    P0, P3, P4, P5, P8, P11, P12, P13 = points
    numerator = (
        np.linalg.norm(P3 - P13) ** 3
        + np.linalg.norm(P4 - P12) ** 3
        + np.linalg.norm(P5 - P11) ** 3
    )
    denominator = 3 * np.linalg.norm(P0 - P8) ** 3
    distance = numerator / denominator
    return distance
# def estimate_head_pose(landmarks, image_size):
#     scale_factor = USER_FACE_WIDTH / 150.0
#     model_points = np.array([
#         (0.0, 0.0, 0.0),
#         (0.0, -330.0 * scale_factor, -65.0 * scale_factor),
#         (-225.0 * scale_factor, 170.0 * scale_factor, -135.0 * scale_factor),
#         (225.0 * scale_factor, 170.0 * scale_factor, -135.0 * scale_factor),
#         (-150.0 * scale_factor, -150.0 * scale_factor, -125.0 * scale_factor),
#         (150.0 * scale_factor, -150.0 * scale_factor, -125.0 * scale_factor)
#     ])
#     focal_length = image_size[1]
#     center = (image_size[1] / 2, image_size[0] / 2)
#     camera_matrix = np.array(
#         [[focal_length, 0, center[0]],
#          [0, focal_length, center[1]],
#          [0, 0, 1]], dtype="double"
#     )
#     dist_coeffs = np.zeros((4, 1))
#     image_points = np.array([
#         landmarks[NOSE_TIP_INDEX],
#         landmarks[CHIN_INDEX],
#         landmarks[LEFT_EYE_LEFT_CORNER_INDEX],
#         landmarks[RIGHT_EYE_RIGHT_CORNER_INDEX],
#         landmarks[LEFT_MOUTH_CORNER_INDEX],
#         landmarks[RIGHT_MOUTH_CORNER_INDEX]
#     ], dtype="double")
#     (success, rotation_vector, translation_vector) = cv.solvePnP(
#         model_points, image_points, camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE
#     )
#     rotation_matrix, _ = cv.Rodrigues(rotation_vector)
#     projection_matrix = np.hstack((rotation_matrix, translation_vector.reshape(-1, 1)))
#     _, _, _, _, _, _, euler_angles = cv.decomposeProjectionMatrix(projection_matrix)
#     pitch, yaw, roll = euler_angles.flatten()[:3]
#     pitch = normalize_pitch(pitch)
#     return pitch, yaw, roll
def normalize_angle(angle, min_angle, max_angle):
    """ Normalize an angle to be within the range [min_angle, max_angle]. """
    while angle < min_angle:
        angle += 360
    while angle > max_angle:
        angle -= 360
    return angle

def normalize_pitch(pitch):
    return normalize_angle(pitch, -90, 90)

def normalize_yaw(yaw):
    return normalize_angle(yaw, -180, 180)

def normalize_roll(roll):
    return normalize_angle(roll, -180, 180)

def blinking_ratio(landmarks):
    right_eye_ratio = euclidean_distance_3D(landmarks[RIGHT_EYE_POINTS])
    left_eye_ratio = euclidean_distance_3D(landmarks[LEFT_EYE_POINTS])
    ratio = (right_eye_ratio + left_eye_ratio + 1) / 2
    return ratio

def process_video(img_path, folder_path):
   
    TOTAL_BLINKS = 0
    EYES_BLINK_FRAME_COUNTER = 0
    BLINK_THRESHOLD = 0.51
    EYE_AR_CONSEC_FRAMES = 2
    LEFT_EYE_IRIS = [474, 475, 476, 477]
    RIGHT_EYE_IRIS = [469, 470, 471, 472]
    LEFT_EYE_OUTER_CORNER = [33]
    LEFT_EYE_INNER_CORNER = [133]
    RIGHT_EYE_OUTER_CORNER = [362]
    RIGHT_EYE_INNER_CORNER = [263]
    MIN_DETECTION_CONFIDENCE = 0.8
    MIN_TRACKING_CONFIDENCE = 0.8
    _indices_pose = [1, 33, 61, 199, 263, 291]
    total_frames = 0
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    try:
        image_path = os.path.join(folder_path, img_path)
        frame = cv.imread(image_path)
        frame = cv.resize(frame, (640, 480))  
        frame_shape1 = frame.shape[1]
        frame_shape0 = frame.shape[0]

        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        total_frames += 1
        del frame 

        results = mp_face_mesh.process(image)
        if results.multi_face_landmarks is None: 
            del image
            print("Face is not facing camera")
            return 44
        
        del image  

        if results.multi_face_landmarks:
            mesh_points = np.array(
                [np.multiply([p.x, p.y], [frame_shape1, frame_shape0]).astype(int)
                    for p in results.multi_face_landmarks[0].landmark]
            )
            mesh_points_3D = np.array(
                [[n.x, n.y, n.z] for n in results.multi_face_landmarks[0].landmark]
            )
            head_pose_points_3D = np.multiply(
                mesh_points_3D[_indices_pose], [frame_shape1, frame_shape0, 1]
            )
            head_pose_points_2D = mesh_points[_indices_pose]
            nose_3D_point = np.multiply(head_pose_points_3D[0], [1, 1, 3000])
            nose_2D_point = head_pose_points_2D[0]
            focal_length = 1 * frame_shape1
            cam_matrix = np.array(
                [[focal_length, 0, frame_shape0 / 2], [0, focal_length, frame_shape1 / 2], [0, 0, 1]]
            )
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            head_pose_points_2D = np.delete(head_pose_points_3D, 2, axis=1)
            head_pose_points_3D = head_pose_points_3D.astype(np.float64)
            head_pose_points_2D = head_pose_points_2D.astype(np.float64)
            success, rot_vec, trans_vec = cv.solvePnP(
                head_pose_points_3D, head_pose_points_2D, cam_matrix, dist_matrix
            )
            rotation_matrix, _ = cv.Rodrigues(rot_vec)
            projection_matrix = np.hstack((rotation_matrix, trans_vec.reshape(-1, 1)))
            _, _, _, _, _, _, euler_angles = cv.decomposeProjectionMatrix(projection_matrix)
            pitch, yaw, roll = euler_angles.flatten()[:3]
            pitch = normalize_pitch(pitch)
            yaw = normalize_yaw(yaw)
            roll = normalize_roll(roll)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv.RQDecomp3x3(rotation_matrix)
            angle_x = angles[0] * 360
            angle_y = angles[1] * 360
            z = angles[2] * 360
            threshold_angle = 10
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_EYE_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_EYE_IRIS])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)
            l_dx, l_dy = vector_position(mesh_points[LEFT_EYE_OUTER_CORNER], center_left)
            r_dx, r_dy = vector_position(mesh_points[RIGHT_EYE_OUTER_CORNER], center_right)      
          #  Return a direction code based on face angle
            if angle_y < -threshold_angle and yaw > 15:
                if is_left((l_cx, l_cy), (r_cx, r_cy), (l_dx, l_dy), (r_dx, r_dy)):
                   # print("head left and Watch_left")
                    return 11
                elif is_right((l_cx, l_cy), (r_cx, r_cy), (l_dx, l_dy), (r_dx, r_dy)):
                   # print("head left watch right")
                    return 12
                elif is_center((l_cx, l_cy), (r_cx, r_cy), (l_dx, l_dy), (r_dx, r_dy)):
                    print("head left watch centre")
                    return 13

            elif angle_y > threshold_angle and yaw < -15:
                if is_left((l_cx, l_cy), (r_cx, r_cy), (l_dx, l_dy), (r_dx, r_dy)):
                    #print("head right and Watch_left")
                    return 21
                elif is_right((l_cx, l_cy), (r_cx, r_cy), (l_dx, l_dy), (r_dx, r_dy)):
                    #print("head right watch right")
                    return 22
                elif is_center((l_cx, l_cy), (r_cx, r_cy), (l_dx, l_dy), (r_dx, r_dy)):
                    #print("head right watch centre")  
                    return 23
            
            elif angle_x < -threshold_angle and pitch < -15:
                if is_left((l_cx, l_cy), (r_cx, r_cy), (l_dx, l_dy), (r_dx, r_dy)):
                    #print("head down and Watch_left")
                    return 31
                elif is_right((l_cx, l_cy), (r_cx, r_cy), (l_dx, l_dy), (r_dx, r_dy)):
                    #print("head down watch right")
                    return 32
                elif is_center((l_cx, l_cy), (r_cx, r_cy), (l_dx, l_dy), (r_dx, r_dy)):
                    #print("head down watch centre")
                    return 33
              
            elif angle_x > threshold_angle and pitch > 15:
                if is_left((l_cx, l_cy), (r_cx, r_cy), (l_dx, l_dy), (r_dx, r_dy)):
                    #print("head up and Watch_left")
                    return 41
                elif is_right((l_cx, l_cy), (r_cx, r_cy), (l_dx, l_dy), (r_dx, r_dy)):
                    #print("head up watch right")
                    return 42
                elif is_center((l_cx, l_cy), (r_cx, r_cy), (l_dx, l_dy), (r_dx, r_dy)):
                    #print("head up watch centre")
                    return 43
            else :
                if is_left((l_cx, l_cy), (r_cx, r_cy), (l_dx, l_dy), (r_dx, r_dy)):
                    #print("head straight and Watch_left")
                    return 51
                elif is_right((l_cx, l_cy), (r_cx, r_cy), (l_dx, l_dy), (r_dx, r_dy)):
                    #print("head straight watch right")
                    return 52
                elif is_center((l_cx, l_cy), (r_cx, r_cy), (l_dx, l_dy), (r_dx, r_dy)):
                    #print("head straight watch centre")
                    return 53                
                
        
              
                del mesh_points, mesh_points_3D, results 
                
   
            del mesh_points, mesh_points_3D, results 
            # If the face direction doesn't match any condition, return 0 (neutral)
        gc.collect()  
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0  # Return None if there's an error

def mycallback(count):
           # Ensure the count is valid
        counts.append(count)
        #print(counts)  # Append the count to the counts list

def long_time_task(image, folder_path):
    return process_video(img_path=image, folder_path=folder_path)
    
def process_in_batches(images, folder_path, batch_size):
    cnt = None
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        with Pool(8) as q:  
            for image in tqdm(batch):
                cnt = q.apply_async(long_time_task, args=(image, folder_path), callback=mycallback)
            q.close()
            q.join()  
        gc.collect()  
        print(f"Processed batch {i // batch_size + 1}")
   # print( "The get ( output ud d )", counts)
    return counts



# if __name__ == "__main__":
#     frame_folder = "/media/almabay/New Volume/interviewer1/Interviewer_1_9th_August/Interviewer_1/Interviewer/datasets/ChaLearn/test_data/good_score"
#     list_dir = os.listdir(frame_folder)
#     process_in_batches(list_dir, frame_folder, batch_size=64)  
#     print("Processing complete. All results are stored in 'big_res'.")
#     print("-----------------------------------------------------------------------------------")
#     for direction, count in counts.items():
#         print(f"{direction}: {count}")