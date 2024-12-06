import cv2 as cv
import numpy as np
import numpy as np
from scipy.interpolate import lagrange
import mediapipe as mp
import math
import socket
from deepface import DeepFace
import argparse
import time
import csv
from datetime import datetime
import os
# from .AngleBuffer import AngleBuffer
import pandas as pd
import numpy as np
import collections
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
class AngleBuffer:
    def __init__(self, size=40):
        self.size = size
        self.buffer = collections.deque(maxlen=size)

    def add(self, angles):
        self.buffer.append(angles)

    def get_average(self):
        return np.mean(self.buffer, axis=0)
def compute_Horizontal(left_eye, right_eye, left_iris, right_iris):
    pupil_left = left_iris[0]/(left_eye[0]*2-10)
    pupil_right = right_iris[0]/(right_eye[0]*2-10)
   # print((pupil_left+pupil_right)/2)
    return (pupil_left+pupil_right)/2

def compute_vertical(left_eye, right_eye, left_iris, right_iris):
    pupil_left = left_iris[1]/(left_eye[1]*2-10)
    pupil_right = right_iris[1]/(right_eye[1]*2-10)
    return (pupil_left+pupil_right)/2

def is_right(left_eye,right_eye,left_iris,right_iris):
    if compute_Horizontal(left_eye,right_eye,left_iris,right_iris) <0.008:
        return True
    else:
        return None

def is_left(left_eye,right_eye,left_iris,right_iris):
    if compute_Horizontal(left_eye,right_eye,left_iris,right_iris) >0.01:
        return True
    else:
        return None

def is_center(left_eye,right_eye,left_iris,right_iris):
    return is_left(left_eye,right_eye,left_iris,right_iris) is not True and is_right(left_eye,right_eye,left_iris,right_iris) is not True
    # else:
    #     return None


def compute_face_direction(pitch, roll, yaw):
    # Convert angles from degrees to radians
    pitch_rad = np.radians(pitch)
    roll_rad = np.radians(roll)
    yaw_rad = np.radians(yaw)
    
    # Define the rotation matrices
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
    
    # Combine the rotation matrices
    R = np.dot(R_z, np.dot(R_y, R_x))
    
    # The initial direction vector (facing straight ahead)
    initial_vector = np.array([0, 0, 1])
    
    # Apply the rotation
    direction_vector = np.dot(R, initial_vector)
    
    return direction_vector


def classify_direction(pitch, roll, yaw):
    if pitch > 15:
        return 'Up'
    elif pitch < -15:
        return 'Down'
    elif yaw > 15:
        return 'Left'
    elif yaw < -15:
        return 'Right'
    elif roll > 15:
        return 'Tilted Left'
    elif roll < -15:
        return 'Tilted Right'
    else:
        return 'Straight'
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def interpolate_list(dataList, original_fps, target_fps):
    # Convert the list to a numpy array for easier handling
    data_array = np.array(dataList)
    original_frames = len(data_array) 
    # Calculate the number of interpolated values needed
    interpolation_factor = target_fps / original_fps
    interpolated_frames = int((original_frames - 1) * interpolation_factor) + 1
    
    # Prepare the new interpolated data array
    interpolated_data = np.zeros(interpolated_frames)
    
    # Time indices for original and interpolated frames
    original_times = np.linspace(0, original_frames - 1, original_frames)
    interpolated_times = np.linspace(0, original_frames - 1, interpolated_frames)
    
    # Create interpolation function
    interp_func = interp1d(original_times, data_array, kind='linear')
    
    # Apply the interpolation function to new time indices
    interpolated_data = interp_func(interpolated_times)
    
    # Convert to DataFrame if needed
   # interpolated_df = pd.DataFrame(interpolated_data, columns=['Value'])
    return interpolated_data.tolist()

def multiply_values_by_3(d):
    # Iterate over dictionary items
    for key in d:
        # Multiply each value by 3
        d[key] *= 3
    return d

def count_facing_directions(data):
    counts = {
        'Up': 0,
        'Down': 0,
        'Left': 0,
        'Right': 0,
        'Tilted Left': 0,
        'Tilted Right': 0,
        'Straight': 0
    }


    for person in data[0]:
        pitch, roll, yaw = person['pitch'], person['roll'], person['yaw']
        direction = classify_direction(pitch, roll, yaw)
        counts[direction] += 1
    # print(watchLeft,watchcenter,watchright)
    original_fps = 10
    target_fps = 30

    print(type(data))
    #interpolated_data1 = interpolate_list(data[1], original_fps, target_fps)

   # interpolated_data2 = interpolate_list(data[2], original_fps, target_fps)

    #interpolated_data3 = interpolate_list(data[3], original_fps, target_fps)

    #interpolated_data4 = interpolate_list(data[4], original_fps, target_fps)


    #interpolated_data5 = interpolate_list(data[4], original_fps, target_fps)
    #print(interpolated_df)

    new_count = multiply_values_by_3(counts) 
    
    return new_count, 3*data[1], 3*data[2], 3*data[3] , 3*data[4] , data[5]


















def faceTrack(filepath):
    dataList=[]
    USER_FACE_WIDTH = 140  # [mm]
    NOSE_TO_CAMERA_DISTANCE = 600  # [mm
    PRINT_DATA = True
    DEFAULT_WEBCAM = 0
    SHOW_ALL_FEATURES = True
    LOG_DATA = True
    LOG_ALL_FEATURES = False

    # ENABLE_HEAD_POSE: Enable the head position and orientation estimator.
    ENABLE_HEAD_POSE = True
    LOG_FOLDER = "logs"
    SHOW_ON_SCREEN_DATA = True

    # TOTAL_BLINKS: Counter for the total number of blinks detected.
    TOTAL_BLINKS = 0

    # EYES_BLINK_FRAME_COUNTER: Counter for consecutive frames with detected potential blinks.
    EYES_BLINK_FRAME_COUNTER = 0

    # BLINK_THRESHOLD: Eye aspect ratio threshold below which a blink is registered.
    BLINK_THRESHOLD = 0.51

    # EYE_AR_CONSEC_FRAMES: Number of consecutive frames below the threshold required to confirm a blink.
    EYE_AR_CONSEC_FRAMES = 2

    ## Head Pose Estimation Landmark Indices
    # These indices correspond to the specific facial landmarks used for head pose estimation.
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

    ## MediaPipe Model Confidence Parameters
    # These thresholds determine how confidently the model must detect or track to consider the results valid.
    MIN_DETECTION_CONFIDENCE = 0.8
    MIN_TRACKING_CONFIDENCE = 0.8

    ## Angle Normalization Parameters
    # MOVING_AVERAGE_WINDOW: The number of frames over which to calculate the moving average for smoothing angles.
    MOVING_AVERAGE_WINDOW = 10

    # Initial Calibration Flags
    # initial_pitch, initial_yaw, initial_roll: Store the initial head pose angles for calibration purposes.
    # calibrated: A flag indicating whether the initial calibration has been performed.
    initial_pitch, initial_yaw, initial_roll = None, None, None
    calibrated = False

    # User-configurable parameters
    PRINT_DATA = True  # Enable/disable data printing
    DEFAULT_WEBCAM = 0  # Default webcam number
    SHOW_ALL_FEATURES = True  # Show all facial landmarks if True
    LOG_DATA = True  # Enable logging to CSV
    LOG_ALL_FEATURES = False  # Log all facial landmarks if True
    LOG_FOLDER = "logs"  # Folder to store log files

    # eyes blinking variables
    SHOW_BLINK_COUNT_ON_SCREEN = True  # Toggle to show the blink count on the video feed
    TOTAL_BLINKS = 0  # Tracks the total number of blinks detected
    EYES_BLINK_FRAME_COUNTER = (
        0  # Counts the number of consecutive frames with a potential blink
    )
    BLINK_THRESHOLD = 0.51  # Threshold for the eye aspect ratio to trigger a blink
    EYE_AR_CONSEC_FRAMES = (
        2  # Number of consecutive frames below the threshold to confirm a blink
    )
    # SERVER_ADDRESS: Tuple containing the SERVER_IP and SERVER_PORT for UDP communication.
    # SERVER_ADDRESS = (SERVER_IP, SERVER_PORT)


    #If set to false it will wait for your command (hittig 'r') to start logging.
    IS_RECORDING = False  # Controls whether data is being logged

    # Command-line arguments for camera source
    # parser = argparse.ArgumentParser(description="Eye Tracking Application")
    # parser.add_argument(
    #     "-c", "--camSource", help="Source of camera", default=str(DEFAULT_WEBCAM)
    # )
    #args = parser.parse_args()

    # Iris and eye corners landmarks indices
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]
    L_H_LEFT = [33]  # Left eye Left Corner
    L_H_RIGHT = [133]  # Left eye Right Corner
    R_H_LEFT = [362]  # Right eye Left Corner
    R_H_RIGHT = [263]  # Right eye Right Corner

    # Blinking Detection landmark's indices.
    # P0, P3, P4, P5, P8, P11, P12, P13
    RIGHT_EYE_POINTS = [33, 160, 159, 158, 133, 153, 145, 144]
    LEFT_EYE_POINTS = [362, 385, 386, 387, 263, 373, 374, 380]

    # Face Selected points indices for Head Pose Estimation
    _indices_pose = [1, 33, 61, 199, 263, 291]

    # Function to calculate vector position
    def vector_position(point1, point2):
        x1, y1 = point1.ravel()
        x2, y2 = point2.ravel()
        return x2 - x1, y2 - y1


    def euclidean_distance_3D(points):
        """Calculates the Euclidean distance between two points in 3D space.

        Args:
            points: A list of 3D points.

        Returns:
            The Euclidean distance between the two points.

            # Comment: This function calculates the Euclidean distance between two points in 3D space.
        """

        # Get the three points.
        P0, P3, P4, P5, P8, P11, P12, P13 = points

        # Calculate the numerator.
        numerator = (
            np.linalg.norm(P3 - P13) ** 3
            + np.linalg.norm(P4 - P12) ** 3
            + np.linalg.norm(P5 - P11) ** 3
        )

        # Calculate the denominator.
        denominator = 3 * np.linalg.norm(P0 - P8) ** 3

        # Calculate the distance.
        distance = numerator / denominator

        return distance

    def estimate_head_pose(landmarks, image_size):
        # Scale factor based on user's face width (assumes model face width is 150mm)
        scale_factor = USER_FACE_WIDTH / 150.0
        # 3D model points.
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0 * scale_factor, -65.0 * scale_factor),        # Chin
            (-225.0 * scale_factor, 170.0 * scale_factor, -135.0 * scale_factor),     # Left eye left corner
            (225.0 * scale_factor, 170.0 * scale_factor, -135.0 * scale_factor),      # Right eye right corner
            (-150.0 * scale_factor, -150.0 * scale_factor, -125.0 * scale_factor),    # Left Mouth corner
            (150.0 * scale_factor, -150.0 * scale_factor, -125.0 * scale_factor)      # Right mouth corner
        ])
        

        # Camera internals
        focal_length = image_size[1]
        center = (image_size[1]/2, image_size[0]/2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]], dtype = "double"
        )

        # Assuming no lens distortion
        dist_coeffs = np.zeros((4,1))

        # 2D image points from landmarks, using defined indices
        image_points = np.array([
            landmarks[NOSE_TIP_INDEX],            # Nose tip
            landmarks[CHIN_INDEX],                # Chin
            landmarks[LEFT_EYE_LEFT_CORNER_INDEX],  # Left eye left corner
            landmarks[RIGHT_EYE_RIGHT_CORNER_INDEX],  # Right eye right corner
            landmarks[LEFT_MOUTH_CORNER_INDEX],      # Left mouth corner
            landmarks[RIGHT_MOUTH_CORNER_INDEX]      # Right mouth corner
        ], dtype="double")


            # Solve for pose
        (success, rotation_vector, translation_vector) = cv.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)

        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv.Rodrigues(rotation_vector)

        # Combine rotation matrix and translation vector to form a 3x4 projection matrix
        projection_matrix = np.hstack((rotation_matrix, translation_vector.reshape(-1, 1)))

        # Decompose the projection matrix to extract Euler angles
        _, _, _, _, _, _, euler_angles = cv.decomposeProjectionMatrix(projection_matrix)
        pitch, yaw, roll = euler_angles.flatten()[:3]


        # Normalize the pitch angle
        pitch = normalize_pitch(pitch)

        return pitch, yaw, roll

    def normalize_pitch(pitch):
        """
        Normalize the pitch angle to be within the range of [-90, 90].

        Args:
            pitch (float): The raw pitch angle in degrees.

        Returns:
            float: The normalized pitch angle.
        """
        # Map the pitch angle to the range [-180, 180]
        if pitch > 180:
            pitch -= 360

        # Invert the pitch angle for intuitive up/down movement
        pitch = -pitch

        # Ensure that the pitch is within the range of [-90, 90]
        if pitch < -90:
            pitch = -(180 + pitch)
        elif pitch > 90:
            pitch = 180 - pitch
            
        pitch = -pitch

        return pitch


    # This function calculates the blinking ratio of a person.
    def blinking_ratio(landmarks):
        """Calculates the blinking ratio of a person.

        Args:
            landmarks: A facial landmarks in 3D normalized.

        Returns:
            The blinking ratio of the person, between 0 and 1, where 0 is fully open and 1 is fully closed.

        """

        # Get the right eye ratio.
        right_eye_ratio = euclidean_distance_3D(landmarks[RIGHT_EYE_POINTS])

        # Get the left eye ratio.
        left_eye_ratio = euclidean_distance_3D(landmarks[LEFT_EYE_POINTS])

        # Calculate the blinking ratio.
        ratio = (right_eye_ratio + left_eye_ratio + 1) / 2

        return ratio


    # Initializing MediaPipe face mesh and camera
    if PRINT_DATA:
      #  print("Initializing the face mesh and camera...")
        if PRINT_DATA:
            head_pose_status = "enabled" if ENABLE_HEAD_POSE else "disabled"
          #  print(f"Head pose estimation is {head_pose_status}.")

    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    # cam_source = int(args.camSource)
    cap = cv.VideoCapture(filepath)    
   # Main loop for video capture and processing
    watchLeft=0
    watchright=0
    blink=0
    watchcenter=0
    professional_shoulder=0
    left_movement=0
    right_movement=0
    total_frames=0
    shoulder_data=()
    try:
        angle_buffer = AngleBuffer(size=MOVING_AVERAGE_WINDOW)  # Adjust size for smoothing
        #image frames[], 

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flipping the frame for a mirror effect
            # I think we better not flip to correspond with real world... need to make sure later...
            #frame = cv.flip(frame, 1)
            image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            total_frames+=1
            # Process the image to find the pose landmarks.
            results = pose.process(image)

            # Draw the image back in BGR color space.
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

            faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                # Extract the face ROI (Region of Interest)
                face_roi = rgb_frame[y:y + h, x:x + w]

                
                # Perform emotion analysis on the face ROI
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                print(result)
                # Determine the dominant emotion
                emotion = result[0]['dominant_emotion']


            if results.pose_landmarks:
                # Extract coordinates for both shoulders.
                left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

                # Convert landmark coordinates to pixel values.
                h, w, _ = image.shape
                left_shoulder_coords = (int(left_shoulder.x * w), int(left_shoulder.y * h))
                right_shoulder_coords = (int(right_shoulder.x * w), int(right_shoulder.y * h))

                # Draw circles on the shoulder points.
                # cv2.circle(image, left_shoulder_coords, 10, (255, 0, 0), -1)  # Blue circle for left shoulder.
                # cv2.circle(image, right_shoulder_coords, 10, (0, 0, 255), -1)  # Red circle for right shoulder.

                # print(left_shoulder_coords[1])
                # print(right_shoulder_coords[1])

                if (left_shoulder_coords[1]-right_shoulder_coords[1]) in range(-20,21):
                    professional_shoulder+=1
                elif (left_shoulder_coords[1]-right_shoulder_coords[1])>21:
                    left_movement+=1
                 #   print("left")
                elif (left_shoulder_coords[1]-right_shoulder_coords[1])<-21:
                    right_movement+=1
                  #  print("right")
                    
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            img_h, img_w = frame.shape[:2]
            results = mp_face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                mesh_points = np.array(
                    [
                        np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                        for p in results.multi_face_landmarks[0].landmark
                    ]
                )

                # Get the 3D landmarks from facemesh x, y and z(z is distance from 0 points)
                # just normalize values
                mesh_points_3D = np.array(
                    [[n.x, n.y, n.z] for n in results.multi_face_landmarks[0].landmark]
                )
                # getting the head pose estimation 3d points
                head_pose_points_3D = np.multiply(
                    mesh_points_3D[_indices_pose], [img_w, img_h, 1]
                )
                head_pose_points_2D = mesh_points[_indices_pose]

                # collect nose three dimension and two dimension points
                nose_3D_point = np.multiply(head_pose_points_3D[0], [1, 1, 3000])
                nose_2D_point = head_pose_points_2D[0]

                # create the camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array(
                    [[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]]
                )

                # The distortion parameters
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                head_pose_points_2D = np.delete(head_pose_points_3D, 2, axis=1)
                head_pose_points_3D = head_pose_points_3D.astype(np.float64)
                head_pose_points_2D = head_pose_points_2D.astype(np.float64)
                # Solve PnP
                success, rot_vec, trans_vec = cv.solvePnP(
                    head_pose_points_3D, head_pose_points_2D, cam_matrix, dist_matrix
                )
                # Get rotational matrix
                rotation_matrix, jac = cv.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv.RQDecomp3x3(rotation_matrix)

                # Get the y rotation degree
                angle_x = angles[0] * 360
                angle_y = angles[1] * 360
                z = angles[2] * 360

                # if angle cross the values then
                threshold_angle = 10
                # See where the user's head tilting
                if angle_y < -threshold_angle:
                    face_looks = "Left"
                elif angle_y > threshold_angle:
                    face_looks = "Right"
                elif angle_x < -threshold_angle:
                    face_looks = "Down"
                elif angle_x > threshold_angle:
                    face_looks = "Up"
                else:
                    face_looks = "Forward"
                if SHOW_ON_SCREEN_DATA:
                    cv.putText(
                        frame,
                        f"Face Looking at {face_looks}",
                        (img_w - 400, 80),
                        cv.FONT_HERSHEY_TRIPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                        cv.LINE_AA,
                    )
                # Display the nose direction
            
                p1 = nose_2D_point
                p2 = (
                    int(nose_2D_point[0] + angle_y * 10),
                    int(nose_2D_point[1] - angle_x * 10),
                )

                cv.line(frame, p1, p2, (255, 0, 255), 3)
                # getting the blinking ratio
                eyes_aspect_ratio = blinking_ratio(mesh_points_3D)
                # print(f"Blinking ratio : {ratio}")
                # checking if ear less then or equal to required threshold if yes then
                # count the number of frame frame while eyes are closed.
                if eyes_aspect_ratio <= BLINK_THRESHOLD:
                    EYES_BLINK_FRAME_COUNTER += 1
                # else check if eyes are closed is greater EYE_AR_CONSEC_FRAMES frame then
                # count the this as a blink
                # make frame counter equal to zero

                else:
                    if EYES_BLINK_FRAME_COUNTER > EYE_AR_CONSEC_FRAMES:
                        TOTAL_BLINKS += 1
                        blink+=1
                    EYES_BLINK_FRAME_COUNTER = 0
                
                # Display all facial landmarks if enabled
                if SHOW_ALL_FEATURES:
                    for point in mesh_points:
                        cv.circle(frame, tuple(point), 1, (0, 255, 0), -1)
                # Process and display eye features
                (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_EYE_IRIS])
                (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_EYE_IRIS])
                center_left = np.array([l_cx, l_cy], dtype=np.int32)
                center_right = np.array([r_cx, r_cy], dtype=np.int32)

                # Highlighting the irises and corners of the eyes
                cv.circle(
                    frame, center_left, int(l_radius), (255, 0, 255), 2, cv.LINE_AA
                )  # Left iris
                cv.circle(
                    frame, center_right, int(r_radius), (255, 0, 255), 2, cv.LINE_AA
                )  # Right iris
                cv.circle(
                    frame, mesh_points[LEFT_EYE_INNER_CORNER][0], 3, (255, 255, 255), -1, cv.LINE_AA
                )  # Left eye right corner
                cv.circle(
                    frame, mesh_points[LEFT_EYE_OUTER_CORNER][0], 3, (0, 255, 255), -1, cv.LINE_AA
                )  # Left eye left corner
                cv.circle(
                    frame, mesh_points[RIGHT_EYE_INNER_CORNER][0], 3, (255, 255, 255), -1, cv.LINE_AA
                )  # Right eye right corner
                cv.circle(
                    frame, mesh_points[RIGHT_EYE_OUTER_CORNER][0], 3, (0, 255, 255), -1, cv.LINE_AA
                )  # Right eye left corner

                # Calculating relative positions
                l_dx, l_dy = vector_position(mesh_points[LEFT_EYE_OUTER_CORNER], center_left)
                r_dx, r_dy = vector_position(mesh_points[RIGHT_EYE_OUTER_CORNER], center_right)

                # Printing data if enabled
                
                if PRINT_DATA:
                   # print(f"Total Blinks: {TOTAL_BLINKS}")
                    if is_left((l_cx,l_cy),(r_cx,r_cy),(l_dx,l_dy),(r_dx,r_cy)):
                        watchLeft+=1
                    #    print("Person is watching Left", watchLeft)
                    elif is_right((l_cx,l_cy),(r_cx,r_cy),(l_dx,l_dy),(r_dx,r_cy)):
                        #print("Person is watching right")
                        watchright+=1
                     #   print("Person is watching Left", watchright)
                    elif is_center((l_cx,l_cy),(r_cx,r_cy),(l_dx,l_dy),(r_dx,r_cy)):
                        # print("Person is watching center")
                        watchcenter+=1
                      #  print("Person is watching Left", watchcenter)
                    # print(blink)
                    # print(f"Left Eye Center X: {l_cx} Y: {l_cy}")
                    # print(f"Right Eye Center X: {r_cx} Y: {r_cy}")
                    # print(f"Left Iris Relative Pos Dx: {l_dx} Dy: {l_dy}")
                    # print(f"Right Iris Relative Pos Dx: {r_dx} Dy: {r_dy}\n")
                    # # Check if head pose estimation is enabled
                    if ENABLE_HEAD_POSE:
                        pitch, yaw, roll = estimate_head_pose(mesh_points, (img_h, img_w))
                        angle_buffer.add([pitch, yaw, roll])
                        pitch, yaw, roll = angle_buffer.get_average()

                        # Set initial angles on first successful estimation or recalibrate
                        if initial_pitch is None or (key == ord('c') and calibrated):
                            initial_pitch, initial_yaw, initial_roll = pitch, yaw, roll
                            calibrated = True
                            # if PRINT_DATA:
                                
                                # print("Head pose recalibrated.")

                        # Adjust angles based on initial calibration
                        if calibrated:
                            pitch -= initial_pitch
                            yaw -= initial_yaw
                            roll -= initial_roll
                        
                        
                        if PRINT_DATA:
                            # print(f"Head Pose Angles: Pitch={pitch}, Yaw={yaw}, Roll={roll}")
                            dataList.append({'pitch': pitch, 'roll': roll, 'yaw': yaw})


            
            key = cv.waitKey(1) & 0xFF
        shoulder_data=((professional_shoulder/total_frames),(left_movement/total_frames),(right_movement/total_frames))
       # print(shoulder_data)  
    except Exception as e:
        print(f"An error occurred: {e}")
    return [dataList,watchLeft,watchcenter,watchright,blink,shoulder_data]


# if __name__=="__main__":
#  data = count_facing_directions(faceTrack("datasets/ChaLearn/test/rbbrebrnbte.mp4"))
#  print(data)