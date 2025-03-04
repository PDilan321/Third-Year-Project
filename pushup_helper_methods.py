import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix, classification_report

import cv2
import mediapipe as mp

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import numpy as np
import pushup_helper_methods as ph


angle_columns = [
    'left_elbow_angle', 'right_elbow_angle', 'left_shoulder_angle', 
    'right_shoulder_angle', 'hip_spine_angle', 'left_knee_angle', 
    'right_knee_angle', 'spine_alignment', 'head_neck_spine_angle'
]

mmpose_angle_columns = [
    'left_elbow_angle', 'right_elbow_angle', 'left_shoulder_angle', 
    'right_shoulder_angle', 'hip_spine_angle', 'left_knee_angle', 
    'right_knee_angle',
]

distance_columns = [
    'elbow_wrist_left', 'elbow_wrist_right',
    'shoulder_elbow_left', 'shoulder_elbow_right',
    'shoulder_hip_left', 'shoulder_hip_right',
    'spine_neck', 'hip_knee_left', 'hip_knee_right',
    'knee_ankle_left', 'knee_ankle_right'
]

mmpose_distance_columns = [
    'elbow_wrist_left', 'elbow_wrist_right',
    'shoulder_elbow_left', 'shoulder_elbow_right',
    'shoulder_hip_left', 'shoulder_hip_right',
    'hip_knee_left', 'hip_knee_right',
    'knee_ankle_left', 'knee_ankle_right'
]

landmark_columns_3d = [
        f'landmark_{i}_{axis}' for i in range(33) for axis in ['x', 'y', 'z']
]

mmpose_landmark_columns = [
    "Pelvis_x", "Pelvis_y", "Pelvis_z",
    "RHip_x", "RHip_y", "RHip_z",
    "RKnee_x", "RKnee_y", "RKnee_z",
    "RAnkle_x", "RAnkle_y", "RAnkle_z",
    "LHip_x", "LHip_y", "LHip_z",
    "LKnee_x", "LKnee_y", "LKnee_z",
    "LAnkle_x", "LAnkle_y", "LAnkle_z",
    "Spine1_x", "Spine1_y", "Spine1_z",
    "Neck_x", "Neck_y", "Neck_z",
    "Head_x", "Head_y", "Head_z",
    "Site_x", "Site_y", "Site_z",
    "LShoulder_x", "LShoulder_y", "LShoulder_z",
    "LElbow_x", "LElbow_y", "LElbow_z",
    "LWrist_x", "LWrist_y", "LWrist_z",
    "RShoulder_x", "RShoulder_y", "RShoulder_z",
    "RElbow_x", "RElbow_y", "RElbow_z",
    "RWrist_x", "RWrist_y", "RWrist_z"
]

mmpose_2d_landmark_columns = [
    "Pelvis_x", "Pelvis_y",
    "RHip_x", "RHip_y",
    "RKnee_x", "RKnee_y",
    "RAnkle_x", "RAnkle_y",
    "LHip_x", "LHip_y",
    "LKnee_x", "LKnee_y", 
    "LAnkle_x", "LAnkle_y", 
    "Spine1_x", "Spine1_y", 
    "Neck_x", "Neck_y", 
    "Head_x", "Head_y", 
    "Site_x", "Site_y",
    "LShoulder_x", "LShoulder_y", 
    "LElbow_x", "LElbow_y", 
    "LWrist_x", "LWrist_y",
    "RShoulder_x", "RShoulder_y",
    "RElbow_x", "RElbow_y", 
    "RWrist_x", "RWrist_y", 
]

mmpose_columns = mmpose_angle_columns + mmpose_distance_columns + mmpose_distance_columns
mmpose_2d_columns = angle_columns + distance_columns + mmpose_2d_landmark_columns

mediapipe_columns = angle_columns + distance_columns + landmark_columns_3d

def calculate_angle(p1, p2, p3):
    """Calculate angle at point p2 given 3 points."""
    v1 = np.array(p1) - np.array(p2)  # Vector from p2 to p1
    v2 = np.array(p3) - np.array(p2)  # Vector from p2 to p3

    # Check if either of the vectors is zero-length
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return float('nan')  # Return NaN if division by zero is possible

    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))  # Cosine of the angle

    # Ensure the cosine value is within the valid range for arccos
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)  # Compute angle in radians
    return np.degrees(angle)  # Convert angle from radians to degrees

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))


def extract_landmarks(landmarks):
    """Extract all 3D landmarks into a flat array."""
    landmark_array = []
    for i in range(33):  # Total 33 landmarks
        landmark_array.extend([landmarks[i].x, landmarks[i].y, landmarks[i].z])
    return landmark_array

def extract_pose_features(video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(video_path)

    angles = []
    distances = [] 
    landmarks_3d = []


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            landmark_features = extract_landmarks(landmarks)
            landmarks_3d.append(landmark_features)

            # Extract coordinates for each landmark
            landmarks_dict = {}
            for i in range(33):  # 33 pose landmarks
                landmarks_dict[i] = {
                    'x': landmarks[i].x,
                    'y': landmarks[i].y,
                    'z': landmarks[i].z
                }
            
            # Calculate the angles
            left_elbow_angle = calculate_angle(
                (landmarks_dict[11]['x'], landmarks_dict[11]['y'], landmarks_dict[11]['z']),
                (landmarks_dict[13]['x'], landmarks_dict[13]['y'], landmarks_dict[13]['z']),
                (landmarks_dict[15]['x'], landmarks_dict[15]['y'], landmarks_dict[15]['z'])
            )

            right_elbow_angle = calculate_angle(
                (landmarks_dict[12]['x'], landmarks_dict[12]['y'], landmarks_dict[12]['z']),
                (landmarks_dict[14]['x'], landmarks_dict[14]['y'], landmarks_dict[14]['z']),
                (landmarks_dict[16]['x'], landmarks_dict[16]['y'], landmarks_dict[16]['z'])
            )

            left_shoulder_angle = calculate_angle(
                (landmarks_dict[13]['x'], landmarks_dict[13]['y'], landmarks_dict[13]['z']),
                (landmarks_dict[11]['x'], landmarks_dict[11]['y'], landmarks_dict[11]['z']),
                (landmarks_dict[0]['x'], landmarks_dict[0]['y'], landmarks_dict[0]['z'])  # Spine/torso
            )

            right_shoulder_angle = calculate_angle(
                (landmarks_dict[14]['x'], landmarks_dict[14]['y'], landmarks_dict[14]['z']),
                (landmarks_dict[12]['x'], landmarks_dict[12]['y'], landmarks_dict[12]['z']),
                (landmarks_dict[0]['x'], landmarks_dict[0]['y'], landmarks_dict[0]['z'])  # Spine/torso
            )

            hip_spine_angle = calculate_angle(
                (landmarks_dict[23]['x'], landmarks_dict[23]['y'], landmarks_dict[23]['z']),
                (landmarks_dict[0]['x'], landmarks_dict[0]['y'], landmarks_dict[0]['z']),
                (landmarks_dict[24]['x'], landmarks_dict[24]['y'], landmarks_dict[24]['z'])
            )

            left_knee_angle = calculate_angle(
                (landmarks_dict[23]['x'], landmarks_dict[23]['y'], landmarks_dict[23]['z']),
                (landmarks_dict[25]['x'], landmarks_dict[25]['y'], landmarks_dict[25]['z']),
                (landmarks_dict[27]['x'], landmarks_dict[27]['y'], landmarks_dict[27]['z'])
            )

            right_knee_angle = calculate_angle(
                (landmarks_dict[24]['x'], landmarks_dict[24]['y'], landmarks_dict[24]['z']),
                (landmarks_dict[26]['x'], landmarks_dict[26]['y'], landmarks_dict[26]['z']),
                (landmarks_dict[28]['x'], landmarks_dict[28]['y'], landmarks_dict[28]['z'])
            )

            spine_alignment = calculate_angle(
                (landmarks_dict[0]['x'], landmarks_dict[0]['y'], landmarks_dict[0]['z']),
                (landmarks_dict[23]['x'], landmarks_dict[23]['y'], landmarks_dict[23]['z']),
                (landmarks_dict[24]['x'], landmarks_dict[24]['y'], landmarks_dict[24]['z'])
            )

            neck_coords = (
                (landmarks_dict[11]['x'] + landmarks_dict[12]['x']) / 2,
                (landmarks_dict[11]['y'] + landmarks_dict[12]['y']) / 2,
                (landmarks_dict[11]['z'] + landmarks_dict[12]['z']) / 2
            )

            head_neck_spine_angle = calculate_angle(
                neck_coords,  # Midpoint of neck
                (landmarks_dict[0]['x'], landmarks_dict[0]['y'], landmarks_dict[0]['z']),  # Head (nose)
                (landmarks_dict[1]['x'], landmarks_dict[1]['y'], landmarks_dict[1]['z'])   # Spine/shoulder center
            )

            angles.append([
                left_elbow_angle, right_elbow_angle, left_shoulder_angle, right_shoulder_angle,
                hip_spine_angle, left_knee_angle, right_knee_angle, spine_alignment, head_neck_spine_angle
            ])

            # Calculate the distances (normalized by shoulder width)
            shoulder_width = calculate_distance(
                (landmarks_dict[11]['x'], landmarks_dict[11]['y'], landmarks_dict[11]['z']),
                (landmarks_dict[12]['x'], landmarks_dict[12]['y'], landmarks_dict[12]['z'])
            )
            
            distances.append([
                calculate_distance(
                    (landmarks_dict[13]['x'], landmarks_dict[13]['y'], landmarks_dict[13]['z']),
                    (landmarks_dict[15]['x'], landmarks_dict[15]['y'], landmarks_dict[15]['z'])
                ) / shoulder_width,
                
                calculate_distance(
                    (landmarks_dict[14]['x'], landmarks_dict[14]['y'], landmarks_dict[14]['z']),
                    (landmarks_dict[16]['x'], landmarks_dict[16]['y'], landmarks_dict[16]['z'])
                ) / shoulder_width,
                
                calculate_distance(
                    (landmarks_dict[11]['x'], landmarks_dict[11]['y'], landmarks_dict[11]['z']),
                    (landmarks_dict[13]['x'], landmarks_dict[13]['y'], landmarks_dict[13]['z'])
                ) / shoulder_width,
                
                calculate_distance(
                    (landmarks_dict[12]['x'], landmarks_dict[12]['y'], landmarks_dict[12]['z']),
                    (landmarks_dict[14]['x'], landmarks_dict[14]['y'], landmarks_dict[14]['z'])
                ) / shoulder_width,
                
                calculate_distance(
                    (landmarks_dict[11]['x'], landmarks_dict[11]['y'], landmarks_dict[11]['z']),
                    (landmarks_dict[23]['x'], landmarks_dict[23]['y'], landmarks_dict[23]['z'])
                ) / shoulder_width,
                
                calculate_distance(
                    (landmarks_dict[12]['x'], landmarks_dict[12]['y'], landmarks_dict[12]['z']),
                    (landmarks_dict[24]['x'], landmarks_dict[24]['y'], landmarks_dict[24]['z'])
                ) / shoulder_width,
                
                calculate_distance(
                    (landmarks_dict[0]['x'], landmarks_dict[0]['y'], landmarks_dict[0]['z']),
                    (landmarks_dict[1]['x'], landmarks_dict[1]['y'], landmarks_dict[1]['z'])
                ) / shoulder_width,
                
                calculate_distance(
                    (landmarks_dict[23]['x'], landmarks_dict[23]['y'], landmarks_dict[23]['z']),
                    (landmarks_dict[25]['x'], landmarks_dict[25]['y'], landmarks_dict[25]['z'])
                ) / shoulder_width,
                
                calculate_distance(
                    (landmarks_dict[24]['x'], landmarks_dict[24]['y'], landmarks_dict[24]['z']),
                    (landmarks_dict[26]['x'], landmarks_dict[26]['y'], landmarks_dict[26]['z'])
                ) / shoulder_width,
                
                calculate_distance(
                    (landmarks_dict[25]['x'], landmarks_dict[25]['y'], landmarks_dict[25]['z']),
                    (landmarks_dict[27]['x'], landmarks_dict[27]['y'], landmarks_dict[27]['z'])
                ) / shoulder_width,
                
                calculate_distance(
                    (landmarks_dict[26]['x'], landmarks_dict[26]['y'], landmarks_dict[26]['z']),
                    (landmarks_dict[28]['x'], landmarks_dict[28]['y'], landmarks_dict[28]['z'])
                ) / shoulder_width,
            ])
            
    cap.release()

    angles = np.nan_to_num(np.array(angles), nan=0.0)
    distances = np.nan_to_num(np.array(distances), nan=0.0)

    return angles, distances, landmarks_3d












def extract_pose_features_single_frame(frame):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    angles = []
    distances = []
    landmarks_3d = []

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        landmark_features = extract_landmarks(landmarks)
        landmarks_3d.append(landmark_features)

        # Extract coordinates for each landmark
        landmarks_dict = {}
        for i in range(33):  # 33 pose landmarks
            landmarks_dict[i] = {
                'x': landmarks[i].x,
                'y': landmarks[i].y,
                'z': landmarks[i].z
            }

        # Calculate the angles
        left_elbow_angle = calculate_angle(
            (landmarks_dict[11]['x'], landmarks_dict[11]['y'], landmarks_dict[11]['z']),
            (landmarks_dict[13]['x'], landmarks_dict[13]['y'], landmarks_dict[13]['z']),
            (landmarks_dict[15]['x'], landmarks_dict[15]['y'], landmarks_dict[15]['z'])
        )

        right_elbow_angle = calculate_angle(
            (landmarks_dict[12]['x'], landmarks_dict[12]['y'], landmarks_dict[12]['z']),
            (landmarks_dict[14]['x'], landmarks_dict[14]['y'], landmarks_dict[14]['z']),
            (landmarks_dict[16]['x'], landmarks_dict[16]['y'], landmarks_dict[16]['z'])
        )

        left_shoulder_angle = calculate_angle(
            (landmarks_dict[13]['x'], landmarks_dict[13]['y'], landmarks_dict[13]['z']),
            (landmarks_dict[11]['x'], landmarks_dict[11]['y'], landmarks_dict[11]['z']),
            (landmarks_dict[0]['x'], landmarks_dict[0]['y'], landmarks_dict[0]['z'])  # Spine/torso
        )

        right_shoulder_angle = calculate_angle(
            (landmarks_dict[14]['x'], landmarks_dict[14]['y'], landmarks_dict[14]['z']),
            (landmarks_dict[12]['x'], landmarks_dict[12]['y'], landmarks_dict[12]['z']),
            (landmarks_dict[0]['x'], landmarks_dict[0]['y'], landmarks_dict[0]['z'])  # Spine/torso
        )

        hip_spine_angle = calculate_angle(
            (landmarks_dict[23]['x'], landmarks_dict[23]['y'], landmarks_dict[23]['z']),
            (landmarks_dict[0]['x'], landmarks_dict[0]['y'], landmarks_dict[0]['z']),
            (landmarks_dict[24]['x'], landmarks_dict[24]['y'], landmarks_dict[24]['z'])
        )

        left_knee_angle = calculate_angle(
            (landmarks_dict[23]['x'], landmarks_dict[23]['y'], landmarks_dict[23]['z']),
            (landmarks_dict[25]['x'], landmarks_dict[25]['y'], landmarks_dict[25]['z']),
            (landmarks_dict[27]['x'], landmarks_dict[27]['y'], landmarks_dict[27]['z'])
        )

        right_knee_angle = calculate_angle(
            (landmarks_dict[24]['x'], landmarks_dict[24]['y'], landmarks_dict[24]['z']),
            (landmarks_dict[26]['x'], landmarks_dict[26]['y'], landmarks_dict[26]['z']),
            (landmarks_dict[28]['x'], landmarks_dict[28]['y'], landmarks_dict[28]['z'])
        )

        spine_alignment = calculate_angle(
            (landmarks_dict[0]['x'], landmarks_dict[0]['y'], landmarks_dict[0]['z']),
            (landmarks_dict[23]['x'], landmarks_dict[23]['y'], landmarks_dict[23]['z']),
            (landmarks_dict[24]['x'], landmarks_dict[24]['y'], landmarks_dict[24]['z'])
        )

        neck_coords = (
            (landmarks_dict[11]['x'] + landmarks_dict[12]['x']) / 2,
            (landmarks_dict[11]['y'] + landmarks_dict[12]['y']) / 2,
            (landmarks_dict[11]['z'] + landmarks_dict[12]['z']) / 2
        )

        head_neck_spine_angle = calculate_angle(
            neck_coords,  # Midpoint of neck
            (landmarks_dict[0]['x'], landmarks_dict[0]['y'], landmarks_dict[0]['z']),  # Head (nose)
            (landmarks_dict[1]['x'], landmarks_dict[1]['y'], landmarks_dict[1]['z'])   # Spine/shoulder center
        )

        angles.append([
            left_elbow_angle, right_elbow_angle, left_shoulder_angle, right_shoulder_angle,
            hip_spine_angle, left_knee_angle, right_knee_angle, spine_alignment, head_neck_spine_angle
        ])

        # Calculate the distances (normalized by shoulder width)
        shoulder_width = calculate_distance(
            (landmarks_dict[11]['x'], landmarks_dict[11]['y'], landmarks_dict[11]['z']),
            (landmarks_dict[12]['x'], landmarks_dict[12]['y'], landmarks_dict[12]['z'])
        )

        distances.append([
            calculate_distance(
                (landmarks_dict[13]['x'], landmarks_dict[13]['y'], landmarks_dict[13]['z']),
                (landmarks_dict[15]['x'], landmarks_dict[15]['y'], landmarks_dict[15]['z'])
            ) / shoulder_width,
            
            calculate_distance(
                (landmarks_dict[14]['x'], landmarks_dict[14]['y'], landmarks_dict[14]['z']),
                (landmarks_dict[16]['x'], landmarks_dict[16]['y'], landmarks_dict[16]['z'])
            ) / shoulder_width,
            
            calculate_distance(
                (landmarks_dict[11]['x'], landmarks_dict[11]['y'], landmarks_dict[11]['z']),
                (landmarks_dict[13]['x'], landmarks_dict[13]['y'], landmarks_dict[13]['z'])
            ) / shoulder_width,
            
            calculate_distance(
                (landmarks_dict[12]['x'], landmarks_dict[12]['y'], landmarks_dict[12]['z']),
                (landmarks_dict[14]['x'], landmarks_dict[14]['y'], landmarks_dict[14]['z'])
            ) / shoulder_width,
            
            calculate_distance(
                (landmarks_dict[11]['x'], landmarks_dict[11]['y'], landmarks_dict[11]['z']),
                (landmarks_dict[23]['x'], landmarks_dict[23]['y'], landmarks_dict[23]['z'])
            ) / shoulder_width,
            
            calculate_distance(
                (landmarks_dict[12]['x'], landmarks_dict[12]['y'], landmarks_dict[12]['z']),
                (landmarks_dict[24]['x'], landmarks_dict[24]['y'], landmarks_dict[24]['z'])
            ) / shoulder_width,
            
            calculate_distance(
                (landmarks_dict[0]['x'], landmarks_dict[0]['y'], landmarks_dict[0]['z']),
                (landmarks_dict[1]['x'], landmarks_dict[1]['y'], landmarks_dict[1]['z'])
            ) / shoulder_width,
            
            calculate_distance(
                (landmarks_dict[23]['x'], landmarks_dict[23]['y'], landmarks_dict[23]['z']),
                (landmarks_dict[25]['x'], landmarks_dict[25]['y'], landmarks_dict[25]['z'])
            ) / shoulder_width,
            
            calculate_distance(
                (landmarks_dict[24]['x'], landmarks_dict[24]['y'], landmarks_dict[24]['z']),
                (landmarks_dict[26]['x'], landmarks_dict[26]['y'], landmarks_dict[26]['z'])
            ) / shoulder_width,
            
            calculate_distance(
                (landmarks_dict[25]['x'], landmarks_dict[25]['y'], landmarks_dict[25]['z']),
                (landmarks_dict[27]['x'], landmarks_dict[27]['y'], landmarks_dict[27]['z'])
            ) / shoulder_width,
            
            calculate_distance(
                (landmarks_dict[26]['x'], landmarks_dict[26]['y'], landmarks_dict[26]['z']),
                (landmarks_dict[28]['x'], landmarks_dict[28]['y'], landmarks_dict[28]['z'])
            ) / shoulder_width,
        ])

    # Convert to numpy arrays and handle NaN values
    angles = np.nan_to_num(np.array(angles), nan=0.0)
    distances = np.nan_to_num(np.array(distances), nan=0.0)

    return angles, distances, landmarks_3d






def euclidean_distance(landmarks, joint1, joint2):
    # Extract the coordinates of the two joints
    x1, y1, z1 = landmarks[joint1.value].x, landmarks[joint1.value].y, landmarks[joint1.value].z
    x2, y2, z2 = landmarks[joint2.value].x, landmarks[joint2.value].y, landmarks[joint2.value].z

    # Compute the Euclidean distance between the two joints
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)


from scipy.stats import mode
import numpy as np

from scipy.stats import mode
import numpy as np

import numpy as np
from scipy.stats import mode

def smooth_states(states, window_size=5):
    """
    Smooth the predicted states by correcting noisy predictions using a moving window.

    Args:
    states (list): List of predicted states (e.g., [2, 1, 2, ...]).
    window_size (int): Size of the smoothing window (must be odd).

    Returns:
    list: Smoothed states.
    """
    half_window = window_size // 2
    smoothed_states = states.copy()

    for i in range(len(states)):
        # Define the window boundaries
        start = max(0, i - half_window)
        end = min(len(states), i + half_window + 1)
        
        # Compute the mode (most frequent value) in the window
        window = states[start:end]
        window_mode = mode(window, keepdims=True).mode[0]
        smoothed_states[i] = window_mode

    # Post-processing: Remove small noisy fluctuations
    for i in range(1, len(smoothed_states) - 1):
        if smoothed_states[i - 1] == smoothed_states[i + 1] and smoothed_states[i] != smoothed_states[i - 1]:
            smoothed_states[i] = smoothed_states[i - 1]  # Replace noise with surrounding state

    return smoothed_states



# def compress_states(states):
#     """
#     Compress the states by removing consecutive duplicates.
    
#     Args:
#     states (list): List of smoothed states (e.g., [2, 2, 1, 1, ...]).
    
#     Returns:
#     list: Compressed states.
#     """
#     compressed_states = [states[0]]  # Initialize with the first state
#     for i in range(1, len(states)):
#         if states[i] != states[i - 1]:  # Add state if it's different from the previous one
#             compressed_states.append(states[i])
#     return compressed_states

def compress_states(states):
    """
    Compress the states by removing consecutive duplicates.
    
    Args:
    states (list): List of smoothed states (e.g., [2, 2, 1, 1, ...]).
    
    Returns:
    list: Compressed states.
    """
    compressed_states = [states[0]]  # Initialize with the first state
    for i in range(1, len(states)):
        if states[i] != states[i - 1]:  # Add state if it's different from the previous one
            compressed_states.append(states[i])
    return compressed_states


def get_state_mapping(smoothed_states, compressed_states):
    mapping = []
    compressed_index = 0
    for i in range(len(smoothed_states)):
        # If the current state is different from the previous one or it's the first element,
        # it should correspond to the next compressed state element
        if i == 0 or smoothed_states[i] != smoothed_states[i - 1]:
            mapping.append((compressed_index, i))  # Store the compressed index and smoothed index
            compressed_index += 1
            if compressed_index >= len(compressed_states):  # Once we run out of compressed states, stop
                break
    return mapping


def getRepsAndSmoothedStates(states, ideal_sequence=[2, 1, 0, 1, 2], window_size=5):
    rep_count = 0
    seq_len = len(ideal_sequence)

    print(f"Predicted States: {states}")

    smoothed_states = smooth_states(states, window_size=window_size)
    print(f"Smoothed States: {smoothed_states}")

    # Step 1: Compress consecutive duplicates
    compressed_states = compress_states(smoothed_states)
    print(f"Compressed States: {compressed_states}")

    state_mappings = get_state_mapping(smoothed_states, compressed_states)
    print(f"State Mappings: {state_mappings}")

    # Step 2: Sliding window with overlapping sequence detection
    i = 0
    reps_by_frame = [0] * len(smoothed_states)  # Initialize all frames with 0
    while i <= len(compressed_states) - seq_len:
        # Check if the current window matches the ideal sequence
        if compressed_states[i:i + seq_len] == ideal_sequence:
            rep_count += 1

            smoothed_state_index = state_mappings[i+seq_len - 1][1]
            for j in range(smoothed_state_index, len(smoothed_states)):
                reps_by_frame[j] = rep_count

        i += 1  # Move one step forward to allow overlap detection

    return smoothed_states, reps_by_frame









def count_reps_robust(states, ideal_sequence=[2, 1, 0, 1, 2],  window_size=5):
    rep_count = 0
    seq_len = len(ideal_sequence)

    print(f"Predicted States: {states}")

    smoothed_states = smooth_states(states, window_size=window_size)
    print(f"Smoothed States: {smoothed_states}")

    # Step 1: Compress consecutive duplicates
    compressed_states = compress_states(smoothed_states)
    print(f"Compressed States: {compressed_states}")

    # Step 2: Sliding window with overlapping sequence detection
    i = 0
    while i <= len(compressed_states) - seq_len:
        # Check if the current window matches the ideal sequence
        if compressed_states[i:i + seq_len] == ideal_sequence:
            rep_count += 1
            # Move only one step forward to allow overlap detection
            i += 1
        else:
            i += 1

    return rep_count


def create_annotated_video(video_path, predicted_states, rep_count, output_path="annotated_pushup.mp4"):
    """
    Create a video with state and repetition counter overlayed on each frame.

    Args:
        video_path (str): Path to the original video.
        predicted_states (list): List of predicted states for each frame.
        rep_count (int): Total repetition count.
        output_path (str): Path to save the annotated video.
    """
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= len(predicted_states):
            break
        
        # Annotate the frame
        state_text = f"State: {predicted_states[frame_idx]}"
        rep_text = f"Reps: {rep_count}"
        
        # Draw the text on the frame
        cv2.putText(frame, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, rep_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255), 2, cv2.LINE_AA)
        
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Annotated video saved to {output_path}")



def process_video_with_model(video_path, model_path, ideal_sequence=[2, 1, 0, 1, 2], window_size=5):
    model = load_model(model_path)

    # Extract pose features from the video
    angles, distances, landmarks_3d = ph.extract_pose_features(video_path)

    # Scale features
    scaler = StandardScaler()
    combined_features = scaler.fit_transform(
        np.hstack([angles, distances, landmarks_3d])
    )

    # Reshape for model input
    video_input = combined_features.reshape(
        -1, 1, len(ph.angle_columns + ph.distance_columns + ph.landmark_columns_3d)
    )

    # Predict states
    probabilities = model.predict(video_input)
    print(probabilities)
    predicted_states = np.argmax(probabilities, axis=1)

    # Count repetitions
    smoothed_states, repetitions = ph.getRepsAndSmoothedStates(
        states=predicted_states,
        ideal_sequence=ideal_sequence,
        window_size=window_size,
    )

    return smoothed_states, repetitions, angles, landmarks_3d