# import os
# import time
# import cv2
# import mediapipe as mp
# import pandas as pd

# # Define the folders and video numbers
# # folders = ['data/push-up-multi-reps/front-facing', 
# #            'data/push-up-multi-reps/left-facing', 
# #            'data/push-up-multi-reps/right-facing']
# folders = ['data/pull-ups']

# video_numbers = list(range(1, 27))  # List of video numbers to process

# output_path = 'pullup-keypoints'
# output_csv_path = os.path.join(output_path, 'pullup-keypoints.csv')

# # Initialize Mediapipe Pose Estimator
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# # Prepare a list to store all data
# all_data = []

# # Process videos in each folder
# for folder in folders:
#     print(f"Processing folder: {folder}")
#     for video_file in os.listdir(folder):
#         if video_file.endswith('.mp4'):
#             video_id = os.path.splitext(video_file)[0]

#             # Check if the video number is in the target list
#             try:
#                 video_num = int(video_id.split('_')[-1])  # Assuming filename ends with the number
#                 if video_num in video_numbers:
#                     print(f"Processing video: {video_id}")
#                     full_video_path = os.path.join(folder, video_file)

#                     # Open the video
#                     cap = cv2.VideoCapture(full_video_path)
#                     frame_idx = 0

#                     start_time = time.time()

#                     # Process each frame of the video
#                     while cap.isOpened():
#                         ret, frame = cap.read()
#                         if not ret:
#                             break

#                         frame_idx += 1
#                         # Convert the frame to RGB (Mediapipe uses RGB)
#                         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                         results = pose.process(frame_rgb)

#                         # Prepare a row of data for this frame
#                         row = {'video_id': video_id, 'frame_id': frame_idx, 'folder': os.path.basename(folder)}

#                         if results.pose_landmarks:
#                             # Collect keypoints for each landmark (33 landmarks, each with 4 values: x, y, z, visibility)
#                             for i, landmark in enumerate(results.pose_landmarks.landmark):
#                                 row[f'landmark_{i}_x'] = landmark.x
#                                 row[f'landmark_{i}_y'] = landmark.y
#                                 row[f'landmark_{i}_z'] = landmark.z
#                                 row[f'landmark_{i}_visibility'] = landmark.visibility
#                         else:
#                             # If no pose is detected, store None for keypoints
#                             for i in range(33):
#                                 row[f'landmark_{i}_x'] = None
#                                 row[f'landmark_{i}_y'] = None
#                                 row[f'landmark_{i}_z'] = None
#                                 row[f'landmark_{i}_visibility'] = None
                        
#                         all_data.append(row)

#                     # Close video capture
#                     cap.release()

#                     end_time = time.time()
#                     print(f"Processed {video_file} in {end_time - start_time:.2f} seconds")
#                 else:
#                     print(f"Skipping video: {video_id} (Not in target range)")
#             except ValueError:
#                 print(f"Skipping video: {video_file} (Invalid format)")

# # After processing all videos, save the collected data to a single CSV file
# df = pd.DataFrame(all_data)
# os.makedirs(output_path, exist_ok=True)

# # Check if the output CSV already exists
# if os.path.exists(output_csv_path):
#     # If it exists, append the new data
#     df_existing = pd.read_csv(output_csv_path)
#     df_combined = pd.concat([df_existing, df], ignore_index=True)
#     df_combined.to_csv(output_csv_path, index=False)
# else:
#     # If it doesn't exist, create a new file
#     df.to_csv(output_csv_path, index=False)

# print(f"All video data saved to {output_csv_path}")


import math
import pandas as pd

import numpy as np

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


# Read the CSV file with keypoints
df = pd.read_csv('pullup-keypoints/pullup-keypoints.csv')

# Calculate joint angles for each row
for index, row in df.iterrows():
    # Extract x, y, z coordinates for each landmark
    landmarks = {}
    for i in range(33):
        landmarks[i] = {
            'x': row[f'landmark_{i}_x'],
            'y': row[f'landmark_{i}_y'],
            'z': row[f'landmark_{i}_z']
        }
    
    # Left Elbow Angle (left_shoulder -> left_elbow -> left_wrist)
    left_elbow_angle = calculate_angle(
        (landmarks[11]['x'], landmarks[11]['y'], landmarks[11]['z']),
        (landmarks[13]['x'], landmarks[13]['y'], landmarks[13]['z']),
        (landmarks[15]['x'], landmarks[15]['y'], landmarks[15]['z'])
    )

    # Right Elbow Angle (right_shoulder -> right_elbow -> right_wrist)
    right_elbow_angle = calculate_angle(
        (landmarks[12]['x'], landmarks[12]['y'], landmarks[12]['z']),
        (landmarks[14]['x'], landmarks[14]['y'], landmarks[14]['z']),
        (landmarks[16]['x'], landmarks[16]['y'], landmarks[16]['z'])
    )

    # Left Shoulder Angle (left_elbow -> left_shoulder -> spine)
    left_shoulder_angle = calculate_angle(
        (landmarks[13]['x'], landmarks[13]['y'], landmarks[13]['z']),
        (landmarks[11]['x'], landmarks[11]['y'], landmarks[11]['z']),
        (landmarks[0]['x'], landmarks[0]['y'], landmarks[0]['z'])  # Spine/torso
    )

    # Right Shoulder Angle (right_elbow -> right_shoulder -> spine)
    right_shoulder_angle = calculate_angle(
        (landmarks[14]['x'], landmarks[14]['y'], landmarks[14]['z']),
        (landmarks[12]['x'], landmarks[12]['y'], landmarks[12]['z']),
        (landmarks[0]['x'], landmarks[0]['y'], landmarks[0]['z'])  # Spine/torso
    )

    # Hip Spine Angle (left_hip -> spine -> right_hip)
    hip_spine_angle = calculate_angle(
        (landmarks[23]['x'], landmarks[23]['y'], landmarks[23]['z']),
        (landmarks[0]['x'], landmarks[0]['y'], landmarks[0]['z']),
        (landmarks[24]['x'], landmarks[24]['y'], landmarks[24]['z'])
    )

    # Left Knee Angle (left_hip -> left_knee -> left_ankle)
    left_knee_angle = calculate_angle(
        (landmarks[23]['x'], landmarks[23]['y'], landmarks[23]['z']),
        (landmarks[25]['x'], landmarks[25]['y'], landmarks[25]['z']),
        (landmarks[27]['x'], landmarks[27]['y'], landmarks[27]['z'])
    )

    # Right Knee Angle (right_hip -> right_knee -> right_ankle)
    right_knee_angle = calculate_angle(
        (landmarks[24]['x'], landmarks[24]['y'], landmarks[24]['z']),
        (landmarks[26]['x'], landmarks[26]['y'], landmarks[26]['z']),
        (landmarks[28]['x'], landmarks[28]['y'], landmarks[28]['z'])
    )

    # Spine Alignment (angle between torso and legs)
    spine_alignment = calculate_angle(
        (landmarks[0]['x'], landmarks[0]['y'], landmarks[0]['z']),  # Spine/torso
        (landmarks[23]['x'], landmarks[23]['y'], landmarks[23]['z']),  # Left hip
        (landmarks[24]['x'], landmarks[24]['y'], landmarks[24]['z'])   # Right hip
    )

    neck_coords = (
        (landmarks[11]['x'] + landmarks[12]['x']) / 2,
        (landmarks[11]['y'] + landmarks[12]['y']) / 2,
        (landmarks[11]['z'] + landmarks[12]['z']) / 2
    )

    head_neck_spine_angle = calculate_angle( neck_coords,  # Midpoint of neck
        (landmarks[0]['x'], landmarks[0]['y'], landmarks[0]['z']),  # Head (nose)
        (landmarks[1]['x'], landmarks[1]['y'], landmarks[1]['z'])   # Spine/shoulder center
    )

    # Add the angles to the row
    df.at[index, 'left_elbow_angle'] = left_elbow_angle
    df.at[index, 'right_elbow_angle'] = right_elbow_angle
    df.at[index, 'left_shoulder_angle'] = left_shoulder_angle
    df.at[index, 'right_shoulder_angle'] = right_shoulder_angle
    df.at[index, 'hip_spine_angle'] = hip_spine_angle
    df.at[index, 'left_knee_angle'] = left_knee_angle
    df.at[index, 'right_knee_angle'] = right_knee_angle
    df.at[index, 'spine_alignment'] = spine_alignment
    df.at[index, 'head_neck_spine_angle'] = head_neck_spine_angle


def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Iterate through the DataFrame to calculate distances
for index, row in df.iterrows():
    # Extract x, y, z coordinates for each landmark
    landmarks = {}
    for i in range(33):
        landmarks[i] = {
            'x': row[f'landmark_{i}_x'],
            'y': row[f'landmark_{i}_y'],
            'z': row[f'landmark_{i}_z']
        }
    
    # Reference for normalization (shoulder width)
    shoulder_width = calculate_distance(
        (landmarks[11]['x'], landmarks[11]['y'], landmarks[11]['z']),
        (landmarks[12]['x'], landmarks[12]['y'], landmarks[12]['z'])
    )
    
    # Calculate and normalize distances
    distances = {
        'elbow_wrist_left': calculate_distance(
            (landmarks[13]['x'], landmarks[13]['y'], landmarks[13]['z']),
            (landmarks[15]['x'], landmarks[15]['y'], landmarks[15]['z'])
        ) / shoulder_width,
        'elbow_wrist_right': calculate_distance(
            (landmarks[14]['x'], landmarks[14]['y'], landmarks[14]['z']),
            (landmarks[16]['x'], landmarks[16]['y'], landmarks[16]['z'])
        ) / shoulder_width,
        'shoulder_elbow_left': calculate_distance(
            (landmarks[11]['x'], landmarks[11]['y'], landmarks[11]['z']),
            (landmarks[13]['x'], landmarks[13]['y'], landmarks[13]['z'])
        ) / shoulder_width,
        'shoulder_elbow_right': calculate_distance(
            (landmarks[12]['x'], landmarks[12]['y'], landmarks[12]['z']),
            (landmarks[14]['x'], landmarks[14]['y'], landmarks[14]['z'])
        ) / shoulder_width,
        'shoulder_hip_left': calculate_distance(
            (landmarks[11]['x'], landmarks[11]['y'], landmarks[11]['z']),
            (landmarks[23]['x'], landmarks[23]['y'], landmarks[23]['z'])
        ) / shoulder_width,
        'shoulder_hip_right': calculate_distance(
            (landmarks[12]['x'], landmarks[12]['y'], landmarks[12]['z']),
            (landmarks[24]['x'], landmarks[24]['y'], landmarks[24]['z'])
        ) / shoulder_width,
        'spine_neck': calculate_distance(
            (landmarks[0]['x'], landmarks[0]['y'], landmarks[0]['z']),
            (landmarks[1]['x'], landmarks[1]['y'], landmarks[1]['z'])
        ) / shoulder_width,
        'hip_knee_left': calculate_distance(
            (landmarks[23]['x'], landmarks[23]['y'], landmarks[23]['z']),
            (landmarks[25]['x'], landmarks[25]['y'], landmarks[25]['z'])
        ) / shoulder_width,
        'hip_knee_right': calculate_distance(
            (landmarks[24]['x'], landmarks[24]['y'], landmarks[24]['z']),
            (landmarks[26]['x'], landmarks[26]['y'], landmarks[26]['z'])
        ) / shoulder_width,
        'knee_ankle_left': calculate_distance(
            (landmarks[25]['x'], landmarks[25]['y'], landmarks[25]['z']),
            (landmarks[27]['x'], landmarks[27]['y'], landmarks[27]['z'])
        ) / shoulder_width,
        'knee_ankle_right': calculate_distance(
            (landmarks[26]['x'], landmarks[26]['y'], landmarks[26]['z']),
            (landmarks[28]['x'], landmarks[28]['y'], landmarks[28]['z'])
        ) / shoulder_width,
    }

    # Add distances to the DataFrame
    for key, value in distances.items():
        df.at[index, key] = value


# Save the updated DataFrame with angles to a new CSV file
df.to_csv('pullup-keypoints/pullup-keypoints-new.csv', index=False)
print("Angles added and distances saved to 'pushup-keypoints-new.csv'")
