import os
import time
import pandas as pd
import time
import torch
import mmpretrain  # This ensures MMPretrain models (like ViTPose) are recognized
from mmpose.apis import MMPoseInferencer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# inferencer = MMPoseInferencer(pose2d='human')
inferencer = MMPoseInferencer(pose2d='vitpose-h')
# inferencer = MMPoseInferencer(pose3d='human3d')

# Define the folders and video numbers
folders = ['data/push-up-multi-reps/front-facing',
           'data/push-up-multi-reps/left-facing', 
           'data/push-up-multi-reps/right-facing']

video_numbers = list(range(1, 27))  # List of video numbers to process

output_path = 'vitpose-h-pushup-keypoints-2d'
output_csv_path = os.path.join(output_path, 'vitpose-h-pushup-keypoints-2d.csv')

# Prepare a list to store all data
all_data = []

# Process videos in each folder
for folder in folders:
    print(f"Processing folder: {folder}")
    for video_file in os.listdir(folder):
        if video_file.endswith('.mp4'):
            video_id = os.path.splitext(video_file)[0]

            # Check if the video number is in the target list
            try:
                video_num = int(video_id.split('_')[-1])  # Assuming filename ends with the number
                if video_num in video_numbers:
                    print(f"Processing video: {video_id}")
                    full_video_path = os.path.join(folder, video_file)

                    # Run inference
                    start_time = time.time()
                    result_generator = inferencer(  
                        full_video_path, pred_out_dir=output_path, return_vis=False, radius=8, thickness=6
                    )
                    for _ in result_generator:
                        pass
                    end_time = time.time()
                    print(f"Processed {video_file} in {end_time - start_time:.2f} seconds")
                else:
                    print(f"Skipping video: {video_id} (Not in target range)")
            except ValueError:
                print(f"Skipping video: {video_file} (Invalid format)")

# After processing all videos, save the collected data to a single CSV file
df = pd.DataFrame(all_data)
os.makedirs(output_path, exist_ok=True)

# Check if the output CSV already exists
if os.path.exists(output_csv_path):
    # If it exists, append the new data
    df_existing = pd.read_csv(output_csv_path)
    df_combined = pd.concat([df_existing, df], ignore_index=True)
    df_combined.to_csv(output_csv_path, index=False)
else:
    # If it doesn't exist, create a new file
    df.to_csv(output_csv_path, index=False)

print(f"All video data saved to {output_csv_path}")

# import math
# import pandas as pd
# import numpy as np

# def calculate_angle(p1, p2, p3):
#     """Calculate angle at point p2 given 3 points."""
#     v1 = np.array(p1) - np.array(p2)  # Vector from p2 to p1
#     v2 = np.array(p3) - np.array(p2)  # Vector from p2 to p3

#     if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
#         return float('nan')  # Return NaN if division by zero is possible

#     cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
#     cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
#     angle = np.arccos(cosine_angle)
#     return np.degrees(angle)

# def calculate_distance(p1, p2):
#     """Calculate Euclidean distance between two points."""
#     return np.linalg.norm(np.array(p1) - np.array(p2))

# # Read the CSV file with MMPose keypoints
# df = pd.read_csv('mmpose-pushup-keypoints/mmpose-pushup-keypoints-neww.csv')

# # Calculate joint angles and normalized distances for each row
# for index, row in df.iterrows():
#     landmarks = {}
#     landmark_names = ['Pelvis', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'Spine1', 'Neck', 'Head', 'Site', 
#                       'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']

#     for name in landmark_names:
#         landmarks[name] = {
#             'x': row[f'{name}_x'],
#             'y': row[f'{name}_y'],
#             'z': row[f'{name}_z']
#         }

#     # Shoulder width for normalization
#     shoulder_width = calculate_distance(
#         (landmarks['LShoulder']['x'], landmarks['LShoulder']['y'], landmarks['LShoulder']['z']),
#         (landmarks['RShoulder']['x'], landmarks['RShoulder']['y'], landmarks['RShoulder']['z'])
#     )

#     # Compute joint angles
#     angles = {
#         'left_elbow_angle': calculate_angle(
#             (landmarks['LShoulder']['x'], landmarks['LShoulder']['y'], landmarks['LShoulder']['z']),
#             (landmarks['LElbow']['x'], landmarks['LElbow']['y'], landmarks['LElbow']['z']),
#             (landmarks['LWrist']['x'], landmarks['LWrist']['y'], landmarks['LWrist']['z'])
#         ),
#         'right_elbow_angle': calculate_angle(
#             (landmarks['RShoulder']['x'], landmarks['RShoulder']['y'], landmarks['RShoulder']['z']),
#             (landmarks['RElbow']['x'], landmarks['RElbow']['y'], landmarks['RElbow']['z']),
#             (landmarks['RWrist']['x'], landmarks['RWrist']['y'], landmarks['RWrist']['z'])
#         ),
#         'left_shoulder_angle': calculate_angle(
#             (landmarks['LElbow']['x'], landmarks['LElbow']['y'], landmarks['LElbow']['z']),
#             (landmarks['LShoulder']['x'], landmarks['LShoulder']['y'], landmarks['LShoulder']['z']),
#             (landmarks['Spine1']['x'], landmarks['Spine1']['y'], landmarks['Spine1']['z'])
#         ),
#         'right_shoulder_angle': calculate_angle(
#             (landmarks['RElbow']['x'], landmarks['RElbow']['y'], landmarks['RElbow']['z']),
#             (landmarks['RShoulder']['x'], landmarks['RShoulder']['y'], landmarks['RShoulder']['z']),
#             (landmarks['Spine1']['x'], landmarks['Spine1']['y'], landmarks['Spine1']['z'])
#         ),
#         'hip_spine_angle': calculate_angle(
#             (landmarks['LHip']['x'], landmarks['LHip']['y'], landmarks['LHip']['z']),
#             (landmarks['Spine1']['x'], landmarks['Spine1']['y'], landmarks['Spine1']['z']),
#             (landmarks['RHip']['x'], landmarks['RHip']['y'], landmarks['RHip']['z'])
#         ),
#         'left_knee_angle': calculate_angle(
#             (landmarks['LHip']['x'], landmarks['LHip']['y'], landmarks['LHip']['z']),
#             (landmarks['LKnee']['x'], landmarks['LKnee']['y'], landmarks['LKnee']['z']),
#             (landmarks['LAnkle']['x'], landmarks['LAnkle']['y'], landmarks['LAnkle']['z'])
#         ),
#         'right_knee_angle': calculate_angle(
#             (landmarks['RHip']['x'], landmarks['RHip']['y'], landmarks['RHip']['z']),
#             (landmarks['RKnee']['x'], landmarks['RKnee']['y'], landmarks['RKnee']['z']),
#             (landmarks['RAnkle']['x'], landmarks['RAnkle']['y'], landmarks['RAnkle']['z'])
#         ),
#         'spine_alignment': calculate_angle(
#             (landmarks['Spine1']['x'], landmarks['Spine1']['y'], landmarks['Spine1']['z']),
#             (landmarks['LHip']['x'], landmarks['LHip']['y'], landmarks['LHip']['z']),
#             (landmarks['RHip']['x'], landmarks['RHip']['y'], landmarks['RHip']['z'])
#         ),
#         'head_neck_spine_angle': calculate_angle(
#             ((landmarks['LShoulder']['x'] + landmarks['RShoulder']['x']) / 2,
#              (landmarks['LShoulder']['y'] + landmarks['RShoulder']['y']) / 2,
#              (landmarks['LShoulder']['z'] + landmarks['RShoulder']['z']) / 2),
#             (landmarks['Spine1']['x'], landmarks['Spine1']['y'], landmarks['Spine1']['z']),
#             (landmarks['Neck']['x'], landmarks['Neck']['y'], landmarks['Neck']['z'])
#         )
#     }

#     # Compute normalized distances
#     distances = {
#         'elbow_wrist_left': calculate_distance(
#             (landmarks['LElbow']['x'], landmarks['LElbow']['y'], landmarks['LElbow']['z']),
#             (landmarks['LWrist']['x'], landmarks['LWrist']['y'], landmarks['LWrist']['z'])
#         ) / shoulder_width,
#         'elbow_wrist_right': calculate_distance(
#             (landmarks['RElbow']['x'], landmarks['RElbow']['y'], landmarks['RElbow']['z']),
#             (landmarks['RWrist']['x'], landmarks['RWrist']['y'], landmarks['RWrist']['z'])
#         ) / shoulder_width,
#         'shoulder_elbow_left': calculate_distance(
#             (landmarks['LShoulder']['x'], landmarks['LShoulder']['y'], landmarks['LShoulder']['z']),
#             (landmarks['LElbow']['x'], landmarks['LElbow']['y'], landmarks['LElbow']['z'])
#         ) / shoulder_width,
#         'shoulder_elbow_right': calculate_distance(
#             (landmarks['RShoulder']['x'], landmarks['RShoulder']['y'], landmarks['RShoulder']['z']),
#             (landmarks['RElbow']['x'], landmarks['RElbow']['y'], landmarks['RElbow']['z'])
#         ) / shoulder_width,
#         'shoulder_hip_left': calculate_distance(
#             (landmarks['LShoulder']['x'], landmarks['LShoulder']['y'], landmarks['LShoulder']['z']),
#             (landmarks['LHip']['x'], landmarks['LHip']['y'], landmarks['LHip']['z'])
#         ) / shoulder_width,
#         'shoulder_hip_right': calculate_distance(
#             (landmarks['RShoulder']['x'], landmarks['RShoulder']['y'], landmarks['RShoulder']['z']),
#             (landmarks['RHip']['x'], landmarks['RHip']['y'], landmarks['RHip']['z'])
#         ) / shoulder_width,
#         'hip_knee_left': calculate_distance(
#             (landmarks['LHip']['x'], landmarks['LHip']['y'], landmarks['LHip']['z']),
#             (landmarks['LKnee']['x'], landmarks['LKnee']['y'], landmarks['LKnee']['z'])
#         ) / shoulder_width,
#         'hip_knee_right': calculate_distance(
#             (landmarks['RHip']['x'], landmarks['RHip']['y'], landmarks['RHip']['z']),
#             (landmarks['RKnee']['x'], landmarks['RKnee']['y'], landmarks['RKnee']['z'])
#         ) / shoulder_width,
#         'knee_ankle_left': calculate_distance(
#             (landmarks['LKnee']['x'], landmarks['LKnee']['y'], landmarks['LKnee']['z']),
#             (landmarks['LAnkle']['x'], landmarks['LAnkle']['y'], landmarks['LAnkle']['z'])
#         ) / shoulder_width,
#         'knee_ankle_right': calculate_distance(
#             (landmarks['RKnee']['x'], landmarks['RKnee']['y'], landmarks['RKnee']['z']),
#             (landmarks['RAnkle']['x'], landmarks['RAnkle']['y'], landmarks['RAnkle']['z'])
#         ) / shoulder_width,
#         'spine_neck': calculate_distance(
#             (landmarks['Spine1']['x'], landmarks['Spine1']['y'], landmarks['Spine1']['z']),
#             (landmarks['Neck']['x'], landmarks['Neck']['y'], landmarks['Neck']['z'])
#         ) / shoulder_width
#     }

#     for key, value in {**angles, **distances}.items():
#         df.at[index, key] = value

# df.to_csv('mmpose-pushup-keypoints/mmpose-pushup-keypoints-updated.csv', index=False)
# print("Updated angles and distances saved.")


