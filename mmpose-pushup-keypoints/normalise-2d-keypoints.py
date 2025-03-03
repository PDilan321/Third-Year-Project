import pandas as pd
import numpy as np


# # Load data
# file_path = "mmpose-pushup-keypoints/mmpose-pushup-keypoints-2d.csv"
# df = pd.read_csv(file_path)

# # Extract keypoint columns (assuming columns start at index 2, after video_id & frame_id)
# keypoints = df.iloc[:, 2:].values.reshape(df.shape[0], -1, 2)  # Reshape into (N, num_keypoints, 2)

# # Normalize keypoints using min-max scaling to [-1, 1]
# x_min, x_max = keypoints[:, :, 0].min(), keypoints[:, :, 0].max()
# y_min, y_max = keypoints[:, :, 1].min(), keypoints[:, :, 1].max()

# keypoints[:, :, 0] = 2 * (keypoints[:, :, 0] - x_min) / (x_max - x_min) - 1
# keypoints[:, :, 1] = 2 * (keypoints[:, :, 1] - y_min) / (y_max - y_min) - 1

# # Flatten keypoints back into DataFrame format
# normalized_data = keypoints.reshape(df.shape[0], -1)
# normalized_df = pd.DataFrame(normalized_data, columns=df.columns[2:])  # Keep original column names

# # Keep the original video_id and frame_id
# normalized_df.insert(0, 'frame_id', df['frame_id'])
# normalized_df.insert(0, 'video_id', df['video_id'])

# # Save normalized keypoints to a new CSV
# normalized_df.to_csv("mmpose-pushup-keypoints/mmpose-pushup-keypoints-2d-normalised.csv", index=False)
# print("Normalization complete. Saved as 'mmpose-normalized.csv'.")

import pandas as pd
import numpy as np

def calculate_angle(p1, p2, p3):
    """Calculate the angle at point p2 given three 2D points."""
    v1 = np.array(p1) - np.array(p2)  # Vector p2 → p1
    v2 = np.array(p3) - np.array(p2)  # Vector p2 → p3

    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return float('nan')  # Return NaN if division by zero occurs

    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Clip to valid range
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two 2D points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Load normalized 2D keypoint data
file_path = 'mmpose-pushup-keypoints/mmpose-pushup-keypoints-2d-normalised.csv'
df = pd.read_csv(file_path)

# Process each row (frame)
for index, row in df.iterrows():
    landmarks = {}
    landmark_names = ['Pelvis', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'Spine1', 'Neck', 'Head', 'Site', 
                      'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']

    for name in landmark_names:
        landmarks[name] = (row[f'{name}_x'], row[f'{name}_y'])

    # Shoulder width for normalization
    shoulder_width = calculate_distance(landmarks['LShoulder'], landmarks['RShoulder'])
    
    # Compute joint angles
    angles = {
        'left_elbow_angle': calculate_angle(landmarks['LShoulder'], landmarks['LElbow'], landmarks['LWrist']),
        'right_elbow_angle': calculate_angle(landmarks['RShoulder'], landmarks['RElbow'], landmarks['RWrist']),
        'left_shoulder_angle': calculate_angle(landmarks['LElbow'], landmarks['LShoulder'], landmarks['Spine1']),
        'right_shoulder_angle': calculate_angle(landmarks['RElbow'], landmarks['RShoulder'], landmarks['Spine1']),
        'hip_spine_angle': calculate_angle(landmarks['LHip'], landmarks['Spine1'], landmarks['RHip']),
        'left_knee_angle': calculate_angle(landmarks['LHip'], landmarks['LKnee'], landmarks['LAnkle']),
        'right_knee_angle': calculate_angle(landmarks['RHip'], landmarks['RKnee'], landmarks['RAnkle']),
        'spine_alignment': calculate_angle(landmarks['Spine1'], landmarks['LHip'], landmarks['RHip']),
        'head_neck_spine_angle': calculate_angle(
            ((landmarks['LShoulder'][0] + landmarks['RShoulder'][0]) / 2, 
             (landmarks['LShoulder'][1] + landmarks['RShoulder'][1]) / 2),
            landmarks['Spine1'],
            landmarks['Neck']
        )
    }

    # Compute normalized distances
    distances = {
        'elbow_wrist_left': calculate_distance(landmarks['LElbow'], landmarks['LWrist']) / shoulder_width,
        'elbow_wrist_right': calculate_distance(landmarks['RElbow'], landmarks['RWrist']) / shoulder_width,
        'shoulder_elbow_left': calculate_distance(landmarks['LShoulder'], landmarks['LElbow']) / shoulder_width,
        'shoulder_elbow_right': calculate_distance(landmarks['RShoulder'], landmarks['RElbow']) / shoulder_width,
        'shoulder_hip_left': calculate_distance(landmarks['LShoulder'], landmarks['LHip']) / shoulder_width,
        'shoulder_hip_right': calculate_distance(landmarks['RShoulder'], landmarks['RHip']) / shoulder_width,
        'hip_knee_left': calculate_distance(landmarks['LHip'], landmarks['LKnee']) / shoulder_width,
        'hip_knee_right': calculate_distance(landmarks['RHip'], landmarks['RKnee']) / shoulder_width,
        'knee_ankle_left': calculate_distance(landmarks['LKnee'], landmarks['LAnkle']) / shoulder_width,
        'knee_ankle_right': calculate_distance(landmarks['RKnee'], landmarks['RAnkle']) / shoulder_width,
        'spine_neck': calculate_distance(landmarks['Spine1'], landmarks['Neck']) / shoulder_width
    }

    # Update dataframe
    for key, value in {**angles, **distances}.items():
        df.at[index, key] = value

# Save updated CSV
output_file = 'mmpose-pushup-keypoints/mmpose-pushup-keypoints-2d-updated.csv'
df.to_csv(output_file, index=False)
print(f"Updated angles and distances saved to '{output_file}'.")
