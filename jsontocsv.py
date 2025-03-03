# import json
# import csv
# import os

# keypoints_list = [
#     'Pelvis', 'RHip', 'RKnee', 'RAnkle', 
#     'LHip', 'LKnee', 'LAnkle', 'Spine1', 
#     'Neck', 'Head', 'Site', 'LShoulder', 
#     'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist'
# ]

# current_json = 'push-up_49'
# json_path = f'output/predictions/{current_json}.json'
# csv_file_path = 'push-ups-keypoints.csv'
# label = 1  # 1 correct form, 0 incorrect form

# # Load the JSON file
# with open(json_path, 'r') as json_file:
#     data = json.load(json_file)

# # Check if the CSV file already exists
# file_exists = os.path.exists(csv_file_path)

# # Open the CSV file in append mode
# with open(csv_file_path, 'a', newline='') as csv_file:
#     csv_writer = csv.writer(csv_file)
    
#     # Write the header only if the file does not exist
#     if not file_exists:
#         header = ['video_id', 'frame_id']
#         for keypoint in keypoints_list:
#             header.extend([f'{keypoint}_x', f'{keypoint}_y', f'{keypoint}_z'])  # Proper names for keypoints
#         header.append('label')  # Add the label column
#         csv_writer.writerow(header)
    
#     # Write data
#     for frame in data:
#         frame_id = frame['frame_id']
#         row = [current_json, frame_id]
        
#         # Check if there are any instances
#         if frame['instances']:
#             # Only select the first instance and skip the rest
#             instance = frame['instances'][0]  # Select only the first instance
            
#             # Add keypoint coordinates for the first instance
#             for keypoint, coords in zip(keypoints_list, instance['keypoints']):
#                 row.extend(coords)  # Append x, y, z for each keypoint
        
#         # Append the label to the row
#         row.append(label)
        
#         # Write the row to the CSV
#         csv_writer.writerow(row)


import os
import json
import csv

# Directory containing JSON prediction files
predictions_dir = 'mmpose-pushup-keypoints-2d'
# predictions_dir = 'multipleoutput/predictions'
csv_file_path = 'mmpose-pushup-keypoints-2d.csv'

# Keypoints list
keypoints_list = [
    'Pelvis', 'RHip', 'RKnee', 'RAnkle', 
    'LHip', 'LKnee', 'LAnkle', 'Spine1', 
    'Neck', 'Head', 'Site', 'LShoulder', 
    'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist'
]

# Check if the CSV file already exists
file_exists = os.path.exists(csv_file_path)

# Open the CSV file in append mode
with open(csv_file_path, 'a', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Write the header only if the file does not exist
    if not file_exists:
        header = ['video_id', 'frame_id']
        for keypoint in keypoints_list:
            header.extend([f'{keypoint}_x', f'{keypoint}_y'])  # Proper names for keypoints
        csv_writer.writerow(header)
    
    # Iterate over all files in the predictions directory
    for json_file_name in os.listdir(predictions_dir):
        if not json_file_name.endswith('.json'):
            continue  # Skip non-JSON files
        
        # Load the JSON file
        json_path = os.path.join(predictions_dir, json_file_name)
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)
        
        # Write data to the CSV
        for frame in data:
            frame_id = frame['frame_id']
            row = [json_file_name.replace('.json', ''), frame_id]
            
            # Check if there are any instances
            if frame['instances']:
                # Only select the first instance and skip the rest
                instance = frame['instances'][0]
                
                # Add keypoint coordinates for the first instance
                for keypoint, coords in zip(keypoints_list, instance['keypoints']):
                    row.extend(coords)  # Append x, y, z for each keypoint
            
            # Write the row to the CSV
            csv_writer.writerow(row)
