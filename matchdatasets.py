import pandas as pd

# File paths
mediapipe_csv = "3D/mmpose-3d/aligned-mmpose-dataset.csv"
mmpose_csv = "3D/mmpose-3d/pseudo-global-mmpose-dataset.csv"
output_csv = "3D/mmpose-3d/pseudo-global-mmpose-dataset-matched.csv"

# Load CSVs as DataFrames
mediapipe_df = pd.read_csv(mediapipe_csv)
mmpose_df = pd.read_csv(mmpose_csv)

# Select relevant columns from MediaPipe
selected_columns = ['video_id', 'frame_id', 'state']
mediapipe_filtered = mediapipe_df[selected_columns]

# Merge MMPose data with MediaPipe filtered rows
aligned_df = mediapipe_filtered.merge(mmpose_df, on=['video_id', 'frame_id'], how='left')

# Save the aligned dataset
aligned_df.to_csv(output_csv, index=False)

print(f"Aligned dataset saved to {output_csv}")
