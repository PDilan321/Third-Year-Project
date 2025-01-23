import os
import time
import torch
from mmpose.apis import MMPoseInferencer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


video_numbers = list(range(1, 27))
# video_numbers = list(range(3, 10)) + list(range(26, 51))


# Set the directory manually here
video_dir = 'data/push-up-multi-reps/'  # Change 'good' to 'bad' when processing bad videos
output_path = 'multirepoutput'

# Initialize the MMPose inferencer
inferencer = MMPoseInferencer(pose3d='video-pose-lift_tcn-243frm-supv_8xb128-160e_h36m')

# Process videos in the specified list
for video_file in os.listdir(video_dir):
    if video_file.endswith('.mp4'):
        video_id = os.path.splitext(video_file)[0]
        
        # Check if the video number is in the target list
        try:
            video_num = int(video_id.split('_')[-1])  # Assuming filename ends with the number
            if video_num in video_numbers:
                print(f"Processing video: {video_id}")
                full_video_path = os.path.join(video_dir, video_file)

                # Run inference
                start_time = time.time()
                result_generator = inferencer(  
                    full_video_path, out_dir=output_path, return_vis=False, radius=8, thickness=6
                )
                for _ in result_generator:
                    pass
                end_time = time.time()
                print(f"Processed {video_file} in {end_time - start_time:.2f} seconds")
            else:
                print(f"Skipping video: {video_id} (Not in target range)")
        except ValueError:
            print(f"Skipping video: {video_file} (Invalid format)")