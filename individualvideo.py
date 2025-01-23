import os
import time
import torch
import cv2
from mmpose.apis import MMPoseInferencer

# def downsample_video(input_path, output_path, width=640, height=480):
#     cap = cv2.VideoCapture(input_path)
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))
        
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         resized_frame = cv2.resize(frame, (width, height))
#         out.write(resized_frame)
    
#     cap.release()
#     out.release()

# downsample_video('data/pull-ups/pull_up_7.mp4', 'data/pull-ups/pull_up_7_downsampled.mp4')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

current_video = 'push-up_49'
vid_path = f'data/push-ups-actual/{current_video}.mp4'  

if not os.path.exists(vid_path):
    raise FileNotFoundError(f"The file {vid_path} does not exist.")

# inferencer = MMPoseInferencer(pose3d='motionbert_dstformer-ft-243frm_8xb32-120e_h36m')
inferencer = MMPoseInferencer(pose3d='video-pose-lift_tcn-243frm-supv_8xb128-160e_h36m')
# inferencer = MMPoseInferencer(pose3d='human3d')
# inferencer = MMPoseInferencer(pose2d='human', device=device)


result_generator = inferencer(vid_path, out_dir='output', return_vis=True, radius=8, thickness=6)

print(f"Inputs: {vid_path}")

start_time = time.time()
for _ in result_generator:
    pass
end_time = time.time()
print(f"Time to process result generator: {end_time - start_time} seconds")

print("Processing complete.")