import streamlit as st
import tempfile
import time
import cv2
import pushup_helper_methods as ph

from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pushup_helper_methods as ph

from sklearn.preprocessing import StandardScaler
import os

import tempfile
import os
import time
import cv2
import streamlit as st

from pdb import set_trace as bp


# def visualize_angle(img, angle, landmark):
#     frame_width = img.shape[1]  # Get actual width
#     frame_height = img.shape[0]  # Get actual height
#     cv2.putText(img, str(int(angle)),
#                 tuple(np.multiply(landmark, [frame_width, frame_height]).astype(int)),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

def project_3d_to_2d(x, y, z, frame_width, frame_height):
    return (int(x * frame_width), int(y * frame_height))  # Assuming normalized coordinates

def visualize_joint(frame, angle, x, y, z, color=(0, 255, 0)):
    frame_width, frame_height = frame.shape[1], frame.shape[0]
    x_2d, y_2d = project_3d_to_2d(x, y, z, frame_width, frame_height)
    
    cv2.circle(frame, (x_2d, y_2d), 5, color, -1)  

    cv2.putText(frame, f"{str(int(angle))}", (x_2d + 5, y_2d - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



def create_annotated_video_dynamic(video_path, output_path, predicted_states, reps_per_frame, angles, landmarks_3d):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 encoding, more compatible
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= len(predicted_states):
            break
        
        state_text = f"State: {predicted_states[frame_idx]}"
        rep_text = f"Reps: {reps_per_frame[frame_idx]}"
         
        cv2.putText(frame, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, rep_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255), 2, cv2.LINE_AA)
        

        right_shoulder_angle = angles[frame_idx][3]
        left_shoulder_angle = angles[frame_idx][2]
        
        right_elbow_angle = angles[frame_idx][1]
        left_elbow_angle = angles[frame_idx][0]

        right_shoulder_x = landmarks_3d[frame_idx][36]
        right_shoulder_y = landmarks_3d[frame_idx][37]
        right_shoulder_z = landmarks_3d[frame_idx][38]

        left_shoulder_x = landmarks_3d[frame_idx][33]
        left_shoulder_y = landmarks_3d[frame_idx][34]
        left_shoulder_z = landmarks_3d[frame_idx][35]

        left_elbow_x = landmarks_3d[frame_idx][39]
        left_elbow_y = landmarks_3d[frame_idx][40]
        left_elbow_z = landmarks_3d[frame_idx][41]

        right_elbow_x = landmarks_3d[frame_idx][42]
        right_elbow_y = landmarks_3d[frame_idx][43]
        right_elbow_z = landmarks_3d[frame_idx][44]

        visualize_joint(frame, right_shoulder_angle, right_shoulder_x, right_shoulder_y, right_shoulder_z, color=(255, 0, 0))  
        visualize_joint(frame, right_elbow_angle, right_elbow_x, right_elbow_y, right_elbow_z, color=(255, 0, 0))  
        visualize_joint(frame, left_shoulder_angle, left_shoulder_x, left_shoulder_y, left_shoulder_z, color=(255, 0, 0))  
        visualize_joint(frame, left_elbow_angle, left_elbow_x, left_elbow_y, left_elbow_z, color=(0, 0, 255))  

        
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    return output_path  


def main():
    st.set_page_config(page_title='Push-Up Repetition Counter', page_icon="🔍", layout='wide')

    st.title('RepCheck - Push-Up Edition')

    options = st.sidebar.selectbox('Select Option', ('Video', 'WebCam'))

    model_path = "3D/Mediapipe/pushup_model_latestBiLSTM.h5"
    # model_path_2d = "2D/rtm-pose/rtm-pose-BiLSTM.h5"
    model = load_model(model_path)

    if options == "Video":
        st.markdown("## Upload Your Video")
        uploaded_video = st.file_uploader("Upload a video of your push-ups (.mp4 or .mov)", type=["mp4", "mov"])

        if uploaded_video:
            st.video(uploaded_video)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
                temp_video_file.write(uploaded_video.read())
                input_video_path = temp_video_file.name

            if st.button("Start Counting Reps"):
                st.write("Processing your video... Please wait.")

                predicted_states, cumulative_reps_per_frame, angles, landmarks_3d = ph.process_video_with_model(input_video_path, model_path)
                st.write("Predicted states: " + str(predicted_states))
                st.write("Reps per frame: " + str(cumulative_reps_per_frame))

                output_video_path = create_annotated_video_dynamic(
                    input_video_path, "annotated_pushup.mp4.mp4", predicted_states, cumulative_reps_per_frame, angles, landmarks_3d
                )

                st.markdown("### Results")

                if os.path.exists(output_video_path):
                    # st.video(output_video_path)

                    with open(output_video_path, "rb") as video_file:
                        video_bytes = video_file.read()

                    st.video(video_bytes)

                    st.write(f"Output video exists: {os.path.exists(output_video_path)}")
                    st.write(f"File size: {os.path.getsize(output_video_path)} bytes")

                    st.download_button(
                        label="Download Annotated Video",
                        data=video_bytes,
                        file_name="annotated_pushup.mp4",
                        mime="video/mp4",
                    )
                else:
                    st.write("Error: Annotated video could not be created.")


        else:
            st.info("Please upload a video to begin.")

    else:
        st.markdown('-------')

        st.sidebar.markdown('-------')

        start_button = st.button("Start Webcam")

        if start_button:
            st.write("Initializing webcam...")

            video_placeholder = st.empty()

            cap = cv2.VideoCapture(0)

            scaler = StandardScaler()
            reps = 0
            processed_states = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.write("Failed to grab frame.")
                    break

                frame = cv2.resize(frame, (640, 480))

                angles, distances, landmarks_3d = ph.extract_pose_features_single_frame(frame)
                combined = np.hstack([angles, distances, landmarks_3d])
                combined_scaled = scaler.fit_transform(combined.reshape(1, -1))



                # Reshape for model input and make predictions
                input_frame = combined_scaled.reshape(-1, 1, combined_scaled.shape[1])
                probabilities = model.predict(input_frame)
                predicted_state = np.argmax(probabilities, axis=1)[0]

                # Append state for rep counting
                processed_states.append(predicted_state)

                # Count reps every 20 frames (to reduce lag)
                if len(processed_states) > 20:
                    reps = ph.count_reps_robust(
                        states=processed_states[-20:],  # Process the last few states
                        ideal_sequence=[2, 1, 0, 1, 2],  # Expected push-up sequence
                        window_size=5
                    )

                # Overlay the state and rep count on the frame
                # cv2.putText(frame, f"State: {predicted_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Reps: {reps}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                # Convert the frame color for Streamlit display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Display the frame in Streamlit
                video_placeholder.image(frame_rgb, channels="RGB")

            # Release the webcam when done
            cap.release()

        st.markdown("## Click 'Stop' to end the session.")


if __name__ == '__main__':
    # def load_css():
    #     with open("static/styles.css", "r") as f:
    #         css = f"<style>{f.read()}</style>"
    #         st.markdown(css, unsafe_allow_html=True)
    main()





