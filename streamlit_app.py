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

def create_annotated_video_dynamic(video_path, predicted_states, reps_per_frame, output_path="annotated_pushup.mp4"):
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
        rep_text = f"Reps: {reps_per_frame[frame_idx]}"  # Use frame-specific rep count
        
        # Draw the text on the frame
        cv2.putText(frame, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, rep_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255), 2, cv2.LINE_AA)
        
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    return output_path  # Return the output file path




def main():
    st.set_page_config(page_title='Push-Up Repetition Counter', page_icon="🔍", layout='wide')

    
    st.title('RepCheck - Push-Up Edition')

    options = st.sidebar.selectbox('Select Option', ('Video', 'WebCam'))

    model_path = "pushup_model_latestBiLSTM.h5"
    model = load_model(model_path)


    if options == "Video":
        st.markdown("## Upload Your Video")
        uploaded_video = st.file_uploader("Upload a video of your push-ups (.mp4 or .mov)", type=["mp4", "mov"])

        if uploaded_video:
            # Display the uploaded video
            st.video(uploaded_video)

            # Save the video to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
                temp_video_file.write(uploaded_video.read())
                input_video_path = temp_video_file.name

            # Start processing button
            if st.button("Start Counting Reps"):
                st.write("Processing your video... Please wait.")

                # Predict states and reps per frame
                predicted_states, cumulative_reps_per_frame = ph.process_video_with_model(input_video_path, model_path)
                
                st.write("Predicted states: " + str(predicted_states))
                st.write("Reps per frame: " + str(cumulative_reps_per_frame))


                # Annotate video
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output_file:
                    output_video_path = create_annotated_video_dynamic(
                        input_video_path, 
                        predicted_states, 
                        cumulative_reps_per_frame, 
                        temp_output_file.name
                    )

                st.markdown("### Results")

                # Verify that the output video path exists
                if os.path.exists(output_video_path):
                    st.write(f"Annotated video saved at: {output_video_path}")

                    # Option 1: Serve the video using its file path
                    st.video(output_video_path)

                    # Option 2: Serve the video using binary content (if needed)
                    with open(output_video_path, "rb") as f:
                        video_bytes = f.read()
                        st.video(video_bytes)

                        # Add download button
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

        # Visualize Video after analysis (analysis based on the selected exercise)
        # st.markdown(' ## Output Video')

    else:
        st.markdown('-------')

        st.sidebar.markdown('-------')

        start_button = st.button("Start Webcam")

        scaler = StandardScaler()
        sample_data = np.random.rand(1, angles.shape[1] + distances.shape[1] + landmarks_3d.shape[1])  # Example data
        scaler.fit(sample_data)

        if start_button:
            st.write("Initializing webcam...")

            # Placeholder to display the video
            video_placeholder = st.empty()

            # Initialize webcam
            cap = cv2.VideoCapture(0)

            # Feature extraction setup
            scaler = StandardScaler()
            reps = 0
            processed_states = []

            # Loop to process frames
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.write("Failed to grab frame.")
                    break

                # Resize the frame for better processing
                frame = cv2.resize(frame, (640, 480))

                # Extract pose features from the frame
                angles, distances, landmarks_3d = ph.extract_pose_features_single_frame(frame)
                combined = np.hstack([angles, distances, landmarks_3d])
                # combined_scaled = scaler.fit_transform(combined.reshape(1, -1))
                combined_scaled = scaler.transform(combined.reshape(1, -1))


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





