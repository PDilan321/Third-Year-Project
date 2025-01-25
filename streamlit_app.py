import streamlit as st
import tempfile
import time
import cv2
import pushup_helper_methods as ph



def main():
    st.set_page_config(page_title='Push-Up Repetition Counter', layout='centered')
    
    st.title('RepCheck - Push-Up Edition')

    options = st.sidebar.selectbox('Select Option', ('Video', 'WebCam'))

    # Define Operations if Video Option is selected
    if options == 'Video':
        st.markdown('-------')

        st.write('## Upload your video')
        st.write("")

        st.sidebar.markdown('-------')

        st.sidebar.markdown('-------')

        # User can upload a video:
        uploaded_video = st.file_uploader("Include useful message !", type=["mp4", "mov"])
        if uploaded_video:
            st.video(uploaded_video)
            
            # Save the uploaded video temporarily
            with open("temp_video.mp4", "wb") as f:
                f.write(uploaded_video.read())

            # Display the "Start Counting Reps" button
            if st.button("Start Counting Reps"):
                st.write("Processing your video... This may take a few moments.")

                # Call the model to process the video
                model_path = "pushup_model_latestBiLSTM.h5"
                predicted_states, reps = ph.process_video_with_model("temp_video.mp4", model_path)

                # Display results
                st.markdown("### Results")
                st.write(f"Predicted States: {predicted_states}")
                st.write(f"Predicted Reps: {reps}")

        st.markdown('-------')

        # Visualize Video after analysis (analysis based on the selected exercise)
        # st.markdown(' ## Output Video')

    else:
        st.markdown('-------')

        st.sidebar.markdown('-------')

        # New button for direct activation
        st.write(' Click button to start training')
        start_button = st.button('Start Exercise')

        if start_button:
            time.sleep(2)  # Add a delay of 2 seconds
            ready = True
            while ready:
                cap = cv2.VideoCapture(0)
                break


if __name__ == '__main__':
    # def load_css():
    #     with open("static/styles.css", "r") as f:
    #         css = f"<style>{f.read()}</style>"
    #         st.markdown(css, unsafe_allow_html=True)
    main()


