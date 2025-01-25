from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix, classification_report


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pushup_helper_methods as ph

 
def process_video_with_model(video_path, model_path, ideal_sequence=[2, 1, 0, 1, 2], window_size=5):
    import os
    from tensorflow.keras.models import load_model
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    import pushup_helper_methods as ph  # Your custom helper module

     model = load_model(model_path)

    # Extract pose features from the video
    angles, distances, landmarks_3d = ph.extract_pose_features(video_path)

    # Scale features
    scaler = StandardScaler()
    combined_features = scaler.fit_transform(
        np.hstack([angles, distances, landmarks_3d])
    )

    # Reshape for model input
    video_input = combined_features.reshape(
        -1, 1, len(ph.angle_columns + ph.distance_columns + ph.landmark_columns_3d)
    )

    # Predict states
    probabilities = model.predict(video_input)
    predicted_states = np.argmax(probabilities, axis=1)

    # Count repetitions
    repetitions = ph.count_reps_robust(
        states=predicted_states,
        ideal_sequence=ideal_sequence,
        window_size=window_size,
    )

    return predicted_states, repetitions

 


# Load the saved model  
model = load_model('pushup_model_latestBiLSTM.h5')

# Path to your new video
new_video_path = ".mp4"
# push-up_3.mp4 works

# Extract features from the video
angles, distances, landmarks_3d = ph.extract_pose_features(new_video_path)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
combined = scaler.fit_transform(np.hstack([angles, distances, landmarks_3d]))

# Reshape for model input
new_video_input = combined.reshape(-1, 1, len(ph.angle_columns + ph.distance_columns + ph.landmark_columns_3d)) #len(angle_columns + distance_columns)

# Predict states
probabilitiess = model.predict(new_video_input)
predicted_states = np.argmax(probabilitiess, axis=1)

reps = ph.count_reps_robust(
        states=predicted_states,
        ideal_sequence=[2, 1, 0, 1, 2],  # Expected sequence for a full rep
        window_size=5,  # Adjust as needed based on noise level
    )
print(f"Predicted repetitions in the new video: {reps}")


# Call the function to create the video
ph.create_annotated_video(new_video_path, predicted_states.tolist(), reps)