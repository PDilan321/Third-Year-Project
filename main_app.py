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



# Load the saved model  
model = load_model('pushup_model_latestBiLSTM.h5')

# Path to your new video
new_video_path = "push-up_3.mp4"
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

reps = ph.count_reps_robust(predicted_states)
print(f"Predicted repetitions in the new video: {reps}")


# Call the function to create the video
ph.create_annotated_video(new_video_path, predicted_states.tolist(), reps)