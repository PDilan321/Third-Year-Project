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


tf.config.run_functions_eagerly(True)

import cv2
import mediapipe as mp

def calculate_angle(p1, p2, p3):
    """Calculate angle at point p2 given 3 points."""
    v1 = np.array(p1) - np.array(p2)  # Vector from p2 to p1
    v2 = np.array(p3) - np.array(p2)  # Vector from p2 to p3

    # Check if either of the vectors is zero-length
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return float('nan')  # Return NaN if division by zero is possible

    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))  # Cosine of the angle

    # Ensure the cosine value is within the valid range for arccos
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)  # Compute angle in radians
    return np.degrees(angle)  # Convert angle from radians to degrees

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))


def extract_landmarks(landmarks):
    """Extract all 3D landmarks into a flat array."""
    landmark_array = []
    for i in range(33):  # Total 33 landmarks
        landmark_array.extend([landmarks[i].x, landmarks[i].y, landmarks[i].z])
    return landmark_array

def extract_pose_features(video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(video_path)

    angles = []
    distances = [] 
    landmarks_3d = []


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            landmark_features = extract_landmarks(landmarks)
            landmarks_3d.append(landmark_features)

            # Extract coordinates for each landmark
            landmarks_dict = {}
            for i in range(33):  # 33 pose landmarks
                landmarks_dict[i] = {
                    'x': landmarks[i].x,
                    'y': landmarks[i].y,
                    'z': landmarks[i].z
                }
            
            # Calculate the angles
            left_elbow_angle = calculate_angle(
                (landmarks_dict[11]['x'], landmarks_dict[11]['y'], landmarks_dict[11]['z']),
                (landmarks_dict[13]['x'], landmarks_dict[13]['y'], landmarks_dict[13]['z']),
                (landmarks_dict[15]['x'], landmarks_dict[15]['y'], landmarks_dict[15]['z'])
            )

            right_elbow_angle = calculate_angle(
                (landmarks_dict[12]['x'], landmarks_dict[12]['y'], landmarks_dict[12]['z']),
                (landmarks_dict[14]['x'], landmarks_dict[14]['y'], landmarks_dict[14]['z']),
                (landmarks_dict[16]['x'], landmarks_dict[16]['y'], landmarks_dict[16]['z'])
            )

            left_shoulder_angle = calculate_angle(
                (landmarks_dict[13]['x'], landmarks_dict[13]['y'], landmarks_dict[13]['z']),
                (landmarks_dict[11]['x'], landmarks_dict[11]['y'], landmarks_dict[11]['z']),
                (landmarks_dict[0]['x'], landmarks_dict[0]['y'], landmarks_dict[0]['z'])  # Spine/torso
            )

            right_shoulder_angle = calculate_angle(
                (landmarks_dict[14]['x'], landmarks_dict[14]['y'], landmarks_dict[14]['z']),
                (landmarks_dict[12]['x'], landmarks_dict[12]['y'], landmarks_dict[12]['z']),
                (landmarks_dict[0]['x'], landmarks_dict[0]['y'], landmarks_dict[0]['z'])  # Spine/torso
            )

            hip_spine_angle = calculate_angle(
                (landmarks_dict[23]['x'], landmarks_dict[23]['y'], landmarks_dict[23]['z']),
                (landmarks_dict[0]['x'], landmarks_dict[0]['y'], landmarks_dict[0]['z']),
                (landmarks_dict[24]['x'], landmarks_dict[24]['y'], landmarks_dict[24]['z'])
            )

            left_knee_angle = calculate_angle(
                (landmarks_dict[23]['x'], landmarks_dict[23]['y'], landmarks_dict[23]['z']),
                (landmarks_dict[25]['x'], landmarks_dict[25]['y'], landmarks_dict[25]['z']),
                (landmarks_dict[27]['x'], landmarks_dict[27]['y'], landmarks_dict[27]['z'])
            )

            right_knee_angle = calculate_angle(
                (landmarks_dict[24]['x'], landmarks_dict[24]['y'], landmarks_dict[24]['z']),
                (landmarks_dict[26]['x'], landmarks_dict[26]['y'], landmarks_dict[26]['z']),
                (landmarks_dict[28]['x'], landmarks_dict[28]['y'], landmarks_dict[28]['z'])
            )

            spine_alignment = calculate_angle(
                (landmarks_dict[0]['x'], landmarks_dict[0]['y'], landmarks_dict[0]['z']),
                (landmarks_dict[23]['x'], landmarks_dict[23]['y'], landmarks_dict[23]['z']),
                (landmarks_dict[24]['x'], landmarks_dict[24]['y'], landmarks_dict[24]['z'])
            )

            neck_coords = (
                (landmarks_dict[11]['x'] + landmarks_dict[12]['x']) / 2,
                (landmarks_dict[11]['y'] + landmarks_dict[12]['y']) / 2,
                (landmarks_dict[11]['z'] + landmarks_dict[12]['z']) / 2
            )

            head_neck_spine_angle = calculate_angle(
                neck_coords,  # Midpoint of neck
                (landmarks_dict[0]['x'], landmarks_dict[0]['y'], landmarks_dict[0]['z']),  # Head (nose)
                (landmarks_dict[1]['x'], landmarks_dict[1]['y'], landmarks_dict[1]['z'])   # Spine/shoulder center
            )

            angles.append([
                left_elbow_angle, right_elbow_angle, left_shoulder_angle, right_shoulder_angle,
                hip_spine_angle, left_knee_angle, right_knee_angle, spine_alignment, head_neck_spine_angle
            ])

            # Calculate the distances (normalized by shoulder width)
            shoulder_width = calculate_distance(
                (landmarks_dict[11]['x'], landmarks_dict[11]['y'], landmarks_dict[11]['z']),
                (landmarks_dict[12]['x'], landmarks_dict[12]['y'], landmarks_dict[12]['z'])
            )
            
            distances.append([
                calculate_distance(
                    (landmarks_dict[13]['x'], landmarks_dict[13]['y'], landmarks_dict[13]['z']),
                    (landmarks_dict[15]['x'], landmarks_dict[15]['y'], landmarks_dict[15]['z'])
                ) / shoulder_width,
                
                calculate_distance(
                    (landmarks_dict[14]['x'], landmarks_dict[14]['y'], landmarks_dict[14]['z']),
                    (landmarks_dict[16]['x'], landmarks_dict[16]['y'], landmarks_dict[16]['z'])
                ) / shoulder_width,
                
                calculate_distance(
                    (landmarks_dict[11]['x'], landmarks_dict[11]['y'], landmarks_dict[11]['z']),
                    (landmarks_dict[13]['x'], landmarks_dict[13]['y'], landmarks_dict[13]['z'])
                ) / shoulder_width,
                
                calculate_distance(
                    (landmarks_dict[12]['x'], landmarks_dict[12]['y'], landmarks_dict[12]['z']),
                    (landmarks_dict[14]['x'], landmarks_dict[14]['y'], landmarks_dict[14]['z'])
                ) / shoulder_width,
                
                calculate_distance(
                    (landmarks_dict[11]['x'], landmarks_dict[11]['y'], landmarks_dict[11]['z']),
                    (landmarks_dict[23]['x'], landmarks_dict[23]['y'], landmarks_dict[23]['z'])
                ) / shoulder_width,
                
                calculate_distance(
                    (landmarks_dict[12]['x'], landmarks_dict[12]['y'], landmarks_dict[12]['z']),
                    (landmarks_dict[24]['x'], landmarks_dict[24]['y'], landmarks_dict[24]['z'])
                ) / shoulder_width,
                
                calculate_distance(
                    (landmarks_dict[0]['x'], landmarks_dict[0]['y'], landmarks_dict[0]['z']),
                    (landmarks_dict[1]['x'], landmarks_dict[1]['y'], landmarks_dict[1]['z'])
                ) / shoulder_width,
                
                calculate_distance(
                    (landmarks_dict[23]['x'], landmarks_dict[23]['y'], landmarks_dict[23]['z']),
                    (landmarks_dict[25]['x'], landmarks_dict[25]['y'], landmarks_dict[25]['z'])
                ) / shoulder_width,
                
                calculate_distance(
                    (landmarks_dict[24]['x'], landmarks_dict[24]['y'], landmarks_dict[24]['z']),
                    (landmarks_dict[26]['x'], landmarks_dict[26]['y'], landmarks_dict[26]['z'])
                ) / shoulder_width,
                
                calculate_distance(
                    (landmarks_dict[25]['x'], landmarks_dict[25]['y'], landmarks_dict[25]['z']),
                    (landmarks_dict[27]['x'], landmarks_dict[27]['y'], landmarks_dict[27]['z'])
                ) / shoulder_width,
                
                calculate_distance(
                    (landmarks_dict[26]['x'], landmarks_dict[26]['y'], landmarks_dict[26]['z']),
                    (landmarks_dict[28]['x'], landmarks_dict[28]['y'], landmarks_dict[28]['z'])
                ) / shoulder_width,
            ])
            
    cap.release()

    angles = np.nan_to_num(np.array(angles), nan=0.0)
    distances = np.nan_to_num(np.array(distances), nan=0.0)

    return angles, distances, landmarks_3d



def euclidean_distance(landmarks, joint1, joint2):
    # Extract the coordinates of the two joints
    x1, y1, z1 = landmarks[joint1.value].x, landmarks[joint1.value].y, landmarks[joint1.value].z
    x2, y2, z2 = landmarks[joint2.value].x, landmarks[joint2.value].y, landmarks[joint2.value].z

    # Compute the Euclidean distance between the two joints
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)


def count_reps_robust(states, ideal_sequence=[2, 1, 0, 1, 2]):
    """
    Count repetitions based on robust state transition tracking with compression and overlapping matching.

    Args:
    states (list): List of detected states in the video (e.g., [2, 1, 0, ...]).
    ideal_sequence (list): The state sequence that defines one rep.

    Returns:
    int: Number of repetitions.
    """
    rep_count = 0
    seq_len = len(ideal_sequence)

    print(f"Predicted States: {states}")

    # Step 1: Compress consecutive duplicates
    compressed_states = [k for idx, k in enumerate(states) if idx == 0 or k != states[idx - 1]]
    print(f"Compressed States: {compressed_states}")

    smoothed_states = moving_average_smoothing(compressed_states, window_size=5) #not currently using
    print(f"Smoothed States: {smoothed_states}")


    # Step 2: Sliding window with overlapping sequence detection
    i = 0
    while i <= len(compressed_states) - seq_len:
        # Check if the current window matches the ideal sequence
        if compressed_states[i:i + seq_len] == ideal_sequence:
            rep_count += 1
            # Move only one step forward to allow overlap detection
            i += 1
        else:
            i += 1

    return rep_count


def create_annotated_video(video_path, predicted_states, rep_count, output_path="annotated_pushup.mp4"):
    """
    Create a video with state and repetition counter overlayed on each frame.

    Args:
        video_path (str): Path to the original video.
        predicted_states (list): List of predicted states for each frame.
        rep_count (int): Total repetition count.
        output_path (str): Path to save the annotated video.
    """
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
        rep_text = f"Reps: {rep_count}"
        
        # Draw the text on the frame
        cv2.putText(frame, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, rep_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255), 2, cv2.LINE_AA)
        
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Annotated video saved to {output_path}")


def moving_average_smoothing(labels, window_size=5):
    """
    Apply moving average smoothing on frame labels.
    
    Args:
        labels (list or np.array): Array of state labels (0, 1, 2).
        window_size (int): Number of frames for the moving average window.

    Returns:
        np.array: Smoothed labels.
    """
    # Convert labels to a Pandas Series for easy handling
    labels_series = pd.Series(labels)
    
    # Apply a rolling window and take the mean, then round to nearest integer
    smoothed = labels_series.rolling(window=window_size, center=True).mean().round()
    
    # Fill the NaN values (from edges of the rolling window)
    smoothed = smoothed.fillna(method='bfill').fillna(method='ffill')
    
    return smoothed.to_numpy().astype(int)

# Load data
csv_file = 'pushup-keypoints/pushup-keypoints-test.csv'
# csv_file = 'all_videos_with_angles_and_distances_new.csv'
df = pd.read_csv(csv_file)

state_counts = df['state'].value_counts()
print(state_counts)

# Extract the camera side (front, left, right) from video_id
df['camera_side'] = df['video_id'].str.extract(r'(front_facing|left_facing|right_facing)')

# Group by camera_side and state, then count occurrences
state_counts_per_camera = df.groupby(['camera_side', 'state']).size().unstack(fill_value=0)

print(state_counts_per_camera)

angle_columns = [
    'left_elbow_angle', 'right_elbow_angle', 'left_shoulder_angle', 
    'right_shoulder_angle', 'hip_spine_angle', 'left_knee_angle', 
    'right_knee_angle', 'spine_alignment', 'head_neck_spine_angle'
]

distance_columns = [
    'elbow_wrist_left', 'elbow_wrist_right',
    'shoulder_elbow_left', 'shoulder_elbow_right',
    'shoulder_hip_left', 'shoulder_hip_right',
    'spine_neck', 'hip_knee_left', 'hip_knee_right',
    'knee_ankle_left', 'knee_ankle_right'
]

landmark_columns_3d = [
        f'landmark_{i}_{axis}' for i in range(33) for axis in ['x', 'y', 'z']
]


# Update X with normalized distances and other features
X = df[angle_columns + distance_columns + landmark_columns_3d].values
y = df['state'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape for BiLSTM (samples, timesteps, features)
X_train = X_train.reshape(-1, 1, X_train.shape[1])
X_test = X_test.reshape(-1, 1, X_test.shape[1])



# Flatten the data from 3D to 2D
X_train_flat = X_train.reshape(X_train.shape[0], -1)

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_embedded = tsne.fit_transform(X_train_flat)

# Plotting the results
plt.figure(figsize=(10, 7))
for state in range(3):  # Assuming 3 classes (up, mid, down)
    plt.scatter(X_embedded[y_train == state, 0], X_embedded[y_train == state, 1], label=f'State {state}')
plt.legend()
plt.title("t-SNE Visualization of Pushup States")
plt.show()



# Compute class weights for imbalanced data
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))
print(f"Class weights: {class_weights}")



model = Sequential([
    Bidirectional(LSTM(units=128, return_sequences=False), input_shape=(1, len(angle_columns + distance_columns + landmark_columns_3d)),), 
    Dropout(rate=0.1),
    Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(3, activation='softmax')
])

# model = Sequential([
#     LSTM(128, return_sequences=True, input_shape=(1, len(angle_columns + distance_columns + landmark_columns_3d))),
#     Dropout(rate=0.1),
#     LSTM(64, return_sequences=False),
#     Dense(32, activation='relu', kernel_regularizer=l2(0.001)),  # Reduced regularization
#     Dense(3, activation='softmax')
# ])



# optimizer = tf.keras.optimizers.Nadam(learning_rate=0.0001) # 0.84

# optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001) # 0.81
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9) # 0.8
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001) # 0.83
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001) # 0.88

# Compile the model
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train, 
    validation_split=0.2, 
    epochs=100,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stopping]
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Predict states
y_pred_probs = model.predict(X_test)  # Softmax probabilities
y_pred = np.argmax(y_pred_probs, axis=1)  # Convert to discrete states

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))


# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# # Save the trained model
# model.save('pushup_model_biLSTM.h5')  # Saves the model to an H5 file


# from tensorflow.keras.models import load_model

# # Load the saved model  
# model = load_model('pushup_model_biLSTM.h5')

# # Path to your new video
# new_video_path = "push-up_1.mp4"
# # push-up_3.mp4 works

# # Extract features from the video
# angles, distances, landmarks_3d = extract_pose_features(new_video_path)
# combined = np.hstack([angles, distances])

# # Reshape for model input
# new_video_input = np.hstack([angles, distances]).reshape(-1, 1, len(angle_columns + distance_columns)) #len(angle_columns + distance_columns)

# # Predict states
# probabilitiess = model.predict(new_video_input)
# predicted_states = np.argmax(probabilitiess, axis=1)

# # Count repetitions
# reps = count_reps_robust(predicted_states.tolist())
# print(f"Predicted repetitions in the new video: {reps}")

# # Call the function to create the video
# create_annotated_video(new_video_path, predicted_states.tolist(), reps)

# # Prepare the input data for a specific video
# example_video = df[df['video_id'] == "front_facing_9"]  # Filter by video_id
# if example_video.empty:
#     raise ValueError("The example_video dataframe is empty. Check your video_id filtering.")

# input_data = example_video[angle_columns+distance_columns].values.reshape(-1, 1, len(angle_columns+distance_columns))  # Reshape for LSTM input
# print(f"Input data shape: {input_data.shape}")

# # Predict states for this specific video
# probabilities = model.predict(input_data)
# print(probabilities[:10])  # Print the first 10 frames' probabilities

# # Convert probabilities to discrete states
# y_pred = np.argmax(probabilities, axis=1)

# repss = count_reps_robust(y_pred)


# plt.figure(figsize=(12, 6))
# plt.plot(y_pred, label="DF prediction")
# plt.plot(example_video["state"].values, label="Real States")
# plt.legend()
# plt.show()





# # Prepare the input data for a specific video
# example_video = df[df['video_id'] == "video_20"]  # Filter by video_id
# if example_video.empty:
#     raise ValueError("The example_video dataframe is empty. Check your video_id filtering.")

# input_data = example_video[angle_columns+distance_columns].values.reshape(-1, 1, len(angle_columns+distance_columns))  # Reshape for LSTM input
# print(f"Input data shape: {input_data.shape}")

# # Predict states for this specific video
# probabilities = model.predict(input_data)
# print(probabilities[:10])  # Print the first 10 frames' probabilities

# # Convert probabilities to discrete states
# y_pred = np.argmax(probabilities, axis=1)

# from sklearn.metrics import confusion_matrix, classification_report
# print(confusion_matrix(example_video['state'].values, y_pred))
# print(classification_report(example_video['state'].values, y_pred))

# # Visualize predicted states
# plt.plot(y_pred, label='Predicted States')  # Predicted states
# plt.plot(example_video['state'].values, label='True States')  # True states
# plt.legend()
# plt.show()

# reps = count_reps_fuzzy(y_pred)
# print(f"Number of reps: {reps}"
# 



# # # Prepare the input data for a specific video
# example_video = df[df['video_id'] == "right_facing_11"]  # Filter by video_id
# if example_video.empty:
#     raise ValueError("The example_video dataframe is empty. Check your video_id filtering.")

# input_dataa = example_video[angle_columns+distance_columns].values
# input_data = input_dataa.reshape(-1, 1, len(angle_columns + distance_columns))  # Reshape for LSTM input
# print(f"Input data shape: {input_data.shape}")

# # Predict states for this specific video
# probabilities = model.predict(input_data)
# y_pred = np.argmax(probabilities, axis=1) 
    
# print(f"Predicted States for Video 21: {y_pred}")
# print(f"Actual States for Video 21: {example_video['state'].values}")



# ================================================================= LSTM REP COUNTING CLASSIFICATION CODES =================================================================

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.optimizers import Adam

# from tensorflow.keras.callbacks import EarlyStopping
# import matplotlib.pyplot as plt


# early_stopping = EarlyStopping(
#     monitor='val_loss',  # Monitor the validation loss
#     patience=5,          # Number of epochs with no improvement after which training will be stopped
#     restore_best_weights=True  # Restore the model weights from the epoch with the best validation loss
# )

# # Load data
# csv_file = 'push-up-reps-keypoints-with-angles.csv'
# df = pd.read_csv(csv_file)

# # Function to calculate angles (you likely already executed this part)
# def calculate_angle(p1, p2, p3):
#     """Calculate angle at point p2 given 3 points."""
#     v1 = np.array(p1) - np.array(p2)
#     v2 = np.array(p3) - np.array(p2)
#     cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
#     angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
#     return np.degrees(angle)



# # Grouping by video and preparing sequences
# grouped = df.groupby('video_id')

# # Determine the maximum sequence length
# max_length = 208  # This is based on the longest video in your dataset

# # Function to pad 2D sequences (frames × features) for each video
# def pad_sequence(sequence, max_length, num_features):
#     padded = np.zeros((max_length, num_features))  # Create a zero-filled array of shape (max_length, num_features)
#     sequence_length = min(len(sequence), max_length)  # Ensure we don't exceed max_length
#     padded[:sequence_length, :] = sequence[:sequence_length]  # Copy the sequence into the padded array
#     return padded

# # Angle columns to extract
# angle_columns = [
#     'left_elbow_angle', 'right_elbow_angle', 'left_shoulder_angle', 
#     'right_shoulder_angle', 'hip_spine_angle', 'left_knee_angle', 
#     'right_knee_angle', 'spine_alignment', 'head_neck_spine_angle'
# ]
# num_features = len(angle_columns)  # Number of features per frame

# # Prepare padded sequences
# padded_sequences = []
# labels = []

# # Adjust labels to rep count
# labels = []

# for video_id, group in grouped:
#     group = group.sort_values('frame_id')  # Sort by frame_id to preserve temporal order
#     sequence = group[angle_columns].values  # Extract angle features
#     rep_count = group['rep_count'].iloc[0]  # Get rep count label
    
#     # Pad sequence
#     padded = pad_sequence(sequence, max_length, num_features)
#     padded_sequences.append(padded)
#     labels.append(rep_count)

# # Convert to numpy arrays
# padded_sequences = np.array(padded_sequences)
# rep_counts = np.array(labels)

# # Split into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(padded_sequences, rep_counts, test_size=0.2, random_state=42)

# # Model for regression
# model = Sequential([
#     LSTM(256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
#     Dropout(0.1),
#     LSTM(128, return_sequences=False),
#     Dense(64, activation='relu'),
#     Dense(1)
# ])

# optimizer = Adam(learning_rate=1e-4)
# model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])


# # Train model
# history = model.fit(
#     X_train, y_train,
#     epochs=100,
#     batch_size=16,
#     validation_split=0.2,
#     callbacks=[early_stopping]
# )

# # Evaluate on test set
# test_loss, test_mae = model.evaluate(X_test, y_test)
# print(f"Test Loss (MSE): {test_loss}")
# print(f"Test MAE: {test_mae}")

# # Predictions
# predictions = model.predict(X_test)
# print(f"Predicted Reps: {predictions.flatten()}")
# print(f"True Reps: {y_test}")

# plt.hist(rep_counts, bins=10)
# plt.xlabel('Rep Count')
# plt.ylabel('Frequency')
# plt.title('Distribution of Rep Counts')
# plt.show()


# plt.scatter(y_test, predictions.flatten())
# plt.xlabel('True Reps')
# plt.ylabel('Predicted Reps')
# plt.title('True vs Predicted Reps')
# plt.show()

# ================================================================= LSTM REP COUNTING CLASSIFICATION CODES =================================================================


# ================================================================= GOOD/BAD REP CLASSIFICATION CODES =================================================================

# # Load data
# csv_file = 'push-up-reps-keypoints-with-angles.csv'
# df = pd.read_csv(csv_file)

# # Function to calculate angles (you likely already executed this part)
# def calculate_angle(p1, p2, p3):
#     """Calculate angle at point p2 given 3 points."""
#     v1 = np.array(p1) - np.array(p2)
#     v2 = np.array(p3) - np.array(p2)
#     cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
#     angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
#     return np.degrees(angle)



# # Grouping by video and preparing sequences
# grouped = df.groupby('video_id')

# # Determine the maximum sequence length
# max_length = 208  # This is based on the longest video in your dataset

# # Function to pad 2D sequences (frames × features) for each video
# def pad_sequence(sequence, max_length, num_features):
#     padded = np.zeros((max_length, num_features))  # Create a zero-filled array of shape (max_length, num_features)
#     sequence_length = min(len(sequence), max_length)  # Ensure we don't exceed max_length
#     padded[:sequence_length, :] = sequence[:sequence_length]  # Copy the sequence into the padded array
#     return padded

# # Angle columns to extract
# angle_columns = [
#     'left_elbow_angle', 'right_elbow_angle', 'left_shoulder_angle', 
#     'right_shoulder_angle', 'hip_spine_angle', 'left_knee_angle', 
#     'right_knee_angle', 'spine_alignment', 'head_neck_spine_angle'
# ]
# num_features = len(angle_columns)  # Number of features per frame

# # Prepare padded sequences
# padded_sequences = []
# labels = []

# for video_id, group in grouped:
#     group = group.sort_values('frame_id')  # Sort by frame_id to preserve temporal order
#     sequence = group[angle_columns].values  # Extract angle features as a 2D array (frames × features)
#     label = group['label'].iloc[0]  # All frames in a video have the same label
    
#     # Pad the sequence to the maximum length
#     padded = pad_sequence(sequence, max_length, num_features)
#     padded_sequences.append(padded)
#     labels.append(label)

# # Convert to numpy arrays for model training
# padded_sequences = np.array(padded_sequences)
# labels = np.array(labels)

# print(f"Padded sequences shape: {padded_sequences.shape}")  # Should be (num_videos, max_length, num_features)
# print(f"Labels shape: {labels.shape}")

# # Splitting the data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# print(f"Training set size: {X_train.shape[0]}")
# print(f"Testing set size: {X_test.shape[0]}")

# print(padded_sequences[0])  # Inspect first padded sequence
# print(labels[0])            # Inspect corresponding label




# # Model definition
# model = Sequential()

# # Add an LSTM layer with 64 units and input shape matching your data
# model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))

# # Add dropout for regularization
# model.add(Dropout(0.2))

# # Add a Dense layer with sigmoid activation for binary classification
# model.add(Dense(1, activation='sigmoid'))

# # Compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Print model summary
# model.summary()


# # Define early stopping to monitor validation loss
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# # Train the model
# history = model.fit(
#     X_train, y_train, 
#     epochs=100,  # Number of complete passes through the data
#     batch_size=16,  # Number of samples processed at a time
#     validation_split=0.2,  # Use 20% of the training data for validation
#     callbacks=[early_stopping]
# )

# # Evaluate on the test set
# test_loss, test_accuracy = model.evaluate(X_test, y_test)
# print(f"Test Loss: {test_loss}")
# print(f"Test Accuracy: {test_accuracy}")



# # Plot accuracy
# plt.plot(history.history['accuracy'], label='Train Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Accuracy over Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# # Plot loss
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Loss over Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()




# from sklearn.metrics import confusion_matrix, classification_report

# # Predict on test set
# predictions = model.predict(X_test)
# predictions = (predictions > 0.5).astype(int)  # Convert probabilities to binary classes

# # Confusion matrix
# cm = confusion_matrix(y_test, predictions)
# print("Confusion Matrix:")
# print(cm)

# # Classification report
# print("Classification Report:")
# print(classification_report(y_test, predictions))

# ================================================================= GOOD/BAD REP CLASSIFICATION CODES =================================================================


# # CALCULATING ANGLES
# angles = []
# for idx, row in df.iterrows():
#     frame_angles = {
#         'left_elbow_angle': calculate_angle(
#             (row['LShoulder_x'], row['LShoulder_y'], row['LShoulder_z']),
#             (row['LElbow_x'], row['LElbow_y'], row['LElbow_z']),
#             (row['LWrist_x'], row['LWrist_y'], row['LWrist_z'])
#         ),
#         'right_elbow_angle': calculate_angle(
#             (row['RShoulder_x'], row['RShoulder_y'], row['RShoulder_z']),
#             (row['RElbow_x'], row['RElbow_y'], row['RElbow_z']),
#             (row['RWrist_x'], row['RWrist_y'], row['RWrist_z'])
#         ),
#         'left_shoulder_angle': calculate_angle(
#             (row['Spine1_x'], row['Spine1_y'], row['Spine1_z']),
#             (row['LShoulder_x'], row['LShoulder_y'], row['LShoulder_z']),
#             (row['LElbow_x'], row['LElbow_y'], row['LElbow_z'])
#         ),
#         'right_shoulder_angle': calculate_angle(
#             (row['Spine1_x'], row['Spine1_y'], row['Spine1_z']),
#             (row['RShoulder_x'], row['RShoulder_y'], row['RShoulder_z']),
#             (row['RElbow_x'], row['RElbow_y'], row['RElbow_z'])
#         ),
#         'hip_spine_angle': calculate_angle(
#             (row['Spine1_x'], row['Spine1_y'], row['Spine1_z']),
#             (row['Pelvis_x'], row['Pelvis_y'], row['Pelvis_z']),
#             (row['RHip_x'], row['RHip_y'], row['RHip_z'])
#         ),
#         'left_knee_angle': calculate_angle(
#             (row['LHip_x'], row['LHip_y'], row['LHip_z']),
#             (row['LKnee_x'], row['LKnee_y'], row['LKnee_z']),
#             (row['LAnkle_x'], row['LAnkle_y'], row['LAnkle_z'])
#         ),
#         'right_knee_angle': calculate_angle(
#             (row['RHip_x'], row['RHip_y'], row['RHip_z']),
#             (row['RKnee_x'], row['RKnee_y'], row['RKnee_z']),
#             (row['RAnkle_x'], row['RAnkle_y'], row['RAnkle_z'])
#         ),
#         'spine_alignment': calculate_angle(
#             (row['Pelvis_x'], row['Pelvis_y'], row['Pelvis_z']),
#             (row['Spine1_x'], row['Spine1_y'], row['Spine1_z']),
#             (row['Neck_x'], row['Neck_y'], row['Neck_z'])
#         ),
#         'head_neck_spine_angle': calculate_angle(
#             (row['Neck_x'], row['Neck_y'], row['Neck_z']),
#             (row['Spine1_x'], row['Spine1_y'], row['Spine1_z']),
#             (row['Head_x'], row['Head_y'], row['Head_z'])
#         )
#     }
#     angles.append(frame_angles)

# # Convert angles to DataFrame and merge with original data
# angles_df = pd.DataFrame(angles)
# df = pd.concat([df, angles_df], axis=1) 

# # Save updated DataFrame
# df.to_csv('push-up-reps-keypoints-with-angles.csv', index=False)



# import pandas as pd
# import numpy as np

# # Load data
# csv_file = 'push-up-clips-keypoints.csv'
# df = pd.read_csv(csv_file)

# # Drop specific videos
# videos_to_remove = ['bad_rep_26', 'good_rep_50']
# df = df[~df['video_id'].isin(videos_to_remove)]  # Keep only rows not in the list

# # Save updated DataFrame without bad_rep_26 and good_rep_50
# filtered_csv_file = 'filtered-push-ups-keypoints.csv'
# df.to_csv(filtered_csv_file, index=False)

# print(f"Saved filtered dataset to {filtered_csv_file}.")

# # Function to compute Euclidean distance
# def euclidean_distance(joint1, joint2):
#     return np.sqrt(
#         (df[f'{joint1}_x'] - df[f'{joint2}_x'])**2 +
#         (df[f'{joint1}_y'] - df[f'{joint2}_y'])**2 +
#         (df[f'{joint1}_z'] - df[f'{joint2}_z'])**2
#     )

# # Calculate reference length (distance between shoulders as normalization factor)
# reference_length = euclidean_distance('RShoulder', 'LShoulder')

# # Calculate normalized distances
# df['elbow_wrist_left'] = euclidean_distance('LElbow', 'LWrist') / reference_length
# df['elbow_wrist_right'] = euclidean_distance('RElbow', 'RWrist') / reference_length
# df['shoulder_elbow_left'] = euclidean_distance('LShoulder', 'LElbow') / reference_length
# df['shoulder_elbow_right'] = euclidean_distance('RShoulder', 'RElbow') / reference_length
# df['shoulder_hip_left'] = euclidean_distance('LShoulder', 'LHip') / reference_length
# df['shoulder_hip_right'] = euclidean_distance('RShoulder', 'RHip') / reference_length
# df['spine_neck'] = euclidean_distance('Spine1', 'Neck') / reference_length
# df['hip_knee_left'] = euclidean_distance('LHip', 'LKnee') / reference_length
# df['hip_knee_right'] = euclidean_distance('RHip', 'RKnee') / reference_length
# df['knee_ankle_left'] = euclidean_distance('LKnee', 'LAnkle') / reference_length
# df['knee_ankle_right'] = euclidean_distance('RKnee', 'RAnkle') / reference_length



# from keras_tuner import Hyperband

# def build_model(hp):
#     # model = Sequential([
#     #     Bidirectional(LSTM(128, return_sequences=False), input_shape=(1, 20)),  # num_units = 128
#     #     Dropout(0.2),  # dropout_rate = 0.2
#     #     Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
#     #     Dense(3, activation='softmax')
#     # ])
#     model = Sequential([
#     # LSTM layer with tunable units
#     Bidirectional(LSTM(units=hp.Int('lstm_units', min_value=64, max_value=256, step=32), return_sequences=False), input_shape=(1, 20)),  # num_units = 128
#     Dropout(rate=hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)),  # Tunable dropout rate
#     Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
#     Dense(3, activation='softmax')
#     ])
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1])),
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy']
#     )
#     return model

# # Define the tuner
# tuner = Hyperband(
#     build_model,
#     objective='val_accuracy',
#     max_epochs=100,
#     factor=2,
#     directory='my_project',
#     project_name='bilstm_tuning'
# )

# # Run the search
# tuner.search(X_train, y_train, validation_split=0.2, epochs=50, batch_size=16)

# # Best hyperparameters
# best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
# print(f"Best Hyperparameters: {best_hps.values}")