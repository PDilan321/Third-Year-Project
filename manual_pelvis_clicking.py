import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
import pushup_helper_methods as ph

# # Directory structure for videos
# VIDEO_DIRS = {
#     'left_facing': 'data/push-up-multi-reps/left-facing',
#     'right_facing': 'data/push-up-multi-reps/right-facing'
# }
# OUTPUT_CSV = '3D/mmpose-3d/pelvis_clicks.csv'

# # Function to click pelvis positions
# def click_pelvis(video_path, video_id, key_frames=10):
#     cap = cv2.VideoCapture(video_path)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
    
#     # Select key frames (evenly spaced)
#     frame_indices = np.linspace(0, total_frames - 1, key_frames, dtype=int)
#     clicks = []

#     def mouse_callback(event, x, y, flags, param):
#         if event == cv2.EVENT_LBUTTONDOWN:
#             clicks.append((frame_idx, x, y))
#             print(f"Clicked pelvis at frame {frame_idx} in {video_id}: ({x}, {y})")

#     cv2.namedWindow('Video')
#     cv2.setMouseCallback('Video', mouse_callback)

#     for frame_idx in frame_indices:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         cv2.imshow('Video', frame)
#         print(f"Click on the pelvis in frame {frame_idx} of {video_id} (press 'q' to continue)")
#         while len(clicks) <= frame_idx // (total_frames // key_frames):
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#     cap.release()
#     cv2.destroyAllWindows()
#     return [(video_id, idx, x, y) for idx, x, y in clicks if idx in frame_indices]

# # Process videos from all directories
# all_clicks = []
# for orientation in VIDEO_DIRS:
#     video_dir = VIDEO_DIRS[orientation]
#     for i in range(1, 26):  # 1 to 25 for each orientation
#         video_id = f"{orientation}_{i}"
#         video_path = os.path.join(video_dir, f"{video_id}.mp4")  # Adjust extension if needed
#         if os.path.exists(video_path):
#             clicks = click_pelvis(video_path, video_id)
#             all_clicks.extend(clicks)
#         else:
#             print(f"Video not found: {video_path}")

# # Save clicks to CSV with video_id format
# clicks_df = pd.DataFrame(all_clicks, columns=['video_id', 'frame_id', 'pelvis_x', 'pelvis_y'])
# clicks_df.to_csv(OUTPUT_CSV, index=False)
# print(f"Saved pelvis clicks to {OUTPUT_CSV}")



############################### Interpolating Pelvis Clicks ########################################

# import pandas as pd
# import numpy as np

# # Load pelvis clicks and MMPose dataset
# clicks_df = pd.read_csv('3D/mmpose-3d/pelvis_clicks.csv')
# mmpose_df = pd.read_csv('3D/mmpose-3d/aligned-mmpose-dataset.csv')

# # Function to map full video frame indices to MMPose trimmed frame indices
# def map_to_mmpose_frames(video_id, full_frame_indices, mmpose_df):
#     mmpose_frames = mmpose_df[mmpose_df['video_id'] == video_id]['frame_id'].sort_values().values
#     if len(mmpose_frames) == 0 or len(full_frame_indices) == 0:
#         return []

#     # Get the number of clicks for this video (should be 10)
#     video_clicks = clicks_df[clicks_df['video_id'] == video_id].sort_values('frame_id')
#     num_clicks = len(video_clicks)
#     if num_clicks == 0:
#         return []

#     # Ensure full_frame_indices matches the number of clicks (10)
#     full_frame_indices = np.linspace(0, 99, num_clicks, dtype=int)  # Assume 100-frame video, adjust if needed

#     # Map clicks to MMPose frames proportionally
#     total_mmpose_frames = len(mmpose_frames)
#     if total_mmpose_frames <= num_clicks:
#         # If fewer MMPose frames than clicks, use all MMPose frames
#         mapping = {i: mmpose_frames[i % total_mmpose_frames] for i in range(num_clicks)}
#     else:
#         # Map evenly spaced clicks to MMPose frames
#         click_positions = np.linspace(0, total_mmpose_frames - 1, num_clicks, dtype=int)
#         mapping = {i: mmpose_frames[idx] for i, idx in enumerate(click_positions)}

#     # Get pelvis X, Y for this video and align with mapping
#     pelvis_x = video_clicks['pelvis_x'].values
#     pelvis_y = video_clicks['pelvis_y'].values

#     return [(video_id, mapping[i], x, y) for i, (x, y) in enumerate(zip(pelvis_x, pelvis_y))]

# # Process each video to align clicks with MMPose frames
# aligned_clicks = []
# for video_id in clicks_df['video_id'].unique():
#     # Get total frames in MMPose for this video (trimmed)
#     mmpose_frames = mmpose_df[mmpose_df['video_id'] == video_id]['frame_id'].sort_values().values
#     if len(mmpose_frames) > 0:
#         # Assume 10 clicks per video, evenly spaced in full video (0 to 99 for 100 frames)
#         full_frame_indices = np.linspace(0, 99, 10, dtype=int)  # Adjust 99 if video length differs
#         aligned = map_to_mmpose_frames(video_id, full_frame_indices, mmpose_df)
#         aligned_clicks.extend(aligned)

# # Save aligned clicks
# aligned_clicks_df = pd.DataFrame(aligned_clicks, columns=['video_id', 'frame_id', 'pelvis_x', 'pelvis_y'])
# aligned_clicks_df.to_csv('3D/mmpose-3d/aligned_pelvis_clicks.csv', index=False)
# print("Saved aligned pelvis clicks to 'aligned_pelvis_clicks.csv'")

# # Interpolate for all MMPose frames per video
# def interpolate_pelvis(video_id, clicks_df, mmpose_df):
#     video_clicks = clicks_df[clicks_df['video_id'] == video_id].sort_values('frame_id')
#     mmpose_frames = mmpose_df[mmpose_df['video_id'] == video_id]['frame_id'].sort_values().values
#     if len(mmpose_frames) == 0 or len(video_clicks) == 0:
#         return pd.DataFrame()
    
#     frame_ids = mmpose_frames
#     pelvis_x = np.interp(frame_ids, video_clicks['frame_id'], video_clicks['pelvis_x'])
#     pelvis_y = np.interp(frame_ids, video_clicks['frame_id'], video_clicks['pelvis_y'])
#     return pd.DataFrame({
#         'video_id': [video_id] * len(frame_ids),
#         'frame_id': frame_ids,
#         'pelvis_x': pelvis_x,
#         'pelvis_y': pelvis_y
#     })

# # Process each video
# interpolated_pelvis = []
# for video_id in aligned_clicks_df['video_id'].unique():
#     interpolated = interpolate_pelvis(video_id, aligned_clicks_df, mmpose_df)
#     if not interpolated.empty:
#         interpolated_pelvis.append(interpolated)

# interpolated_df = pd.concat(interpolated_pelvis)
# interpolated_df.to_csv('3D/mmpose-3d/pelvis_interpolated.csv', index=False)
# print("Saved interpolated pelvis positions to 'pelvis_interpolated.csv'")



###################################### Use interpolated csv to convert the relative coordinates to global coordinates ########################################


# # Load MMPose dataset and interpolated pelvis positions
# mmpose_df = pd.read_csv('3D/mmpose-3d/aligned-mmpose-dataset.csv')
# pelvis_df = pd.read_csv('3D/mmpose-3d/pelvis_interpolated.csv')

# # Merge to align pelvis 2D positions with MMPose 3D data
# merged_df = mmpose_df.merge(
#     pelvis_df[['video_id', 'frame_id', 'pelvis_x', 'pelvis_y']],
#     on=['video_id', 'frame_id'],
#     how='left'
# )

# def manual_normalize_pelvis_xyz(group):
#     pelvis_x = group['pelvis_x'].values
#     pelvis_y = group['pelvis_y'].values
#     x_norm = (pelvis_x - pelvis_x.min()) / (pelvis_x.max() - pelvis_x.min()) if pelvis_x.max() != pelvis_x.min() else 0
#     y_norm = (pelvis_y - pelvis_y.min()) / (pelvis_y.max() - pelvis_y.min()) if pelvis_y.max() != pelvis_y.min() else 0
#     group['pelvis_x_offset'] = x_norm * 0.2  # 0.2m for X (lateral)
#     group['pelvis_y_offset'] = y_norm * 0.1  # 0.1m for Y (depth)
#     group['pelvis_z_offset'] = y_norm * 0.5  # 0.5m for Z (pushup height, tied to Y)
#     return group

# pelvis_xyz_normalized = merged_df.groupby('video_id').apply(manual_normalize_pelvis_xyz)
# pelvis_xyz_normalized = pelvis_xyz_normalized.reset_index(drop=True)

# # Apply offsets to create pseudo-global coordinates
# for col in [c for c in pelvis_xyz_normalized.columns if '_x' in c and '_global' not in c]:
#     pelvis_xyz_normalized[f'{col}_global'] = pelvis_xyz_normalized[col] + pelvis_xyz_normalized['pelvis_x_offset']
# for col in [c for c in pelvis_xyz_normalized.columns if '_y' in c and '_global' not in c]:
#     pelvis_xyz_normalized[f'{col}_global'] = pelvis_xyz_normalized[col] + pelvis_xyz_normalized['pelvis_y_offset']
# for col in [c for c in pelvis_xyz_normalized.columns if '_z' in c and '_global' not in c]:
#     pelvis_xyz_normalized[f'{col}_global'] = pelvis_xyz_normalized[col] + pelvis_xyz_normalized['pelvis_z_offset']

# # Add angles, distances, and metadata
# global_x_cols = [col for col in pelvis_xyz_normalized.columns if '_x_global' in col]
# global_y_cols = [col for col in pelvis_xyz_normalized.columns if '_y_global' in col]
# global_z_cols = [col for col in pelvis_xyz_normalized.columns if '_z_global' in col]
# all_cols = (global_x_cols + global_y_cols + global_z_cols +
#             ph.mmpose_angle_columns + ph.mmpose_distance_columns + ['state', 'video_id', 'frame_id'])

# # Save to new CSV
# pelvis_xyz_normalized[all_cols].to_csv('3D/mmpose-3d/pelvis-click-global-mmpose-dataset.csv', index=False)
# print("Saved manual pelvis-click pseudo-global coordinates to 'pelvis-click-global-mmpose-dataset.csv'")



################################################################ Train on new global coordinates ################################################################


# Train single-timestep model
global_df = pd.read_csv('3D/mmpose-3d/pelvis-click-global-mmpose-dataset.csv')
global_df.dropna(inplace=True)

all_cols = [col for col in global_df.columns if col not in ['state', 'video_id', 'frame_id']]
X = global_df[all_cols].values
y = global_df['state'].values
print("X shape:", X.shape)

scaler = StandardScaler()
X = scaler.fit_transform(X)
X = X.reshape(-1, 1, X.shape[1])  # Single timestep

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

model = Sequential([
    Bidirectional(LSTM(units=128, return_sequences=False), input_shape=(1, len(all_cols))),
    Dropout(rate=0.1),
    Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(3, activation='softmax')
])

optimizer = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stopping]
)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
print("Classification Report:")
print(classification_report(y_test, y_pred))

model.save('pelvis-click-global-model-single-timestamp.h5')

print("Validation Accuracy:", history.history['val_accuracy'])
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()