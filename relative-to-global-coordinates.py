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

# # Load MMPose 3D dataset (with original angles and distances)
# csv_file = '3D/mmpose-3d/aligned-mmpose-dataset.csv'
# df = pd.read_csv(csv_file)
# df.dropna(inplace=True)

# # Load RTM-Pose 2D dataset
# rtm_df = pd.read_csv('2D/rtm-pose/rtm-pose-pushup-dataset.csv')
# rtm_df.dropna(inplace=True)

# def normalize_pelvis_xyz(group):
#     y_values = group['Pelvis_y'].values
#     z_norm = (y_values - y_values.min()) / (y_values.max() - y_values.min())
#     group['pelvis_z_offset'] = z_norm * 0.5
#     x_values = group['Pelvis_x'].values
#     x_norm = (x_values - x_values.min()) / (x_values.max() - x_values.min())
#     group['pelvis_x_offset'] = x_norm * 0.2
#     group['pelvis_y_offset'] = z_norm * 0.1
#     return group

# pelvis_xyz_normalized = rtm_df[['video_id', 'frame_id', 'Pelvis_x', 'Pelvis_y']].groupby('video_id').apply(normalize_pelvis_xyz)
# pelvis_xyz_normalized = pelvis_xyz_normalized.reset_index(drop=True)

# # Merge offsets into MMPose data
# mmpose_df = df.merge(
#     pelvis_xyz_normalized[['video_id', 'frame_id', 'pelvis_x_offset', 'pelvis_y_offset', 'pelvis_z_offset']],
#     on=['video_id', 'frame_id'],
#     how='left'
# )

# # Apply pseudo-global offsets to XYZ
# for col in [c for c in mmpose_df.columns if '_x' in c and '_global' not in c]:
#     mmpose_df[f'{col}_global'] = mmpose_df[col] + mmpose_df['pelvis_x_offset']
# for col in [c for c in mmpose_df.columns if '_y' in c and '_global' not in c]:
#     mmpose_df[f'{col}_global'] = mmpose_df[col] + mmpose_df['pelvis_y_offset']
# for col in [c for c in mmpose_df.columns if '_z' in c and '_global' not in c]:
#     mmpose_df[f'{col}_global'] = mmpose_df[col] + mmpose_df['pelvis_z_offset']

# # Filter to left/right-facing and prepare for CSV
# mmpose_df['camera_side'] = mmpose_df['video_id'].str.extract(r'(front_facing|left_facing|right_facing)')
# mmpose_filtered = mmpose_df[mmpose_df['camera_side'].isin(['left_facing', 'right_facing'])].copy()
# mmpose_filtered.dropna(inplace=True)

# # Select columns for new CSV: pseudo-global XYZ, angles, distances, state, metadata
# global_x_cols = [col for col in mmpose_filtered.columns if '_x_global' in col]
# global_y_cols = [col for col in mmpose_filtered.columns if '_y_global' in col]
# global_z_cols = [col for col in mmpose_filtered.columns if '_z_global' in col]
# all_cols = (global_x_cols + global_y_cols + global_z_cols +
#             ph.mmpose_angle_columns + ph.mmpose_distance_columns + ['state', 'video_id', 'frame_id', 'camera_side'])

# # Save to new CSV
# mmpose_filtered[all_cols].to_csv('3D/mmpose-3d/pseudo-global-mmpose-dataset.csv', index=False)
# print("Saved pseudo-global coordinates with angles, distances, and state to 'pseudo-global-mmpose-dataset.csv'")

# Load the new CSV for single-timestep training
global_df = pd.read_csv('3D/mmpose-3d/pseudo-global-mmpose-dataset-matched.csv')
global_df.dropna(inplace=True)

# Prepare data for single-timestep model
all_cols = [col for col in global_df.columns if col not in ['state', 'video_id', 'frame_id', 'camera_side']]
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

model.save('relative-to-global-model-single-timestamp.h5')  # Saves the model to an H5 file

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







# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.manifold import TSNE
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.regularizers import l2
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# import pushup_helper_methods as ph
# import time
# import tensorflow as tf

# # Load MMPose 3D dataset (with original angles and distances)
# csv_file = '3D/mmpose-3d/aligned-mmpose-dataset.csv'
# df = pd.read_csv(csv_file)
# df.dropna(inplace=True)

# # Load RTM-Pose 2D dataset
# rtm_df = pd.read_csv('2D/rtm-pose/rtm-pose-pushup-dataset.csv')
# rtm_df.dropna(inplace=True)

# def normalize_pelvis_xyz(group):
#     y_values = group['Pelvis_y'].values
#     z_norm = (y_values - y_values.min()) / (y_values.max() - y_values.min())
#     group['pelvis_z_offset'] = z_norm * 0.5
#     x_values = group['Pelvis_x'].values
#     x_norm = (x_values - x_values.min()) / (x_values.max() - x_values.min())
#     group['pelvis_x_offset'] = x_norm * 0.2
#     group['pelvis_y_offset'] = z_norm * 0.1
#     return group

# pelvis_xyz_normalized = rtm_df[['video_id', 'frame_id', 'Pelvis_x', 'Pelvis_y']].groupby('video_id').apply(normalize_pelvis_xyz)
# pelvis_xyz_normalized = pelvis_xyz_normalized.reset_index(drop=True)

# mmpose_df = df.merge(
#     pelvis_xyz_normalized[['video_id', 'frame_id', 'pelvis_x_offset', 'pelvis_y_offset', 'pelvis_z_offset']],
#     on=['video_id', 'frame_id'],
#     how='left'
# )

# for col in [c for c in mmpose_df.columns if '_x' in c and '_global' not in c]:
#     mmpose_df[f'{col}_global'] = mmpose_df[col] + mmpose_df['pelvis_x_offset']
# for col in [c for c in mmpose_df.columns if '_y' in c and '_global' not in c]:
#     mmpose_df[f'{col}_global'] = mmpose_df[col] + mmpose_df['pelvis_y_offset']
# for col in [c for c in mmpose_df.columns if '_z' in c and '_global' not in c]:
#     mmpose_df[f'{col}_global'] = mmpose_df[col] + mmpose_df['pelvis_z_offset']

# mmpose_df['camera_side'] = mmpose_df['video_id'].str.extract(r'(front_facing|left_facing|right_facing)')
# mmpose_filtered = mmpose_df[mmpose_df['camera_side'].isin(['left_facing', 'right_facing'])].copy()
# mmpose_filtered.dropna(inplace=True)

# global_x_cols = [col for col in mmpose_filtered.columns if '_x_global' in col]
# global_y_cols = [col for col in mmpose_filtered.columns if '_y_global' in col]
# global_z_cols = [col for col in mmpose_filtered.columns if '_z_global' in col]
# all_cols = (global_x_cols + global_y_cols + global_z_cols +
#             ph.mmpose_angle_columns + ph.mmpose_distance_columns)
# print("Features:", len(all_cols), all_cols)

# X = mmpose_filtered[all_cols].values
# y = mmpose_filtered['state'].values
# print("X shape:", X.shape)

# def create_sequences(X, y, seq_length=10, step=1):
#     X_seq, y_seq = [], []
#     for i in range(0, len(X) - seq_length + 1, step):
#         X_seq.append(X[i:i + seq_length])
#         y_seq.append(y[i + seq_length - 1])
#     return np.array(X_seq), np.array(y_seq)

# seq_length = 10
# X_seq, y_seq = create_sequences(X, y, seq_length=seq_length)
# scaler = StandardScaler()
# X_seq = scaler.fit_transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)
# print("X_seq shape:", X_seq.shape)

# # Add noise for data augmentation
# def add_noise(X, noise_level=0.01):
#     noise = np.random.normal(0, noise_level, X.shape)
#     return X + noise

# X_seq_noisy = add_noise(X_seq)

# X_train, X_test, y_train, y_test = train_test_split(X_seq_noisy, y_seq, test_size=0.2, random_state=42)
# class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
# class_weights = dict(enumerate(class_weights))

# model = Sequential([
#     LSTM(units=128, return_sequences=False, input_shape=(seq_length, len(all_cols))),
#     Dropout(rate=0.3),
#     Dense(16, activation='relu', kernel_regularizer=l2(0.03)),
#     Dropout(rate=0.3),  # Additional dropout
#     Dense(3, activation='softmax')
# ])

# optimizer = Adam(learning_rate=0.0005)
# model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
# checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# start_time = time.time()
# history = model.fit(
#     X_train, y_train,
#     validation_split=0.2,
#     epochs=100,
#     batch_size=32,
#     class_weight=class_weights,
#     callbacks=[early_stopping, checkpoint],
#     verbose=2
# )
# print(f"Training took {time.time() - start_time:.2f} seconds")

# # Load best model
# model.load_weights('best_model.h5')

# loss, accuracy = model.evaluate(X_test, y_test)
# print(f"Test Accuracy: {accuracy:.2f}")

# y_pred_probs = model.predict(X_test)
# y_pred = np.argmax(y_pred_probs, axis=1)
# print("Classification Report:")
# print(classification_report(y_test, y_pred))

# print("Validation Accuracy:", history.history['val_accuracy'])
# plt.plot(history.history['accuracy'], label='Train Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Accuracy over Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Loss over Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(6,6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()