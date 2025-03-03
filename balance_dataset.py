import pandas as pd
from sklearn.utils import resample

# Load your dataset
df = pd.read_csv("3D/mediapipe/mediapipe-pushup-dataset-balanced.csv")
# df = pd.read_csv("mmpose-pushup-keypoints/mmpose-pushup-keypoints-2d-updated-final.csv")

# Separate classes
df_0 = df[df['state'] == 0]
df_1 = df[df['state'] == 1]  
df_2 = df[df['state'] == 2]

# df_1_upsampled = resample(df_1, replace=True, n_samples=len(df_0), random_state=42)

# df_balanced = pd.concat([df_0, df_1_upsampled, df_2])

# # Shuffle dataset
# df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# # Save to CSV
# df_balanced.to_csv("3D/mediapipe/mediapipe-pushup-dataset.csv", index=False)



import numpy as np

def augment_with_noise(df, state_label, num_samples, noise_std=0.02):
    """Augment dataset by adding slight Gaussian noise to joint coordinates."""
    df_state = df[df['state'] == state_label]
    
    augmented_samples = []
    
    for _ in range(num_samples):
        sample = df_state.sample(n=1, random_state=None).copy()
        
        # Apply Gaussian noise to all columns except 'state'
        noise = np.random.normal(0, noise_std, sample.shape[1] - 1)
        sample.iloc[:, :-1] += noise  # Apply noise to numerical features
        
        augmented_samples.append(sample)
    
    return pd.concat(augmented_samples)

# Augment state 1 to balance dataset
augmented_df_1 = augment_with_noise(df, state_label=1, num_samples=(len(df_0) - len(df_1)))

df_augmented = pd.concat([df, augmented_df_1]).sample(frac=1, random_state=42)
df_augmented.to_csv("3D/mediapipe/mediapipe-pushup-dataset-balanced-with-noise.csv", index=False)