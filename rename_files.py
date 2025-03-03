import os

# Directory containing your videos
video_directory = r"C:\Users\Dilan Patel\Desktop\3rd Year Project\Third-Year-Project\data\pull-ups"


# Prefix for the renamed files
prefix = "pull_up"

# File extension of your video files
file_extension = ".mp4"  # Change this if your files have a different extension (e.g., ".mov")

# List all files in the directory
video_files = [f for f in os.listdir(video_directory) if f.endswith(file_extension)]

# Rename files
for idx, video_file in enumerate(video_files, start=1):
    # Construct the new filename
    new_name = f"{prefix}_{idx}{file_extension}"
    
    # Get full paths for renaming
    old_path = os.path.join(video_directory, video_file)
    new_path = os.path.join(video_directory, new_name)
    
    # Rename the file
    os.rename(old_path, new_path)
    print(f"Renamed: {video_file} -> {new_name}")

print("All files renamed successfully!")
