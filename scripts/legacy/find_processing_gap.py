import os
from datetime import datetime, timedelta

# Set the path to your folder
folder_path = '/scratch/users/databases/deepextremes-video-XXpsnr'

# Get all folders and their creation times
folders = [
    (f, os.path.getctime(os.path.join(folder_path, f)))
    for f in os.listdir(folder_path)
    if os.path.isdir(os.path.join(folder_path, f))
]

# Sort folders by creation time
folders.sort(key=lambda x: x[1])

# Find gaps greater than 1 minute
gap_threshold = timedelta(minutes=10)
for i in range(1, len(folders)):
    prev_time = datetime.fromtimestamp(folders[i - 1][1])
    curr_time = datetime.fromtimestamp(folders[i][1])
    gap = curr_time - prev_time
    if gap > gap_threshold:
        print(f"Gap detected at {i/len(folders)*100:.2f}% between '{folders[i - 1][0]}' and '{folders[i][0]}': {gap} ({curr_time=})")