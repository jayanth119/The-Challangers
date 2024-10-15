import cv2
import os
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector

def reduce_redundant_frames(input_video_path, output_folder, threshold=30, split_frames=10):
    """
    Function to reduce redundant frames from a video and split into groups of frames.
    
    Parameters:
    - input_video_path (str): Path to the input video file.
    - output_folder (str): Folder to save the output frames.
    - threshold (int): Scene change detection threshold. Higher values will result in fewer scene changes.
    - split_frames (int): Number of frames per group.
    
    Returns:
    - None
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open video
    video = open_video(input_video_path)
    scene_manager = SceneManager()

    # Add a content detector with a threshold for scene detection
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    # Process video to detect scenes
    scene_manager.detect_scenes(video)

    # Get the list of scenes and their frame intervals
    scenes = scene_manager.get_scene_list()

    # Prepare frame extraction
    cap = cv2.VideoCapture(input_video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    all_frames = []
    current_frame = 0

    # Process scenes to extract frames from them
    for start_time, end_time in scenes:
        start_frame = int(start_time.get_seconds() * frame_rate)
        end_frame = int(end_time.get_seconds() * frame_rate)

        # Extract frames from each detected scene
        for frame_num in range(start_frame, end_frame):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                all_frames.append(frame)

    cap.release()

    # Group frames into chunks of 'split_frames'
    frame_groups = [all_frames[i:i+split_frames] for i in range(0, len(all_frames), split_frames)]

    # Save frames in groups
    for i, frame_group in enumerate(frame_groups):
        group_folder = os.path.join(output_folder, f'group_{i + 1}')
        if not os.path.exists(group_folder):
            os.makedirs(group_folder)

        # Save each frame in the group
        for j, frame in enumerate(frame_group):
            frame_filename = os.path.join(group_folder, f'frame_{i * split_frames + j + 1}.jpg')
            cv2.imwrite(frame_filename, frame)

    print(f"Frames have been saved into {len(frame_groups)} groups in '{output_folder}'.")

# Example usage:
reduce_redundant_frames("input_video.mp4", "output_frames", threshold=30, split_frames=10)
