import cv2
import numpy as np
import os

class VideoProcessor:
    def __init__(self, video_path, output_folder):
        self.video_path = video_path
        self.output_folder = output_folder
        
        # Ensure the output directory exists
        os.makedirs(output_folder, exist_ok=True)
        
    def process_video(self):
        # Capture the video
        video_cap = cv2.VideoCapture(self.video_path)
        
        if not video_cap.isOpened():
            raise ValueError("Error opening video file")
        
        # Get video properties
        fps = int(video_cap.get(cv2.CAP_PROP_FPS))  # Frames per second
        total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames
        duration = int(total_frames / fps)  # Duration of the video in seconds
        
        print(f"FPS: {fps}, Total Frames: {total_frames}, Duration: {duration} seconds")

        current_sec = 0
        frame_count = 0
        frame_list = []

        while video_cap.isOpened():
            ret, frame = video_cap.read()
            
            if not ret:
                break
            
            # Convert frame to numpy array and store in list
            frame_list.append(frame)
            frame_count += 1
            
            # When we reach the target 25 frames (1 second worth of frames at 25 fps)
            if frame_count == 25:
                self._save_frames(frame_list, current_sec)
                current_sec += 1
                frame_count = 0
                frame_list = []  # Reset for the next second

        video_cap.release()
        cv2.destroyAllWindows()
    
    def _save_frames(self, frame_list, current_sec):
        """
        Save the frames of the current second as numpy arrays with filenames '{sec,frame}.npy'.
        """
        for idx, frame in enumerate(frame_list):
            filename = f"{self.output_folder}/{current_sec},{idx}.npy"
            np.save(filename, frame)
            print(f"Saved frame {idx} of second {current_sec} to {filename}")

# Example usage:
video_processor = VideoProcessor(video_path=r"C:\Users\HARSHA PC\Pictures\003.mp4", output_folder='output_frames')
video_processor.process_video()
