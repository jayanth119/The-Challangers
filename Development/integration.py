from .reduceImageRedunant import ReduceRedunant
from  .stringSummarizer import LangchainSummarizer
from .vid2frame import VideoProcessor
from .video_analyser import ImageProcessor
import os

class Video2Text():
    def __init__(self, video_path, output_folder, model_path):
        self.video_path = video_path
        self.output_folder = output_folder
        self.model_path = model_path
        self.video_processor = VideoProcessor(video_path, output_folder)
        self.image_processor = ImageProcessor(model_path)
        self.langchain_summarizer = LangchainSummarizer()
        self.reduce_redunant = ReduceRedunant()

    def process_video(self):
        frames_1sec = self.video_processor.process_video()
        print("Video to Frames conversion completed successfully")
        return frames_1sec
    
    def remove_redundant(self, images):
        reduced_images = self.reduce_redunant.reduce_images(images, 5)
        self.reduce_redunant.save_images(reduced_images, save_path='reduced_images', num_images=5)
        print("Redundant frames removed successfully")
        return reduced_images
    
    def captioning(self, images):
        paths_list=[]
        for img_path in os.listdir(images):
            paths_list.append(os.path.join(images, img_path))
        model = self.image_processor('OpenGVLab/InternVL2-8B')
        captions_fps = model.frame_iter(paths_list)
        print("Captioned every 5 frames successfully")
        return captions_fps
    
    def summarize_text(self, captions):
        summaries_1sec = []
        for caption in captions:
            summary = self.langchain_summarizer.summarize_with_langchain(caption)
            summaries_1sec.append(summary)
        print("Text Summarized successfully")
        return summaries_1sec
    
    def forward(self):
        frames_1sec = self.process_video()
        reduced_images = self.remove_redundant(frames_1sec)
        captions_fps = self.captioning(reduced_images)
        summaries_1sec = self.summarize_text(captions_fps)
        return summaries_1sec

