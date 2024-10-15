# Importing the required libraries
import os
import openai
from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI

# Setting the OPEN AI API Key from Environment Variable
openai.api_key = os.environ["OPENAI_API_KEY"]

# Load GPT-3.5-turbo via LangChain
llm = ChatOpenAI(openai_api_key=openai.api_key, model_name="gpt-3.5-turbo", temperature=0.5)

# Function to summarize the text using GPT-3.5-turbo
class LangchainSummarizer:
    def summarize_with_langchain(self, list_text):
        text = " ".join(list_text)
        document = Document(page_content=text)
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = chain.run([document])
        return summary

# Example working of the function
# list_demo = ['This image captures a busy street scene in Krishnagiri, India, recorded on December 12, 2023, at 15:47:29. The scene includes multiple vehicles such as cars, buses, and motorcycles navigating through the well-marked intersection. A construction barrier separates the different lanes. A prominent display of the location includes the text "Krishnan Lankha Nethaji Bridge," identifying the bridge in the vicinity. The environment is bustling with activity, reflecting the vibrancy of urban life in the area.', 'The image displays a car entering a parking garage through a gate. The car in the foreground is a white hatchback, and other vehicles are seen driving down the road with shipping boxes in the back. There is a metal gate partially closed, and the surrounding area appears modern and organized. The environment suggests an urban setting with a controlled entrance and exit point for vehicles.', 'The image displays a traffic collision involving a white car and another light-colored vehicle near a building. A truck is seen parked alongside the curb, partially obstructing the view of the crash site. The scene appears to involve a narrow, divided urban road with visible traffic markings and a pedestrian crossing. The building reflects a sense of urban infrastructure with a balcony on the left side.', 'The image shows a busy street with cars, motorcycles, and pedestrians in a congested area. A traffic light is displayed above the road, indicating a red signal for the vehicles. There appears to be a dense flow of traffic, and the image has an overcast sky. The timestamp at the bottom indicates the capture was taken on May 5th, 2011. The words "CAM ID 010" appear above the timestamp.', 'Two people are attending to a fallen person lying on the ground near a motorbike that appears to have crashed. The surrounding traffic is congested with cars and a pink taxi in the immediate vicinity.']
# res_summary = LangchainSummarizer.summarize_with_langchain(list_demo)
# print("The summary : ")
# print(res_summary)

