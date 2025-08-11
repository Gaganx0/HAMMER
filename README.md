# HAMMER
Multimodal disinformation detection using HAMMER model by Shao et. al. Supervised by Dr. Priyanka Singh and Dr.Xue Li. The University of Queensland. Gagandeep Singh.

## Familiarisation 
To familiarise yourself with the current state of the project, I would suggest you read the journal paper on HAMMER itself. https://github.com/rshaojimmy/MultiModal-DeepFake. The paper is linked in the README. 

Also I have attached my thesis paper itself for the current stage of the research. 
## Setup 
I recommend using separate Conda requirements as some of the methods below have conflicting requirements. 

**1. HAMMER and DGM4** 
Follow the steps in the README (https://github.com/rshaojimmy/MultiModal-DeepFake) and download and set up both the architecture and the dataset. This will be the basis for the rest of the project. 

**2. StyleCLIP**
At the root of the project setup, StyleCLIP from https://github.com/orpatashnik/StyleCLIP. This helps to manipulate facial features with text prompts. We would need only the 'Editing via Global Direction' part of the StyleCLIP implementation. Again, the README has the installation guide. 

**3. HyperStyle**
Before we can process faces through StyleCLIP, we need landmarks of the image, which can be reproduced by using HyperStyle. https://github.com/yuval-alaluf/hyperstyle?tab=readme-ov-file. This would again go in the root of the directory. You should set WSL2 if on Windows. 

**4. Image Restoration.**
We use Restormer https://github.com/swz30/Restormer and Real-ESRGAN https://github.com/xinntao/Real-ESRGAN experimentally to restore distorted images before processing.   

**5. FG-BG consistency**
We use SAM by Meta to separate background from foreground. https://github.com/facebookresearch/segment-anything 
We will use coco torchvision to prompt SAM model. https://download.pytorch.org/whl/cpu
We will then use CLIP to compare the foreground and background. https://github.com/openai/CLIP

## Usage
After setting everything up, use testing.py to check the installation. There are minimal changes to testing.py compared to the standard test file to make it work on windows. The current test.py helps investigate each input pair by computing its manipulation probability and uses Streamlit for the display. 

Select your batch of photos and put them through steps 3 and 2 of the Setup. Use edit_latents_per_emotion.py by putting it into the styleclip folder and flip emotions. At this point, we would have some processed images. Now run align_faces.py to realign faces. 

We then put these realigned image-text pairs back into the HAMMER model to compare performance. 

The scripts in the restoration folder are a work in progress and can be used as a preprocessing step at any moment in the flow to experiment with the output. 

Web scraping inputs for alternative testing methods was done using the Bing reverse search API, but it has been discontinued, and we will be exploring a different way to scrape these input. 








    

 
