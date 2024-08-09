import datetime
import logging
import logging.handlers
import os
import pdb
import sys
from pathlib import Path

import requests

from llava.constants import LOGDIR

import random

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

handler = None


EVAL_TASKS = ['text_vqa', 'visual_spatial_reasoning', 'pope_rand', 'cifar_100',
              'cifar_10', 'mnist', 'snli_ve_classification', 'pope_adv', 'pope_pop']


VISION_FLAN_DEF = {'PlotQA+visual_question_answering': 'You are provided with a chart image and a question related to the chart. Answer the question based on the information given in the chart.', 'VQG+caption_generation': 'Given an image, you will generate a caption for the image.', 'VQA-E+visual_question_answering': 'You are provided with an image and a question related to the image. Answer the question based on the information given in the image.', 'VQG+question_generation': 'Given an image, generate a relevant question about the content of the image.', 'VisDA-2017+object_classification_train': 'You are given an image which contains a 3D rendered object. Your goal is to identify the category of the object present in the image from the given options.', 'GTSRB+image_classification': "In this task, you will classify traffic signs in the given image. The image is in color and contains various traffic signs captured in real-world conditions. Your goal is to accurately identify the type of traffic sign present in the image. Consider factors such as the sign's shape, color, and symbols to ensure correct classification. A sample output should look like this: 'Speed limit (80km/h)'", 'Winoground+image_captioning': 'In this task, you will be provided with an image and two captions. Your task is to identify which of the two captions correctly describes the image.', 'spot-the-diff+image_diff_identification': 'You are provided with an image which contains two pictures side by side. Your task is to identify the differences between the two pictures. Separate the differences with a comma.', 'SentiCap+image_sentiment_captioning': 'You are provided with a picture, write a caption with a specific sentiment (positive or negative) related to the picture. Note that the sentiment in the caption should match the requested sentiment.', 'KVQA+image_question_answer': 'You are provided with a picture and a question related to the picture. Your job is to correctly answer the question. Note that any references to directions (left, right, center, etc.) in the questions are from the perspective of the person depicted in the image.', 'KVQA+image_captioning': 'You are provided with a picture. Write a caption for the image mentioning the people present in the image and also the occasion depicted.', 'MemeCap+image_captioning': 'You are given a meme and your goal is to generate a caption that describes the image. Ignore the text in the meme.', 'VisDA-2017+object_classification_validation': 'You are given an image which contains an object. Your goal is to identify the category of the object present in the image from the given options.', 'ConceptualCaptions+image_captioning': 'Generate a short and abstract caption for the given image.', 'MemeCap+meme_captioning': 'Generate an explanation for the given meme.', 'VQA-E+image_captioning': 'You are provided with an image. Explain what is going on in the image.', 'Caltech101+Living_Thing_classification': 'In this task, you have to classify if the setting contains a living thing or not. The object in the image is among a total of 102 classes such as Airplane, Ant, Butterfly, Chair,... The classes of the image are a diverse set ranging from objects to living beings. Pay attention to details as the object in the image can be in any format(sketch, painting, captured photo, etc) So, your answer should be if the object is a living thing or not.', 'ObjectNet+Object_classfication': 'Your task is to recognize the object depicted in the given image. The object can be any item commonly used in our everyday lives, such as kitchen tools, food items, stationery, clothing, and more. To correctly identify the object, carefully observe its color, shape, and size characteristics.', 'Caltech101+Image_classification': 'In this task, you have to classify the object in the image among a total of 102 classes such as Airplane, Ant, Butterfly, Chair,... The classes of the image are a diverse set ranging from objects to living beings. Pay attention to details as the object in the image can be in any format(sketch, painting, captured photo, etc) So, your answer should be the class of the object in the image', 'Places205+Image_env_classification': 'In this task, you have to identify if the place or scene pictured is indoor or outdoor. In the image is among a total of 205 classes such as Hospital, Bridge, Courtyard, Motel,.... The classes of the images are a diverse set of places or scenes. Pay attention to the details as some of the images may contain an object that relates to a specific place while some images may directly show the place or scenary. So, your answer should be the place or scene shown in the image', 'Core50+Object_detection': "Your task is to identify the item shown in the picture. The images contain everyday objects such as a plug adapter, mobile phone, scissors, and more. It is important to carefully consider the object's shape, size, and color characteristics in order to accurately classify the image.", 'Cars+car_brand_classification': 'In this task, you have to identify the brand of the car such as Audi, BMW, Bentley,... This means you have to identify the company which manufactured the car. For this, you need to look at the logo shown in the car image. Based on the detailing shown for the car image, the company model of the car can be identified. So, your answer should be the brand name of the car.', 'NUS-WIDE+Animal_classification': "Identify if the given image contains any animal in it. Pay attention to each object in the image as well as the background environment while making this classification. If the image contains an animal, the answer should be 'yes'. Otherwise, 'no'.", 'Cars+car_classification': 'In this task, based on the given image dataset of different cars, you have to identify the model + car make + Year of Make of a car in the image among a total of 196 categories such as Audi A5 Coupe 2012, BMW 3 Series Sedan 2012, Bentley Arnage Sedan 2009,... Pay attention to details such as the size, logo, type of the car to identify the model. So by looking at a car image, Give your answer in the following format:  Model of the Car++Make of the Car++Year of Make', 'ImageNet-A+image_classification': 'In this task, given an image, please identify what the image contains a. The image could contain, among other things, animals, birds, daily objects, insects', 'recipe-qa+visual_coherence': 'The given image contains 4 image choices numbered A through D. Select the least similar image among the gorup', 'textcaps+caption_generation': 'Write a caption for the image. When you write the caption, also consider the text on the image and decide the best way to fit them into your caption.', 'visdial+answer_question_6': 'Here is a conversation between 2 people. They are talking about a picture and discussing some questions and answers about it. Read the conversation and then answer as the next person.', 'crowdhuman+count_numbers': 'You are given a picture of a crowd of people, there could be a varying number of people in this image. Please tell me how many people are in this particular image.', 'fairface+image_classification_age': "You are given an image of a person's face. This person can be of different ages, your task is to identify the person's age", 'eurosat+image_classification_agri': 'Given an image taken from a satellite, identify if there are agricultural or forest lands in this image. There could be different types of cover, some man-made or naturally occurring.', 'visdial+answer_question_4': 'This is a short conversation between two people, represented as A and B. They are talking about the given picture and are discussing some questions about it. Pay attention to the conversation and answer the question.', 'ImageNet-C+image_classification_weather': 'Given an image, identify what kind of weather conditions might have corrupted the image. It can be different types of bad weather or outdoor conditions', 'visdial+answer_question_9': 'Here is a detailed conversation between a person and a robot. They are discussing some questions and answers about a picture. From the conversation, answer the question as the robot.', 'eurosat+image_classification_natural': 'Given this satellite image, categorize if this image contains crops, water bodies, industrial structures, or types of common vegetation.', 'cinic-10+object_presence_animal': 'The given image can contain some animals; they can be animals typically found in the wild or domesticated animals. The picture could also contain something that does not fit this description. Your job is to identify if the subject of the image is an animal or not.', 'cinic-10+image_classification_animal': 'The given image can contain various types of animals. Some of these animals are found in forests, drylands, or other natural areas. Some of them could also be domesticated pets. Please identify the animal in the picture.', 'coco+image_classification_sports': 'Given an image of sporting goods, identify what the object is. It could be used to play a team sport or an individual activity. The objects can also be used in different kinds of sports and sometimes make it easier for the wearer to play the sport.', 'semart+image_description': 'Here is a picture of fine art, can you tell me some background about it.', 'iconqa+fill_in_blank': 'Hey, here is a picture and a question I had about it, can you help me complete my answer?', 'eurosat+image_classification_land': 'Based on the input satellite image, can you classify the predominant land cover type present in the image, such as but not limited to: forests, urban areas, and agricultural fields?', 'semart+image_school': 'I would like to know what school of art this painting could be from. Some possible schools of art are Italian, German, or Spanish among others', 'cinic-10+image_classification_shipping': 'The given image can contain different types of shipping equipment. They can carry goods across water or land, and they carry all types of materials required around the world. Please identify the type of shipping option in the picture.', 'semart+image_technique': 'Can you please tell me what technique is used to create the painting in this picture? Among some other techniques, it could be an oil painting', 'infographicvqa+single_document_question': "Here is a picture of a business or industry document, can you please answer my question based on the text in the document?", 'eurosat+image_classification_crops': 'Can you classify different types of crops or agricultural fields in a satellite image? I would like to know about the vegetation and crop, if possible.', 'coco+image_classification_kitchen': 'Given an image of something from the kitchen, identify what it could be. The image could be of cooking tools or items that are used for eating. It could also be used for serving food or storing it.', 'coco+image_classification_vehicle': 'Given an image of a vehicle, identify the kind of vehicle it is. The vehicle can be of different types; it could be something used, personal, or public transport. It could carry one or more people at the same time.', 'fairface+image_classification_gender': "Here is a picture of a person. Based only upon this picture, what would you guess this person's gender is?", 'coco+image_classification_appliance': 'Given an image of a common electronic appliance from around the house, identify the type of object it is. It could be an appliance that is commonly used in the kitchen to cook or store food.', 'iconqa+choose_txt': 'I have a question about what is happening in this picture, can you please give me an answer?', 'vizwiz+question_answer': "A blind person asks you a question about this image, answer the question in the best way possible.", 'semart+image_type': 'Here is an image of some art. I want to know what type of painting it is. Among others, some types could be: religious, self-portraits, or landscapes', 'cinic-10+object_presence_shipping': 'The given image can contain some vehicles used for transporting goods and materials across large distances, even around the world. The picture could also contain something that does not fit this description. Your job is to identify if the subject of the image can be used for shipping goods or not.', 'expw+expression_detection': 'Given an image of a person, look at their face and tell me what their facial expression is like', 'coco+image_classification_furniture': 'Given an image of a piece of furniture in a house, identify the type of furniture. It is usually used to make the house look better and can be made of different kinds of material.', 'Set5+image_recognition': 'In this task, recognize the subject in the image from among 5 subjects, namely - baby, bird, butterfly, head, woman.', 'infographicvqa+question_answer': "Here is an infographic representing some data in a visual form. Please answer my question using the data in the image", 'ImageNet-C+image_classification_general': 'In this task, identify the type of corruption given a corrupted image. It could be digitally altered, contain natural distortions or contain other corruptions', 'eurosat+image_classification_manmade': 'Can you identify man-made structures like highways or industrial facilities in this image taken from a satellite?', 'cinic-10+image_classification_transport': 'The given image can contain different types of transport vehicles. People use these vehicles to travel around in their day-to-day lives. It could be air travel or a slower means of transport on the ground. Please identify the type of transport option in the picture.', 'semart+image_timeframe': 'Here is a picture of some fine art created in the past, I would like to know roughly what period in time it could have been made in', 'eurosat+image_classification_water': 'Is it possible to classify different types of water bodies, such as rivers or lakes, in the given satellite image?', 'visdial+answer_question_2': 'Here is a short conversation between a human and an AI assistant. They are talking about a particular picture. They are discussing some questions and answers about it. Pay attention to the conversation and then answer the question at the end.', 'ImageNet-C+image_classification_noise': 'Given an image, identify the type of corruption in the image. The image can have digitally generated noise, blur, or other distortions', 'ImageNet-C+image_classification_blur': 'Given a blurred picture, identify the type of blur in the image, it can be blurred in different ways', 'fairface+image_classification_race': "What could be a good guess for this person's race in the given image?", 'coco+image_classification_animal': 'Given an image of an animal, identify the kind of animal in the image. The picture could be of more popular animals that are visible around zoos or are sometimes domesticated at home. They could also sometimes be found in the wild.', 'PICKAPIC+image_short_description': 'Check this image. Can you give me a short description for it?', 'INATURALIST+phylum_classification': 'Phylum is defined as a principal taxonomic category that ranks above class and below kingdom. Identify the phylum of the organism in the image.', 'PACS+giraffe_image_category_classification': 'You will be given an image of a guitar. The image could be of different categories like painting, cartoon, photograph, or sketch. Identify the image category.', 'INATURALIST+genus_classification': 'The genus is a taxonomic rank above the species and below the family. Identify the genus of the organism in the image.', 'INATURALIST+class_classification': 'Taxonomic category is a rank or group of organisms developed on the basis of their fundamental characteristics, similarities and dissimilarities. A class is a taxonomic rank above the order and below the phylum. Identify the class of the organism in the image.', 'PACS+photo_object_classification': 'The input is a photograph of an object. Identify the main object in the image.', 'INATURALIST+supercategory_classification': 'You will be given an image of an organism. Analyze the image and pick the super category for this organism from the options provided.', 'PACS+horse_image_category_classification': 'You will be given an image of a horse. The image could be of different categories like painting, cartoon, photograph, or sketch. Identify the image category.', 'REDCAPS+reddit_caption_2': 'Imagine I am posting this image on a casual community based social media platform. What could be a possible caption for this image? I could describe the image, write something fun, give context or describe my emotions about the image.', 'PACS+art_painting_object_classification': 'You will be given an art painting image as input. Identify the main object in the image.', 'NOCAPS+image_caption': 'Give me a list of captions for this image.', 'PACS+cartoon_object_classification': 'You will be given an image of a cartoon. Identify the main object in the image.', 'PACS+guitar_image_category_classification': 'You will be given an image of a guitar. The image could be of different categories like painting, cartoon, photograph, or sketch. Identify the image category.', 'CLEVR+question_answer': 'Analyze the image provided and answer the question provided in the input text.', 'CHART2TEXT+chart_caption': 'I will give you a chart. Analyze it and give me as many details as possible.', 'FLICKR30K+caption_image': 'Each image has something going on. Carefully analyze the image and generate 5 captions for each image.', 'DOMAIN_NET+image_category_classification': 'You will be given an image. Answer 2 questions. What kind of image is this? Choose from clip art, info graph, painting, rough sketch, painting, real and sketch. Second question, what is the main object in the image? Answer it like "This is a clip art of an apple."', 'PACS+house_image_category_classification': 'You will be given an image of a house. The image could be of different categories like painting, cartoon, photograph, or sketch. Identify the image category.', 'PACS+elephant_image_category_classification': 'You will be given an image of an elephant. The image could be of different categories like painting, cartoon, photograph, or sketch. Identify the image category.', 'DOMAIN_NET+real_image_classification': 'Identify the main object in the image.', 'MEMOTION+sentiment_detection': 'Can you please read the image and extract all the text from it?', 'DAQUAR+object_question_answer': "The input text will contain a question about the image", 'DOMAIN_NET+painting_image_classification': 'The input for this task is a painting. Identify the main object in the painting.', 'REDCAPS+reddit_caption_1': 'I am posting this image on Reddit. Can you give me a caption for this image? Something suitable for Reddit.', 'LOC_NARRATIVES+flickr30k_images_caption': 'Please give me a detailed caption about the input image.', 'INATURALIST+latin_english_name_classification': 'Identify the organism in the image. Give the english name(also called common name) followed by the scientific name(also called latin name). For example : "The organism in the image is Common Earthworm. Its scientific name is Lumbricus terrestris."', 'STVQA+image_question_answer': "The input text contains a question about the image. Please answer the question.", 'CONCADIA+image_description': 'Can you describe the visual appearance of this image?', 'DTD+coarse_grained_texture_classification': "Texture is defined as the feel, appearance or consistency of a surface or substance from a human's perspective. Detect the primary texture represented in the image.", 'DOMAIN_NET+clipart_image_classification': 'Clip art is defined as simple pictures or symbols used in documents and presentations. The input is a clip art image. Identify the main object in the image.', 'HICO+human_activity_detection': 'Answer a simple question. What is the person in the image doing? If there is no action being performed, describe the main object in the image.', 'PACS+dog_image_category_classification': 'You will be given an image of a dog. The image could be of different categories like painting, cartoon, photograph, or sketch. Identify the image category.', 'CONCADIA+image_caption_context_2': 'Caption this image. Along with the caption, give me some background knowledge regarding this image.', 'RAF_DB+human_emotion_detection': 'Give me details about the human in the image. What is their gender, race and age? What emotion are they depicting?', 'SKETCH+object_detection': 'Each image is a human drawn sketch of a object. Identify the main object in the image.', 'HICO+object_classification': 'What is the object in the image?', 'PACS+person_image_category_classification': 'You will be given an image of a person. The image could be of different categories like painting, cartoon, photograph, or sketch. Identify the image category.', 'GEOMETRY3K+geometry_question_answer': 'I will give you a figure with some geometrical information. Analyze the image and data in the input text and answer the question.',
                   'LOC_NARRATIVES+ade20k_images_caption': 'Can you give me a detailed description about this image?', 'DVQA+charts_question_answer_2': 'Check this graph and give me a suitable answer for the question in the input text.', 'DTD+all_texture_detection': "Texture is defined as the feel, appearance or consistency of a surface or substance from a human's perspective. Detect all the textures in the image. Present it as a comma separated list", 'CONCADIA+image_caption_context_1': 'Can you give me a caption and some background knowledge about this image?', 'WIKIART+art_classification': 'For the given painting, give me the artist name. Also provide the genre and style, if possible.', 'RAVEN+next_pattern': 'Each image has 8 images labeled as Image 1 to Image 8. These 8 images follow a specific pattern. Detect the pattern and select the next image in the sequence from the 8 available options.', 'LOC_NARRATIVES+open_images_caption': 'What is going in this image? Give me a detailed description.', 'DOMAIN_NET+quickdraw_image_classification': 'In this task, the input will be a rough sketch of something. Identify the main object depicted in the rough sketch.', 'VIQUAE+question_answer': "With the help of this image, can you answer the question given in the input text?", 'FUNSD+text_detection': 'Identify all the text in the image. Any ordering of the text is acceptable. Each chunk of text should be surrounded by double quotes and separated by commas.', 'WIT+detailed_description': 'Give me some background knowledge about this image.', 'SKETCH+living_organism_detection': 'In this task, you will identify whether the picture contains a living organism. The images given are black and white sketches drawn by human beings. If the picture depicts a living organism or part of a living organism, the output should be "Living". Otherwise, print "Non-Living"', 'INATURALIST+order_classification': 'Taxonomic category is a rank or group of organisms developed on the basis of their fundamental characteristics, similarities and dissimilarities. The order is a taxonomic rank above the family and below the class. Identify the order of the organism in the image.', 'VQARAD+question_answer': 'I will give you a radiology image(scan of a body part). Analyze it and answer the question given in the input text.', 'LOC_NARRATIVES+coco_images_caption': 'I want to know more about this image. Can you please describe it?', 'VIZWIZ+image_captioning': 'Give me multiple captions for this image.', 'VOC2007+object_detection': 'Identify some objects that are present in the image. Give a comma separated list as the output.', 'DOMAIN_NET+infograph_image_classification': 'An info graph is a visual image like a poster that is used to represent information or data about any object. For this task, the input will be a info graph. Identify the main object of the info graph.', 'LAD+object_detection_details': "I'll give you an image. What is the main object in it? Please give additional details.", 'STANFORD_DOGS+dog_classification': 'Identify the breed of the dog in the image.', 'DOCVQA+question_answer': "Check the image and answer the question given in the input text.", 'INATURALIST+family_classification': 'The family is a taxonomic rank above the genus and below the order. Identify the family of the organism in the image.', 'DVQA+charts_question_answer_1': "I am trying to analyse this graph. Can you answer the question given in the input text?", 'LFW+face_recognition': "In this task, you will be presented with a face image of an individual. Your objective is to accurately classify the image by identifying the person's identity it represents. To accomplish this, you must meticulously examine the facial features present in the image, such as the shape and structure of the face, eyes, nose, mouth, hair, and any other distinguishing features such as moles, scars, or birthmarks that can provide valuable clues for determining the identity. For instance, certain facial proportions, distinct eye color, or unique hair style could be defining characteristics of an individual's identity. Just as one might identify a bicycle by its wheels or a sunflower by its petals in other datasets, in this case, a person can be identified by their unique set of facial features. Once you've made an informed determination based on these visual clues, provide your answer as the identity of the person.", 'ayahoo_test_images+animal_object_vehicle_image_classification': "In this task, you are given an image from a dataset, which contains images from different categories of animals, objects, and vehicles. These categories further divide into subcategories. Your job is to classify the given image into one of these subcategories, which could be anything from an aeroplane to a zebra. Your classification should be based on key identifiers like size, shape, color, distinctive features, and the context or environment depicted in the image. For example, if you're given an image of a zebra, your answer would simply be zebra. Remember that images could be of objects or vehicles as well. Your answer should be a single word representing the appropriate subcategory for the image, emphasizing specificity beyond the broad categories.", '300w+human_portrait_classification': "In this task, you will be presented with an image depicting a human portrait image. Your objective is to accurately classify the image by identifying the two categories it belongs to which are indoor and outdoor. To do so, carefully examine the visual elements present in the image, such as the background, people's clothes and any distinguishing features that can provide valuable clues for determining the category. For instance, if a person is at a baseball game outdoors, the category is outdoors. Once you have determined the category, provide your answer as the name of the category. ", 'trainSet+image_classification': 'Using one word, classify the type of distortion / style is applied to the image or what type of setting is represented in the image.', 'Office-Home+Image_classification': "Your task involves classifying object images into their respective categories like Bed, Sink, Sneakers, Table, TV and so on; for instance, if the model is presented with an image of a laptop, it should correctly identify and categorize the image as 'Laptop'.", 'model-vs-human+image_style_classification': 'What is the artistic style of this image?', 'Clevr+Question_Answer_Matching': 'You will be given an Image of 3D-rendered objects, a number of Questions and same number of Answers. The task here is to match the questions to the right answers according to the image you see. The format of the output shoud be something like: Q1A3,Q2A5,Q3A2,Q4A1,Q5A1', 'CLEVR_CoGenT+Question_Answering': 'The input for this task is an image of 3D-rendered objects and a question that fall into different categories. The questions fall into five classes of tasks: Exist, Count, Compare Integer, Query Attribute, and Compare Attribute. The task here is to answer the question.', 'FGVC_Aircraft+Aircraft_Classification_Manufacturer': 'Determine the manufacturer of the provided aircraft image.\nThe manufacturer refers to the company that designs, builds, and assembles the aircraft, possessing the expertise and experience in the aviation industry necessary for production and delivery.', 'Office_31+Image_Classification_ObjectAndCategory': "The input for this task is an image of the commonly encountered office objects such as keyboards, file cabinets, and laptops from three different categories(AMAZON, DSLR, WEBCAM). The output of this task is to name the domain of image(AMAZON, DSLR, or WEBCAM) as well as the object in the image in the following format: 'Category ObjectName' (exp. amazon mug) AMAZON: these images were captured from a website of online merchants, they are captured against clean background. DSLR cameras offer improved image quality when compared to standard webcams.", 'CUB-200-2011+Bird_Classification': 'Your objective is to identify the species of the bird depicted in the provided image.', 'STL-10+Image_Classification': 'You will be presented with an image and your objective is to identify and name the object depicted in the image.', 'FGVC_Aircraft+Aircraft_Classification_Variant': 'Your objective is to analyze an aircraft image and provide the variant of the aircraft. (e.g., A300B4).\n Variant: A variant indicates a variation of a particular aircraft model, often incorporating specific modifications, improvements, or customizations compared to the base model.', 'CLEVR_CoGenT+VQA_context': 'You are given some question and answer pairs as context, and you will answer the question at the end based on the image.', 'Clevr+Multiple_Question_Answering': 'The input for this task is an image of 3D-rendered objects and a set of questions that fall into different categories. The questions fall into five classes of tasks: Exist, Count, Compare Integer, Query Attribute, and Compare Attribute. The output of this task is a set of answers to the given questions for each image. The answers should be generated based on the content of the image and the category of the question. The output should be in the form of text.', 'CLEVR_CoGenT+Question_Answer_Matching': 'In this task, you will be presented with an image containing 3D-rendered objects along with a set of questions and corresponding answers. Your goal is to correctly match each question with its corresponding answer based on the visual content of the image. The output format should follow this pattern: Q1A3, Q2A5, Q3A2, Q4A1, Q5A1, indicating the question number followed by the corresponding answer number.', 'Office_31+Image_Classification_Category': 'The input for this task is an image of commonly encountered office objects such as keyboards, file cabinets, and laptops from three different categories(AMAZON, DSLR, WEBCAM). The output of this task is to name the domains of the image(AMAZON, DSLR, or WEBCAM). AMAZON: These images were captured from a website of online merchants, they are captured against a clean background. DSLR cameras offer improved image quality when compared to standard webcams.', 'CLEVR_CoGenT+Multiple_Question_Answering': 'The input for this task is an image of 3D-rendered objects and a set of questions that fall into different categories. The questions fall into five classes of tasks: Exist, Count, Compare Integer, Query Attribute, and Compare Attribute. The output of this task is a set of answers to the given questions for each image. The answers should be generated based on the content of the image and the category of the question. The output should be in the form of text.', 'Clevr+Question_Answering': 'The input for this task is an image of 3D-rendered objects and a question that fall into different categories. The questions fall into five classes of tasks: Exist, Count, Compare Integer, Query Attribute, and Compare Attribute. The task here is to answer the question.', 'DeepFashion_highres_Attribute_and_Category+Cloth_Classification': 'Can you write a very short description of the cloth?', 'LSUN+Image_Classification': 'In this task you will be provided with a picture of a scene(dining room, bedroom, kitchen, outdoor church, and so on) and you have to classify images into their corresponding scene categories. Your answer should be the name of the place.', 'FGVC_Aircraft+Aircraft_Classification_Family': 'From the image provided to you, guess the family of the aircraft.\nHint: Family:  A family represents a collection of aircraft models produced by the same manufacturer, sharing common characteristics, design principles, or technological platforms.', 'FGVC_Aircraft+Aircraft_Classification': 'Your objective is to analyze an aircraft image and provide the manufacturer, family, and variant of the aircraft in the specified order: manufacturer; family; variant (e.g., Airbus; A300; A300B4).\nManufacturer: The manufacturer refers to the company that designs, builds, and assembles the aircraft, possessing the expertise and experience in the aviation industry necessary for production and delivery.\nFamily: A family represents a collection of aircraft models produced by the same manufacturer, sharing common characteristics, design principles, or technological platforms.\nVariant: A variant indicates a variation of a particular aircraft model, often incorporating specific modifications, improvements, or customizations compared to the base model.', 'Clevr+VQA_context': 'you are given some question and answer pairs as context, and you will answer the question at the end based on the image.', 'places365+Image_Classification': 'Your task involves analyzing an image of a scene and identifying the appropriate name for that particular scene. Examples of scene names could include airfield, airplane cabin, airport terminal, alcove, alley, amphitheater, amusement arcade, etc. ', 'Office_31+Image_Classification_Object': 'The input for this task is an image of commonly encountered office objects such as keyboards, file cabinets, and laptops. from three different categories (Pictures from AMAZON, pictures taken with a DSLR camera, and pictures taken by WEBCAM). The output of this task is to name the object in the image.', 'ImageNet-Sketch+image_classification': 'You are given a sketch of an object. Tell me the name of the object in the image.', 'Dark-Zurich+time_classification': 'Identify the time of the day when the image is captured. Options are: daytime, nighttime, twilight.', 'A-OKVQA+answer_rationales_matching': 'Answer the question and provide an explanation.', 'JHU-CROWD+scene_type_recognition': 'Provide the location of the scene. It could be a water park, marathon, protest, stadium, or any other possible location.', 'NABirds+bird_species_detection': 'Identify the species of the bird in the image, considering its overall appearance, including size, shape, color, and patterns.', 'A-OKVQA+rationales_generation': 'Provide 3 rationales for the given question and answer.', 'CrisisMMD+humanitarian_categories_classification': "Determine the humanitarian categories of the tweet text and the image. The categoies are: 'affected individuals', 'infrastructure and utility damage', 'injured or dead people', 'missing or found people', 'not humanitarian', 'other relevant information', 'rescue volunteering or donation effort', 'vehicle damage'. Output the categories of the tweet text and the image in the format: 'The tweet text is <category> and the image is <category>.'", 'AI2D+visual_question_answering': 'Answer the multiple choice question based on the image. The answer should be one of the choices.', 'FFHQ-Text+text-to-face_generation': 'Generate multiple descriptions for the face in the given image.', 'ExDark+object_recognition': 'Identify the object in the image, including bicycle, boat, bottle, bus, car, and other object.', 'CoVA+webpage_recognition': 'What is the name of the website based on the given image?', 'SCUT-CTW1500+text_detection': 'Detect the all text in the image.', 'FoodLogoDet-1500+food_logo_recognition': 'What is the logo in the image?', 'MNIST-M+number_recognition': 'In this task, you will be presented with a grayscale image containing a handwritten digit overlaid on a natural image background. Your objective is to correctly identify the digit in the image.', 'NABirds+body_part_detection': 'Tell me what body parts of the bird you can see in the image. provide the output in the following format: "The visible body parts are bill; crown; nape; left eye; belly; breast; back; tail; right wing."', 'A-OKVQA+visual_question_answering': 'Answer the question based on the image.', 'MVTecAD+anomaly_detection': 'The primary objective of this task is to accurately identify the type and cause of anomalies in the object present in the provided image. The image depicts a specific category of object and texture, and within this category, there are defect-free images as well as images exhibiting different types of defects. Your task is to carefully examine the image and meticulously identify the specific type and cause of any deviations from the normal appearance of the object or texture. Pay close attention to irregularities in lines, shading, color scheme, and level of detail. Additionally, analyze the unique characteristics of the category, including shape, color, and texture. Your focus should be on precisely identifying the particular type and cause of the anomaly. The potential anomalies to consider encompass a wide range, such as gray strokes, bent objects, holes, missing wires, and more.', 'Road-Anomaly+anomaly_detection': 'Detect the unusual dangers which can be encountered by a vehicle on the road.', 'Caltech-256+image_classification': 'Your task is to identify the object category of a real-world image. The image can contain different objects like an American flag, bear, cake, and more. Analyze the shape, color, and texture of the object to determine its category. Consider the specific details of the label. Provide the name of the object based on your classification.', 'MVTecAD+image_classification': 'Your objective is to classify an image based on its corresponding object category. The image provided encompasses a diverse range of industrial items, including a bottle, cable, carpet, and more. Focus on the overall visual appearance of the image, paying attention to details such as lines, shading, color scheme, and level of detail. It is crucial to analyze the distinctive characteristics of the object, such as its shape, color, and texture, as these features may vary significantly between different object categories. Once you have completed the classification process, output the appropriate object name based on your analysis.', 'Total-Text+text_detection': 'Detect and tell me all the text on the image. Separate them with semicolons.', 'ImageNet-R+image_classification': 'Your task is to classify the image using various categories. You need to carefully observe the details of the object in the image, including its shape, color, and texture, as these characteristics may vary across different renditions. Output the appropriate object name as the result of your classification process.', 'FlickrLogos-27+logo_detection': 'Detect and provide the logo name in the image.', 'CrisisMMD+Informativeness_classification': "Determine the informativeness of the tweet text and image. Informativeness is measured by the extent to which the tweet provides information about the crisis event. For instance, a tweet such as 'I am safe' is considered non-informative, whereas a tweet like 'I am safe. The earthquake is 10 miles away from me' is considered informative. Provide an output indicating whether the tweet text and image are informative or not informative.", 'AID+aerial_scene_classification': 'You are given an aerial image. Tell me the scene in the image.', 'DeepWeeds+weed_species_recognition': 'Identify weed species native to Australia in their natural habitat, alongside neighboring flora.', 'ImageNet-R+image_domain_classification': "Your goal is to classify the image based on its domain, which can be 'videogame', 'painting', 'sketch', 'cartoon', 'art', 'toy', 'deviantart', 'graphic', 'sculpture', 'misc', 'embroidery', 'sticker', 'graffiti', 'origami', or 'tattoo'. Your final output should specify the identified domain of the image.", 'CrisisMMD+damage_severity_classification': "Determine the damage severity of the scene in the image, including 'little or no damage', 'mild damage', and 'severe damage'.", 'Yoga-82+yoga_pose_recognition': 'What is the name of the yoga pose?', 'VisDA-2017+image_classification': 'Your task is to classify an image based on its corresponding object category. The image contains a variety of objects distributed among 12 categories, including aeroplane, horse, knife, person, plant, and others. To accurately classify the image, carefully analyze its visual characteristics, such as shape, color, texture, and spatial context relations, as these attributes can vary significantly across different domains. Once you have identified the object category of the image, output the appropriate label for your classification.', 'scienceqa+question_answering': "Answer a question about the given image", 'scienceqa+rational_question_answering': 'answer the following question with a detailed explanation.', 'scienceqa+explanation': 'Explain the rational behind the answer to the question.', 'scienceqa+detailed_question_answering': "Let's think step by step and provide detailed rationale before selecting the correct answer.", 'scienceqa+detailed_explanation': "Explain the rational behind the answer to the question. Please provide very comprehesive and detailed response.", 'scienceqa+background_question_answering': 'First think about the background knowledge needed to answer the question and then select the correct answer.'}

VISION_FLAN_ID = {'PlotQA+visual_question_answering': 29, 'VQG+caption_generation': 30, 'VQA-E+visual_question_answering': 31, 'VQG+question_generation': 32, 'VisDA-2017+object_classification_train': 33, 'GTSRB+image_classification': 34, 'Winoground+image_captioning': 35, 'spot-the-diff+image_diff_identification': 36, 'SentiCap+image_sentiment_captioning': 37, 'KVQA+image_question_answer': 38, 'KVQA+image_captioning': 39, 'MemeCap+image_captioning': 40, 'VisDA-2017+object_classification_validation': 41, 'ConceptualCaptions+image_captioning': 42, 'MemeCap+meme_captioning': 43, 'VQA-E+image_captioning': 44, 'Caltech101+Living_Thing_classification': 45, 'ObjectNet+Object_classfication': 46, 'Caltech101+Image_classification': 47, 'Places205+Image_env_classification': 48, 'Core50+Object_detection': 49, 'Cars+car_brand_classification': 50, 'NUS-WIDE+Animal_classification': 51, 'Cars+car_classification': 52, 'ImageNet-A+image_classification': 53, 'recipe-qa+visual_coherence': 54, 'textcaps+caption_generation': 55, 'visdial+answer_question_6': 56, 'crowdhuman+count_numbers': 57, 'fairface+image_classification_age': 58, 'eurosat+image_classification_agri': 59, 'visdial+answer_question_4': 60, 'ImageNet-C+image_classification_weather': 61, 'visdial+answer_question_9': 62, 'eurosat+image_classification_natural': 63, 'cinic-10+object_presence_animal': 64, 'cinic-10+image_classification_animal': 65, 'coco+image_classification_sports': 66, 'semart+image_description': 67, 'iconqa+fill_in_blank': 68, 'eurosat+image_classification_land': 69, 'semart+image_school': 70, 'cinic-10+image_classification_shipping': 71, 'semart+image_technique': 72, 'infographicvqa+single_document_question': 73, 'eurosat+image_classification_crops': 74, 'coco+image_classification_kitchen': 75, 'coco+image_classification_vehicle': 76, 'fairface+image_classification_gender': 77, 'coco+image_classification_appliance': 78, 'iconqa+choose_txt': 79, 'vizwiz+question_answer': 80, 'semart+image_type': 81, 'cinic-10+object_presence_shipping': 82, 'expw+expression_detection': 83, 'coco+image_classification_furniture': 84, 'Set5+image_recognition': 85, 'infographicvqa+question_answer': 86, 'ImageNet-C+image_classification_general': 87, 'eurosat+image_classification_manmade': 88, 'cinic-10+image_classification_transport': 89, 'semart+image_timeframe': 90, 'eurosat+image_classification_water': 91, 'visdial+answer_question_2': 92, 'ImageNet-C+image_classification_noise': 93, 'ImageNet-C+image_classification_blur': 94, 'fairface+image_classification_race': 95, 'coco+image_classification_animal': 96, 'PICKAPIC+image_short_description': 97, 'INATURALIST+phylum_classification': 98, 'PACS+giraffe_image_category_classification': 99, 'INATURALIST+genus_classification': 100, 'INATURALIST+class_classification': 101, 'PACS+photo_object_classification': 102, 'INATURALIST+supercategory_classification': 103, 'PACS+horse_image_category_classification': 104, 'REDCAPS+reddit_caption_2': 105, 'PACS+art_painting_object_classification': 106, 'NOCAPS+image_caption': 107, 'PACS+cartoon_object_classification': 108, 'PACS+guitar_image_category_classification': 109, 'CLEVR+question_answer': 110, 'CHART2TEXT+chart_caption': 111, 'FLICKR30K+caption_image': 112, 'DOMAIN_NET+image_category_classification': 113, 'PACS+house_image_category_classification': 114, 'PACS+elephant_image_category_classification': 115, 'DOMAIN_NET+real_image_classification': 116, 'MEMOTION+sentiment_detection': 117, 'DAQUAR+object_question_answer': 118, 'DOMAIN_NET+painting_image_classification': 119, 'REDCAPS+reddit_caption_1': 120, 'LOC_NARRATIVES+flickr30k_images_caption': 121, 'INATURALIST+latin_english_name_classification': 122, 'STVQA+image_question_answer': 123, 'CONCADIA+image_description': 124, 'DTD+coarse_grained_texture_classification': 125, 'DOMAIN_NET+clipart_image_classification': 126,
                  'HICO+human_activity_detection': 127, 'PACS+dog_image_category_classification': 128, 'CONCADIA+image_caption_context_2': 129, 'RAF_DB+human_emotion_detection': 130, 'SKETCH+object_detection': 131, 'HICO+object_classification': 132, 'PACS+person_image_category_classification': 133, 'GEOMETRY3K+geometry_question_answer': 134, 'LOC_NARRATIVES+ade20k_images_caption': 135, 'DVQA+charts_question_answer_2': 136, 'DTD+all_texture_detection': 137, 'CONCADIA+image_caption_context_1': 138, 'WIKIART+art_classification': 139, 'RAVEN+next_pattern': 140, 'LOC_NARRATIVES+open_images_caption': 141, 'DOMAIN_NET+quickdraw_image_classification': 142, 'VIQUAE+question_answer': 143, 'FUNSD+text_detection': 144, 'WIT+detailed_description': 145, 'SKETCH+living_organism_detection': 146, 'INATURALIST+order_classification': 147, 'VQARAD+question_answer': 148, 'LOC_NARRATIVES+coco_images_caption': 149, 'VIZWIZ+image_captioning': 150, 'VOC2007+object_detection': 151, 'DOMAIN_NET+infograph_image_classification': 152, 'LAD+object_detection_details': 153, 'STANFORD_DOGS+dog_classification': 154, 'DOCVQA+question_answer': 155, 'INATURALIST+family_classification': 156, 'DVQA+charts_question_answer_1': 157, 'LFW+face_recognition': 158, 'ayahoo_test_images+animal_object_vehicle_image_classification': 159, '300w+human_portrait_classification': 160, 'trainSet+image_classification': 161, 'Office-Home+Image_classification': 162, 'model-vs-human+image_style_classification': 163, 'Clevr+Question_Answer_Matching': 164, 'CLEVR_CoGenT+Question_Answering': 165, 'FGVC_Aircraft+Aircraft_Classification_Manufacturer': 166, 'Office_31+Image_Classification_ObjectAndCategory': 167, 'CUB-200-2011+Bird_Classification': 168, 'STL-10+Image_Classification': 169, 'FGVC_Aircraft+Aircraft_Classification_Variant': 170, 'CLEVR_CoGenT+VQA_context': 171, 'Clevr+Multiple_Question_Answering': 172, 'CLEVR_CoGenT+Question_Answer_Matching': 173, 'Office_31+Image_Classification_Category': 174, 'CLEVR_CoGenT+Multiple_Question_Answering': 175, 'Clevr+Question_Answering': 176, 'DeepFashion_highres_Attribute_and_Category+Cloth_Classification': 177, 'LSUN+Image_Classification': 178, 'FGVC_Aircraft+Aircraft_Classification_Family': 179, 'FGVC_Aircraft+Aircraft_Classification': 180, 'Clevr+VQA_context': 181, 'places365+Image_Classification': 182, 'Office_31+Image_Classification_Object': 183, 'ImageNet-Sketch+image_classification': 184, 'Dark-Zurich+time_classification': 185, 'A-OKVQA+answer_rationales_matching': 186, 'JHU-CROWD+scene_type_recognition': 187, 'NABirds+bird_species_detection': 188, 'A-OKVQA+rationales_generation': 189, 'CrisisMMD+humanitarian_categories_classification': 190, 'AI2D+visual_question_answering': 191, 'FFHQ-Text+text-to-face_generation': 192, 'ExDark+object_recognition': 193, 'CoVA+webpage_recognition': 194, 'SCUT-CTW1500+text_detection': 195, 'FoodLogoDet-1500+food_logo_recognition': 196, 'MNIST-M+number_recognition': 197, 'NABirds+body_part_detection': 198, 'A-OKVQA+visual_question_answering': 199, 'MVTecAD+anomaly_detection': 200, 'Road-Anomaly+anomaly_detection': 201, 'Caltech-256+image_classification': 202, 'MVTecAD+image_classification': 203, 'Total-Text+text_detection': 204, 'ImageNet-R+image_classification': 205, 'FlickrLogos-27+logo_detection': 206, 'CrisisMMD+Informativeness_classification': 207, 'AID+aerial_scene_classification': 208, 'DeepWeeds+weed_species_recognition': 209, 'ImageNet-R+image_domain_classification': 210, 'CrisisMMD+damage_severity_classification': 211, 'Yoga-82+yoga_pose_recognition': 212, 'VisDA-2017+image_classification': 213, 'scienceqa+question_answering': 214, 'scienceqa+rational_question_answering': 215, 'scienceqa+explanation': 216, 'scienceqa+detailed_question_answering': 217, 'scienceqa+detailed_explanation': 218, 'scienceqa+background_question_answering': 219}


def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True)
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        flagged = False
    except KeyError as e:
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"


def get_task_def(task):
    if task == 'image_caption':
        instructs = [
            f"""Generate textual description of the given image."""
        ]
    elif task == 'GQA':
        instructs = [
            f"""Answer the compositional question in natural language based on the content of the image. The questions involve multiple reasoning skills, spatial understanding and multi-step inference""",
        ]
    elif task == 'VQAv2':
        instructs = [
            f"""Answer the question in natural language based on the content of the image.""",
        ]
    elif task == 'visualgenome_vqa':
        instructs = [
            f"""Answer the question in natural language based on the content of the image.""",
        ]
    elif task == 'ok_vqa':
        instructs = [
            f"""Answer the question in natural language based on the content of the image. The questions require external knowledge to answer.""",
        ]
    elif task == 'VQA':
        instructs = [
            f"In this task, we ask you a question about an image and provide you with some options containing the correct answer. You should select the best answer from the options based on the content of the image."]
    elif task == 'GC':  # same as region_caption
        instructs = [
            f"""The goal of this task is to generate description for part of the image within a bounding box. We provide you with the coordinate of top-left corner of the bounding box denoted as x1 y1 and the coordinate of bottom-right corner of the bouding box denoted as x2 y2. The format of the input bounding box is x1 y1 x2 y2.""",
        ]
    elif task == 'GC_selection':  # same as region_caption
        instructs = [
            f"""In this task, you are given some natual langugae sentences in the options and you are required to select the sentence that works best as a caption for part of the image within a bounding box. A bounding box is an imaginary rectangle that outlines an object or part of the scene in an image. We provide you with the coordinate of top-left corner of the bounding box denoted as x1 y1 and the coordinate of bottom-right corner of the bouding box denoted as x2 y2. The format of the input bounding box is "x1 y1 x2 y2".""",
        ]

    elif task == 'VG':
        instructs = [
            f"""In this task, you are asked to localize the region in an image that is described by the given text. The region should be specified via a bounding box which is an imaginary rectangle that outlines part of the scene in an image. Your output should contain the coordinate of top-left corner x1 y1 and the coordinate of bottom-right corner x2 y2 of the bouding box. Specifically, the output should look like "x1 y1 x2 y2"."""
        ]

    elif task == 'VG_selection':  # same as region_caption
        instructs = [
            f"""We give you a caption for part of the image specified by a bounding box. A bounding box is an imaginary rectangle that outlines part of the scene in an image. We provide you with some candidate bounding boxes in the options. The format of the bounding boxes is x1 y1 x2 y2. x1 y1 denotes the coordinate of top-left corner and x2 y2 denotes the bottom-right corner of a bounding box. You need to select the bounding box that aligns best with the given caption.""",
        ]

    elif task == 'object_grounding':
        instructs = [
            f"""In this task, we ask you to identify the object in a region of an image. The region is specified via the coordinates of a rectangle. The input format of the region is x1 y1 x2 y2 where x1 y1 denotes the coordinate of the top-left corner and x2 y2 denotes the coordinate of the bottom-right corner of the rectangle. You need to first localize the rectangular region in the image and then identify what is the main object in that region."""
        ]
    elif task == 'object_region_match':
        instructs = [
            f"""In this task, we will provide you with an object name and a bounding box, and you will decide if the object that we give you is the same object in the bounding box. We specify the bounding box via two coordinates x1 y1 and x2 y2 which denotes the position of top-left corner and the bottom-right corner of the rectangular bounding box, respectively. Instead of answering the question by using your own words, select the best answer from options.""",
        ]
    elif task == 'object_match':
        instructs = [
            f"""In this task, we provide you with two rectangular regions (i.e., region 1 and region 2) in an image and you will decide if the object in region 1 is the same as the object in region 2. Each region is a imaginary rectangular box in the image that outlines an object. The region format is x1 y1 x2 y2 where x1 y1 denotes the coordinate of the top-left corner and x2 y2 denotes the coordinate of the rectangle. You should select the best answer from options."""
        ]
    elif task == 'question_image_match':
        instructs = [
            f"""In this task, you need to decide if the image contains enough information to answer a visual question. You will select your answer from options.""",
        ]
    elif task == 'object_region_selection':
        instructs = [
            f"""In this task, we provide you with an object name and you need to select the region from options that contains the object. We define a region as a rectangular box and the format is x1 y1 x2 y2 where x1 y1 denotes the coordinate of the top-left corner and x2 y2 denotes the coordinate of the rectangle.""",
        ]
    # modify
    elif task == 'missing_object_selection':
        instructs = [
            f"""In this task, we provide you with some regions and you need to select an object from options that do not appear in any of the region. A region is a rectangular box and the format is x1 y1 x2 y2 where x1 y1 denotes the coordinate of the top-left corner and x2 y2 denotes the coordinate of the rectangle. Select "None" if all the objects appear in all regions.""",
        ]
    elif task == 'ITM':
        instructs = [
            f"""In this task you are given some text and you need to decide if the text describe the image.""",
        ]
    # modify
    elif task == 'region_object_selection':
        instructs = [
            f"""Select objects from the options that appear in at least one of the regions. Select "None" if you can't find any object that appears in any region.""",
        ]
    # modify
    elif task == 'region_generation':  # mscoco
        instructs = [
            f"""In this task, you are asked to identify all the regions in the image that contain the given object. The region is defined as a rectangular box and the format is x1 y1 x2 y2 where x1 y1 denotes the coordinate of the top-left corner and x2 y2 denotes the coordinate of the bottom-right corner the rectangle. In your output you should use a single space to separate the regions.""",
        ]

    elif task == 'region_caption_match':
        instructs = [
            f"""We provide you with a natural langugae sentence and the coordinates of a region in the image. You need to decide if the sentence matches the content in the region. Do not consider the content of the image outside of the region. The region is defined as a rectangular box and the format is x1 y1 x2 y2 where x1 y1 denotes the coordinate of the top-left corner and x2 y2 denotes the coordinate of the rectangle.""",
        ]
    elif task == 'object_relationship':
        instructs = [
            f"""In this task, you will decide the relationship between two objects in the image. The relationship can be about the relative position of two objects such as "on", "in", "behind" ... and the action of one object performing to another object such as "taking", "researching for", "feeding"... One of the objects is the "subject" and the other object is the "object" in the relationship. The two objects are specified via their regions. A region is an imaginary rectangular box in the image. The format of the box is x1 y1 x2 y2 where x1 y1 denotes the coordinate of the top-left corner and x2 y2 denotes the coordinate of the bottom-right corner of the box."""
        ]
    elif task == 'visual_object_identification':
        instructs = [
            f"""In this task, you need to predict the name of the "object" given a "subject" in a visual relationship. The subject is specified via its bounding box which is a rectangualr region in the image. The rectangle is specified via x1 y1 x2 y2 where x1 y1 denotes the coordinate of the top-left corner and x2 y2 denotes the coordinate of the bottom-right corner."""
        ]
    elif task == 'visual_subject_identification':
        instructs = [
            f"""In this task, you need to predict the name of the "subject" given an "object" in a visual relationship. The object is specified via its bounding box which is a rectangualr region in the image. The rectangle is specified via x1 y1 x2 y2 where x1 y1 denotes the coordinate of the top-left corner and x2 y2 denotes the coordinate of the bottom-right corner.""",
        ]
    elif task == 'visual_object_region':
        instructs = [
            f"""In this task, you need to predict the region of the "object" given the name of a "subject" in a visual relationship. The subject is specified via its bounding box which is a rectangualr region in the image. The rectangle is specified via x1 y1 x2 y2 where x1 y1 denotes the coordinate of the top-left corner and x2 y2 denotes the coordinate of the bottom-right corner.""",
        ]
    elif task == 'visual_subject_region':
        instructs = [
            f"""In this task, you need to predict the region of the "subject" given an "object" in a visual relationship. The object is specified via its bounding box which is a rectangualr region in the image. The rectangle is specified via x1 y1 x2 y2 where x1 y1 denotes the coordinate of the top-left corner and x2 y2 denotes the coordinate of the bottom-right corner."""
        ]
    elif task == 'descriptive_object_region_generate':
        instructs = [
            f"""In this task, we provide you a description of an object in the image. The description is about distinct features such as "shape", "color", and "position" of the object. You need to identify a single object in the image that satisfies the description. Once you identify the object, output its region in the format of x1 y1 x2 y2 where x1 y1 denotes the coordinate of the top-left corner and x2 y2 denotes the coordinate of the bottom-right corner of the bounding box.""",
        ]
    elif task == 'descriptive_object_region_select':
        instructs = [
            f"""In this task, we provide you a description of an object in the image. The description is about distinct features such as "shape", "color", and "position" of the object. You need to identify a single object in the image that satisfies the description. Once you identify the object, output its region in the format of x1 y1 x2 y2 where x1 y1 denotes the coordinate of the top-left corner and x2 y2 denotes the coordinate of the bottom-right corner of the bounding box.""",
        ]
    elif task == 'object_description_generate':
        instructs = [
            f"""In this task, we ask you to generate a referring expression about an object in the image. The referring expression is a natual language sentence that describes the distinct properties of an object so that people can identify the object by reading it. To aviod confusion, we specify the object via its bounding box. The bounding box has the format x1 y1 x2 y2 where x1 y1 specifies its coordinate of the top-left corner and x2 y2 specifies its coordinate of the bottom-right corner of the rectangular box.""",
        ]
    elif task == 'image_quality':
        instructs = [
            f"""You are given an image and you need to decide the quality issue of the image. If the image is not clear or out of focus, select "blur". If the main object is at a cornern of the image or most part of the main object is not in the image, select "bad framing". If the object of interest in the image is blocked by another object that is very closed to the camera, select "obscured". If the scene or objects in the image is upside down, or in a bad rotated view, select "rotation". If there is no quality issue with the image, select "no flaws". If the quality issue is not listed in the options, select "other"."""
        ]
    elif task == 'text_localization':
        instructs = [
            f"""There is some text on the image and we want you to localize some letters in the text. We provide you with the letters (can be a single letter or a sequence of letters) and you need to select a bounding box from options for the provided letters. The format of the bounding box is x1 y1 x2 y2 where x1 y1 specifies its coordinate of the top-left corner and x2 y2 specifies its coordinate of the bottom-right corner of the rectangular box."""
        ]
    elif task == 'text_legibility':
        instructs = [
            f"""Decide if the text in the given region is legible (clear and complete) or illegible (not clear and not complete). The region is a imaginary rectangle in the image and the format of the region is x1 y1 x2 y2 where x1 y1 specifies its coordinate of the top-left corner and x2 y2 specifies its coordinate of the bottom-right corner.""",
        ]
    elif task == 'text_type':
        instructs = [
            f"""Look at the text in the given region and select the text type from options. The text types are "handwritten" and "machine printed". The region is a imaginary rectangle on the image and the format of the region is x1 y1 x2 y2 where x1 y1 specifies its coordinate of the top-left corner and x2 y2 specifies its coordinate of the bottom-right corner."""
        ]
    elif task == 'region_text_match':
        instructs = [
            f"""In this task, we provide you with some text and an image. You need to decide if the text on part of the image is the same as the text we provided.  We specify the part of the image via a bounding box. A bounding box is a imaginary rectangle and its format is x1 y1 x2 y2. x1 and y1 denotes the coordinate of the top-lefp corner and the x2 y2 denotes the coordinate of the right-bottom corner."""
        ]
    elif task == 'multimodal_factual_checking':
        instructs = [
            f"Determine if the given claim is factually supported by both the image and the context Choose your answer from the provided options."
        ]
    elif task == 'wikihow_next_step':
        instructs = [
            f"Given the task, the history of completed steps, and the current step with its corresponding image, determine the next step for this task. Consider the task's goal and the context provided to generate the appropriate subsequent step.",
        ]
    elif task == 'wikihow_text_image_step_order':
        instructs = [
            f"Given the task and the current step, determine if the content of the image represents the next or previous step in the process. Choose your answer from the provided options.",
        ]
    elif task == 'wikihow_image_text_step_order':
        instructs = [
            f"Given the task and the current step represented by the image, determine if the provided text describes the next or previous step in the process. Consider the overall goal and the context of the step shown in the image to make your decision. Choose your answer from the provided options.",
        ]
    elif task == 'wikihow_immediate_next_step_selection':
        instructs = [
            f"Given the task and the current step represented by the image, identify the immediate next step in the process. Consider the overall goal and the context of the step shown in the image, and select the correct next step from the provided options.",
        ]
    elif task == 'image_text_selection':
        instructs = [f"""Examine the image provided and choose the text from the options that best describes it. Consider the content and context of the image, and select the most accurate caption from the given options.""",
                     ]
    elif task == 'visual_attribute':
        instructs = [
            f"""Examine the image and the specified region within it. A bounding box is an imaginary rectangle defined by coordinates x1 y1 x2 y2, where x1 y1 represents the top-left corner and x2 y2 represents the bottom-right corner. Consider the object's properties and characteristics within the specified region, identify the attribute of the object, and select the correct option from the given choices.""",
        ]
    # image generation tasks
    elif task == 'infilling':
        instructs = [
            f"Examine the image containing a filled black box representing the missing portion. Your task is to generate only the content of the missing part, considering the context and content of the visible area. Do not recreate the entire image with the missing part filled in; focus solely on generating the content for the missing region itself.",
        ]
    elif task == 'im_region_extraction':
        instructs = [
            f"Examine the image and concentrate on the specified region. The region is specified via x1 y1 x2 y2 where x1 y1 denotes the coordinate of the top-left corner and x2 y2 denotes the coordinate of the bottom-right corner. Extract the portion of the original image defined by the given region to create a new image. The extracted image should appear as if it was directly taken from the original image, maintaining the same visual quality and content as the designated area.",
        ]
    elif task == 'im_descriptive_infilling':
        instructs = [
            f"Examine the image containing a filled black box representing the missing or obscured portion. Use the provided description to generate the content of the missing region. Focus solely on generating the content for the missing region, taking into account the context provided by the description. Ensure that the generated content integrates seamlessly with the visible parts of the image.",
        ]
    elif task == 'image_completion':
        instructs = [
            f"Examine the image containing a filled black box, which represents the missing or obscured portion. Using the provided description, generate a complete version of the image that includes the content described by the text in the previously missing area. Make sure the generated content integrates seamlessly with the visible parts of the image, replacing the black box and forming a cohesive and complete image, without altering any existing visible areas.",
        ]
    elif task == 'image_completion_w_region_caption':
        instructs = [
            f"Examine the image with a filled black box representing the missing or obscured portion, along with the provided caption describing the content of the missing area. Based on the caption, generate the content for the missing region and integrate it seamlessly into the image, creating a complete version of the image with the missing area filled in. Ensure that the generated content reflects the caption accurately and integrates well with the visible parts of the image, without creating or modifying any existing visible areas.",
        ]
    elif task == 'image_completion_w_image_caption':
        instructs = [
            f"Examine the image with a missing area represented by a black box. Use the provided image caption to generate a complete version of the image, including the content described by the text. Ensure that the generated content integrates seamlessly with the visible parts of the image, creating a cohesive and complete image without modifying any existing visible areas.",
        ]
    elif task == 'VQA_activity_recognition':
        instructs = [
            f"""Examine the image and identify the activity being performed by the animals or people present in the image, based on the given question. Select the most appropriate answer from the provided options."""
        ]
    elif task == 'VQA_attribute':
        instructs = [
            f"""In this task, you will be asked a question about the attribute of an object in the image. Look at the image and answer the question by identifying the object and selecting its attribute from the given options. The question will ask about a specific attribute of the object, and you must choose the best answer from the options provided."""
        ]
    elif task == 'VQA_color':
        instructs = [
            f"""Given an image, answer the question about the color of a specific object in the image by selecting the best answer from the given options. The question will ask about the color of the object in the image, and you must identify the object first before selecting the correct color option."""
        ]

    elif task == 'VQA_counting':
        instructs = [
            f"""Examine the image and count the number of specific objects as asked in the given question. Your task is to select the correct answer from the given options based on your count."""
        ]

    elif task == 'VQA_object_presence':
        instructs = [
            f"""Given an image and a question asking about the presence of a specific object in the image, select the answer from the given options. The question will include a reference to the object of interest.""",
        ]

    elif task == 'VQA_object_recognition':
        instructs = [
            f"""Examine the image and answer a question about the type or subclass of an object in the image. Choose the best answer from the given options. """

        ]

    elif task == 'VQA_positional_reasoning':
        instructs = [
            f"""Examine the image and analyze the spatial relationships between objects. Based on this analysis, answer the given question about the position of objects within the image. Consider the relative locations of objects in the image and select the best answer from the given options."""
        ]

    elif task == 'VQA_scene_recognition':
        instructs = [
            f"""In this task, you are presented with an image depicting a certain environment or scene. Your goal is to understand the scene in the image and select the correct answer to the provided question, which is related to the overall environment or scene. You should carefully analyze the scene and choose the answer that best fits the provided question from the given options."""
        ]

    elif task == 'VQA_sentiment_understanding':
        instructs = [
            f"""Examine the image and interpret the sentiment depicted within it. Answer the provided question regarding the emotion conveyed in the image, and select the best answer from the given options."""
        ]

    elif task == 'VQA_sport_recognition':
        instructs = [
            f"""Examine the image and answer the question about the sport depicted in it. Choose the correct answer from the given options based on the sports that are taking place in the image."""
        ]

    elif task == 'VQA_utility_affordance':
        instructs = [
            f"""You will answer a question about the utility affordance of an object in the image. Utility affordance refers to the potential usefulness or practical value of an object for achieving a particular goal or fulfilling a specific need.""",
        ]

    elif task == 'select_overlap_most_region':
        instructs = [
            f"""Given the a region, you need to decide which region in the options overlaps most with given region. The region is specified via x1 y1 x2 y2 where x1 y1 denotes the coordinate of the top-left corner and x2 y2 denotes the coordinate of the bottom-right corner. In order to find the region that overlaps most with the given region, you need to compute their overlapped area.""",
        ]
    elif task == 'select_overlap_least_region':
        instructs = [
            f"""Given the a region, you need to decide which region in the options overlaps least with given region. The region is specified via x1 y1 x2 y2 where x1 y1 denotes the coordinate of the top-left corner and x2 y2 denotes the coordinate of the bottom-right corner. In order to find the region that overlaps least with the given region, you need to compute their overlapped area.""",
        ]
    elif task == 'region_area':
        instructs = [
            f"""You are given a bounding box and you need to find the area of the bounding box. The bounding box is specified via x1 y1 x2 y2 where x1 y1 denotes the coordinate of the top-left corner and x2 y2 denotes the coordinate of the bottom-right corner. The region can be compute by the equation: (x2-x1) multiply (y2-y1).""",
        ]
    elif task == 'select_overlaped_region':
        instructs = [
            f"""Given the a region, you need to select a region in the options that overlaps with given region. The region is specified via x1 y1 x2 y2 where x1 y1 denotes the coordinate of the top-left corner and x2 y2 denotes the coordinate of the bottom-right corner. Two regions are overlapped if their overlapped area is not zero.""",

        ]
    elif task == 'select_nonoverlaped_region':
        instructs = [
            f"""Given the a region, you need to select a region in the options that does not overlap with given region. The region is specified via x1 y1 x2 y2 where x1 y1 denotes the coordinate of the top-left corner and x2 y2 denotes the coordinate of the bottom-right corner. Two regions are overlapped if their overlapped area is zero."""
        ]
    elif task == 'if_region_overlap':
        instructs = [
            f"""Given two regions, you need to decide if two regions are overlapped. Two regions are overlapped if their overlapped area is not zero."""
        ]
    # --------------- llava tasks -----------------
    elif task == 'llava_conversation':
        instructs = [
            f"""Given the history of a conversation between a User and an AI Assistant, generate the next response."""
        ]
    elif task == 'llava_detailed_caption':
        instructs = [
            f"""Generate a detailed caption of the image."""
        ]
    elif task == 'llava_complex_reasoning':
        instructs = [
            f"""Answer a complex question about the image. The question can be related to the background knowledge of the objects in the image, or about events happening in the image."""
        ]

    elif task == 'textcaps':
        instructs = [
            f"""Generate a caption involving the text in the image."""
        ]
    elif task == 'scienceqa_exp':
        instructs = [
            f"""given the question and the answer, generate an explanation."""
        ]
    elif task == 'aok_vqa':
        instructs = [
            f"""select the correct answer for the given question."""
        ]
    elif task == 'science_qa':
        instructs = [
            f"""select the correct answer for the given question. the question is about sicence."""
        ]
    elif task == 'visit':
        instructs = [
            f"""generate a detailed answer for the given question."""
        ]
    elif task == 'text_vqa':
        instructs = [
            f"""answer a question about some text in the image."""
        ]
    elif task == 'visual_spatial_reasoning':
        instructs = [
            f"""answer a question about the spatial relationship between objects in the image."""
        ]
    elif task == 'natural_language_visual_reasoning':
        instructs = [
            f"""answer a question about the image."""
        ]
    elif task == 'winoground':
        instructs = [
            f"""select the more accurate caption from two similar captions for the image."""
        ]
    elif task == 'medic_damage_severity':
        instructs = [
            f"""classify the damage severity in the image."""
        ]
    elif task == 'medic_informative':
        instructs = [
            f"""classify if the image is informative about a disaster."""
        ]
    elif task == 'medic_disaster_types':
        instructs = [
            f"""identify the type of disaster happenining in the image."""
        ]
    elif task == 'medic_humanitarian':
        instructs = [
            f"""decide the humanitarian of the image."""
        ]
    elif task == 'aokvqa_rational':
        instructs = [
            f"""given a questions and an answer, explain why it is the answer to the question."""
        ]
    elif task == 'cifar_10':
        instructs = [
            f"""classify the main object in the image."""
        ]
    elif task == 'cifar_100':
        instructs = [
            f"""classify the main object in the image."""
        ]
    elif task == 'miniImagenet':
        instructs = [
            f"""classify the main object in the image."""
        ]
    elif task == 'mnist':
        instructs = [
            f"""classify the number in the image."""
        ]
    elif task == 'pope_adv':
        instructs = [
            f"""decide if the mentioned object is in the image."""
        ]
    elif task == 'pope_pop':
        instructs = [
            f"""decide if the mentioned object is in the image."""
        ]
    elif task == 'pope_rand':
        instructs = [
            f"""decide if the mentioned object is in the image."""
        ]
    elif task == 'snli_ve_answer_choice':
        instructs = [
            f"""select the caption fits the sentence best."""
        ]
    elif task == 'snli_ve_classification':
        instructs = [
            f"""decide if the content of the image support the sentence."""
        ]
    elif task in VISION_FLAN_DEF:
        instruct = VISION_FLAN_DEF.get(
            task, 'Answer the following question based on the image.')
        instructs = [
            instruct
        ]

    elif task == 'object_localization.jsonl':
        instructs = [
            f"""Answer a question about the location of an object in the image."""
        ]
    elif task == 'image_style.jsonl':
        instructs = [
            f"""Identify the art style of this image."""
        ]
    elif task == 'celebrity_recognition.jsonl':
        instructs = [
            f"""recognize the celebrity in the image."""
        ]
    elif task == 'physical_property_reasoning.jsonl':
        instructs = [
            f"""Read a passage and answer a question about the physical property of an object in the image."""
        ]
    elif task == 'image_quality.jsonl':
        instructs = [
            f"""select the image with the quality mentioned in the text."""
        ]
    elif task == 'function_reasoning.jsonl':
        instructs = [
            f"""identify the function of the demonstrated object."""
        ]
    elif task == 'attribute_comparison.jsonl':
        instructs = [
            f"""Comparing attributes of two objects."""
        ]
    elif task == 'nature_relation.jsonl':
        instructs = [
            f"""Decide the nature relations of these animals or humans in the image."""
        ]
    elif task == 'identity_reasoning.jsonl':
        instructs = [
            f"""Identify the best options based on the image and input text."""
        ]
    elif task == 'image_emotion.jsonl':
        instructs = [
            f"""Identify the emotion in the image."""
        ]
    elif task == 'image_topic.jsonl':
        instructs = [
            f"""select the best caption describing the image."""
        ]
    elif task == 'future_prediction.jsonl':
        instructs = [
            f"""predict a future event based on the image."""
        ]
    elif task == 'ocr.jsonl':
        instructs = [
            f"""Answer a question about the text in the image."""
        ]
    elif task == 'structuralized_imagetext_understanding.jsonl':
        instructs = [
            f"""Answer a question about a chart of a table which has structured text on it."""
        ]
    elif task == 'physical_relation.jsonl':
        instructs = [
            f"""answer a question about the physical relationship between objects in the image."""
        ]
    elif task == 'image_scene.jsonl':
        instructs = [
            f"""select the best caption describing the image."""
        ]
    elif task == 'attribute_recognition.jsonl':
        instructs = [
            f"""recognize the attributes of an object in the image."""
        ]
    elif task == 'spatial_relationship.jsonl':
        instructs = [
            f"""answer a question about the saptial relationship in the image."""
        ]
    elif task == 'social_relation.jsonl':
        instructs = [
            f"""decide the social relationship between the two persons in this image."""
        ]
    elif task == 'action_recognition.jsonl':
        instructs = [
            f"""Decide what kind of human behavior does this picture describe."""
        ]
    elif task == 'mm_vet.jsonl':
        instructs = [
            f"""Perform the task based on the instruction. Some of the questions can be answered with short phrases and some other more open-ended questions require you to generate detailed respones."""
        ]

    elif task == 'landmark.jsonl':
        instructs = [
            f"""Answer a yes-or-no question about the landmark in the image."""
        ]
    elif task == 'text_translation.jsonl':
        instructs = [
            f"""Decide if the translation of some text in the image is correct or not."""
        ]
    elif task == 'color.jsonl':
        instructs = [
            f"""Answer a yes-or-no question about the color of an object in the image."""
        ]
    elif task == 'celebrity.jsonl':
        instructs = [
            f"""Answer a yes-or-no question about the celebrity in the image."""
        ]
    elif task == 'scene.jsonl':
        instructs = [
            f"""Answer a yes-or-no question about the scene in the image."""
        ]
    elif task == 'numerical_calculation.jsonl':
        instructs = [
            f"""Decide if the answer to a arithmetic question shown in the image is correct or not."""
        ]
    elif task == 'commonsense_reasoning.jsonl':
        instructs = [
            f"""Answer a yes-or-no question related to commonsense reasoning about the image."""
        ]
    elif task == 'code_reasoning.jsonl':
        instructs = [
            f"""Answer a yes-or-no question about some computer code in the image."""
        ]
    elif task == 'count.jsonl':
        instructs = [
            f"""Answer a yes-or-no question about counting objects in the image."""
        ]
    elif task == 'OCR.jsonl':
        instructs = [
            f"""Answer a yes-or-no question about the text in the image."""
        ]
    elif task == 'existence.jsonl':
        instructs = [
            f"""Decide the presence of an object in the image."""
        ]
    elif task == 'artwork.jsonl':
        instructs = [
            f"""Answer a yes-or-no question about an artwork."""
        ]
    elif task == 'posters.jsonl':
        instructs = [
            f"""Answer a yes-or-no question about a poster."""
        ]
    elif task == 'position.jsonl':
        instructs = [
            f"""Answer a yes-or-no question about the position of an object in the image."""
        ]

    # elif task == 'Scene_Understanding':
    # elif task == 'Visual_Reasoning':
    # elif task == 'Instances_Counting':
    # elif task == 'Instance_Interaction':
    # elif task == 'Instance_Attributes':
    # elif task == 'Text_Understanding':
    # elif task == 'Instance_Identity':
    # elif task == 'Instance_Location':
    # elif task == 'Spatial_Relation':

    else:
        instructs = [
            f"""None"""
        ]
        print(
            f'warning: {task} does not have a valid definition. plz write it!!!!')

    return instructs[0]


def get_taks_id(task_name):
    task_id_dict = {'GQA': 0, 'VQAv2': 1, 'visualgenome_vqa': 2, 'ok_vqa': 3, 'image_quality': 4, 'VQA_attribute': 5, 'image_text_selection': 6, 'VQA_counting': 7, 'wikihow_immediate_next_step_selection': 8, 'VQA_scene_recognition': 9, 'VQA_object_presence': 10, 'wikihow_next_step': 11, 'VQA_positional_reasoning': 12, 'wikihow_text_image_step_order': 13, 'question_image_match': 14,
                    'ITM': 15, 'VQA': 16, 'VQA_object_recognition': 17, 'wikihow_image_text_step_order': 18, 'VQA_color': 19, 'VQA_sport_recognition': 20, 'multimodal_factual_checking': 21, 'VQA_activity_recognition': 22, 'image_caption': 23, 'VQA_sentiment_understanding': 24, 'VQA_utility_affordance': 25, 'llava_conversation': 26, 'llava_detailed_caption': 27, 'llava_complex_reasoning': 28}
    if task_name in task_id_dict:
        return task_id_dict[task_name]
    elif task_name in VISION_FLAN_ID:
        return VISION_FLAN_ID[task_name]
    else:
        print(f"warning: {task_name} does not have a valid task id")
        return -1


def get_new_test_list(test_type, question_dir=None):
    test_type2list = {
        'mme': ['landmark.jsonl', 'text_translation.jsonl', 'color.jsonl', 'celebrity.jsonl', 'scene.jsonl', 'numerical_calculation.jsonl', 'commonsense_reasoning.jsonl', 'code_reasoning.jsonl', 'count.jsonl', 'OCR.jsonl', 'existence.jsonl', 'artwork.jsonl', 'posters.jsonl', 'position.jsonl'],
    }
    question_dir = Path(question_dir)
    test_file_path = {
        'mme': question_dir / 'mme/MME_Benchmark_release_version',
    }

    return test_file_path[test_type], test_type2list[test_type]


def get_mme_test_list(question_dir=None):
    test_type_list = ['landmark.jsonl', 'text_translation.jsonl', 'color.jsonl', 'celebrity.jsonl', 'scene.jsonl', 'numerical_calculation.jsonl',
                      'commonsense_reasoning.jsonl', 'code_reasoning.jsonl', 'count.jsonl', 'OCR.jsonl', 'existence.jsonl', 'artwork.jsonl', 'posters.jsonl', 'position.jsonl']
    question_dir = Path(question_dir)
    test_file_path = question_dir / 'MME_Benchmark_release_version'

    return test_file_path, test_type_list


if __name__ == "__main__":
    get_taks_id('PlotQA+visual_question_answering')
