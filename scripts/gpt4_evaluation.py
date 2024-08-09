import json
import openai
import sys
import pdb
from tqdm import tqdm


SYS_PROMPT = "Given a question about an image, a reference answer, and two predictions from two vision-language models, you are asked to decide which prediction is more accurate based on the reference answer. Please note being more accurate is more important than being comprehensive so do not bias towards the longer prediction."

# USER_PROMPT = f"The question is: {}\nThe reference answer is: {}\nPrediction 1: {}\n Prediction 2: {}. If you think Prediction 1 is more accurate based on the reference answer, output Prediction 1. If you think Prediction 2 is more accurate based on the reference answer, output Prediction 2. If you think two predictions are equally accurate based on the reference answer, output Equal."

input_path_1 = sys.argv[1]
input_path_2 = sys.argv[2]

predictions_1 = [ json.loads(line) for line in open(input_path_1, 'r')]
predictions_2 = [ json.loads(line) for line in open(input_path_2, 'r')]

def gpt_eval(q, ref, pred1, pred2):
    USER_PROMPT = f"The question is: {q}\nThe reference answer is: {ref}\nPrediction 1: {pred1}\n Prediction 2: {pred2}. If you think Prediction 1 is more accurate based on the reference answer, output Prediction 1. If you think Prediction 2 is more accurate based on the reference answer, output Prediction 2. If you think two predictions are equally accurate based on the reference answer, output Equal. Note: Do not output any other text than \"Prediction 1\"; \"Prediction 2\"; \"Equal\". Accuracy is more important than comprehensiveness so do not bias towards the longer prediction"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": USER_PROMPT }
        ]
    )
    return response.choices[0].message.content

results = []
for line1, line2 in tqdm(zip(predictions_1, predictions_2)):    
    response = gpt_eval(line1['prompt'], line1['target'], line1['predict'], line2['predict'])
    results.append(response)
    
pdb.set_trace()
    
    

