import pandas as pd 
import numpy as np 
import cv2 
import torch 
import os 
import json 
from sys import argv
from tqdm import tqdm
from models.MAT import MAT

input_csv = argv[1]
output_csv = argv[2]

root_db = os.path.dirname(input_csv)
tab = pd.read_csv(input_csv)

def load_sample(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = cv2.resize(image, (200, 200))
    image = np.transpose(image, (2,0,1))[np.newaxis, ...]
    image_tensor = torch.from_numpy(image).float()
    return image_tensor

def load_model(chk_path, config_path):
    with open(config_path, 'r') as pf:
        model_config = json.load(pf)

    model = MAT(**model_config)
    model_checkpoint = torch.load(chk_path)
    model.load_state_dict(model_checkpoint['state_dict'])
    model.eval()

    return model 

def get_pred(model, image):
    sample_output = torch.softmax(model(image), dim=-1)
    label = torch.argmax(sample_output).numpy()
    return label.numpy()

model = load_model('./data/ckpt_4.pth', './data/model_config.json')

for index, dat in tqdm(tab.iterrows(), total=len(tab)):
    filename = os.path.join(root_db, dat['filename'])
    sample = load_sample(filename)
    logit = get_pred(model, sample)
    tab.loc[index, 'logit'] = logit 

tab.to_csv(output_csv, index=False)
