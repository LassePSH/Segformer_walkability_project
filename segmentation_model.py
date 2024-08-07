import torch 
import torchvision
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import requests
from tqdm import tqdm
from segmentation_dataset_def import Dataset
from torch.utils.data import DataLoader
import os

def id_to_PIL(file_id):
        path = '/scratch/lpsha/data/img/' #Titans
        
        file_path = path+str(file_id)+'.jpg'
        PIL_image = Image.open(file_path)

        if PIL_image.mode != 'RGB':
            PIL_image=PIL_image.convert('RGB')

        return PIL_image

def explode_array_column(row):
    return pd.Series(row['vector'])

def epoch(model,data_loader,device):
    df_out = pd.DataFrame(columns=['img_id', 'vector'])
    model = model.eval()

    with torch.no_grad():
        for d in tqdm(data_loader):
            img_ids = d['img_id']
      
            input_images = []     
            for im_id in img_ids:
                try: input_images.append(id_to_PIL(im_id))
                except: print('Failed to load: ' + str(img_id)) 

            tensor_feat = feature_extractor(images=input_images, return_tensors="pt")
            tensor_input = tensor_feat.to(device)

            outputs = model(**tensor_input)
            logits= outputs.logits
            armax = torch.argmax(logits,dim=1)
            batch_size = armax.size(0)

            vectors = []
            for i in range(batch_size):
                empty_vector = np.zeros(19)
                c = armax[i].cpu().numpy().ravel()
                unique_values_index, counts = np.unique(c, return_counts=True)
                density_vector = counts/counts.sum()
                empty_vector[unique_values_index] = density_vector
                vectors.append(empty_vector)
            
            data = {'img_id': img_ids, 'vector': vectors}
            iteration_df = pd.DataFrame(data)
            df_out=pd.concat([df_out, iteration_df])
        
        # expand vector
        expanded_cols = df_out.apply(explode_array_column, axis=1)
        expanded_cols.columns = ['class_{}'.format(i) for i in range(expanded_cols.shape[1])]
        df_out = pd.concat([df_out, expanded_cols], axis=1)

        df_out.to_csv('feature_vector.csv',index=False)

if __name__ == "__main__":

    ## LOAD MODEL ##
    feature_extractor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b1-finetuned-cityscapes-1024-1024")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b1-finetuned-cityscapes-1024-1024")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device: ',device)
    
    if torch.cuda.is_available():
        model.cuda()

    ## LOAD DATA ##
    df=pd.read_csv('/home/lpsha/data/img_features_s.csv') # titans
    print('Data shape: ', df.shape)

    df = df.loc[df.camera_type == 'perspective']
    print('Data shape: ', df.shape)

    is_ids = os.listdir('/scratch/lpsha/data/img/')
    df['img_id']=df['id'].astype(int).astype(str)
    df=df.loc[(df['img_id']+'.jpg').isin(is_ids)]
    print('Data shape: ', df.shape)

    # dataloader
    id_dataset = Dataset(img_ids=df['id'].to_numpy()) 
    id_dataloader = DataLoader(id_dataset, batch_size=15,num_workers=1)

    ## RUN ##
    epoch(model, id_dataloader, device)(base)
