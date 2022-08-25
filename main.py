import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from sys import argv
from tqdm import tqdm
import torchvision.transforms as transforms
from collections import OrderedDict

from models import F3Net

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class VIPDataset(Dataset):
    def __init__(self, file_ids, data_dir='data'):
        self.file_ids = file_ids
        self.data_dir = data_dir
        self.transform = transforms.PILToTensor()

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, index):
        image_id = self.file_ids[index]
        x = self.transform(Image.open(os.path.join(self.data_dir, image_id)))
        x = 2*x.float()/255.0 - 1
        return x

if __name__ == '__main__':
    input_csv = argv[1]
    output_csv = argv[2]

    test_list = pd.read_csv(input_csv)['filename'].to_numpy()

    test_ds = VIPDataset(test_list)

    batch_size = 3
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=1)

    model = F3Net()
    state_dict = torch.load('code/checkpoints/VIP-32.pth')
    state_dict = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    output = []
    for data in tqdm(test_dl):
        stu_fea, stu_cla = model(data.to(device))
        output.extend(torch.argmax(stu_cla.sigmoid(), dim=-1).flatten().tolist())
    output = np.array(output)
    output = np.vstack([test_list, output])
    pd.DataFrame(output.T, columns=['filename', 'logit']).to_csv(output_csv, index=False)
