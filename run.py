import torch
import torch.nn as nn
import os
#到时候文件夹只需要保留run.py,deeplabv3.py,build_aspp.py,build_backbone.py,build_decoder.py
from deeplabv3 import DeepLab
from PIL import Image
from os.path import splitext
from glob import glob
import logging
import numpy as np
from os import listdir
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from dataset_ import BasicDataset
import time


#预测函数
def predict_img(net,full_img,device,scale_factor=1):

    net.eval()
    #将数组转换为tensor
    img = torch.from_numpy(BasicDataset.preprocess(full_img,scale_factor))
    img = img.unsqueeze(0)
    img = img.to(device=device,dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        probs = torch.sigmoid(output)
        probs = probs.squeeze(0)
        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(full_img.size[1]),
            transforms.ToTensor()
        ])
        probs = tf(probs.cpu())
        full_mask = probs.squeeze().numpy()
    return full_mask



if __name__ == '__main__':


    start = time.time()
    #文件存放路径
    in_files_path = '../input_path/'
    out_files_path = '../output_path/'
    in_put = os.listdir(in_files_path)
    out_files = []
    in_files = []
    for f in in_put:
        pathsplit = os.path.splitext(f)
        in_files.append(in_files_path+f)
        out_files.append(out_files_path+pathsplit[0]+'.png')

    net = DeepLab(backbone='mobilenet',output_stride=16)
    model = './param.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.to(device)
    net.load_state_dict(torch.load(model,map_location=device),False)

    for ii,fn in enumerate(in_files):
        img = Image.open(fn)
        img_rows,img_cols = img.size
        mask = predict_img(net=net,full_img=img,device=device,scale_factor=1)
        mask = mask > 0.5

        out_fn = out_files[ii]

        result = Image.fromarray((mask*255).astype(np.uint8))
        result = result.resize((img_rows,img_cols),Image.ANTIALIAS)
        rows,cols = result.size
        pixel = result.load()
        for i in range(rows):
            for j in range(cols):
                if pixel[i,j] <= 127:
                    pixel[i,j] = 0
                else:
                    pixel[i,j] = 255
        result.save(out_fn)


