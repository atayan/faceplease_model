import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import Nets
from os import listdir
from os.path import isfile, join
import csv
from itertools import izip


class FaceRater():
    def __init__(self):
        self.net = Nets.ResNet(block = Nets.BasicBlock, layers = [2, 2, 2, 2], num_classes = 1)
        self.load_model(torch.load('./models/resnet18.pth', map_location=torch.device('cpu')), self.net)
        self.net.eval()
        self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])  

    def read_img(self, filepath):
        img = Image.open(filepath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def load_model(self, pretrained_dict, new):
        model_dict = new.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict['state_dict'].items() if k in model_dict}
        model_dict.update(pretrained_dict)
        new.load_state_dict(model_dict)

    def calculate_score(self, path):
        img = self.read_img(path)
        with torch.no_grad():
            img = img.unsqueeze(0)
            output = self.net(img).squeeze(1)
            prediction = round(float(output.cpu()[0]), 2)
        return prediction


def main():
    dir_path = "faces"
    images = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]

    rater = FaceRater()
    res = []
    print("total images count: {}".format(str(len(images))))
    for i, image in enumerate(images):
        path = os.path.join(dir_path, image)
        score = rater.calculate_score(path)
        print(i)
        res.append((path, score))
    
    with open('results.csv', 'wb') as f:
        writer = csv.writer(f)
        # for path, score in res:
        writer.writerows(res)
    

if __name__ == '__main__':
    main()
