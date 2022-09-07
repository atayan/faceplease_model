import os
import torch
import torchvision.transforms as transforms
import Nets
import io
import socket
import struct

from PIL import Image


UNIX_SOCK_ADDR = '/code/socket.s'


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

    def read_img(self, fp):
        img = Image.open(fp).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def load_model(self, pretrained_dict, new):
        model_dict = new.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict['state_dict'].items() if k in model_dict}
        model_dict.update(pretrained_dict)
        new.load_state_dict(model_dict)

    def calculate_score(self, fp):
        img = self.read_img(fp)
        with torch.no_grad():
            img = img.unsqueeze(0)
            output = self.net(img).squeeze(1)
            prediction = round(float(output.cpu()[0]), 2)
        return prediction


def main():
    try:
        os.unlink(UNIX_SOCK_ADDR)
    except OSError:
        if os.path.exists(UNIX_SOCK_ADDR):
            raise

    rater = FaceRater()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(UNIX_SOCK_ADDR)
    server.listen(100)
    while True:
        conn, _ = server.accept()
        try:
            sz_data = struct.unpack('>I', conn.recv(4))
            sz_data = sz_data[0]
            buf = []
            while len(buf) < sz_data:
                datagram = conn.recv(2 ** 20)
                buf.extend(datagram)
            bts = io.BytesIO(''.join(buf))
            score = rater.calculate_score(bts)
            conn.sendall(struct.pack('>d', score))
        finally:
            conn.close()
    

if __name__ == '__main__':
    main()
