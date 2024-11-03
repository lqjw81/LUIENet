import sys
sys.path.append('/home/work/data/myfive-org')
# from archs.FIVE_APLUS import FIVE_APLUSNet
from archs.use_gabor import FIVE_APLUSNet

import torch
import cv2,datetime,os
import argparse
from tqdm import tqdm
from torchvision import transforms
import numpy as np
from os.path import join
satellite = 'Five'
epoch = '480'

def tensor2img(img):
    img = img.data.cpu().numpy()
    img[img > 1] = 1
    img[img < 0] = 0
    img *= 255
    img = img.astype(np.uint8)[0]
    img = img.transpose((1, 2, 0))
    return img

def img2tensor(np_img):# [h,w,c]
    tensor = get_transforms()(np_img).cuda() # [c,h,w] [-1,1]
    tensor = tensor.unsqueeze(0) # [b,c,h,w] [-1,1]
    return tensor

def get_transforms():
    transform = transforms.Compose([
        transforms.ToTensor(),# H,W,C -> C,H,W && [0,255] -> [0,1]
    ])
    return transform

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument('--img_folder',type=str,default=r'\test_data\input/',help='input image path')
parser.add_argument('--load_modke_folder',type=str,default=r'\results\model_save/',help='output folder')
parser.add_argument('--output_folder',type=str,default=r'\results\output_test-60/',help='output folder')
args = parser.parse_args()

if __name__ == "__main__":
    net = FIVE_APLUSNet().to(device)
    net.eval()
    with torch.no_grad():
        # checkpoint_model = join(args.load_modke_folder, '{}-model-epochs-bestSSIM{}.pth'.format(satellite, epoch))
        checkpoint_model = join(args.load_modke_folder, '{}-model-epochs{}.pth'.format(satellite, epoch))
        checkpoint = torch.load(checkpoint_model, map_location='cpu')
        net.load_state_dict(checkpoint['model'])
        img_folder = args.img_folder
        pbar = tqdm(os.listdir(img_folder))

        for img_name in os.listdir(img_folder):
            img_path_raw = os.path.join(img_folder, img_name)
            img_raw = cv2.cvtColor(cv2.imread(img_path_raw), cv2.COLOR_BGR2RGB)
            img_raw = cv2.resize(img_raw, (256, 256))
            img_raw_tensor = img2tensor(img_raw)
            output_tensor, no_use = net.forward(img_raw_tensor)
            output_img = tensor2img(output_tensor)
            save_folder = args.output_folder
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            save_path = os.path.join(save_folder, img_name)
            cv2.imwrite(save_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
            pbar.update(1)
