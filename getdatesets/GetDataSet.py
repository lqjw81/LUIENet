import os
import random
import cv2
import torchvision.transforms as tfs

from PIL import Image
from torchvision.transforms import functional as FF
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

def get_transforms():
    # transformes.Compose用来串联多个图片变换操作。
    transform = transforms.Compose([
        transforms.ToTensor(),  # H,W,C -> C,H,W && [0,255] -> [0,1]
        transforms.Resize([256, 256])
    ])
    return transform

#首先，在初始化函数中调用get_imglist函数获取文件夹路径下的所有文件的列表。
#然后，在Dataloader类实例化以后的对象调用时，就会用过[index]的方式使用__getitem__函数
#获取文件路径，获取的个数根据batch_size来定。
class MYDataSet(Dataset):
    def __init__(self, src_data_path, dst_data_path, train_flag):

        self.train_A_imglist = self.get_imglist(src_data_path)
        self.train_B_imglist = self.get_imglist(dst_data_path)
        self.transform = get_transforms()
        self.train_flag = train_flag

    def get_imglist(self, img_dir):
        img_name_list = sorted(os.listdir(img_dir))
        img_list = []
        for img_name in img_name_list:
            img_path = os.path.join(img_dir, img_name)
            img_list.append(img_path)
        return img_list

    def __len__(self):
        return len(self.train_A_imglist)

    def __getitem__(self, index):
        if self.train_flag:

            rand_hor = random.randint(0,1)
            rand_rot = random.randint(0,3)

            train_A_img_path = self.train_A_imglist[index]
            train_B_img_path = self.train_B_imglist[index]

            train_A_img = Image.open(train_A_img_path)
            train_B_img = Image.open(train_B_img_path)

            train_A_img = train_A_img.convert('RGB')
            train_B_img = train_B_img.convert('RGB')

            train_A_img = tfs.RandomHorizontalFlip(rand_hor)(train_A_img)
            train_B_img = tfs.RandomHorizontalFlip(rand_hor)(train_B_img)

            if rand_rot:
                train_A_img = FF.rotate(train_A_img, 90 * rand_rot)
                train_B_img = FF.rotate(train_B_img, 90 * rand_rot)

            train_A_tensor = self.transform(train_A_img)
            train_B_tensor = self.transform(train_B_img)

            return [train_A_tensor, train_B_tensor]

        else:
            train_A_img_path = self.train_A_imglist[index]
            train_B_img_path = self.train_B_imglist[index]

            train_A_img = cv2.imread(train_A_img_path)
            train_B_img = cv2.imread(train_B_img_path)

            train_A_img = cv2.cvtColor(train_A_img, cv2.COLOR_BGR2RGB) #颜色空间转换函数 cv2.cvtColor
            train_B_img = cv2.cvtColor(train_B_img, cv2.COLOR_BGR2RGB)

            train_A_tensor = self.transform(train_A_img)
            train_B_tensor = self.transform(train_B_img)

            return [train_A_tensor, train_B_tensor]
