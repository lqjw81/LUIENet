import sys
sys.path.append('/home/work/data/myfive-org')
import os
from archs.use_gabor import FIVE_APLUSNet
from torch import optim
from tqdm import tqdm

from utils.image_utils import torchPSNR, torchSSIM

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from torch.utils.data import DataLoader
import time
from getdatasets.GetDataSet import MYDataSet
from os.path import join
from losses.CL1 import L1_Charbonnier_loss
from losses.Perceptual import PerceptualLoss
from losses.SSIMLoss import SSIMLoss
from config.config import FLAGES
# from archs.FIVE_APLUS import FIVE_APLUSNet

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
## 超参数设置
satellite = 'Five'
start_epochs = 0
total_epochs = 500
# model_backup_freq = 2
model_backup_freq = 1
num_workers = 4


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_dataset = MYDataSet(src_data_path=FLAGES.trainA_path, dst_data_path=FLAGES.trainB_path, train_flag=True)
train_datasetloader = DataLoader(train_dataset, batch_size=FLAGES.train_batch_size, shuffle=True, num_workers=num_workers)

val_dataset = MYDataSet(src_data_path=FLAGES.valA_path, dst_data_path=FLAGES.valB_path, train_flag=False)
val_datasetloader = DataLoader(val_dataset, batch_size=FLAGES.val_batch_size, shuffle=False, num_workers=num_workers)

loss_f = L1_Charbonnier_loss()
ssim_loss = SSIMLoss()
loss_per = PerceptualLoss()

class DataPrefetcher():
# 这个类使得在模型训练时可以异步加载下一个数据批次，以减少训练过程中的等待时间。
    def __init__(self, loader):
        self.loader = iter(loader)
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return

    def next(self):
        batch = self.batch
        self.preload()
        return batch

# 实例化模型
model = FIVE_APLUSNet()
# 将模型加载到设备上，用的cpu就放到cpu，gpu就放到gpu
model.to(device)
# 定义一个优化器，传入要更新的模型参数、超参数以及学习率等。
optimizer = torch.optim.Adam(model.parameters(), lr=FLAGES.lr)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0004, max_lr=1.2 * 0.0004,
                                              cycle_momentum=False)
# optimizer = torch.optim.Adam(model.parameters(), lr=FLAGES.lr, betas=[0.9, 0.999])  # ,weight_decay=self.weight_decay)
# optimizer = optim.Adam(model.parameters(), lr=FLAGES.lr, betas=(0.9, 0.999), eps=1e-8)

warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epochs, eta_min=1e-6)
# scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
#                                    after_scheduler=scheduler_cosine)

# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0004, max_lr=1.2 * FLAGES.lr, cycle_momentum=False)
# @torchsnooper.snoop()
# 模型训练
def train(model, train_datasetloader, start_epoch):
    log = open("./logs.txt", mode="a", encoding="utf-8")
    best_psnr, best_ssim, best_epoch_ssim, best_epoch_psnr = 0., 0., 0., 0.
    print('===>Begin Training!')
    # 如果模型中用到了BN和Dropout，则需要使用model.train()启用，
    # 而在测试时怎用model.eval()关闭启用，以此保证BN层的均值和方差不变且不丢弃神经元。
    model.train()
    steps_per_epoch = len(train_datasetloader) #len(datasetloader)就是我以batchsize大小要多少次才能遍历完数据集。
    total_iterations = total_epochs * steps_per_epoch #遍历的总次数，total_epoch是要对数据集读取多少遍*没遍要以batchsize大小读多少次。
    print('total_iterations:{}'.format(total_iterations))
    # train_loss_record = open('%s/train_loss_record.txt' % FLAGES.record_dir, "w")
    for epoch in range(start_epoch + 1, total_epochs + 1):
        start = time.time()  # 记录每轮训练的开始时刻
        prefetcher_train = DataPrefetcher(train_datasetloader)  # 预加载数据
        data = prefetcher_train.next()
        print('Fetching training UIEB spends {} seconds'.format(time.time()-start))
        while data is not None:
            # 将数据放到设备上
            raw, label = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()  #清空梯度

            pred, predforvgg = model(raw)  #模型训练，获取结果

            #计算损失
            loss_d = loss_f(pred, label)
            # loss_d = loss_f(pred, label)
            train_loss = loss_d+0.2*loss_per(pred, label)+0.5*ssim_loss(pred, label)+0.2*loss_per(predforvgg, label)
            # train_loss = ssim_loss(pred, label)+loss_d

            train_loss.backward()  #反向传播
            optimizer.step()  #更新梯度
            scheduler.step()

            data = prefetcher_train.next() #预加载数据
            print('=> {}-Epoch[{}/{}]: train_loss: {:.4f}'.format(satellite, epoch, total_epochs, train_loss.item(),))

        ## Evaluation
        if epoch % model_backup_freq == 0:
            model.eval()
            PSNRs = []
            SSIMs = []
            pbar = tqdm(val_datasetloader)
            for ii, data_val in enumerate(pbar, 0):
                input_ = data_val[0].cuda()
                target_enh = data_val[1].cuda()
                restored_enh, temp = model(input_)
                with torch.no_grad():
                    for res, tar in zip(restored_enh, target_enh):
                        temp_psnr = torchPSNR(res, tar)
                        temp_ssim = torchSSIM(restored_enh, target_enh)
                        PSNRs.append(temp_psnr)
                        SSIMs.append(temp_ssim)

            PSNRs = torch.stack(PSNRs).mean().item()
            SSIMs = torch.stack(SSIMs).mean().item()

            # Save the best PSNR model of validation
            if PSNRs > best_psnr:
                best_psnr = PSNRs
                best_epoch_psnr = epoch
                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                torch.save(state, join(FLAGES.backup_model_dir, '{}-model-epochs{}.pth'.format(satellite, epoch)))
            print("[PSNR] {:.4f} [Best_PSNR] {:.4f} (epoch {})".format(PSNRs, best_psnr, best_epoch_psnr))
            print("[PSNR] {:.4f} [Best_PSNR] {:.4f} (epoch {})".format(PSNRs, best_psnr, best_epoch_psnr), file=log)

            # Save the best SSIM model of validation
            if SSIMs > best_ssim:
                best_ssim = SSIMs
                best_epoch_ssim = epoch
                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                torch.save(state, join(FLAGES.backup_model_dir, '{}-model-epochs{}.pth'.format(satellite, epoch)))
            print("[SSIM] {:.4f}  [Best_SSIM] {:.4f} (epoch {})".format(SSIMs, best_ssim, best_epoch_ssim))
            print("[SSIM] {:.4f}  [Best_SSIM] {:.4f} (epoch {})".format(SSIMs, best_ssim, best_epoch_ssim), file=log)


            # Save each epochs of model
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, join(FLAGES.backup_model_dir, '{}-model-epochs{}.pth'.format(satellite, epoch)))


        # backup a model every epoch
        # if epoch % model_backup_freq == 0:  # 每间隔model_backup_freq轮保存一次权重。
        #     model.eval()
        #     PSNRs=[]
        #     SSIMs=[]
        #     prefetcher_val = DataPrefetcher(val_datasetloader)  # 预加载数据
        #     data_val = prefetcher_val.next()
        #     while data_val is not None:
        #         raw, label = data_val[0].to(device), data_val[1].to(device)
        #         pred, predforvgg = model(raw)
        #         data_val = prefetcher_train.next()  # 预加载数据
        #         with torch.no_grad():
        #             for res, tar in zip(pred, label):
        #                 temp_psnr = torchPSNR(res, tar)
        #                 temp_ssim = torchSSIM(pred, label)
        #                 PSNRs.append(temp_psnr)
        #                 SSIMs.append(temp_ssim)
        #
        #     PSNRs = torch.stack(PSNRs).mean().item()
        #     SSIMs = torch.stack(SSIMs).mean().item()
        #
        #     # Save the best PSNR model of validation
        #     if PSNRs > best_psnr:
        #         best_psnr = PSNRs
        #         best_epoch_psnr = epoch
        #         state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        #         torch.save(state, join(FLAGES.backup_model_dir, '{}-model-epochs{}-bestPSNR{}.pth'.format(satellite, epoch, best_epoch_psnr)))
        #
        #     print("[PSNR] {:.4f} [Best_PSNR] {:.4f} (epoch {})".format(PSNRs, best_psnr, best_epoch_psnr))
        #
        #     # Save the best SSIM model of validation
        #     if SSIMs > best_ssim:
        #         best_ssim = SSIMs
        #         best_epoch_ssim = epoch
        #         state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        #         torch.save(state, join(FLAGES.backup_model_dir, '{}-model-epochs{}-bestSSIM{}.pth'.format(satellite, epoch, best_ssim)))
        #     print("[SSIM] {:.4f}  [Best_SSIM] {:.4f} (epoch {})".format(SSIMs, best_ssim, best_epoch_ssim))
        #
        #     # Save each epoches of model
        #     state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        #     torch.save(state, join(FLAGES.backup_model_dir, '{}-model-epochs{}.pth'.format(satellite, epoch)))

        # 输出每轮训练花费时间
        time_epoch = (time.time() - start)
        print('==>No:epoch {} training costs {:.4f}min'.format(epoch, time_epoch / 60))

def main():
    start_epoch = start_epochs
    if start_epoch == 0:
        print('==> 无保存模型，将从头开始训练！')
    else:
        print('模型加载')
    train(model, train_datasetloader, start_epoch)

if __name__ == '__main__':
    main()