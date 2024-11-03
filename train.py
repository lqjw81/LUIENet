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

model = FIVE_APLUSNet()

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=FLAGES.lr)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0004, max_lr=1.2 * 0.0004,
                                              cycle_momentum=False)

warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epochs, eta_min=1e-6)
def train(model, train_datasetloader, start_epoch):
    log = open("./logs.txt", mode="a", encoding="utf-8")
    best_psnr, best_ssim, best_epoch_ssim, best_epoch_psnr = 0., 0., 0., 0.
    print('===>Begin Training!')
    model.train()
    steps_per_epoch = len(train_datasetloader)。
    total_iterations = total_epochs * steps_per_epoch 
    print('total_iterations:{}'.format(total_iterations))
    # train_loss_record = open('%s/train_loss_record.txt' % FLAGES.record_dir, "w")
    for epoch in range(start_epoch + 1, total_epochs + 1):
        start = time.time() 
        prefetcher_train = DataPrefetcher(train_datasetloader)
        data = prefetcher_train.next()
        print('Fetching training UIEB spends {} seconds'.format(time.time()-start))
        while data is not None:
            raw, label = data[0].to(device), data[1].to(device)

            optimizer.zero_grad() 

            pred, predforvgg = model(raw) 
            loss_d = loss_f(pred, label)
            # loss_d = loss_f(pred, label)
            train_loss = loss_d+0.2*loss_per(pred, label)+0.5*ssim_loss(pred, label)+0.2*loss_per(predforvgg, label)
            # train_loss = ssim_loss(pred, label)+loss_d

            train_loss.backward() 
            optimizer.step() 
            scheduler.step()

            data = prefetcher_train.next() 
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
