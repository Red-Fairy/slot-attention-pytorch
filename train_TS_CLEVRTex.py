import os
import argparse
from dataset import *
from model_TS_CLEVRTex import *
from tqdm import tqdm
import time
import datetime
import torch.optim as optim
import torch
from utils import logger, visual
import math


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument
parser.add_argument('--model_dir', default='./test', type=str, help='where to save models' )
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_slots', default=7, type=int, help='Number of slots in Slot Attention.')
parser.add_argument('--num_iterations', default=3, type=int, help='Number of attention iterations.')
parser.add_argument('--hid_dim', default=64, type=int, help='hidden dimension size')

parser.add_argument('--learning_rate', default=0.0004, type=float)
parser.add_argument('--warmup_steps', default=50000, type=int, help='Number of warmup steps for the learning rate.')
parser.add_argument('--num_steps', default=500000, type=int, help='Number of total step.')

parser.add_argument('--num_workers', default=2, type=int, help='number of workers for loading data')
parser.add_argument('--num_epochs', default=1000, type=int, help='number of workers for loading data')

opt = parser.parse_args()
resolution = (256, 192)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

opt.model_dir = os.path.join('experiments',opt.model_dir)
os.makedirs(opt.model_dir, exist_ok=True)
os.makedirs(os.path.join(opt.model_dir,'weights'), exist_ok=True)
os.makedirs(os.path.join(opt.model_dir,'visuals'), exist_ok=True)

log = logger(opt.model_dir)

log.info(str(opt))

train_set = CLEVRTex(root='/viscam/u/redfairy/datasets/clevrtex_vbg')
model = SlotAttentionAutoEncoder(resolution, opt.num_slots, opt.num_iterations, opt.hid_dim).to(device)
# model.load_state_dict(torch.load('./tmp/model6.ckpt')['model_state_dict'])

criterion = nn.MSELoss()

params = [{'params': model.parameters()}]

train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size,
                        shuffle=True, num_workers=opt.num_workers)

optimizer = optim.Adam(params, lr=opt.learning_rate)

start = time.time()
i = 0
epoch = 0

while True:
    epoch += 1

    model.train()
    total_loss = 0

    for sample in tqdm(train_dataloader):
        i += 1

        if i < opt.warmup_steps:
            learning_rate = opt.learning_rate * (i / opt.warmup_steps)
        else:
            # cosine learning rate decay
            learning_rate = 0.5 * opt.learning_rate * (
                1 + math.cos(math.pi * (i - opt.warmup_steps) /
                             (opt.num_steps - opt.warmup_steps)))

        optimizer.param_groups[0]['lr'] = learning_rate
        
        image = sample['image'].to(device)
        recon_combined, recons, masks, slots = model(image)
        loss = criterion(recon_combined, image)
        total_loss += loss.item()

        del recons, masks, slots

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 500 == 0:
            model.eval()
            visual(model, opt.model_dir, i, img_path='/viscam/u/redfairy/datasets/clevrtex_vbg/0/CLEVRTEX_vbg_000000.png')
            model.train()

    total_loss /= len(train_dataloader)

    log.info("Epoch: {}, Loss: {}, Time: {}".format(epoch, total_loss,
        datetime.timedelta(seconds=time.time() - start)))

    torch.save({
        'model_state_dict': model.state_dict(),
        }, os.path.join(opt.model_dir, 'weights/model_{}.ckpt'.format(epoch)))
        
    if i > opt.num_steps:
        break
