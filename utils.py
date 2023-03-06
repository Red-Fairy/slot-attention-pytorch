import os
from torchvision.transforms.functional import to_tensor, to_pil_image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class logger(object):
    def __init__(self, path):
        self.path = path

    def info(self, msg):
        print(msg)
        with open(os.path.join(self.path, "log.txt"), 'a') as f:
            f.write(msg + "\n")

def visual(model, model_dir, epoch=0, num_slots=7):
    image = Image.open('/viscam/data/clevr/train/images/CLEVR_train_000001.png').convert('RGB')
    image = image.resize((128, 128))
    image = to_tensor(image).cuda().unsqueeze(0)
    fig, ax = plt.subplots(1, num_slots + 2, figsize=(15, 2))

    recon_combined, recons, masks, slots = model(image)
    
    image = image.squeeze(0)
    recon_combined = recon_combined.squeeze(0)
    recons = recons.squeeze(0)
    masks = masks.squeeze(0)
    image = image.permute(1,2,0).cpu().numpy()
    recon_combined = recon_combined.permute(1,2,0)
    recon_combined = recon_combined.cpu().detach().numpy()
    recons = recons.cpu().detach().numpy()
    masks = masks.cpu().detach().numpy()

    ax[0].imshow(image)
    ax[0].set_title('Image')
    ax[1].imshow(recon_combined)
    ax[1].set_title('Recon.')

    for i in range(7):
        picture = recons[i] * masks[i] + (1 - masks[i])
        ax[i + 2].imshow(picture)
        ax[i + 2].set_title('Slot %s' % str(i + 1))
    for i in range(len(ax)):
        ax[i].grid(False)
        ax[i].axis('off')

    # save
    fig.savefig(f'{model_dir}/visuals/epoch_{epoch}.png', bbox_inches='tight', pad_inches=0)
