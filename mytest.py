from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer
import argparse
from torch.autograd import Variable
from trainer import aclgan_Trainer
import torch.backends.cudnn as cudnn
import torch
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
import shutil
import torchvision.utils as vutils
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./configs/male2female.yaml', help='Path to the config file.')
parser.add_argument('--trainer', type=str, default='aclgan', help="aclgan")
parser.add_argument('--checkpoint', type=str, default='D:/dachuang/ACL-GAN-master\outputs\male2female\checkpoints\gen_00050000.pt',help="checkpoint of autoencoders")
opts = parser.parse_args()

config = get_config(opts.config)
display_size = config['display_size']
if opts.trainer == 'aclgan':
    trainer = aclgan_Trainer(config)
    state_dict = torch.load(opts.checkpoint)
    trainer.gen_AB.load_state_dict(state_dict['AB'])
    trainer.gen_BA.load_state_dict(state_dict['BA'])
trainer.cuda()
train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)
test_display_images_a = torch.stack([test_loader_a.dataset[i] for i in range(0,100)]).cuda()
test_display_images_b = torch.stack([test_loader_b.dataset[i] for i in range(0,100)]).cuda()

with torch.no_grad():
    test_image_outputs = trainer.sample(test_display_images_a, test_display_images_b)
    """
    image_outputs = [images.expand(-1, 3, -1, -1) for images in test_image_outputs]
    for images in image_outputs:
        vutils.save_image(images[:display_size].data,'./test_output/%s.jpg' % (str(torch.rand(1))),nrow=1,normalize=True,padding=0)
    """
    write_2images(test_image_outputs, 100, './test_output', 'test')
    

    