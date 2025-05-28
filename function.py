import argparse
import os
import shutil
import sys
import tempfile
import time
from collections import OrderedDict
from datetime import datetime
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from einops import rearrange
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from PIL import Image
from skimage import io
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from tensorboardX import SummaryWriter
# from dataset import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from losses import *

import cfg
import models.sam.utils.transforms as samtrans
import pytorch_ssim
# from models.discriminatorlayer import discriminator
from conf import settings
from utils import *

# from lucent.modelzoo.util import get_model_layers
# from lucent.optvis import render, param, transform, objectives
# from lucent.modelzoo import inceptionv1

args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device) if torch.cuda.is_available() else torch.device('cpu')
pos_weight = torch.ones([1]).cuda(device=GPUdevice) * 2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
seed = torch.randint(1, 11, (args.b, 7))

torch.backends.cudnn.benchmark = True
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
# scaler = torch.cuda.amp.GradScaler()
scaler = torch.amp.GradScaler(device='cuda')
max_iterations = settings.EPOCH
post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []

def generate_click_prompt(img, msk, pt_label = 1):
    # return: prompt, prompt mask
    pt_list = []
    msk_list = []
    # b, c, h, w, d = msk.size()
    if msk.dim() == 5:
        b, c, h, w, d = msk.size()
    elif msk.dim() == 4:
        b, c, h, w = msk.size()
        d = 1
        msk = msk.unsqueeze(-1)
        img = img.unsqueeze(-1)
        # Possibly flatten depth dimension or loop over it
        img = rearrange(img, 'b c h w d -> (b d) c h w')
        b, c, h, w = img.size()
    else:
        raise ValueError(f"Unexpected imgs shape: {img.shape}")

    msk = msk[:,0,:,:,:]
    for i in range(d):
        pt_list_s = []
        msk_list_s = []
        for j in range(b):
            msk_s = msk[j,:,:,i]
            indices = torch.nonzero(msk_s)
            if indices.size(0) == 0:
                # generate a random array between [0-h, 0-h]:
                random_index = torch.randint(0, h, (2,)).to(device = msk.device)
                new_s = msk_s
            else:
                random_index = random.choice(indices)
                label = msk_s[random_index[0], random_index[1]]
                new_s = torch.zeros_like(msk_s)
                # convert bool tensor to int
                new_s = (msk_s == label).to(dtype = torch.float)
                # new_s[msk_s == label] = 1
            pt_list_s.append(random_index)
            msk_list_s.append(new_s)
        pts = torch.stack(pt_list_s, dim=0)
        msks = torch.stack(msk_list_s, dim=0)
        pt_list.append(pts)
        msk_list.append(msks)
    pt = torch.stack(pt_list, dim=-1)
    msk = torch.stack(msk_list, dim=-1)

    msk = msk.unsqueeze(1)

    return img, pt, msk #[b, 2, d], [b, c, h, w, d]



def train_sam(args, net: nn.Module, optimizer, scaler, train_loader,
              epoch, writer, schedulers=None, vis=50):

    hard = 0
    epoch_loss = 0
    ind = 0
    # train mode
    net.train()
    optimizer.zero_grad()
    train_save_dir = 'D:/2025-research/Polyp-KAN-AdapterNet/output/samples/Trainval'
    os.makedirs(train_save_dir, exist_ok=True)

    epoch_loss = 0
    # GPUdevice = torch.device('cuda:' + str(args.gpu_device))

    # New loss composition
    class CombinedLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.dice = DiceCELoss(sigmoid=True, squared_pred=True,
                                   reduction='mean') if args.thd else nn.BCEWithLogitsLoss()
            self.focal = FocalLoss()
            self.surface = SurfaceLoss()

        def forward(self, pred, target):
            return (self.dice(pred, target)
                    + 0.5 * self.focal(pred, target)
                    + 0.2 * self.surface(pred, target))

    lossfunc = CombinedLoss().to(GPUdevice)
    # if args.thd:
    #     lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    # else:
    #     lossfunc = criterion_G

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            # torch.cuda.empty_cache()
            imgs = pack['image'].to(dtype=torch.float32, device=GPUdevice)
            masks = pack['mask'].to(dtype=torch.float32, device=GPUdevice)
            if imgs.shape[1] != 3:  # Only permute if channel is not in 2nd dim
                imgs = imgs.permute(0, 3, 1, 2)  # NHWC → NCHW

            if 'pt' not in pack:
                imgs, pt, masks = generate_click_prompt(imgs, masks)
                point_labels = torch.ones(imgs.shape[0], device=imgs.device)  # or shape (B,) with default label 1
            else:
                pt = pack['pt']
                point_labels = pack['p_label']
            name = pack['image_meta_dict']['filename_or_obj']


            if args.thd:
                imgs, pt, masks = generate_click_prompt(imgs, masks)

                pt = rearrange(pt, 'b n d -> (b d) n')
                imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                masks = rearrange(masks, 'b c h w d -> (b d) c h w ')

                # imgs = imgs.repeat(1, 3, 1, 1)
                point_labels = torch.ones(imgs.size(0))

                # Ensure input is 3 channels, required by most pretrained backbones
                if imgs.shape[1] == 1:
                    imgs = imgs.repeat(1, 3, 1, 1)  # from grayscale
                elif imgs.shape[1] != 3:
                    raise ValueError(f"Expected 1 or 3 channels, but got {imgs.shape[1]}")
                # print(f"Input to model shape: {imgs.shape}")
                # Print shape after transformation
                print(f"[DEBUG] After transformation - imgs.shape = {imgs.shape}")

                imgs = F.interpolate(imgs, size=(args.image_size, args.image_size), mode='bilinear',
                                     align_corners=False)
                masks = F.interpolate(masks, size=(args.out_size, args.out_size), mode='bilinear', align_corners=False)
                # imgs = torchvision.transforms.Resize((args.image_size, args.image_size))(imgs)
                # masks = torchvision.transforms.Resize((args.out_size, args.out_size))(masks)
            showp = pt

            mask_type = torch.float32
            ind += 1
            # b_size, c, w, h = imgs.size() original is taking width before height but usual convention is height before width
            b_size, c, h, w = imgs.size() # corrected image tensor size is [batch, channels, height, width]
            longsize = w if w >= h else h

            if point_labels.clone().flatten()[0] != -1:
                # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                point_coords = pt
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                if (len(point_labels.shape) == 1):  # only one point prompt
                    coords_torch, labels_torch, showp = coords_torch[None, :, :], labels_torch[None, :], showp[None, :,
                                                                                                         :]
                pt = (coords_torch, labels_torch)

            '''init'''
            if hard:
                true_mask_ave = (true_mask_ave > 0.5).float()
                # true_mask_ave = cons_tensor(true_mask_ave)
            # imgs = imgs.to(dtype = mask_type,device = GPUdevice)

            '''Train'''
            if args.mod == 'sam_adpt':
                # Trainable components
                trainable_params = [
                    'adapt1', 'adapt2', 'adapt3',  # Adapters
                    'prompt_emb',  # Hyp-Ada prompt
                    'decoder1', 'decoder2',  # First 2 decoder stages
                    'final'  # Final conv layer
                ]

                # Freeze ALL parameters first
                for param in net.parameters():
                    param.requires_grad = False

                # Unfreeze specific components
                for name, param in net.named_parameters():
                    if any(key in name for key in trainable_params):
                        param.requires_grad = True
                # # print("UKAN attributes:", dir(net))
                # # Define which parts are adapters and which aren't
                # adapter_names = ['adapt1', 'adapt2', 'adapt3']
                # encoder_names = ['encoder1', 'encoder2', 'encoder3',
                #                  'patch_embed3', 'patch_embed4',
                #                  'block1', 'block2', 'norm3', 'norm4']
                #
                # # Freeze encoder layers
                # for name in encoder_names:
                #     if hasattr(net, name):
                #         module = getattr(net, name)
                #         for n, value in module.named_parameters():
                #             value.requires_grad = False
                #     else:
                #         print(f"[WARNING] Encoder module '{name}' not found.")
                #
                # # Unfreeze adapters
                # for name in adapter_names:
                #     if hasattr(net, name):
                #         module = getattr(net, name)
                #         for n, value in module.named_parameters():
                #             value.requires_grad = True
                #     else:
                #         print(f"[WARNING] Adapter module '{name}' not found.")
                # # for n, value in net.image_encoder.named_parameters():
                # #     if "Adapter" not in n:
                # #         value.requires_grad = False
                # #     else:
                # #         value.requires_grad = True
            elif args.mod == 'sam_lora' or args.mod == 'sam_adalora':
                from models.common import loralib as lora
                lora.mark_only_lora_as_trainable(net.image_encoder)
                if args.mod == 'sam_adalora':
                    # Initialize the RankAllocator
                    rankallocator = lora.RankAllocator(
                        net.image_encoder, lora_r=4, target_rank=8,
                        init_warmup=500, final_warmup=1500, mask_interval=10,
                        total_step=3000, beta1=0.85, beta2=0.85,
                    )
            else:
                for n, value in net.image_encoder.named_parameters():
                    value.requires_grad = True

            # imge = net.image_encoder(imgs)
            if hasattr(net, 'image_encoder'):
                imge = net.image_encoder(imgs)
            else:
                imge = net(imgs)  # fallback to UKAN forward()

            with torch.cuda.amp.autocast(enabled=args.amp):  # Mixed precision context
                with torch.set_grad_enabled(args.net == 'polyp_kan'): # enables gradients only for polyp_kan
                    if args.net == 'polyp_kan':

                        imgs = pack['image'].requires_grad_(True)  # Force input grad tracking
                        imgs = imgs.to(GPUdevice)  # Make sure this matches your model device
                        pred = net(imgs)  # Direct UKAN forward pass

                    elif args.net == 'sam' or args.net == 'mobile_sam':
                        se, de = net.prompt_encoder(
                        points=pt,
                        boxes=None,
                        masks=None,
                        )
                    elif args.net == "efficient_sam":
                        coords_torch, labels_torch = transform_prompt(coords_torch, labels_torch, h, w)
                        se = net.prompt_encoder(
                        coords=coords_torch,
                        labels=labels_torch,
                        )

            if args.net == 'sam':
                pred, _ = net.mask_decoder(
                    image_embeddings=imge,
                    image_pe=net.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de,
                    multimask_output=(args.multimask_output > 1),
                )
            elif args.net == 'mobile_sam':
                pred, _ = net.mask_decoder(
                    image_embeddings=imge,
                    image_pe=net.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de,
                    multimask_output=False,
                )
            elif args.net == "efficient_sam":
                se = se.view(
                    se.shape[0],
                    1,
                    se.shape[1],
                    se.shape[2],
                )
                pred, _ = net.mask_decoder(
                    image_embeddings=imge,
                    image_pe=net.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=se,
                    multimask_output=False,
                )

            # Resize to the ordered output size
            # pred = F.interpolate(pred, size=(args.out_size, args.out_size))
            # Resize prediction to match mask dimensions
            pred = F.interpolate(pred, size=(352, 352), mode='bilinear', align_corners=False)

            # Ensure masks have correct dimensions [B, C, H, W]
            masks = masks.squeeze(-1)  # Remove last dimension if present

            assert pred.shape[-2:] == masks.shape[-2:], \
                f"Shape mismatch: pred {pred.shape} vs mask {masks.shape}"

            loss = lossfunc(pred, masks)
            if loss.requires_grad:
                loss.retain_grad()

            pbar.set_postfix(**{'loss (batch)': loss.item()})
            epoch_loss += loss.item()

            # Unified gradient handling
            scaler.scale(loss).backward()  # Scale loss
            if args.mod == 'sam_adalora':
                lora_loss = loss + lora.compute_orth_regu(net, regu_weight=0.1)
                scaler.scale(lora_loss).backward()

                # Gradient clipping
            scaler.unscale_(optimizer)  # Unscale before clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()

            # Lion-specific zero_grad
            optimizer.zero_grad(set_to_none=True)  # Critical for Lion

            # nn.utils.clip_grad_value_(net.parameters(), 0.1)
            # if args.mod == 'sam_adalora':
            #     (loss + lora.compute_orth_regu(net, regu_weight=0.1)).backward()
            #     optimizer.step()
            #     rankallocator.update_and_mask(net, ind)
            # else:
            #     loss.backward()
            #     optimizer.step()
            #
            # optimizer.zero_grad()

            '''vis images'''
            if vis:
                if ind % vis == 0:
                    namecat = 'Train'
                    for na in name[:2]:
                        # namecat = namecat + na.split('/')[-1].split('.')[0] + '+'
                        # save_path = os.path.join(train_save_dir, namecat + 'epoch+' + str(epoch) + '.jpg')
                        base_name = os.path.basename(na)  # Extract filename only
                        img_name = os.path.splitext(base_name)[0]
                        namecat += img_name + '+'
                    save_path = os.path.join(train_save_dir, f'{namecat}epoch_{epoch}.jpg')
                    vis_image(imgs, pred, masks, save_path, reverse=False, points=showp)
                    # vis_image(imgs, pred, masks,
                    #           os.path.join(args.path_helper['sample_path'], namecat + 'epoch+' + str(epoch) + '.jpg'),
                    #           reverse=False, points=showp)

            pbar.update()

    return loss


def validation_sam(args, val_loader, epoch, net: nn.Module, writer, clean_dir=True):
    save_dir = 'D:/2025-research/Polyp-KAN-AdapterNet/output/samples/Testval'
    os.makedirs(save_dir, exist_ok=True)  # <--- Add this line

    # eval mode
    net.eval()

    mask_type = torch.float32
    n_val = len(val_loader)  # the number of batch
    ave_res, mix_res = (0, 0, 0, 0), (0,) * args.multimask_output * 2
    rater_res = [(0, 0, 0, 0) for _ in range(6)]
    tot = 0
    hard = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    # GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    # device = GPUdevice

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            imgsw = pack['image'].to(dtype=torch.float32, device=GPUdevice)
            masksw = pack['mask'].to(dtype=torch.float32, device=GPUdevice)
            # for k,v in pack['image_meta_dict'].items():
            #     print(k)
            if 'pt' not in pack or args.thd:
                imgsw, ptw, masksw = generate_click_prompt(imgsw, masksw)
                point_labels = torch.ones(imgsw.shape[0], device=imgsw.device)
            else:
                ptw = pack['pt']
                point_labels = pack['p_label']
            name = pack['image_meta_dict']['filename_or_obj']

            buoy = 0
            if args.evl_chunk:
                evl_ch = int(args.evl_chunk)
            else:
                evl_ch = int(imgsw.size(-1))

            while (buoy + evl_ch) <= imgsw.size(-1):
                if args.thd:
                    pt = ptw[:, :, buoy: buoy + evl_ch]
                else:
                    pt = ptw

                imgs = imgsw[..., buoy:buoy + evl_ch]
                masks = masksw[..., buoy:buoy + evl_ch]
                buoy += evl_ch

                if args.thd:
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                    masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                    imgs = imgs.repeat(1, 3, 1, 1)
                    point_labels = torch.ones(imgs.size(0))

                    imgs = torchvision.transforms.Resize((args.image_size, args.image_size))(imgs)
                    masks = torchvision.transforms.Resize((args.out_size, args.out_size))(masks)

                showp = pt

                mask_type = torch.float32
                ind += 1
                # b_size, c, w, h = imgs.size() original is taking width before height but usual convention is height before width
                b_size, c, h, w = imgs.size()  # corrected image tensor size is [batch, channels, height, width]
                longsize = w if w >= h else h

                if point_labels.clone().flatten()[0] != -1:
                    # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                    point_coords = pt
                    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                    if (len(point_labels.shape) == 1):  # only one point prompt
                        coords_torch, labels_torch, showp = coords_torch[None, :, :], labels_torch[None, :], showp[None,
                                                                                                             :, :]
                    pt = (coords_torch, labels_torch)

                '''init'''
                if hard:
                    true_mask_ave = (true_mask_ave > 0.5).float()
                    # true_mask_ave = cons_tensor(true_mask_ave)
                imgs = imgs.to(dtype=mask_type, device=GPUdevice)

                '''test'''
                if hasattr(net, 'image_encoder'):
                    imge = net.image_encoder(imgs)
                else:
                    imge = net(imgs)  # fallback to UKAN forward()
                with torch.no_grad():
                    # imge = net.image_encoder(imgs)
                    if args.net == 'polyp_kan':
                        imge = net(imgs)  # Directly use UKAN's forward pass
                    elif args.net == 'sam' or args.net == 'mobile_sam':
                        se, de = net.prompt_encoder(
                            points=pt,
                            boxes=None,
                            masks=None,
                        )
                    elif args.net == "efficient_sam":
                        coords_torch, labels_torch = transform_prompt(coords_torch, labels_torch, h, w)
                        se = net.prompt_encoder(
                            coords=coords_torch,
                            labels=labels_torch,
                        )

                    if args.net == 'polyp_kan':
                        pred = net(imgs)  # Direct UKAN prediction
                        # Ensure shape: [B, 1, H, W]
                        if masks.ndim == 5 and masks.shape[-1] == 1:
                            masks = masks.squeeze(-1)

                        if masks.ndim == 4 and masks.shape[1] != 1 and masks.shape[-1] == 1:
                            masks = masks.permute(0, 3, 1, 2)

                        # Resize mask to match prediction shape
                        masks = F.interpolate(masks.float(), size=pred.shape[2:], mode='bilinear', align_corners=False)

                        # Make sure prediction and mask shapes match now
                        assert pred.shape == masks.shape, f"Shape mismatch: pred {pred.shape}, mask {masks.shape}"
                    elif args.net == 'sam':
                        pred, _ = net.mask_decoder(
                            image_embeddings=imge,
                            image_pe=net.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=se,
                            dense_prompt_embeddings=de,
                            multimask_output=(args.multimask_output > 1),
                        )
                    elif args.net == 'mobile_sam':
                        pred, _ = net.mask_decoder(
                            image_embeddings=imge,
                            image_pe=net.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=se,
                            dense_prompt_embeddings=de,
                            multimask_output=False,
                        )
                    elif args.net == "efficient_sam":
                        se = se.view(
                            se.shape[0],
                            1,
                            se.shape[1],
                            se.shape[2],
                        )
                        pred, _ = net.mask_decoder(
                            image_embeddings=imge,
                            image_pe=net.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=se,
                            multimask_output=False,
                        )
                    if args.net != 'polyp_kan':
                        # Resize prediction for SAM variants to match expected output size
                        pred = F.interpolate(pred,
                                             size=(args.out_size, args.out_size),
                                             mode='bilinear',
                                             align_corners=False)

                    # Resize to the ordered output size
                    # pred = F.interpolate(pred, size=(args.out_size, args.out_size)) # works for other models but not polyp_kan
                    prompt_norm = torch.norm(net.prompt_emb).item()
                    writer.add_scalar('Val/PromptNorm', prompt_norm, epoch)
                    tot += lossfunc(pred, masks)

                    '''vis images'''
                    if ind % args.vis == 0:
                        namecat = 'Test'
                        for na in name[:2]:
                            # img_name = na.split('/')[-1].split('.')[0]
                            # namecat = namecat + img_name + '+'
                            # save_path = os.path.join(save_dir, namecat + 'epoch+' + str(epoch) + '.jpg')
                            # Extract ONLY filename without path
                            base_name = os.path.basename(na)  # Handles path separators
                            img_name = os.path.splitext(base_name)[0]  # Remove extension
                            namecat += img_name + '+'
                        save_path = os.path.join(save_dir, f'{namecat}epoch_{epoch}.jpg')
                        vis_image(imgs, pred, masks, save_path, reverse=False, points=showp)

                        # vis_image(imgs, pred, masks, os.path.join(args.path_helper['sample_path'],namecat + 'epoch+' + str(epoch) + '.jpg'), reverse=False, points=showp)

                    temp = eval_seg(pred, masks, threshold)
                    mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

            pbar.update()

    if args.evl_chunk:
        n_val = n_val * (imgsw.size(-1) // evl_ch)

    return tot / n_val, tuple([a / n_val for a in mix_res])


def transform_prompt(coord, label, h, w):
    coord = coord.transpose(0, 1)
    label = label.transpose(0, 1)

    coord = coord.unsqueeze(1)
    label = label.unsqueeze(1)

    batch_size, max_num_queries, num_pts, _ = coord.shape
    num_pts = coord.shape[2]
    rescaled_batched_points = get_rescaled_pts(coord, h, w)

    decoder_max_num_input_points = 6
    if num_pts > decoder_max_num_input_points:
        rescaled_batched_points = rescaled_batched_points[
                                  :, :, : decoder_max_num_input_points, :
                                  ]
        label = label[
                :, :, : decoder_max_num_input_points
                ]
    elif num_pts < decoder_max_num_input_points:
        rescaled_batched_points = F.pad(
            rescaled_batched_points,
            (0, 0, 0, decoder_max_num_input_points - num_pts),
            value=-1.0,
        )
        label = F.pad(
            label,
            (0, decoder_max_num_input_points - num_pts),
            value=-1.0,
        )

    rescaled_batched_points = rescaled_batched_points.reshape(
        batch_size * max_num_queries, decoder_max_num_input_points, 2
    )
    label = label.reshape(
        batch_size * max_num_queries, decoder_max_num_input_points
    )

    return rescaled_batched_points, label


def get_rescaled_pts(batched_points: torch.Tensor, input_h: int, input_w: int):
    return torch.stack(
        [
            torch.where(
                batched_points[..., 0] >= 0,
                batched_points[..., 0] * 1024 / input_w,
                -1.0,
            ),
            torch.where(
                batched_points[..., 1] >= 0,
                batched_points[..., 1] * 1024 / input_h,
                -1.0,
            ),
        ],
        dim=-1,
    )