"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import sys
sys.path.append("../")  #给系统添加路径
sys.path.append("./")
import torchvision.utils as vutils
import numpy as np
import torch as th
import torch.distributed as dist

from cm.utils import staple

from cm import dist_util, logger
from cm.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from cm.random_util import get_generator
from cm.karras_diffusion import karras_sample
import torchvision.transforms as transforms
from cm.bratsloader import BRATSDataset3D

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img



def calculate_dice(pred, target):
    intersection = th.sum(pred * target)
    union = th.sum(pred) + th.sum(target)
    dice = (2 * intersection + 1e-5) / (union + 1e-5)
    return dice

# 假设我们有10个样本，其中8个样本的分类结果正确，2个样本的分类结果错误
import numpy as np



def calculate_iou(image1, image2):
   
    # 计算像素值的差值
    diff = image1 - image2

    # 计算差值的平方
    intersection = th.sum(image1*image2)
    union = th.sum(image1) + th.sum(image2) - intersection
    iou = (intersection + 1e-5) / (union + 1e-5)
    
    diff_squared = diff ** 2

    # 求和所有像素的差值平方
    sum_diff_squared = th.sum(diff_squared)

    # 开平方根得到IO u值
    

    return iou


#io_u = calculate_iou(image1, image2)



def main():
    args = create_argparser().parse_args()
    
    # args = create_argparser().parse_args()

    dist_util.setup_dist(args)
    # logger.configure()
    logger.configure(dir=args.out_dir)

    logger.log("creating data loader...")
    if args.data_name == 'ISIC':
        tran_list = [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ]
        transform_train = transforms.Compose(tran_list)

        # ds = ISICDataset(args, args.data_dir, transform_train)
        args.in_ch = 4
    elif args.data_name == 'BRATS':
        tran_list = [
            transforms.Resize((args.image_size, args.image_size)),
        ]
        transform_train = transforms.Compose(tran_list)

        ds = BRATSDataset3D(args.data_dir, transform_train, test_flag=False)
        args.in_ch = 5
    else:
        tran_list = [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ]
        transform_train = transforms.Compose(tran_list)
        print("Your current directory : ", args.data_dir)
        # ds = CustomDataset(args, args.data_dir, transform_train)
        args.in_ch = 4

    data_list = th.utils.data.DataLoader(ds,
                                     batch_size=args.batch_size,
                                     shuffle=True,drop_last = True)
    data = iter(data_list)

    # dist_util.setup_dist()
    # logger.configure()

    if "consistency" in args.training_mode:
        distillation = True
    else:
        distillation = False

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
        distillation=distillation,
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    # model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
        
    if args.multi_gpu:
        model = th.nn.DataParallel(
            model, device_ids=[int(id) for id in args.multi_gpu.split(',')])
        model.to(device=th.device('cuda', int(args.gpu_dev)))
    else:
        model.to(dist_util.dev())
    model.eval()
    dice = []
    iou = []
    
    for _ in range(len(data)):
        try:
                b, m, path = next(data)
        except StopIteration:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader
                # data_iter = iter(self.dataloader)
                b, m, path = next(data)                    
        # b, m, path = next(data)  #should return an image from the dataloader "data"   b image m math
        c = th.randn_like(b)   #定义了一个和数据一样大的
        # img = th.cat((b, c), dim=1)     #add a noise channel$
        img = b
        if args.data_name == 'ISIC':
            slice_ID=path[0].split("_")[-1].split('.')[0]
        elif args.data_name == 'BRATS':
            # slice_ID=path[0].split("_")[2] + "_" + path[0].split("_")[4]
            slice_ID=path[0].split("_")[-3] + "_" + path[0].split("slice")[-1].split('.nii')[0]
        # slice_ID = 'ce'
        logger.log("sampling...")
        if args.sampler == "multistep":
            assert len(args.ts) > 0
            ts = tuple(int(x) for x in args.ts.split(","))
        else:
            ts = None
        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)
        enslist = []
        masklist = []
        yu = []
        # dice_value = 0
        # iou_value = 0

        # all_images = []
        # all_labels = []
        generator = get_generator(args.generator, args.num_samples, args.seed)
        
        for i in range(args.num_ensemble): 
            

    # while len(all_images) * args.batch_size < args.num_samples:
            model_kwargs = {}
            if args.class_cond:
                classes = th.randint(
                    low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
                )
                model_kwargs["y"] = classes

            output, sample, cal = karras_sample(
                diffusion,
                model,
                (args.batch_size, 1, args.image_size, args.image_size),img = img,
                steps=args.steps,
                model_kwargs=model_kwargs,
                device=dist_util.dev(),
                clip_denoised=args.clip_denoised,
                sampler=args.sampler,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                s_churn=args.s_churn,
                s_tmin=args.s_tmin,
                s_tmax=args.s_tmax,
                s_noise=args.s_noise,
                generator=generator,
                ts=ts,
            )
            
            m = m.to(dist_util.dev())
            
            
      
            end.record()
            th.cuda.synchronize()
            dice_value_s = calculate_dice(cal,m)
            iou_value_s = calculate_iou(cal, m)
            print(dice_value_s)
            print(iou_value_s)
        
            enslist.append(cal)
            masklist.append(m)
            # print(b[:,0,:,:].shape)
            yu.append(img)
            
        # elif args.data_name == 'BRATS':
            # s = th.tensor(sample)[:,-1,:,:].unsqueeze(1)
            # m = th.tensor(m.to(device = 'cuda:0'))[:,0,:,:].unsqueeze(1)

            
        # ensres = staple(th.stack(enslist,dim=0)).squeeze(0)
        # vutils.save_image(ensres, fp = os.path.join(args.out_dir, str(slice_ID)+'_output_ens'+".jpg"), nrow = 1, padding = 10)
            
        # a=np.sum(np.sum(ensres))   
        ensres = staple(th.stack(enslist,dim=0)).squeeze(0)
        m_ensres = staple(th.stack(masklist,dim=0)).squeeze(0)
        non_zero_indices = m_ensres.nonzero()
        num_non_zero = len(non_zero_indices)

        if num_non_zero > 0.001:
            o1 = th.tensor(img)[:,0,:,:].unsqueeze(1)
            o2 = th.tensor(img)[:,1,:,:].unsqueeze(1)
            o3 = th.tensor(img)[:,2,:,:].unsqueeze(1)
            o4 = th.tensor(img)[:,3,:,:].unsqueeze(1)
            # c = th.tensor(cal)

            tup = (o1/o1.max(),o2/o2.max(),o3/o3.max(),o4/o4.max())

            compose = th.cat(tup,0)
            vutils.save_image(compose, fp = os.path.join(args.out_dir, str(slice_ID)+'_c'+ str(i)+".png"), nrow = 8, padding = 10, pad_value = 100)  
            # vutils.save_image(compose[0], fp = os.path.join(args.out_dir, str(slice_ID)+'_c_0'+".png"), nrow = 8, padding = 10, pad_value = 100)    
            # vutils.save_image(compose[10], fp = os.path.join(args.out_dir, str(slice_ID)+'_c_1'+".png"), nrow = 8, padding = 10, pad_value = 100)        
            vutils.save_image(ensres, fp = os.path.join(args.out_dir, str(slice_ID)+'_o'+".png"), nrow = 8, padding = 10, pad_value = 100)
            
            vutils.save_image(m_ensres, fp = os.path.join(args.out_dir, str(slice_ID)+'_m'+".png"), nrow = 8, padding = 10, pad_value = 100)
        # print(a)
        # y_ensres = staple(th.stack(yu,dim=0)).unsqueeze(0)
        # vutils.save_image(y_ensres[-1, ...], fp = os.path.join(args.out_dir, str(slice_ID)+'_c_0'+".png"), nrow = 8, padding = 10)
        # dice_value = calculate_dice(ensres,m_ensres)
        # print(diiou_value_s
        dice.append(dice_value_s)
        iou.append(iou_value_s)
        # ensres = staple(th.stack(enslist,dim=0)).squeeze(0)
        # vutils.save_image(ensres, fp = os.path.join(args.out_dir, str(slice_ID)+'_output_cal'+".png"), nrow = 1, padding = 10)

        # gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        # dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        # all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        # if args.class_cond:
        #     gathered_labels = [
        #         th.zeros_like(classes) for _ in range(dist.get_world_size())
        #     ]
        #     dist.all_gather(gathered_labels, classes)
        #     all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        # logger.log(f"created {len(all_images) * args.batch_size} samples")
        # logger.log(f"dice{dice_value}")
        # logger.log(f"dice{dice_value}")
    dice_avg = sum(dice)/len(dice)
    iou_avg = sum(iou)/len(iou)
    print("-------------------------------------------------")
    print(dice_avg)
    logger.log(f"dice{dice_avg}")
    logger.log(f"iou{iou_avg}")
    # arr = np.concatenate(all_images, axis=0)
    # arr = arr[: args.num_samples]
    # if args.class_cond:
    #     label_arr = np.concatenate(all_labels, axis=0)
    #     label_arr = label_arr[: args.num_samples]
    # if dist.get_rank() == 0:
    #     shape_str = "x".join([str(x) for x in arr.shape])
    #     out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
    #     logger.log(f"saving to {out_path}")
    #     if args.class_cond:
    #         np.savez(out_path, arr, label_arr)
    #     else:
    #         np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        data_name='BRATS',
        data_dir="D:/data/Brast/HGG/test",
        multi_gpu=None,
        training_mode="consistency_distillation",
        gpu_dev="0",
        generator="determ",
        clip_denoised=True,
        num_samples=10000,
        batch_size=8,
        sampler="onestep",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        steps=40,
        model_path="D:/code/consistency_models-main/0129/model519000.pt",# model411000.pt 0.8160 model489000.pt 0.8712   model522000 0.8939
        seed=42,
        ts="",
        num_ensemble=1,
        out_dir='./picture1/'
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
