"""
Train a diffusion model on images.
"""
import sys
import argparse

sys.path.append("../")  #给系统添加路径
sys.path.append("./")
import argparse

from cm import dist_util, logger
# from cm.image_datasets import load_data, BRATSDataset3D
from cm.resample import create_named_schedule_sampler
from cm.bratsloader import BRATSDataset3D
from cm.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from cm.train_util import TrainLoop
import torch.distributed as dist
import torchvision.transforms as transforms
import torch

def main():
    args = create_argparser().parse_args()  #创建参数

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

    data_list = torch.utils.data.DataLoader(ds,
                                     batch_size=args.batch_size,
                                     shuffle=True)
    data = iter(data_list)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    if args.multi_gpu:
        model = torch.nn.DataParallel(
            model, device_ids=[int(id) for id in args.multi_gpu.split(',')])
        model.to(device=torch.device('cuda', int(args.gpu_dev)))
    else:
        model.to(dist_util.dev())
        
    # import netron
    # import onnx
    # from onnx import shape_inference
    # img = torch.randn( 8,5, 256, 256) .to(torch.device("cuda"))
    # t = torch.randn(8,).to(torch.device("cuda"))   
    # # img = th.rand((8, 5, 256, 256)) 
    # # t = th.rand((8))
    # torch.onnx.export(model,(img,t), f='model.onnx', input_names=['image','t'], output_names=['feature_map'])
    # onnx.save(onnx.shape_inference.infer_shapes(onnx.load("model.onnx")), "model.onnx")
    # netron.start("model.onnx")    
    
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)





    # logger.log("creating data loader...")
    # if args.batch_size == -1:
    #     batch_size = args.global_batch_size // dist.get_world_size()
    #     if args.global_batch_size % dist.get_world_size() != 0:
    #         logger.log(
    #             f"warning, using smaller global_batch_size of {dist.get_world_size()*batch_size} instead of {args.global_batch_size}"
    #         )
    # else:
    #     batch_size = args.batch_size

    # data = load_data(
    #     data_dir=args.data_dir,
    #     batch_size=batch_size,
    #     image_size=args.image_size,
    #     class_cond=args.class_cond,
    # )

    # logger.log("creating data loader...")

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        dataloader=data_list,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_name='BRATS',
        data_dir="D:/code/MedSegDiff-master/data/training",
        schedule_sampler="uniform",  # N
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        gpu_dev="0",
        multi_gpu=None,
        batch_size=8,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,   # 日志间隔
        save_interval=10000,   #保存的间隔
        resume_checkpoint=None,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        out_dir='./results_edm/'
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
