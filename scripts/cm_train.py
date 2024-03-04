"""
Train a diffusion model on images.
"""
import sys
import argparse
sys.path.append("../")  #给系统添加路径
sys.path.append("./")

from cm import dist_util, logger
# from cm.image_datasets import load_data
from cm.bratsloader import BRATSDataset3D
from cm.resample import create_named_schedule_sampler
from cm.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    cm_train_defaults,
    args_to_dict,
    add_dict_to_argparser,
    create_ema_and_scales_fn,
)
from cm.train_util import CMTrainLoop
import torch.distributed as dist
import copy
import torchvision.transforms as transforms
import torch


def main():
    args = create_argparser().parse_args()

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
    ema_scale_fn = create_ema_and_scales_fn(
        target_ema_mode="adaptive",
        start_ema=0.95,
        scale_mode="progressive",
        start_scales=2,
        end_scales=200,
        total_steps=args.total_training_steps,
        distill_steps_per_iter=args.distill_steps_per_iter,
    )
    if args.training_mode == "progdist":
        distillation = False
    elif "consistency" in args.training_mode:
        distillation = True
    else:
        raise ValueError(f"unknown training mode {args.training_mode}")

    model_and_diffusion_kwargs = args_to_dict(
        args, model_and_diffusion_defaults().keys()
    )
    model_and_diffusion_kwargs["distillation"] = distillation
    model, diffusion = create_model_and_diffusion(**model_and_diffusion_kwargs)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'模型参数数量: {num_params}')
    model.train()
    if args.use_fp16:
        model.convert_to_fp16()
        
    if args.multi_gpu:
        model = torch.nn.DataParallel(
            model, device_ids=[int(id) for id in args.multi_gpu.split(',')])
        model.to(device=torch.device('cuda', int(args.gpu_dev)))
    else:
        model.to(dist_util.dev())
        
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    if len(args.teacher_model_path) > 0:  # path to the teacher score model.
        logger.log(f"loading the teacher model from {args.teacher_model_path}")
        teacher_model_and_diffusion_kwargs = copy.deepcopy(model_and_diffusion_kwargs)
        teacher_model_and_diffusion_kwargs["dropout"] = args.teacher_dropout
        teacher_model_and_diffusion_kwargs["distillation"] = False
        teacher_model, teacher_diffusion = create_model_and_diffusion(
            **teacher_model_and_diffusion_kwargs,
        )

        teacher_model.load_state_dict(
            dist_util.load_state_dict(args.teacher_model_path, map_location="cpu"),
        )

        teacher_model.to(dist_util.dev())
        teacher_model.eval()

        for dst, src in zip(model.parameters(), teacher_model.parameters()):
            dst.data.copy_(src.data)

        if args.use_fp16:
            teacher_model.convert_to_fp16()

    else:
        teacher_model = None
        teacher_diffusion = None

    # load the target model for distillation, if path specified.

    logger.log("creating the target model")
    target_model, _ = create_model_and_diffusion(
        **model_and_diffusion_kwargs,
    )
    num_params = sum(p.numel() for p in target_model.parameters())
    print(f'模型2参数数量: {num_params}')
    if args.multi_gpu:
        model_t = torch.nn.DataParallel(
            model_t, device_ids=[int(id) for id in args.multi_gpu.split(',')])
        model_t.to(device=torch.device('cuda', int(args.gpu_dev)))
    else:
        target_model.to(dist_util.dev())

    # target_model.to(dist_util.dev())
    target_model.train()

    # dist_util.sync_params(target_model.parameters())
    dist_util.sync_params(target_model.buffers())

    for dst, src in zip(target_model.parameters(), model.parameters()):
        dst.data.copy_(src.data)

    if args.use_fp16:
        target_model.convert_to_fp16()

    logger.log("training...")
    CMTrainLoop(
        model=model,
        target_model=target_model,
        teacher_model=teacher_model,
        teacher_diffusion=teacher_diffusion,
        training_mode=args.training_mode,
        ema_scale_fn=ema_scale_fn,
        total_training_steps=args.total_training_steps,
        diffusion=diffusion,
        data=data,
        dataloader=data_list,
        batch_size=8,
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
        data_dir="D:/data/Brast/HGG/train",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        gpu_dev="0",
        multi_gpu=None,
        batch_size=8,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        target_ema_mode = "adaptive",
        start_ema=0.95,
        scale_mode="progressive",
        start_scales=2,
        end_scales=150,
        training_mode="consistency_training",
        total_training_steps =1000000,
        # ema_rate="0.9999,0.99994,0.9999432189950708" ,
        log_interval=10,
        save_interval=10000,
        resume_checkpoint=None,
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(cm_train_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

