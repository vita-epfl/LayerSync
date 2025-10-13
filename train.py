# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse, ast
import logging
import os

from models import SiT_models
from download import find_model_resume, find_model
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from train_utils import parse_transport_args
import wandb_utils
from datasets import load_dataset




#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new SiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    local_batch_size = int(args.global_batch_size // dist.get_world_size())

    # Setup an experiment folder:
    if rank == 0:
        if (args.ckpt is not None) and (not args.fine_tune):
            checkpoint_dir = "/".join(args.ckpt.split("/")[:-1])
            experiment_dir = "/".join(checkpoint_dir.split("/")[:-1])
        else:
            os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
            experiment_index = len(glob(f"{args.results_dir}/*"))
            model_string_name = args.model.replace("/", "-")  # e.g., SiT-XL/2 --> SiT-XL-2 (for naming folders)
            experiment_name = f"{experiment_index:03d}-{model_string_name}-" \
                            f"{args.wandb_run_name}"
            experiment_dir = f"{args.results_dir}/{experiment_name}"  # Create an experiment folder
            checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
            os.makedirs(checkpoint_dir, exist_ok=True)


        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        entity = os.environ["ENTITY"]
        project = os.environ["PROJECT"]
        if args.wandb:
            wandb_utils.initialize(args, entity, args.wandb_run_name, project)
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        encoder_depth=args.encoder_depth,
        gt_encoder_depth=args.gt_encoder_depth,
    )

    # Note that parameter initialization is done within the SiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training


    if (args.ckpt is not None) and (not args.fine_tune):
        ckpt_path = args.ckpt
        state_dict = find_model(ckpt_path)
        model.load_state_dict(state_dict)
        ema.load_state_dict(state_dict)


        
    if args.fine_tune:
        ckpt_path = args.ckpt
        state_dict = find_model_resume(ckpt_path)
        model.load_state_dict(state_dict)
        ema.load_state_dict(state_dict)
        

    

    requires_grad(ema, False)
    
    model = DDP(model.to(device), device_ids=[rank])
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps,
    )  # default: velocity; 
    transport_sampler = Sampler(transport)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.requires_grad_(False)
    logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    # if (args.ckpt is not None) and (not args.fine_tune):
    #     opt.load_state_dict(state_dict["opt"])

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    

    dataset_train = load_dataset(
            args.dataset_name,
            split='train',
            streaming=False,
            cache_dir=args.cache_dir,
            trust_remote_code=True
        )
        
    # Convert HuggingFace dataset to ImageFolder-like format
    # This assumes the dataset has 'image' and 'label' fields
    class HFDatasetWrapper(torch.utils.data.Dataset):
        def __init__(self, hf_dataset, transform=None):
            self.hf_dataset = hf_dataset
            self.transform = transform
            
        def __len__(self):
            return len(self.hf_dataset)
            
        def __getitem__(self, idx):
            try:
                item = self.hf_dataset[idx]
                image = item['image']
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                label = item['label']
                
                if self.transform:
                    image = self.transform(image)
                return image, label
            except Exception as e:
                logger.warning(f"Error loading image at index {idx}: {e}")
                return self.__getitem__((idx + 1) % len(self.hf_dataset))                    
                
        
    dataset = HFDatasetWrapper(dataset_train, transform=transform)


    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )


    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    if (args.ckpt != None) and (not args.fine_tune):
        train_steps = int(args.ckpt.split(".")[0].split("/")[-1])
    log_steps = 0
    running_loss = 0
    running_diff_loss = 0
    running_reg_loss = 0
    start_time = time()

    # Labels to condition the model with (feel free to change):
    ys = torch.randint(1000, size=(local_batch_size,), device=device)
    use_cfg = args.cfg_scale > 1.0
    # Create sampling noise:
    n = ys.size(0)
    zs = torch.randn(n, 4, latent_size, latent_size, device=device)

    # Setup classifier-free guidance:
    if use_cfg:
        zs = torch.cat([zs, zs], 0)
        y_null = torch.tensor([1000] * n, device=device)
        ys = torch.cat([ys, y_null], 0)
        sample_model_kwargs = dict(y=ys, cfg_scale=args.cfg_scale)
        model_fn = ema.forward_with_cfg
    else:
        sample_model_kwargs = dict(y=ys)
        model_fn = ema.forward

    logger.info(f"Training for {args.epochs} epochs...")
    start_epoch = train_steps // len(loader)
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:


            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            model_kwargs = dict(y=y)
            loss_dict = transport.training_losses(model, x, model_kwargs, loss_type=args.loss_type,  multi_blocks=args.multi_blocks_match)
            diff_loss = loss_dict["loss"].mean()
            reg = loss_dict["reg"].mean()
            loss = diff_loss + args.reg_weight*reg
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            running_diff_loss += diff_loss.item()
            running_reg_loss += reg.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_diff_loss = torch.tensor(running_diff_loss / log_steps, device=device)
                avg_reg_loss = torch.tensor(running_reg_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                if args.wandb:
                    wandb_utils.log(
                        { "train loss": avg_loss, "train diff loss": avg_diff_loss, "train reg loss": avg_reg_loss, "train steps/sec": steps_per_sec },
                        step=train_steps
                    )
                # Reset monitoring variables:
                running_loss = 0
                running_diff_loss = 0
                running_reg_loss = 0
                log_steps = 0
                start_time = time()

            # Save SiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()
            
            if train_steps % args.sample_every == 0 and train_steps > 0:
                logger.info("Generating EMA samples...")
                sample_fn = transport_sampler.sample_ode() # default to ode sampling
                samples = sample_fn(zs, model_fn, **sample_model_kwargs)[-1]
                dist.barrier()

                if use_cfg: #remove null samples
                    samples, _ = samples.chunk(2, dim=0)
                samples = vae.decode(samples / 0.18215).sample
                out_samples = torch.zeros((args.global_batch_size, 3, args.image_size, args.image_size), device=device)
                dist.all_gather_into_tensor(out_samples, samples)
                if args.wandb:
                    wandb_utils.log_image(out_samples, train_steps)
                logging.info("Generating EMA samples done.")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train SiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="imagenet-1k")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--gt-encoder-depth", type=int, default=8)   
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=10000)
    parser.add_argument("--ckpt-every", type=int, default=10_000) 
    parser.add_argument("--sample-every", type=int, default=10_000) 
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--reg-weight", type=float, default=0.2)
    parser.add_argument("--loss-type", type=str, default="layer_sync")
    parser.add_argument("--cache-dir", type=str, default="./cache/")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-run-name", type=str, default="SiT-Training")
    parser.add_argument("--multi-blocks-match", type=ast.literal_eval, default=[], help="List of (block, match) tuples, e.g. '[(1,2), (10,5)]'")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a custom SiT checkpoint")
    parser.add_argument("--fine-tune", action="store_true")

    parse_transport_args(parser)
    args = parser.parse_args()
    main(args)
