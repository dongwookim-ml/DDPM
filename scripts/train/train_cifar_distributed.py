"""
CIFAR-10 Distributed Training Script for DDPM

This script trains a DDPM model on the CIFAR-10 dataset using multiple GPUs with
PyTorch's DistributedDataParallel (DDP) for faster training.

Features:
- Multi-GPU distributed training
- Automatic GPU detection and setup
- Synchronized logging and checkpointing
- Improved batch size scaling

Usage:
    # Single node, multiple GPUs
    python -m torch.distributed.launch --nproc_per_node=4 train_cifar_distributed.py --project_name "ddpm-cifar10-4gpu"
    
    # Or using torchrun (PyTorch 1.10+)
    torchrun --nproc_per_node=4 train_cifar_distributed.py --project_name "ddpm-cifar10-4gpu"
"""

import argparse
import datetime
import os
import warnings
import torch
import torch.distributed as dist
import wandb

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets
from ddpm.utils import script_utils

# Suppress gradient stride warnings that can slow down training
warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides")


def setup_distributed():
    """Initialize distributed training environment."""
    # Initialize the process group
    dist.init_process_group(backend='nccl')
    
    # Get rank and world size
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    # Set device for this process
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    return rank, world_size, local_rank, device


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def main():
    """Main distributed training loop for DDPM on CIFAR-10."""
    # Setup distributed training
    rank, world_size, local_rank, device = setup_distributed()
    
    # Parse arguments
    args = create_argparser().parse_args()
    
    # Adjust batch size for distributed training
    # Total effective batch size = batch_size * world_size
    args.batch_size = args.batch_size // world_size
    if rank == 0:
        print(f"Using {world_size} GPUs")
        print(f"Per-GPU batch size: {args.batch_size}")
        print(f"Effective total batch size: {args.batch_size * world_size}")

    try:
        # Create diffusion model and move to device
        diffusion = script_utils.get_diffusion_from_args(args).to(device)
        
        # Wrap model with DDP
        diffusion = DDP(
            diffusion, 
            device_ids=[local_rank], 
            output_device=local_rank,
            find_unused_parameters=True  # Handle EMA parameters that don't receive gradients
        )
        
        # Create optimizer
        optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.learning_rate)

        # Load model and optimizer checkpoints if provided (only on rank 0)
        if rank == 0:
            if args.model_checkpoint is not None:
                checkpoint = torch.load(args.model_checkpoint, map_location=device)
                diffusion.module.load_state_dict(checkpoint)
                print(f"Loaded model checkpoint from {args.model_checkpoint}")
            if args.optim_checkpoint is not None:
                optimizer.load_state_dict(torch.load(args.optim_checkpoint, map_location=device))
                print(f"Loaded optimizer checkpoint from {args.optim_checkpoint}")

        # Initialize Weights & Biases logging (only on rank 0)
        if args.log_to_wandb and rank == 0:
            if args.project_name is None:
                raise ValueError("args.log_to_wandb set to True but args.project_name is None")

            run = wandb.init(
                project=args.project_name,
                entity='dongwoo-kim-postech',
                config=vars(args),
                name=args.run_name,
            )
            wandb.watch(diffusion.module)  # Track model gradients and parameters

        # Prepare CIFAR-10 datasets
        train_dataset = datasets.CIFAR10(
            root='./cifar_train',
            train=True,
            download=(rank == 0),  # Only download on rank 0
            transform=script_utils.get_transform(),
        )

        test_dataset = datasets.CIFAR10(
            root='./cifar_test',
            train=False,
            download=(rank == 0),  # Only download on rank 0
            transform=script_utils.get_transform(),
        )
        
        # Wait for rank 0 to download data
        if world_size > 1:
            dist.barrier()

        # Create distributed samplers
        train_sampler = DistributedSampler(
            train_dataset, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=True
        )
        
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )

        # Create data loaders with distributed samplers
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=4,  # Increased for multi-GPU
            pin_memory=True,
            drop_last=True,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            sampler=test_sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        
        # Create cycling iterator for training
        train_iter = script_utils.cycle(train_loader)
        
        acc_train_loss = 0  # Accumulated training loss for logging

        # Main training loop
        for iteration in range(1, args.iterations + 1):
            # Set epoch for sampler (important for proper shuffling)
            train_sampler.set_epoch(iteration)
            
            diffusion.train()

            # Get next training batch
            x, y = next(train_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Forward pass: compute diffusion loss
            if args.use_labels:
                loss = diffusion(x, y)  # Conditional generation
            else:
                loss = diffusion(x)     # Unconditional generation

            acc_train_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update EMA model weights (only on module, not DDP wrapper)
            diffusion.module.update_ema()
            
            # Evaluation and logging (only on rank 0)
            if iteration % args.log_rate == 0 and rank == 0:
                # Compute test set loss for evaluation
                test_loss = 0
                test_count = 0
                with torch.no_grad():
                    diffusion.eval()
                    for x, y in test_loader:
                        x = x.to(device, non_blocking=True)
                        y = y.to(device, non_blocking=True)

                        if args.use_labels:
                            loss = diffusion(x, y)
                        else:
                            loss = diffusion(x)

                        test_loss += loss.item()
                        test_count += 1
                
                # Generate sample images for visual evaluation
                if args.use_labels:
                    # Generate one sample per class (0-9 for CIFAR-10)
                    samples = diffusion.module.sample(10, device, y=torch.arange(10, device=device))
                else:
                    # Generate 10 unconditional samples
                    samples = diffusion.module.sample(10, device)
                
                # Convert samples to [0, 1] range and proper format for logging
                samples = ((samples + 1) / 2).clip(0, 1).permute(0, 2, 3, 1).cpu().numpy()

                # Calculate average losses
                test_loss /= test_count
                acc_train_loss /= args.log_rate

                print(f"Iteration {iteration:6d} | Train Loss: {acc_train_loss:.4f} | Test Loss: {test_loss:.4f}")

                # Log metrics and samples to W&B
                if args.log_to_wandb:
                    wandb.log({
                        "iteration": iteration,
                        "test_loss": test_loss,
                        "train_loss": acc_train_loss,
                        "samples": [wandb.Image(sample) for sample in samples],
                    })

                acc_train_loss = 0
            
            # Save model and optimizer checkpoints (only on rank 0)
            if iteration % args.checkpoint_rate == 0 and rank == 0:
                # Expand log_dir path
                log_dir = os.path.expanduser(args.log_dir)
                os.makedirs(log_dir, exist_ok=True)
                
                model_filename = f"{log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration}-model.pth"
                optim_filename = f"{log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration}-optim.pth"

                # Save the module state dict (not the DDP wrapper)
                torch.save(diffusion.module.state_dict(), model_filename)
                torch.save(optimizer.state_dict(), optim_filename)
                print(f"Saved checkpoint at iteration {iteration}")
        
        # Clean up W&B run (only on rank 0)
        if args.log_to_wandb and rank == 0:
            run.finish()
            
    except KeyboardInterrupt:
        if rank == 0:
            print("Keyboard interrupt, run finished early")
        # Handle graceful shutdown on Ctrl+C
        if args.log_to_wandb and rank == 0:
            run.finish()
    finally:
        # Clean up distributed training
        cleanup_distributed()


def create_argparser():
    """
    Create command line argument parser for distributed training script.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    # Generate unique run name with timestamp
    run_name = datetime.datetime.now().strftime("ddpm-%Y-%m-%d-%H-%M-%S")
    
    # Default training hyperparameters (scaled for multi-GPU)
    defaults = dict(
        learning_rate=2e-4,   # Learning rate for Adam optimizer
        batch_size=512,       # Total batch size (will be divided by number of GPUs)
        iterations=100000,    # Total training iterations (reduced for faster testing)

        log_to_wandb=True,    # Whether to log to W&B
        log_rate=500,         # How often to log metrics and samples
        checkpoint_rate=2000, # How often to save checkpoints
        log_dir="~/ddpm_logs", # Directory for saving checkpoints
        project_name=None,    # W&B project name
        run_name=run_name,    # W&B run name with timestamp

        model_checkpoint=None, # Path to model checkpoint to resume from
        optim_checkpoint=None, # Path to optimizer checkpoint to resume from

        schedule_low=1e-4,    # Starting beta value for linear schedule
        schedule_high=0.02,   # Ending beta value for linear schedule
    )
    
    # Add diffusion model defaults
    defaults.update(script_utils.diffusion_defaults())

    parser = argparse.ArgumentParser()
    script_utils.add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
