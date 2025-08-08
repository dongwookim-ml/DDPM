"""
CIFAR-10 Training Script for DDPM

This script trains a DDPM model on the CIFAR-10 dataset. It supports both
unconditional and class-conditional generation, with optional Weights & Biases logging.

The training process:
1. Loads CIFAR-10 dataset with proper preprocessing
2. Creates diffusion model with specified architecture
3. Trains using the DDPM objective (predicting added noise)
4. Regularly evaluates on test set and generates samples
5. Saves model checkpoints periodically

Usage:
    python train_cifar.py --iterations 100000 --batch_size 128 --learning_rate 2e-4
"""

import argparse
import datetime
import torch
import wandb

from torch.utils.data import DataLoader
from torchvision import datasets
from ddpm import script_utils


def main():
    """Main training loop for DDPM on CIFAR-10."""
    args = create_argparser().parse_args()
    device = args.device

    try:
        # Create diffusion model and optimizer
        diffusion = script_utils.get_diffusion_from_args(args).to(device)
        optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.learning_rate)

        # Load model and optimizer checkpoints if provided
        if args.model_checkpoint is not None:
            diffusion.load_state_dict(torch.load(args.model_checkpoint))
        if args.optim_checkpoint is not None:
            optimizer.load_state_dict(torch.load(args.optim_checkpoint))

        # Initialize Weights & Biases logging if requested
        if args.log_to_wandb:
            if args.project_name is None:
                raise ValueError("args.log_to_wandb set to True but args.project_name is None")

            run = wandb.init(
                project=args.project_name,
                entity='treaptofun',  # Replace with your W&B username
                config=vars(args),
                name=args.run_name,
            )
            wandb.watch(diffusion)  # Track model gradients and parameters

        batch_size = args.batch_size

        # Prepare CIFAR-10 datasets
        train_dataset = datasets.CIFAR10(
            root='./cifar_train',
            train=True,
            download=True,
            transform=script_utils.get_transform(),  # Normalize to [-1, 1]
        )

        test_dataset = datasets.CIFAR10(
            root='./cifar_test',
            train=False,
            download=True,
            transform=script_utils.get_transform(),
        )

        # Create data loaders
        # Training loader cycles infinitely for easier training loop
        train_loader = script_utils.cycle(DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,  # Ensure consistent batch sizes
            num_workers=2,
        ))
        # Test loader for evaluation
        test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, num_workers=2)
        
        acc_train_loss = 0  # Accumulated training loss for logging

        # Main training loop
        for iteration in range(1, args.iterations + 1):
            diffusion.train()

            # Get next training batch
            x, y = next(train_loader)
            x = x.to(device)
            y = y.to(device)

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

            # Update EMA model weights
            diffusion.update_ema()
            
            # Evaluation and logging
            if iteration % args.log_rate == 0:
                # Compute test set loss for evaluation
                test_loss = 0
                with torch.no_grad():
                    diffusion.eval()
                    for x, y in test_loader:
                        x = x.to(device)
                        y = y.to(device)

                        if args.use_labels:
                            loss = diffusion(x, y)
                        else:
                            loss = diffusion(x)

                        test_loss += loss.item()
                
                # Generate sample images for visual evaluation
                if args.use_labels:
                    # Generate one sample per class (0-9 for CIFAR-10)
                    samples = diffusion.sample(10, device, y=torch.arange(10, device=device))
                else:
                    # Generate 10 unconditional samples
                    samples = diffusion.sample(10, device)
                
                # Convert samples to [0, 1] range and proper format for logging
                samples = ((samples + 1) / 2).clip(0, 1).permute(0, 2, 3, 1).numpy()

                # Calculate average losses
                test_loss /= len(test_loader)
                acc_train_loss /= args.log_rate

                # Log metrics and samples to W&B
                wandb.log({
                    "test_loss": test_loss,
                    "train_loss": acc_train_loss,
                    "samples": [wandb.Image(sample) for sample in samples],
                })

                acc_train_loss = 0
            
            # Save model and optimizer checkpoints
            if iteration % args.checkpoint_rate == 0:
                model_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration}-model.pth"
                optim_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration}-optim.pth"

                torch.save(diffusion.state_dict(), model_filename)
                torch.save(optimizer.state_dict(), optim_filename)
        
        # Clean up W&B run
        if args.log_to_wandb:
            run.finish()
    except KeyboardInterrupt:
        # Handle graceful shutdown on Ctrl+C
        if args.log_to_wandb:
            run.finish()
        print("Keyboard interrupt, run finished early")


def create_argparser():
    """
    Create command line argument parser for training script.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    # Set default device based on availability
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Generate unique run name with timestamp
    run_name = datetime.datetime.now().strftime("ddpm-%Y-%m-%d-%H-%M-%S")
    
    # Default training hyperparameters
    defaults = dict(
        learning_rate=2e-4,   # Learning rate for Adam optimizer
        batch_size=128,       # Training batch size
        iterations=800000,    # Total training iterations

        log_to_wandb=True,    # Whether to log to W&B
        log_rate=1000,        # How often to log metrics and samples
        checkpoint_rate=1000, # How often to save checkpoints
        log_dir="~/ddpm_logs", # Directory for saving checkpoints
        project_name=None,    # W&B project name
        run_name=run_name,    # W&B run name with timestamp

        model_checkpoint=None, # Path to model checkpoint to resume from
        optim_checkpoint=None, # Path to optimizer checkpoint to resume from

        schedule_low=1e-4,    # Starting beta value for linear schedule
        schedule_high=0.02,   # Ending beta value for linear schedule

        device=device,        # Device for training
    )
    
    # Add diffusion model defaults
    defaults.update(script_utils.diffusion_defaults())

    parser = argparse.ArgumentParser()
    script_utils.add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()