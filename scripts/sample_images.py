"""
Image Sampling Script for DDPM

This script generates images from a trained DDPM model. It supports both
unconditional and class-conditional generation, saving the results as
individual image files.

The sampling process:
1. Loads a trained diffusion model from checkpoint
2. Generates specified number of images
3. Converts samples from [-1, 1] to [0, 1] range
4. Saves each generated image as a PNG file

Usage:
    python sample_images.py --model_path path/to/model.pth --save_dir ./samples --num_images 1000
"""

import argparse
import torch
import torchvision

from ddpm import script_utils


def main():
    """Main function for generating and saving images from trained DDPM model."""
    args = create_argparser().parse_args()
    device = args.device

    try:
        # Create diffusion model with same architecture as training
        diffusion = script_utils.get_diffusion_from_args(args).to(device)
        # Load trained model weights
        diffusion.load_state_dict(torch.load(args.model_path))

        if args.use_labels:
            # Class-conditional generation: generate equal numbers for each class
            for label in range(10):  # CIFAR-10 has 10 classes
                # Create label tensor for this class
                y = torch.ones(args.num_images // 10, dtype=torch.long, device=device) * label
                # Generate samples for this class
                samples = diffusion.sample(args.num_images // 10, device, y=y)

                # Save each generated image
                for image_id in range(len(samples)):
                    # Convert from [-1, 1] to [0, 1] range and clamp
                    image = ((samples[image_id] + 1) / 2).clip(0, 1)
                    # Save with class label in filename
                    torchvision.utils.save_image(image, f"{args.save_dir}/{label}-{image_id}.png")
        else:
            # Unconditional generation: generate specified number of images
            samples = diffusion.sample(args.num_images, device)

            # Save each generated image
            for image_id in range(len(samples)):
                # Convert from [-1, 1] to [0, 1] range and clamp
                image = ((samples[image_id] + 1) / 2).clip(0, 1)
                # Save with sequential numbering
                torchvision.utils.save_image(image, f"{args.save_dir}/{image_id}.png")
    except KeyboardInterrupt:
        print("Keyboard interrupt, generation finished early")


def create_argparser():
    """
    Create command line argument parser for sampling script.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    # Set default device based on availability
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Default sampling parameters
    defaults = dict(
        num_images=10000,  # Number of images to generate
        device=device      # Device for inference
    )
    
    # Add diffusion model defaults (for model architecture)
    defaults.update(script_utils.diffusion_defaults())

    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save generated images")
    # Optional arguments from defaults
    script_utils.add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()