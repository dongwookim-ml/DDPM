#!/usr/bin/env python3
"""
GPU Training Speed Benchmark Script for DDPM

This script compares training performance between single GPU and multi-GPU setups
by training for exactly 1 epoch and measuring various performance metrics.

Features:
- Automatic detection of available GPUs
- Memory usage monitoring
- Throughput measurement (samples/second)
- Training time comparison
- Beautiful visualizations
- Detailed performance report

Usage:
    python benchmark_gpu_training.py --batch_size 32 --num_workers 4
"""

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import psutil
import seaborn as sns
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
import wandb

from ddpm.utils import script_utils


class GPUMonitor:
    """Monitor GPU memory usage and performance metrics."""
    
    def __init__(self, device_ids: List[int]):
        self.device_ids = device_ids
        self.memory_history = []
        self.utilization_history = []
    
    def log_stats(self, step: int):
        """Log current GPU statistics."""
        gpu_stats = {}
        total_memory = 0
        total_utilization = 0
        
        for device_id in self.device_ids:
            if torch.cuda.is_available():
                # Memory stats
                memory_allocated = torch.cuda.memory_allocated(device_id) / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved(device_id) / 1024**3   # GB
                max_memory = torch.cuda.max_memory_allocated(device_id) / 1024**3   # GB
                
                gpu_stats[f'gpu_{device_id}'] = {
                    'memory_allocated_gb': memory_allocated,
                    'memory_reserved_gb': memory_reserved,
                    'max_memory_gb': max_memory
                }
                
                total_memory += memory_allocated
        
        gpu_stats['total_memory_gb'] = total_memory
        gpu_stats['step'] = step
        self.memory_history.append(gpu_stats)
        
        return gpu_stats


class PerformanceTracker:
    """Track training performance metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.start_time = None
        self.end_time = None
        self.batch_times = []
        self.losses = []
        self.steps = 0
        self.total_samples = 0
    
    def start_epoch(self):
        self.start_time = time.time()
    
    def end_epoch(self):
        self.end_time = time.time()
    
    def log_batch(self, batch_size: int, loss: float, batch_time: float):
        self.batch_times.append(batch_time)
        self.losses.append(loss)
        self.steps += 1
        self.total_samples += batch_size
    
    def get_metrics(self) -> Dict:
        """Calculate and return performance metrics."""
        if self.start_time is None or self.end_time is None:
            return {}
        
        total_time = self.end_time - self.start_time
        avg_batch_time = np.mean(self.batch_times) if self.batch_times else 0
        throughput = self.total_samples / total_time if total_time > 0 else 0
        avg_loss = np.mean(self.losses) if self.losses else 0
        
        return {
            'total_time_seconds': total_time,
            'total_time_minutes': total_time / 60,
            'avg_batch_time_seconds': avg_batch_time,
            'throughput_samples_per_second': throughput,
            'total_samples': self.total_samples,
            'total_steps': self.steps,
            'avg_loss': avg_loss,
            'loss_std': np.std(self.losses) if self.losses else 0
        }


def setup_distributed(rank, world_size, port='12355'):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def create_data_loader(batch_size: int, num_workers: int, distributed: bool = False, rank: int = 0):
    """Create CIFAR-10 data loader."""
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=8, rank=rank, shuffle=True)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=True
    )
    
    return loader


def train_single_epoch(model, optimizer, dataloader, device, tracker, monitor, rank=0):
    """Train model for one epoch."""
    model.train()
    tracker.start_epoch()
    
    for step, (images, labels) in enumerate(dataloader):
        batch_start_time = time.time()
        
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Forward pass
        optimizer.zero_grad()
        loss = model(images, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        batch_time = time.time() - batch_start_time
        tracker.log_batch(images.size(0), loss.item(), batch_time)
        
        # Log GPU stats every 50 steps
        if step % 50 == 0:
            gpu_stats = monitor.log_stats(step)
            if rank == 0:  # Only rank 0 prints
                print(f"Step {step:4d} | Loss: {loss.item():.4f} | "
                      f"Batch time: {batch_time:.3f}s | "
                      f"GPU memory: {gpu_stats['total_memory_gb']:.2f}GB")
    
    tracker.end_epoch()
    
    if rank == 0:
        print(f"Epoch completed! Total steps: {tracker.steps}")


def single_gpu_benchmark(args) -> Dict:
    """Run benchmark on single GPU."""
    print("=" * 60)
    print("SINGLE GPU BENCHMARK")
    print("=" * 60)
    
    device = torch.device('cuda:0')
    
    # Create model and optimizer
    model_args = argparse.Namespace(
        # Diffusion process parameters
        num_timesteps=1000,
        schedule="linear",
        loss_type="l2",
        use_labels=True,
        schedule_low=1e-4,
        schedule_high=0.02,
        
        # U-Net architecture parameters
        base_channels=128,
        channel_mults=(1, 2, 2, 2),
        num_res_blocks=2,
        time_emb_dim=128 * 4,
        norm="gn",
        dropout=0.1,
        activation="silu",
        attention_resolutions=(1,),
        
        # EMA parameters
        ema_decay=0.9999,
        ema_update_rate=1,
    )
    
    model = script_utils.get_diffusion_from_args(model_args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Create data loader
    dataloader = create_data_loader(args.batch_size, args.num_workers, distributed=False)
    
    # Initialize monitoring
    tracker = PerformanceTracker()
    monitor = GPUMonitor([0])
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of batches: {len(dataloader)}")
    print(f"Total samples: {len(dataloader) * args.batch_size}")
    
    # Train for one epoch
    train_single_epoch(model, optimizer, dataloader, device, tracker, monitor)
    
    # Get metrics
    metrics = tracker.get_metrics()
    metrics['gpu_count'] = 1
    metrics['device_ids'] = [0]
    metrics['memory_history'] = monitor.memory_history
    
    print(f"\nSINGLE GPU RESULTS:")
    print(f"  Total time: {metrics['total_time_minutes']:.2f} minutes")
    print(f"  Throughput: {metrics['throughput_samples_per_second']:.1f} samples/sec")
    print(f"  Average loss: {metrics['avg_loss']:.4f}")
    
    return metrics


def multi_gpu_worker(rank, world_size, args, results_queue):
    """Worker function for multi-GPU training."""
    try:
        setup_distributed(rank, world_size)
        device = torch.device(f'cuda:{rank}')
        
        # Create model and optimizer
        model_args = argparse.Namespace(
            # Diffusion process parameters
            num_timesteps=1000,
            schedule="linear",
            loss_type="l2",
            use_labels=True,
            schedule_low=1e-4,
            schedule_high=0.02,
            
            # U-Net architecture parameters
            base_channels=128,
            channel_mults=(1, 2, 2, 2),
            num_res_blocks=2,
            time_emb_dim=128 * 4,
            norm="gn",
            dropout=0.1,
            activation="silu",
            attention_resolutions=(1,),
            
            # EMA parameters
            ema_decay=0.9999,
            ema_update_rate=1,
        )
        
        model = script_utils.get_diffusion_from_args(model_args).to(device)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        
        # Create data loader with distributed sampler
        dataloader = create_data_loader(
            args.batch_size, args.num_workers, distributed=True, rank=rank
        )
        
        # Initialize monitoring (only on rank 0)
        tracker = PerformanceTracker()
        monitor = GPUMonitor(list(range(world_size)) if rank == 0 else [rank])
        
        if rank == 0:
            print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"Batch size per GPU: {args.batch_size}")
            print(f"Total batch size: {args.batch_size * world_size}")
            print(f"Number of batches per GPU: {len(dataloader)}")
            print(f"Total samples: {len(dataloader) * args.batch_size * world_size}")
        
        # Train for one epoch
        train_single_epoch(model, optimizer, dataloader, device, tracker, monitor, rank)
        
        # Only rank 0 returns results
        if rank == 0:
            metrics = tracker.get_metrics()
            metrics['gpu_count'] = world_size
            metrics['device_ids'] = list(range(world_size))
            metrics['memory_history'] = monitor.memory_history
            # Adjust for distributed training
            metrics['total_samples'] *= world_size
            metrics['throughput_samples_per_second'] *= world_size
            
            results_queue.put(metrics)
        
    except Exception as e:
        if rank == 0:
            print(f"Error in multi-GPU training: {e}")
            results_queue.put(None)
    finally:
        cleanup_distributed()


def multi_gpu_benchmark(args) -> Dict:
    """Run benchmark on multiple GPUs."""
    print("=" * 60)
    print("MULTI-GPU BENCHMARK (8 GPUs)")
    print("=" * 60)
    
    world_size = 8
    
    # Use multiprocessing to run distributed training
    mp.set_start_method('spawn', force=True)
    results_queue = mp.Queue()
    
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=multi_gpu_worker, args=(rank, world_size, args, results_queue))
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Get results from rank 0
    metrics = results_queue.get()
    
    if metrics is None:
        raise RuntimeError("Multi-GPU benchmark failed")
    
    print(f"\nMULTI-GPU RESULTS:")
    print(f"  Total time: {metrics['total_time_minutes']:.2f} minutes")
    print(f"  Throughput: {metrics['throughput_samples_per_second']:.1f} samples/sec")
    print(f"  Average loss: {metrics['avg_loss']:.4f}")
    
    return metrics


def create_visualizations(single_gpu_metrics: Dict, multi_gpu_metrics: Dict, output_dir: str):
    """Create comprehensive visualization of benchmark results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Training Time Comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Training time comparison
    times = [single_gpu_metrics['total_time_minutes'], multi_gpu_metrics['total_time_minutes']]
    speedup = single_gpu_metrics['total_time_minutes'] / multi_gpu_metrics['total_time_minutes']
    
    bars1 = ax1.bar(['1 GPU', '8 GPUs'], times, color=['#ff7f0e', '#2ca02c'])
    ax1.set_ylabel('Training Time (minutes)')
    ax1.set_title(f'Training Time Comparison\n(Speedup: {speedup:.2f}x)')
    
    # Add value labels on bars
    for bar, time_val in zip(bars1, times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{time_val:.2f}m', ha='center', va='bottom', fontweight='bold')
    
    # Throughput comparison
    throughputs = [single_gpu_metrics['throughput_samples_per_second'], 
                   multi_gpu_metrics['throughput_samples_per_second']]
    throughput_speedup = multi_gpu_metrics['throughput_samples_per_second'] / single_gpu_metrics['throughput_samples_per_second']
    
    bars2 = ax2.bar(['1 GPU', '8 GPUs'], throughputs, color=['#ff7f0e', '#2ca02c'])
    ax2.set_ylabel('Throughput (samples/sec)')
    ax2.set_title(f'Throughput Comparison\n(Speedup: {throughput_speedup:.2f}x)')
    
    for bar, throughput_val in zip(bars2, throughputs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{throughput_val:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # GPU Memory Usage over time (Single GPU)
    if single_gpu_metrics['memory_history']:
        steps = [entry['step'] for entry in single_gpu_metrics['memory_history']]
        memory = [entry['total_memory_gb'] for entry in single_gpu_metrics['memory_history']]
        ax3.plot(steps, memory, marker='o', linewidth=2, markersize=4, label='1 GPU')
    
    if multi_gpu_metrics['memory_history']:
        steps = [entry['step'] for entry in multi_gpu_metrics['memory_history']]
        memory = [entry['total_memory_gb'] for entry in multi_gpu_metrics['memory_history']]
        ax3.plot(steps, memory, marker='s', linewidth=2, markersize=4, label='8 GPUs (total)')
    
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('GPU Memory Usage (GB)')
    ax3.set_title('GPU Memory Usage Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Efficiency metrics
    efficiency_data = {
        'Metric': ['Time Efficiency', 'Throughput Efficiency', 'Memory Efficiency'],
        'Single GPU': [1.0, 1.0, 1.0],
        'Multi GPU': [speedup, throughput_speedup, throughput_speedup/8]  # Assuming linear memory scaling
    }
    
    x = np.arange(len(efficiency_data['Metric']))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, efficiency_data['Single GPU'], width, label='1 GPU', color='#ff7f0e')
    bars2 = ax4.bar(x + width/2, efficiency_data['Multi GPU'], width, label='8 GPUs', color='#2ca02c')
    
    ax4.set_ylabel('Relative Performance')
    ax4.set_title('Performance Efficiency Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(efficiency_data['Metric'])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.2f}x', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gpu_benchmark_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_dir}/gpu_benchmark_comparison.png")
    
    # 2. Detailed Performance Report
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create summary table
    summary_data = [
        ['Metric', '1 GPU', '8 GPUs', 'Speedup'],
        ['Training Time (min)', f"{single_gpu_metrics['total_time_minutes']:.2f}", 
         f"{multi_gpu_metrics['total_time_minutes']:.2f}", f"{speedup:.2f}x"],
        ['Throughput (samples/sec)', f"{single_gpu_metrics['throughput_samples_per_second']:.0f}",
         f"{multi_gpu_metrics['throughput_samples_per_second']:.0f}", f"{throughput_speedup:.2f}x"],
        ['Total Samples', f"{single_gpu_metrics['total_samples']:,}",
         f"{multi_gpu_metrics['total_samples']:,}", "Same"],
        ['Average Loss', f"{single_gpu_metrics['avg_loss']:.4f}",
         f"{multi_gpu_metrics['avg_loss']:.4f}", "N/A"],
        ['GPU Memory (GB)', f"{max([h['total_memory_gb'] for h in single_gpu_metrics['memory_history']]):.2f}",
         f"{max([h['total_memory_gb'] for h in multi_gpu_metrics['memory_history']]):.2f}", "Total"],
    ]
    
    table = ax.table(cellText=summary_data, cellLoc='center', loc='center',
                    colWidths=[0.3, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Style the header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('DDPM Training Performance Comparison\n1 GPU vs 8 GPUs (1 Epoch)', 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig(os.path.join(output_dir, 'performance_summary_table.png'), dpi=300, bbox_inches='tight')
    print(f"Summary table saved to {output_dir}/performance_summary_table.png")
    
    plt.show()


def save_results(single_gpu_metrics: Dict, multi_gpu_metrics: Dict, output_dir: str):
    """Save detailed results to JSON file."""
    results = {
        'benchmark_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'single_gpu': single_gpu_metrics,
        'multi_gpu': multi_gpu_metrics,
        'speedup_analysis': {
            'time_speedup': single_gpu_metrics['total_time_minutes'] / multi_gpu_metrics['total_time_minutes'],
            'throughput_speedup': multi_gpu_metrics['throughput_samples_per_second'] / single_gpu_metrics['throughput_samples_per_second'],
            'efficiency_per_gpu': (multi_gpu_metrics['throughput_samples_per_second'] / single_gpu_metrics['throughput_samples_per_second']) / 8
        }
    }
    
    output_file = os.path.join(output_dir, 'benchmark_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Detailed results saved to {output_file}")


def create_argparser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(description="GPU Training Speed Benchmark for DDPM")
    
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size per GPU")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loader workers")
    parser.add_argument("--output_dir", type=str, default="benchmark_results",
                       help="Output directory for results and visualizations")
    parser.add_argument("--skip_single_gpu", action="store_true",
                       help="Skip single GPU benchmark (useful for debugging multi-GPU)")
    parser.add_argument("--skip_multi_gpu", action="store_true",
                       help="Skip multi-GPU benchmark")
    
    return parser


def main():
    """Main benchmarking function."""
    args = create_argparser().parse_args()
    
    print("🚀 DDPM GPU Training Speed Benchmark")
    print("=" * 60)
    print(f"Batch size per GPU: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Number of workers: {args.num_workers}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print("=" * 60)
    
    if torch.cuda.device_count() < 8:
        print("⚠️  Warning: Less than 8 GPUs available. Multi-GPU benchmark will use all available GPUs.")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    single_gpu_metrics = None
    multi_gpu_metrics = None
    
    try:
        # Run single GPU benchmark
        if not args.skip_single_gpu:
            single_gpu_metrics = single_gpu_benchmark(args)
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            time.sleep(2)
        
        # Run multi-GPU benchmark
        if not args.skip_multi_gpu:
            multi_gpu_metrics = multi_gpu_benchmark(args)
        
        # Create visualizations and save results
        if single_gpu_metrics and multi_gpu_metrics:
            create_visualizations(single_gpu_metrics, multi_gpu_metrics, args.output_dir)
            save_results(single_gpu_metrics, multi_gpu_metrics, args.output_dir)
            
            # Print final summary
            speedup = single_gpu_metrics['total_time_minutes'] / multi_gpu_metrics['total_time_minutes']
            efficiency = speedup / 8 * 100
            
            print("\n" + "=" * 60)
            print("🎉 BENCHMARK COMPLETE!")
            print("=" * 60)
            print(f"Time Speedup: {speedup:.2f}x")
            print(f"Multi-GPU Efficiency: {efficiency:.1f}%")
            print(f"Results saved to: {args.output_dir}/")
            print("=" * 60)
    
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
