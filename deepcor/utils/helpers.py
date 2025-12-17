"""Helper utilities for DeepCor."""

import torch
import time


def check_gpu_and_speedup(tensor_size=(1000, 1000), n_iter=100):
    """
    Check if GPU is available and calculate speedup vs CPU.

    Args:
        tensor_size: Size of random tensors for benchmark
        n_iter: Number of iterations to average

    Returns:
        Dictionary with GPU information and speedup metrics
    """
    # Generate random data
    a_cpu = torch.randn(tensor_size)
    b_cpu = torch.randn(tensor_size)

    # Time on CPU
    start_cpu = time.time()
    for _ in range(n_iter):
        c_cpu = torch.mm(a_cpu, b_cpu)
    end_cpu = time.time()
    cpu_time = (end_cpu - start_cpu) / n_iter

    gpu_available = torch.cuda.is_available()
    gpu_time = None
    speedup = None
    gpu_name = None

    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        a_gpu = a_cpu.to('cuda')
        b_gpu = b_cpu.to('cuda')

        # Warm up GPU
        for _ in range(5):
            _ = torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()

        # Time on GPU
        start_gpu = time.time()
        for _ in range(n_iter):
            c_gpu = torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()
        end_gpu = time.time()
        gpu_time = (end_gpu - start_gpu) / n_iter
        speedup = cpu_time / gpu_time if gpu_time > 0 else None

    return {
        'gpu_available': gpu_available,
        'gpu_name': gpu_name,
        'cpu_time': cpu_time,
        'gpu_time': gpu_time,
        'speedup': speedup
    }
