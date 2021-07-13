import gc
import time
import torch

cpu = torch.device('cpu')
gpu0 = torch.device('cuda:0')
gpu1 = torch.device('cuda:1')

def swap_time_d2h(size=1024):
    a = torch.rand((size//16, 4), device=gpu0)
    torch.cuda.current_stream(gpu0).synchronize()
    tic = time.time()
    a.to(cpu)
    torch.cuda.current_stream(gpu0).synchronize()
    toc = time.time()
    return toc - tic

def swap_time_h2d(size=1024):
    a = torch.rand((size//16, 4), device=cpu)
    torch.cuda.current_stream(gpu0).synchronize()
    tic = time.time()
    a.to(gpu0)
    torch.cuda.current_stream(gpu0).synchronize()
    toc = time.time()
    return toc - tic

def swap_time_d2d(size=1024):
    a = torch.rand((size//16, 4), device=gpu0)
    torch.cuda.current_stream(gpu0).synchronize()
    torch.cuda.current_stream(gpu1).synchronize()
    tic = time.time()
    a.to(gpu1)
    torch.cuda.current_stream(gpu0).synchronize()
    torch.cuda.current_stream(gpu1).synchronize()
    toc = time.time()
    return toc - tic

for size_pow in range(15, 31):
    size = 2**size_pow

    gc.collect()
    swap_time_d2h(size)
    times_d2h = [ swap_time_d2h(size) for _ in range(10) ]

    print(f"message size: 2^{size_pow} bytes, d2h time: {sum(times_d2h) * 100}ms, speed: {size/1024/1024/(sum(times_d2h) * 100)} MB/ms", flush=True)

    gc.collect()
    swap_time_h2d(size)
    times_h2d = [ swap_time_h2d(size) for _ in range(10) ]

    print(f"message size: 2^{size_pow} bytes, h2d time: {sum(times_h2d) * 100}ms, speed: {size/1024/1024/(sum(times_h2d) * 100)} MB/ms", flush=True)

    gc.collect()
    swap_time_d2d(size)
    times_d2d = [ swap_time_d2d(size) for _ in range(10) ]

    print(f"message size: 2^{size_pow} bytes, d2d time: {sum(times_d2d) * 100}ms, speed: {size/1024/1024/(sum(times_d2d) * 100)} MB/ms", flush=True)

    print()
