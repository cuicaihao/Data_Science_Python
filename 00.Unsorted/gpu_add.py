from numba import cuda
import numpy as np
import math
from time import time


@cuda.jit
def gpu_add(a, b, result, n):
    # input : a, b
    # output: results
    # n-dim vector
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if (idx < n):
        result[idx] = a[idx] + b[idx]


def main():
    n = 2000*10000
    x = np.arange(n).astype(np.int32)
    y = 2*x

    # copy data to device
    # cuda.to_device()
    # cuda.copy_to_host()
    # cuda.device_array() // numpy.empty()
    x_device = cuda.to_device(x)
    y_device = cuda.to_device(y)

    # gpu_results = np.zeros(n)
    # init a memory for results
    gpu_results = cuda.device_array(n)
    # cpu_results = np.zeros(n)
    cpu_results = np.empty(n)

    threads_per_block = 1024
    blocks_per_grid = math.ceil(n / threads_per_block)

    start = time()
    gpu_add[blocks_per_grid, threads_per_block](x, y, gpu_results, n)
    cuda.synchronize()
    print("GPU vector add time "+str(time()-start))

    start = time()
    cpu_results = np.add(x, y)
    print("CPU vector add time "+str(time()-start))

    if(np.array_equal(gpu_results, cpu_results)):
        print("results correct!")
    else:
        print("wrong")


if __name__ == "__main__":
    main()
