from numba import cuda


def cpu_print(N):
    for i in range(N):
        print(i)


@cuda.jit
def gpu_print(N):
    # GPU kennel function
    # [gridDim, blockDim]
    # blockDim is Num of thread in block. threadIdx <= blockDim
    # gridDim is Num of block. blockIdx <= gridDim
    # block and grid can be 2D or 3D
    # example: if you want to start 1000 threads. set blockDim 128, then grid dim ceil(1000/128)=8
    # gpuWork[8, 128], 1024 threads are initialized but only 1000 will be executed.
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if (idx < N):
        print(idx)


def main():
    print("GPU:")
    gpu_print[2, 4](8)
    cuda.synchronize()
    print("CPU:")
    cpu_print(8)


if __name__ == "__main__":
    main()

# run the program by executing
# CUDA_VISIBLE_DEVICES='0' python gpu_print.py
