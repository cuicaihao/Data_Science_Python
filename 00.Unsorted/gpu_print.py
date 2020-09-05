from numba import cuda


def cpu_print():
    print("CPU print")


@cuda.jit
def gpu_print():
    # GPU kennel function
    print("GPU print")


def main():
    gpu_print[1, 2]()
    cuda.synchronize()
    cpu_print


if __name__ == "__main__":
    main()

# run the program by executing
# CUDA_VISIBLE_DEVICES='0' python gpu_print.py
