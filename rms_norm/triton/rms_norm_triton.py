


@triton.jit
def rms_norm_kernel(
    x_ptr, output_ptr,
    partial_sums_ptr,
    m, n, 
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0) # row
    pid_n = tl.program_id(1) # col

    # compute offsets

    # load values for block

    # compute squared sum for block
    # do an atomic add for partial sum and put into shared memory partial sum ptr

    # synchronize threads 

    # each first column block for each row 
        # computes RMS value for the block 
        # store in global partial sums ptr 

    # synchronize thread

    # each block for each row divides each element by the RMS value 

# Lunches kernel
# grid of size (rows, n // BLOCK_SIZE)
# global memory for partial sums ptr of size (rows, 1)
