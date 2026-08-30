import sys
import torch
import time
import argparse
import logging

logging.basicConfig(stream=sys.stdout, format="%(asctime)s %(levelname)s : %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

DTYPE = {"float32": torch.float32,
          "float16": torch.float16,
          "bfloat16": torch.bfloat16,
         }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multiply 2 random matrices")
    parser.add_argument("dim_A_1",
                        type=int,
                        help="First dimension of first matrix."
                       )
    parser.add_argument("dim_A_2",
                        type=int,
                        help="Second dimension of first matrix."
                       )
    parser.add_argument("dim_B_1",
                        type=int,
                        help="First dimension of second matrix."
                       )
    parser.add_argument("dim_B_2",
                        type=int,
                        help="Second dimension of second matrix."
                       )
    parser.add_argument("--dtype",
                        required=False,
                        type=str,
                        help="Data type of matrices (`float32`, `float16`, or `bfloat16`).",
                        default="float32"
                       )
    parser.add_argument("--output",
                        required=False,
                        type=str,
                        help="Profile output path.",
                        default="./profile.pickle"
                       )
    
    args = parser.parse_args()

    p = args.dim_A_1
    q = args.dim_A_2
    r = args.dim_B_1
    s = args.dim_B_2

    dtype = DTYPE[args.dtype]
    output = args.output

    logger.info(f"Multiplying matrices of shape ({p}, {q}) and ({r}, {s}) with dtype {dtype}.")

    # Enable memory profiler
    torch.cuda.memory._record_memory_history(max_entries=100000)

    # Initialize random tensor
    mat_a = torch.rand(p, q, dtype=dtype, device="cuda")
    time.sleep(1)
    mat_b = torch.rand(r, s, dtype=dtype, device="cuda")

    # Multiply tensor
    time.sleep(3)
    res = mat_a @ mat_b

    # Stop profiler
    torch.cuda.memory._dump_snapshot(output)
    torch.cuda.memory._record_memory_history(enabled=None)