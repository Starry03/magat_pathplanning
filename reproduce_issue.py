import torch
import sys
import os

# Add project root to path so we can import test.layer.time_graph
sys.path.append(os.path.abspath("/home/starry/Documents/uni_project/magat_pathplanning"))

from test.layer.time_graph import TimeDelayedAggregation

def run_test():
    # Mock dimensions based on error
    # shape '[640, 128]' is invalid for input of size 5242880
    # 640 = B_N -> B=64, N=10?
    B = 64
    N = 10
    C = 128
    T = 3 # nGraphFilterTaps = 3 in Model
    E = 1

    print(f"B={B}, N={N}, C={C}, T={T}, E={E}")

    # Setup
    time_gnn = TimeDelayedAggregation(128, T)
    x = torch.randn(B * N, C)
    x_prev = torch.randn(B, N, T-1, C)

    # Problem case: S with dimension [B, 1, N, N]
    S_bad = torch.randn(B, E, N, N)

    print("\n--- Testing with 4D S [B, 1, N, N] ---")
    try:
        time_gnn(x, x_prev, S_bad)
        print("Success (Unexpected)")
    except RuntimeError as e:
        print(f"Caught expected error: {e}")
    except Exception as e:
        print(f"Caught unexpected exception: {type(e).__name__}: {e}")

    # Fix case: S with dimension [B, N, N]
    print("\n--- Testing with 3D S [B, N, N] ---")
    S_good = S_bad.squeeze(1)
    try:
        time_gnn(x, x_prev, S_good)
        print("Success (Fixed)")
    except Exception as e:
        print(f"Caught unexpected error: {e}")

if __name__ == "__main__":
    run_test()
