#!.venv/bin/python
import os, sys
import numpy as np
import scipy.io as sio

def infer_and_print(mat):
    def shp(k):
        return tuple(np.array(mat[k]).shape)
    Hm, Wm = shp("map")
    T, N, C, HL, WL = shp("inputTensor")
    A = shp("target")[-1]
    print("== Symbols ==")
    print(f"T={T} (steps), N={N} (agents), C={C} (channels), HLxWL={HL}x{WL} (patch), A={A} (actions), Hm x Wm={Hm}x{Wm} (map)")
    if HL == WL:
        FOV = HL - 2
        print(f"FOV={FOV}  (since HL=WL=FOV+2)")
    print("\n== File keys and shapes ==")
    for k, v in mat.items():
        if k.startswith("__"): continue
        arr = np.array(v)
        print(f"{k:>12}: shape={arr.shape}, dtype={arr.dtype}")
    # Sanity checks
    ok = True
    ok &= shp("GSO") == (T, N, N)
    ok &= shp("target") == (T, N, A)
    ok &= shp("inputState") == (T, N, 2)
    ok &= shp("goal") == (N, 2)
    print("\nSanity checks:",
          "OK" if ok else f"Mismatch: GSO={shp('GSO')}, target={shp('target')}, inputState={shp('inputState')}, goal={shp('goal')}")
    print("\n== What the model sees per batch ==")
    print(f"step_input_tensor: [B, {N}, {C}, {HL}, {WL}]")
    print(f"step_input_GSO:    [B, {N}, {N}]")
    print(f"step_target:       [B, {N}, {A}]  -> target_idx: [B*{N}] via argmax")

def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        root = "./offlineExpert/trainset"
        # pick first file found recursively
        path = None
        for mode in ("train", "valid", "test"):
            mode_dir = next((os.path.join(root, d, mode)
                             for d in os.listdir(root)
                             if os.path.isdir(os.path.join(root, d, mode))), None)
            if mode_dir and os.path.isdir(mode_dir):
                files = sorted(f for f in os.listdir(mode_dir) if f.endswith(".mat"))
                if files:
                    path = os.path.join(mode_dir, files[0]); break
        if path is None:
            print("No .mat found. Pass a path: readmat.py /path/to/file.mat"); return
    mat = sio.loadmat(path)
    print(f"File: {os.path.basename(path)}")
    infer_and_print(mat)

if __name__ == "__main__":
    main()