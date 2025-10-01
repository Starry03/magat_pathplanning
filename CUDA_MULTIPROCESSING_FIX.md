# CUDA Multiprocessing Error Fix

## Problem Summary
The error occurred when using `torch.multiprocessing.spawn()` to create child processes for parallel testing. The issue was that CUDA tensors were being pickled and sent to child processes, which is not supported in PyTorch.

### Error Message
```
torch.AcceleratorError: CUDA error: invalid argument
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
```

### Root Causes
1. **Model with CUDA tensors**: The model was being passed to child processes while still on GPU
2. **Model internal state**: The `addGSO()` method stores `self.S` as a CUDA tensor, which persists in the model
3. **Config with CUDA device**: The `config.device` object (torch.device) cannot be pickled properly across processes

## Solution Applied

### Files Modified
- `agents/decentralplannerlocal_OnlineExpert.py`
- `agents/decentralplannerlocal_OnlineExpert_GAT.py`

### Changes Made

#### 1. Model Preparation Before Spawning
```python
# Move model to CPU and clear any CUDA state (like self.S in addGSO)
model_for_spawn = copy.deepcopy(self.model).cpu()
# Reset any internal CUDA tensors to None to ensure clean pickling
if hasattr(model_for_spawn, 'S'):
    model_for_spawn.S = None
```

#### 2. Config Serialization
```python
# Pass device info instead of device object
config_dict = vars(self.config).copy()
device_type = str(self.config.device)
config_dict['device'] = device_type
config_dict['gpu_device'] = self.config.gpu_device
```

#### 3. Child Process Setup
```python
def test_thread(thread_subid, thread_index, config_dict, model, lock, task_queue,
                recorder_queue, switch_toOnlineExpert):
    # Reconstruct config from dict and set device properly
    from easydict import EasyDict
    config = EasyDict(config_dict)
    # Recreate the device object in this process
    if 'cuda' in config.device:
        config.device = torch.device("cuda:{}".format(config.gpu_device))
    else:
        config.device = torch.device("cpu")
    
    # Move model to device in this process
    model = model.to(config.device)
    model.eval()
```

## How It Works

1. **Parent Process**:
   - Creates a CPU copy of the model with cleared CUDA state
   - Serializes config to a dict (string representation of device)
   - Spawns child processes with CPU model and dict config

2. **Child Process**:
   - Receives CPU model (picklable)
   - Reconstructs config from dict
   - Recreates proper torch.device object
   - Moves model to GPU in its own CUDA context
   - Performs inference

## Key Principles

1. **Never pass CUDA tensors across process boundaries**
2. **Each process must create its own CUDA context**
3. **Use CPU for inter-process communication**
4. **Serialize non-picklable objects (like torch.device) as strings/dicts**

## Testing

To verify the fix works:
```bash
# Run the training script
./scripts/train_DMap.sh

# Or run with debug mode
CUDA_LAUNCH_BLOCKING=1 ./scripts/train_DMap.sh
```

## Additional Notes

- The fix preserves the original functionality while making it compatible with multiprocessing
- Each child process has its own GPU context, avoiding conflicts
- The model weights are shared via pickle serialization (CPU-to-CPU)
- Performance should be similar or better due to proper parallelization

## Related Files That May Need Similar Fixes

If you encounter similar errors in other agent files, apply the same pattern to:
- `agents/decentralplannerlocal_OnlineExpert_LoadPreTrained.py`
- `agents/decentralplannerlocal_OnlineExpert_GAT_LoadPreTrained.py`
- `agents/decentralplannerlocal_OnlineExpert_GAT_returnGSO.py`
- `agents/decentralplannerlocal.py`
