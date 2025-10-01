# Deadlock Fix Documentation

## Problem Summary
The multiprocessing deadlock occurred in the `test_multi` function when using multiple processes for testing. The main issue was with the unreliable `task_queue.qsize() > 0` condition in the worker threads.

### Symptoms Observed
- Processes start and initialize tasks successfully
- Worker threads print "thread X initiated" and "thread X started"
- CPU and GPU usage drops to near zero
- Process appears to hang indefinitely

### Root Cause
The deadlock was caused by the following problematic code pattern in worker threads:

```python
while task_queue.qsize() > 0:
    try:
        input, load_target, makespanTarget, tensor_map, ID_dataset, mode, tmp_path = task_queue.get(block=False)
        # ... process task
    except Exception as e:
        print(e)
        return
```

**Problems with this approach:**
1. `qsize()` is unreliable in multiprocessing environments
2. Race condition: `qsize() > 0` might be true, but `get(block=False)` might still fail
3. Workers could exit prematurely when `qsize()` returns 0, even if more tasks are being added
4. No clear termination signal for workers

## Solution Applied

### Files Modified
- `agents/decentralplannerlocal_OnlineExpert_GAT.py`
- `agents/decentralplannerlocal_OnlineExpert.py` 
- `agents/decentralplannerlocal_OnlineExpert_GAT_returnGSO.py`

### Changes Made

#### 1. Worker Thread Logic (test_thread function)
**Before:**
```python
while task_queue.qsize() > 0:
    try:
        input, load_target, makespanTarget, tensor_map, ID_dataset, mode, tmp_path = task_queue.get(block=False)
        print('thread {} gets task {}'.format(thread_index, ID_dataset))
    except Exception as e:
        print(e)
        return
```

**After:**
```python
while True:
    try:
        task_data = task_queue.get(timeout=10)
        if task_data is None:  # Stop signal
            print('thread {} received stop signal'.format(thread_index))
            return
            
        input, load_target, makespanTarget, tensor_map, ID_dataset, mode, tmp_path = task_data
        print('thread {} gets task {}'.format(thread_index, ID_dataset))
    except Exception as e:
        print('thread {} finished or timeout: {}'.format(thread_index, e))
        return
```

#### 2. Main Process Logic (test_multi function)
**Before:**
```python
for input, target, makespan, _, tensor_map in dataloader:
    # ... add tasks to queue
    
# Wait for all processes done
for p in ps:
    p.join()

# Read results
while count_task < size_dataset:
    # ... read from recorder_queue
```

**After:**
```python
for input, target, makespan, _, tensor_map in dataloader:
    # ... add tasks to queue

# Send stop signals to all worker processes
for _ in range(NUM_PROCESSES):
    task_queue.put(None)  # None signals worker to stop

# Read results
while count_task < size_dataset:
    # ... read from recorder_queue

# Wait for all processes done
for p in ps:
    p.join()
```

## How the Fix Works

### 1. Reliable Task Retrieval
- Uses `task_queue.get(timeout=10)` instead of checking `qsize()`
- Blocking get with timeout prevents race conditions
- Timeout ensures workers don't hang indefinitely

### 2. Explicit Stop Signaling
- Main process sends `None` as stop signal to each worker
- Workers check for `None` and exit cleanly when received
- Ensures all workers terminate after processing all real tasks

### 3. Proper Process Coordination
- Main process sends all tasks first
- Then sends stop signals
- Collects all results
- Finally waits for worker processes to join

## Key Benefits

1. **No Race Conditions**: Workers use blocking get with timeout
2. **Guaranteed Termination**: Explicit stop signals ensure workers exit
3. **Robust Error Handling**: Timeout prevents indefinite hanging
4. **Maintains Performance**: All workers continue processing until explicitly stopped

## Testing

To verify the fix works:
```bash
# Run the test script
./test_cuda_fix.sh

# Or run with a small test case
python main.py configs/test_train.json --mode test --test_num_processes 4
```

## Files That Still Need Similar Fixes

If similar deadlock issues occur, check these files for the same pattern:
- `agents/decentralplannerlocal_OnlineExpert_LoadPreTrained.py`
- `agents/decentralplannerlocal_OnlineExpert_GAT_LoadPreTrained.py` 
- `agents/decentralplannerlocal.py`

Look for `while task_queue.qsize() > 0` patterns and apply the same fix.