# Fork details

Made to run on Orin nano (arm) with 8GB RAM.

## Dependencies

```sh
./pyenv_torch.sh
```

Note: pytorch and torchvision wheels are for arch64 architecture (jetson orin nano).

If you need to install for another architecture (which is not the purpose of this fork), change the following lines in `pyenv_torch.sh`:

```sh
# torch
pip install https://pypi.jetson-ai-lab.io/jp6/cu126/+f/590/92ab729aee2b8/torch-2.8.0-cp310-cp310-linux_aarch64.whl#sha256=59092ab729aee2b8937d80cc1b35d1128275bd02a7e1bc911e7efa375bd97226

# torchvision
pip install https://pypi.jetson-ai-lab.io/jp6/cu126/+f/1c0/3de08a69e9554/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl#sha256=1c03de08a69e95542024477e0cde95fab3436804917133d3f00e67629d3fe902
```

## Dataset generation

Dataset generation has many parameters that can be changed in the script itself.

It will take a long time to generate the dataset on the Orin nano, so be patient.

```sh
cd ./offlineExpert
./dataset.sh
```
### Troubleshooting

| Multiprocessing and deadlocks

On Orin Nano specifically, troubleshooting may cause deadlocks (I tried to fix this but it will deadlock during model training). If you look trough the commits, you will see several changes to the original code from the main repo, well I can't fix this. If you see just 1 deadlock, set the following flag everywhere
```sh
--test_num_processes 0
```

## Training
