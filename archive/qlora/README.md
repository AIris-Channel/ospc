# Qwen-QLoRA

Based on [QwenLM / Qwen](https://github.com/QwenLM/Qwen)

## System Requirements

- **CUDA**: Version 11.8 is preferred.

- **GCC**: Version 11.2 is preferred.

## Environment Setup

```bash
pip install -r requirements.txt
conda install mpi4py
```

## Training Scripts

There are two scripts provided for training:

- For multi-GPU training, use `finetune_qlora_ds.sh`.
- For single-GPU training, use `finetune_qlora_single_gpu.sh`.

### Single-GPU Training

To fine-tune the model using single GPU, follow these steps:

1. Clone the model from [Qwen/Qwen1.5-14B-Chat-GPTQ-Int4](https://huggingface.co/Qwen/Qwen1.5-14B-Chat-GPTQ-Int4)

2. Modify the `finetune_qlora_single_gpu.sh` script as follows:
   - Line 7: Set `MODEL="Qwen1.5-14B-Chat-GPTQ-Int4"` to specify the model.
   - Line 10: Set `DATA="dataset.json"` to specify the location of your dataset.
   - Line 47: Set `--output_dir output_qlora` to specify the location where the QLoRA model should be saved.

3. Run the script with `./finetune_qlora_single_gpu.sh`.

### Multi-GPU Training

To fine-tune the model using multiple GPUs, follow these steps:

1. Clone the model from [Qwen/Qwen1.5-14B-Chat-GPTQ-Int4](https://huggingface.co/Qwen/Qwen1.5-14B-Chat-GPTQ-Int4)

2. Modify the `finetune_qlora_ds.sh` script as follows:
   - Line 28: Set `MODEL="Qwen1.5-14B-Chat-GPTQ-Int4"` to specify the model.
   - Line 31: Set `DATA="dataset.json"` to specify the location of your dataset.
   - Line 74: Set `--output_dir output_qlora` to specify the location where the QLoRA model should be saved.

3. Run the script with `./finetune_qlora_ds.sh`.

## Note

Please ensure that your environment meets the hardware requirements for multi-GPU training before attempting to run the multi-GPU script.
