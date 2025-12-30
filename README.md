# PEFT-FT: Fine-Tuning Phi-2 for Dialogue Summarization

This project demonstrates **Parameter-Efficient Fine-Tuning (PEFT)** using **QLoRA** to adapt Microsoft's [Phi-2](https://huggingface.co/microsoft/phi-2) language model for dialogue summarization tasks.

## 🎯 Overview

The goal is to fine-tune Phi-2 (2.7B parameters) to generate concise summaries of conversations using minimal computational resources. By leveraging QLoRA (Quantized Low-Rank Adaptation), we achieve efficient training with only **1.36% of trainable parameters** (~21M out of 1.54B).

### Key Results

| Metric    | Original Model | Fine-Tuned Model | Improvement |
|-----------|----------------|------------------|-------------|
| ROUGE-1   | 0.248          | 0.396            | +14.77%     |
| ROUGE-2   | 0.079          | 0.130            | +5.08%      |
| ROUGE-L   | 0.184          | 0.273            | +8.91%      |
| ROUGE-Lsum| 0.194          | 0.279            | +8.51%      |

## 🛠️ Tech Stack

- **Base Model**: [microsoft/phi-2](https://huggingface.co/microsoft/phi-2) (2.7B parameters)
- **Dataset**: [neil-code/dialogsum-test](https://huggingface.co/datasets/neil-code/dialogsum-test)
- **Fine-Tuning Method**: QLoRA (4-bit quantization + LoRA)
- **Libraries**: 
  - `transformers` - Model loading and training
  - `peft` - Parameter-efficient fine-tuning
  - `bitsandbytes` - 4-bit quantization
  - `datasets` - Data loading
  - `trl` - Reinforcement learning from human feedback utilities
  - `evaluate` - ROUGE score evaluation

## 📁 Project Structure

```
PEFT-FT/
├── FINETUNING USING PEFT.ipynb   # Main Kaggle notebook with full training pipeline
├── PEFT-FT/
│   └── train.py                   # Standalone training script for local execution
├── README.md
└── .gitignore
```

## 🔧 Configuration

### QLoRA Settings
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)
```

### LoRA Configuration
```python
LoraConfig(
    r=32,                                    # Rank
    lora_alpha=32,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'dense'],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
```

### Training Hyperparameters
| Parameter                    | Value           |
|-----------------------------|-----------------|
| Learning Rate               | 2e-4            |
| Batch Size                  | 1               |
| Gradient Accumulation Steps | 4               |
| Max Steps                   | 1000            |
| Optimizer                   | paged_adamw_8bit |
| Max Sequence Length         | 2048            |

## 🚀 Quick Start

### Prerequisites
```bash
pip install -q -U bitsandbytes transformers peft accelerate datasets scipy einops evaluate trl rouge_score
```

### Running the Training Script
```bash
cd PEFT-FT
python train.py
```

### Using the Notebook
1. Upload `FINETUNING USING PEFT.ipynb` to Kaggle
2. Enable GPU accelerator (T4 or P100 recommended)
3. Run all cells sequentially
4. Login to Hugging Face when prompted

## 📊 Training Process

1. **Load Dataset**: DialogSum test dataset with train/validation/test splits
2. **Quantize Model**: Load Phi-2 in 4-bit quantization using NF4
3. **Apply LoRA**: Attach low-rank adapters to attention layers
4. **Train**: Fine-tune on dialogue-summary pairs
5. **Evaluate**: Compute ROUGE scores against human baselines

### Prompt Format
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruct: Summarize the below conversation.
{dialogue}

### Output:
{summary}

### End
```

## 📈 Sample Output

**Input Dialogue:**
> #Person1#: You're finally here! What took so long?
> #Person2#: I got stuck in traffic again...

**Human Summary:**
> #Person2# complains to #Person1# about the traffic jam, #Person1# suggests quitting driving and taking public transportation instead.

**Fine-Tuned Model Output:**
> #Person2# got stuck in traffic again. #Person1# suggests #Person2# should start taking public transport system to work, but #Person2# is reluctant because #Person2# misses having the freedom with a car.

## 💡 Key Takeaways

- **Efficiency**: Only 1.36% of parameters are trainable, reducing VRAM requirements significantly
- **Quality**: Significant improvements across all ROUGE metrics
- **Accessibility**: Can run on consumer GPUs (RTX 3050 supported with reduced batch size)
- **Format Awareness**: The fine-tuned model learns to use the dataset's `#Person1#/#Person2#` format

## 📝 License

This project is for educational purposes. Please refer to the licenses of the underlying models and datasets:
- Phi-2: [Microsoft Research License](https://huggingface.co/microsoft/phi-2)
- DialogSum: Check dataset card on Hugging Face

## 🙏 Acknowledgments

- Microsoft Research for the Phi-2 model
- Hugging Face for the transformers and PEFT libraries
- The DialogSum dataset creators
