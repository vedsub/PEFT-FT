import torch
import time
import os
from datasets import load_dataset
from functools import partial
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)

# 1. Define Helper Functions
def create_prompt_formats(sample):
    """
    Format the data so the model understands:
    Instruct -> Context -> Output
    """
    INTRO = "Instruct: Summarize the following conversation."
    END = "Output:"
    
    # Create the full text string
    formatted_prompt = f"{INTRO}\n{sample['dialogue']}\n{END}\n{sample['summary']}"
    sample["text"] = formatted_prompt
    return sample

def get_max_length(model):
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max length: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length

def preprocess_batch(batch, tokenizer, max_length):
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )

def preprocess_dataset(tokenizer, max_length, seed, dataset):
    print("Preprocessing dataset...")
    # Format the prompts
    dataset = dataset.map(create_prompt_formats)
    
    # Tokenize
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=['id', 'topic', 'dialogue', 'summary', 'text'],
    )
    
    # Filter long samples
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)
    dataset = dataset.shuffle(seed=seed)
    return dataset

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params} || all model parameters: {all_model_params} || percentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

# 2. Main Training Script
def main():
    # Configuration
    model_name = "microsoft/phi-2"
    dataset_name = "neil-code/dialogsum-test"
    output_dir = "./results"
    
    # Load Dataset
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    
    # Load Model (4-bit mode for RTX 3050)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto"
    )
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Preprocess Data
    max_length = get_max_length(model)
    train_dataset = preprocess_dataset(tokenizer, max_length, 42, dataset['train'])
    # Using a small subset of test for validation to save time/memory
    eval_dataset = preprocess_dataset(tokenizer, max_length, 42, dataset['test'].select(range(50)))
    
    # Prepare for QLoRA
    print("Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)
    
    config = LoraConfig(
        r=32, 
        lora_alpha=32,
        target_modules=['q_proj','k_proj','v_proj','dense'],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, config)
    print(print_number_of_trainable_model_parameters(model))
    
    # Training Arguments
    peft_training_args = TrainingArguments(
        output_dir=output_dir,
        warmup_steps=1,
        per_device_train_batch_size=1,  # Keep at 1 for RTX 3050
        gradient_accumulation_steps=4,
        max_steps=200,                  # Reduced steps for testing
        learning_rate=2e-4,
        optim="paged_adamw_8bit",
        logging_steps=25,
        save_strategy="steps",
        save_steps=50,
        eval_strategy="steps",          # Fixed from evaluation_strategy
        eval_steps=50,
        do_eval=True,
        gradient_checkpointing=True,
        group_by_length=True,
        report_to="none"                # Disable wandb
    )
    
    model.config.use_cache = False
    
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=peft_training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model("final_adapter_model")
    print("Done!")

if __name__ == "__main__":
    main()
