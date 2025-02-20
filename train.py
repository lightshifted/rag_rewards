# Use PatchFastRL before all functions to patch GRPO and other RL algorithms
from unsloth import FastLanguageModel, PatchFastRL
from unsloth import is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer
from rewards import *
from data_prep import get_medical_procedures

# Use PatchFastRL to patch GRPO and other RL algorithms
PatchFastRL("GRPO", FastLanguageModel)

max_seq_length = 3000 # Can increase for longer reasoning traces
lora_rank = 64 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/meta-Llama-3.1-8B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.6, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3141,
)

training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 6, # Decrease if out of memory
    max_prompt_length = 512,
    max_completion_length = 3000,
    num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 1000,
    save_steps = 50,
    max_grad_norm = 0.1,
    report_to = ["wandb"], # Can use Weights & Biases
    output_dir = "outputs",
)

if __name__ == "__main__":
    dataset = get_medical_procedures()
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            reasoning_content_reward_func,
            list_format_reward_func,
            code_match_reward_func,
            code_prefix_match_reward_func,
            code_topk_reward_func,
            code_topk_prefix_reward_func,
        ],
        args = training_args,
        train_dataset = dataset
    )
    trainer.train()