## Setup
1. SSH into Lambda instance

2. Copy `setup_env.sh`, `train.py`, `data_prep.py`, and `rewards.py` to the server

3. Initialize Conda environment.
    ```bash
    source ~/miniconda3/bin/activate
    ```
    Or, create a new environment and activate it
    ```bash
    conda create -n <env_name> python=3.10
    ```
    ```bash
    conda activate <env_name>
    ```

    **If Conda's not installed**

    Download the Miniconda installer script:
    ```bash
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    ```

    Make the installer executable:
    ```bash
    chmod +x Miniconda3-latest-Linux-x86_64.sh
    ```

    Run the installer:
    ```bash
    ./Miniconda3-latest-Linux-x86_64.sh -b
    ```
    
    Initialize Conde
    ```bash
    source ~/miniconda3/bin/activate
    ```

4. Install dependencies
    ```bash
    chmod +x setup_env.sh
    ./setup_env.sh
    ```

5. Setup your reward function(s) in `rewards.py`. All functions should return a `List[float]` and take as input a `List[Dict[str, str]]` with values for `role` and `content` keys.

    For example, 
    ```python
    [   
        {'role': '...', 'content': '...'},
        {'role': '...', 'content': '...'},
        ...
    ]
    ``` 
    

6. Setup your dataset loading function in `data_prep.py`. Ensure the output is a HuggingFace `Dataset` object. You'll pass this function to `GRPOTrainer` as the `train_dataset` argument.

7. In `train.py`, pass your reward function(s) to `GRPOTrainer` as a list.
    ```python
    # line 62
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [ #<---- Add your reward functions here
        ],
        args = training_args,
        train_dataset = dataset
    )
    ```

8. Setup your training arguments and run the script
    ```bash
    python train.py
    ```
## Example GPU Usage
**GPU:** A100 40GB

**LLM:** Llama-3.1-8B-Instruct

**Precision:** 4-bit

| Component            | Memory (GiB) | Percentage | Notes                                    |
|---------------------|--------------|------------|------------------------------------------|
| Model Weights       | 5.60         | 14.00%     | Tensors                |
| Non-PyTorch Memory  | 0.09         | 0.23%      | System overhead                          |
| PyTorch Activations | 1.34         | 3.35%      | Peak memory during operations            |
| KV Cache           | 16.35        | 40.88%     | Reserved for attention key-value pairs   |
| Available Memory    | 16.62        | 41.55%     | Remaining for additional operations      |
| **Total GPU Memory**| **40.00**    | **100%**   | **A100 40GB Total Capacity**            |


## Resources
For more information on GRPOTrainer's arguments, see [here](https://huggingface.co/docs/trl/main/en/grpo_trainer).

The training script is based on the [unsloth](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb#scrollTo=2ROBilj7nY33) implementation of GRPO training.

## Tips
1. Don't use few-shot prompting in your training prompts.
2. Your prompt should include a pointed question for the model to reason about. I've found this leads to better results over a full prompt with a task description.
3. If recording results to W&B, plots for each reward function are available in the `wandb` tab (super useful). 
4. Checkpoints are saved in the `output` directory (unless otherwise specified). After your training run, I recommend a lightweight script that loads checkpoints sequentially and generations 3-5 completions. This allows you to observe any improvements in the model's reasoning traces over training. 