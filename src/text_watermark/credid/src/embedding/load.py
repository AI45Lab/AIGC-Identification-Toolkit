import os
import json
from datasets import load_dataset, load_from_disk
from src.utils.text_tools import truncate
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

def load_prompts(prompt_file):
    with open(prompt_file, 'r') as file:
        prompts = json.load(file)
    return prompts

def prepare_input_from_prompts(device,tokenizer, config, project_root):
    # Define the path for the prompts JSON file
    prompts_path = os.path.join(project_root, 'config', 'prompts.json')
    input_text = None

    try:
        # Try to load prompts from the JSON file
        prompts = load_prompts(prompts_path)
        
        # Choose a prompt based on the sample index
        sample_idx = config.get('sample_idx', 0)
        if isinstance(sample_idx, int) and 'prompts' in prompts and sample_idx < len(prompts['prompts']):
            input_text = prompts['prompts'][sample_idx]
        elif 'sample_idx' in prompts:
            input_text = prompts['sample_idx']
        
    except (FileNotFoundError, IndexError, KeyError) as e:
        print(f"Could not load from prompts.json: {e}")
        
    if input_text is None:
        try:
            # Try to load from dataset
            dataset_path = config['dataset_path']
            
            # Check if it's a JSON file or a dataset directory
            if dataset_path.endswith('.json'):
                # Load simple JSON dataset
                import json
                with open(dataset_path, 'r') as f:
                    dataset = json.load(f)
                sample_idx = config.get('sample_idx', 0)
                if 'train' in dataset and sample_idx < len(dataset['train']):
                    input_text = dataset['train'][sample_idx]['text']
            else:
                # Load HuggingFace dataset
                c4_sliced_and_filted = load_from_disk(dataset_path)
                c4_sliced_and_filted = c4_sliced_and_filted['train'].shuffle(seed=42).select(range(100))
                
                sample_idx = config.get('sample_idx', 0)
                input_text = c4_sliced_and_filted[sample_idx]['text']
        except Exception as e:
            print(f"Could not load from dataset: {e}")
            # Final fallback: use the prompt from config
            input_text = config.get('prompt', "A new study has found that the universe is made of")
            print(f"Using config prompt as fallback: {input_text}")
    
    # Tokenize the input text (removed truncation for unlimited length)
    tokenized_input = tokenizer(input_text, return_tensors='pt').to(device)
    # tokenized_input = truncate(tokenized_input, max_length=300)  # 移除截断限制

    return tokenized_input



def load_model_and_tokenizer(config):
    """
    Load the tokenizer and model based on the provided configuration.

    Args:
        config (dict): Configuration dictionary containing model details.

    Returns:
        tokenizer: Loaded tokenizer.
        model: Loaded model.
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(config['model_name'], device_map="auto")
    
    return tokenizer, model
