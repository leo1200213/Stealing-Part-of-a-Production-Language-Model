
from full_layer_extraction import get_Q, dim_extraction, layer_extraction
from basic_logprob_free_attack import binary_search_extraction
from datasets import load_dataset
from torch.utils.data import DataLoader, RandomSampler
import numpy as np
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
import warnings
from tqdm import tqdm

# Suppress specific warnings and logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning)


# Load Pre-trained Model and Tokenizer (e.g., EleutherAI/gpt-neo-125m)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = "EleutherAI/gpt-neo-125m"
model_path = "./models/gpt-neo-125m"  # Specify the path to save the model
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_path, padding_side='left')

# Add a padding token if the tokenizer does not have one
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))


    
if __name__ == '__main__':
    # Step 1: Create tokenized prompts using the original token IDs from 1 to 2000
    num_samples = 2000
    token_ids = list(range(1, num_samples + 1))
    tokenized_prompts = [torch.tensor([[token_id]]).to(device) for token_id in token_ids]

    # Step 2: Perform queries to construct matrix Q
    Q = get_Q(tokenized_prompts, model)
    
    # Step 3: Extract hidden dimensionality
    U, S, h = dim_extraction(Q)
    print(f"--------------------Estimated hidden layer dimension--------------------\nh :{h}")
    
    # Step 4: Extract layer weights approximation
    W_tilde = layer_extraction(U, S, h)
    print(f"--------------------Reconstructed W--------------------\nW_tilde :{W_tilde}")
    
    # Step 5: Perform Basic Logprob-free Attack
    epsilon = 0.000005
    prompt = 'I want to find logit vector!'
    estimated_logit_vector = []
    bias = 50
    inputs_tokenized = tokenizer(prompt, return_tensors="pt").to(device)

    for token_id in tqdm(range(0, 100), desc="Estimating logit vector (from id 0 to 99)"):        
        estimated_bias, gt_logit_vector= binary_search_extraction(model, epsilon, inputs_tokenized, token_id, bias)
        estimated_logit_vector.append(estimated_bias)
    
    print(f'\nEstimatied logit vector:{estimated_logit_vector}')
    
    distance = np.linalg.norm(np.array(estimated_logit_vector) - np.array(gt_logit_vector))
    print(f"L2 distance between ground truth logit vector and estimated logit vector: {distance}")



    


