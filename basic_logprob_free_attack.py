
import torch

def binary_search_extraction(model, epsilon, inputs_tokenized, target_token, bias):
    """
    Perform a basic logprob-free attack to find minimal logit bias for a target token.
    
    Args:
        model: The language model.
        epsilon (float): Precision threshold for binary search.
        inputs_tokenized (dict): Tokenized input prompt.
        target_token (int): The target token ID.
        bias (float): Initial bias value for binary search.
    
    Returns:
        float: Estimated minimal logit bias.
    """
    alpha_i = 0
    beta_i = bias
    input_ids = inputs_tokenized["input_ids"]

    original_output = model.generate(
                input_ids=input_ids,
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True,
                temperature=0
            )
    

    top_token_id = torch.argmax(original_output.scores[0][0]).item()
    top_logit = original_output.scores[0][0].max().item()
    
    #print("top_token=",top_token)
    #print("top_logit",top_logit)
    if (top_token_id==target_token):
        return top_logit, original_output.scores[0][0][:100].tolist() if target_token == 99 else None
    
    while beta_i - alpha_i > epsilon:
        bias_i = (alpha_i + beta_i) / 2
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True,
                temperature=0
            )
            logits = output.scores[0][0].cpu()
            logits[target_token] += bias_i
            most_likely_token_id = torch.argmax(logits).item()
            #print("most_likely_token_id=",most_likely_token_id)

            if most_likely_token_id == target_token:
                beta_i = bias_i
                
            else:
                alpha_i = bias_i
    estimated_logit = top_logit - ((alpha_i + bias_i) / 2)
    right_logit_vector = original_output.scores[0][0][:100].tolist() if target_token == 99 else None
            
    return estimated_logit, right_logit_vector
