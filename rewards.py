import re
import ast
from typing import List, Dict, Tuple, Union
from ragatouille import RAGPretrainedModel


RAG = RAGPretrainedModel.from_index(".ragatouille/colbert/indexes/icd_cm")

def reasoning_content_reward_func(completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
    """
    Reward function that checks if completions have non-empty <reasoning> tags.
    Returns 0.5 for each content object that has reasoning tags with content.

    Args:
        completions: List of completions, where each completion is a list containing 
                    dictionaries with 'content' key
        **kwargs: Additional keyword arguments (unused)

    Returns:
        list[float]: List of rewards (0.5 for valid reasoning content, 0.0 otherwise)
    """
    def has_reasoning_content(content: str) -> bool:
        pattern = r"<reasoning>\s*(\S[\s\S]*?\S)\s*</reasoning>"
        match = re.search(pattern, content)
        return bool(match)

    rewards = []
    for completion in completions:
        # Process each content object in the completion
        for content_obj in completion:
            rewards.append(0.5 if has_reasoning_content(content_obj["content"]) else 0.0)
    return rewards

def list_format_reward_func(
        completions: List[List[Dict[str, str]]], 
        **kwargs
    ) -> List[float]:
    matches: List[str] = [extract_xml_answer(completions[i][0]['content']) for i in range(len(completions))]
    def process_list(data: List[str]) -> List[Union[float, str]]:
        def is_valid_list_string(s: str) -> bool:
            if not (isinstance(s, str) and s.startswith('[') and s.endswith(']')):
                return False
            items = s[1:-1].split(',')
            return len(items) > 0 and all(item.strip() for item in items)

        return [0.5 if is_valid_list_string(s) else 0.0 for s in data]

    return process_list(matches)

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def code_match_reward_func(
    completions: List[List[Dict[str, str]]], # set of completions 
    gt_codes: List[str], # set of ground truth labels
    **kwargs,
    ) -> List[float]:

    gt_labels = gt_codes[0].split(";")
    k = len(gt_labels)
    matches: List[str] = [extract_xml_answer(completions[i][0]['content']) for i in range(len(completions))]
    search_results: List[List[Dict[str, str]]] = RAG.search(matches, k=k)
    pred_codes: List[str] = [[row['document_id'] for row in code_group] for code_group in search_results]

    eval_score = []
    # normalize by the maximum possible score
    max_possible_score = 1.0 * len(gt_labels)
    for codes in pred_codes:
        tally = 0
        for pred_code in codes:
            tally += [1.0 if pred_code in gt_labels else 0.0][0]
        eval_score.append(round(tally / max_possible_score, 3))
    return eval_score

def code_prefix_match_reward_func(
    completions: List[List[Dict[str, str]]], 
    gt_codes: List[str],
    n: int=6, # maximum prefix length
    **kwargs
    ) -> List[float]:
    matches: List[str] = [extract_xml_answer(completions[i][0]['content']) for i in range(len(completions))]

    gt_labels = [gt_codes[0].split(";")]
    k = len(gt_labels)
    search_results: List[List[Dict[str, str]]] = RAG.search(matches, k=k)
    pred_codes: List[str] = [[row['document_id'] for row in code_group] for code_group in search_results]

    n = 6 # prefix length
    truncated_gts = [item[:n] if len(item) >= n else item for item in gt_labels[0]]
    truncated_preds = [[item[:n] if len(item) >= n else item for item in sublist] for sublist in pred_codes]
    
    for i, sublist in enumerate(truncated_preds):
        # Truncate and remove duplicates
        truncated_preds[i] = list(dict.fromkeys(code[:n] if len(code) >= 3 else code for code in sublist))
    
    eval_score = []
    
    # normalize by the maximum possible score
    max_possible_score = 0.25 * len(truncated_gts)
    for pred_code in truncated_preds:
        tally = 0
        for code in pred_code:
            tally += 0.25 if code in truncated_gts else 0.0
        eval_score.append(round(tally / max_possible_score, 3))
    return eval_score

def recall_reward_func(
    completions: List[List[Dict[str, str]]], # set of completions 
    gt_codes: List[str], # set of ground truth labels
    **kwargs
    ) -> List[float]:

    gt_labels = gt_codes[0].split(";")
    k = len(gt_labels)
    matches: List[str] = [extract_xml_answer(completions[i][0]['content']) for i in range(len(completions))]
    search_results: List[List[Dict[str, str]]] = RAG.search(matches, k=k)
    pred_codes: List[str] = [[row['document_id'] for row in code_group] for code_group in search_results]

    eval_score = []

    n = 7
    truncated_gts = [item[:n] if len(item) >= n else item for item in gt_labels]
    truncated_preds = [[item[:n] if len(item) >= n else item for item in sublist] for sublist in pred_codes]

    for i, sublist in enumerate(truncated_preds):
        # Truncate and remove duplicates
        truncated_preds[i] = list(dict.fromkeys(code[:n] if len(code) >= 3 else code for code in sublist))

    eval_score = []
    # Truncate codes to prefix length n
    for i, pred in enumerate(truncated_preds):
        # Count true positives (codes that appear in both prediction and ground truth)
        true_positives = len(set(pred).intersection(set(truncated_gts)))

        # Calculate recall: true_positives / total_ground_truths
        recall = true_positives / len(truncated_gts)

        eval_score.append(recall)

    return eval_score