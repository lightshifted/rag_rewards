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
    completions: List[List[Dict[str, str]]], 
    icd_cm_target: List[str],
    **kwargs
    ) -> List[float]:
    matches: List[str] = [extract_xml_answer(completions[i][0]['content']) for i in range(len(completions))]
    gt_labels = [icd_cm_target[0].split(";")]
    search_results: List[List[Dict[str, str]]] = RAG.search(matches)
    pred_codes: List[str] = [[row['document_id'] for row in code_group] for code_group in search_results]
    results = [1.0 if gt in pred else 0.0 for gt, pred in zip(gt_labels, pred_codes)]
    return results

def code_prefix_match_reward_func(
    completions: List[List[Dict[str, str]]], 
    icd_cm_target: List[str],
    **kwargs
    ) -> List[float]:
    matches: List[str] = [extract_xml_answer(completions[i][0]['content']) for i in range(len(completions))]
    gt_labels = [icd_cm_target[0].split(";")]
    search_results: List[List[Dict[str, str]]] = RAG.search(matches)
    pred_codes: List[str] = [[row['document_id'] for row in code_group] for code_group in search_results]
    results = [1.0 if any(gt[:3] == pred[:3] for pred in preds) else 0.0 
              for gt, preds in zip(gt_labels, pred_codes)]
    return results

def code_topk_reward_func(
    completions: List[List[Dict[str, str]]], 
    icd_cm_target: List[str],
    **kwargs
    ) -> List[float]:
    matches: List[str] = [extract_xml_answer(completions[i][0]['content']) for i in range(len(completions))]
    gt_labels = [icd_cm_target[0].split(";")]
    search_results: List[List[Dict[str, str]]] = RAG.search(matches)
    pred_codes: List[str] = [[row['document_id'] for row in code_group] for code_group in search_results]
    results = [1/(pred.index(gt) + 1) if gt in pred else 0.0
              for gt, pred in zip(gt_labels, pred_codes)]
    return results

def code_topk_prefix_reward_func(
    completions: List[List[Dict[str, str]]],
    icd_cm_target: List[str],
    **kwargs
    ) -> List[float]:
    matches: List[str] = [extract_xml_answer(completions[i][0]['content']) for i in range(len(completions))]
    gt_labels = [icd_cm_target[0].split(";")]
    search_results: List[List[Dict[str, str]]] = RAG.search(matches)
    pred_codes: List[str] = [[row['document_id'] for row in code_group] for code_group in search_results]
    def get_prefix_match_position(gt, preds):
        for i, pred in enumerate(preds):
            if gt[:3] == pred[:3]:
                return i
        return -1
    results = [1/(get_prefix_match_position(gt, preds) + 1) if get_prefix_match_position(gt, preds) != -1 else 0.0 
               for gt, preds in zip(gt_labels, pred_codes)]
    return results
