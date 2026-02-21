# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

example = "<reasoning>Response A directly addresses the question by identifying Clark Terry as the third trumpeter on the album The Trumpet Summit Meets the Oscar Peterson Big 4 and providing his birth year, 1920. This response is concise, relevant, and accurate, aligning well with the information gap. It efficiently fills the knowledge gap and provides the final answer without any redundancy or tangential information. The reasoning is logical and grounded in the provided search results.</reasoning> - Scores: \\boxed{9}"

def extract_solution(solution_str):
    """Extract the answer from the solution string.
    
    Customize this based on your dataset's expected output format.
    For example, if answers are in <answer></answer> tags, or in a specific format.
    """
    # Extract from \boxed{} format
    boxed_pattern = r"\\boxed\{([^}]+)\}"
    boxed_match = re.search(boxed_pattern, solution_str)
    if boxed_match:
        content = boxed_match.group(1).strip()
        # Extract numbers from the content and return as list
        numbers = re.findall(r'\d+(?:\.\d+)?', content)
        if numbers:
            return [float(num) if '.' in num else int(num) for num in numbers]
        return content

    return None


def extract_think_process(solution_str):
    """Extract the answer from the solution string.
    
    Customize this based on your dataset's expected output format.
    For example, if answers are in <answer></answer> tags, or in a specific format.
    """
    # Example 1: Extract from answer tags
    answer_pattern = r"<reasoning>(.*?)</reasoning>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    if len(matches) >= 1:
        return matches[-1].group(1).strip()
    
    return None

def _validate_format(solution_str: str) -> bool:
    """
    Validate that the solution follows the exact format:
    <reasoning>...</reasoning> - \\boxed{Correct/Wrong}

    Requirements:
    1. Starts with <reasoning> (possibly after whitespace)
    2. Exactly one <reasoning>...</reasoning> block
    3. After </reasoning>, only whitespace and "- " before \\boxed
    4. Exactly one \\boxed{Correct} or \\boxed{Wrong}
    5. No other tags allowed
    """
    # Check for any tags other than <reasoning></reasoning>
    all_tags = re.findall(r'</?[^>]+>', solution_str)

    # Only allow <reasoning> and </reasoning> tags
    allowed_tags = {'<reasoning>', '</reasoning>'}
    for tag in all_tags:
        if tag not in allowed_tags:
            return False

    # 1) exactly one <reasoning>...</reasoning>
    reasoning_pattern = r"<reasoning>(.*?)</reasoning>"
    reasoning_matches = list(re.finditer(reasoning_pattern, solution_str, re.DOTALL))
    if len(reasoning_matches) != 1:
        return False

    # 2) <reasoning> should appear at the start (after optional whitespace)
    reasoning_match = reasoning_matches[0]
    before_reasoning = solution_str[:reasoning_match.start()].strip()
    if before_reasoning:
        return False

    # 3) exactly one \boxed{...}
    boxed_pattern = r"\\boxed\{([^}]*)\}"
    boxed_matches = list(re.finditer(boxed_pattern, solution_str))
    if len(boxed_matches) != 1:
        return False

    # 4) After </reasoning>, only whitespace and "- " before \boxed
    between_content = solution_str[reasoning_match.end():boxed_matches[0].start()]
    # Allow only whitespace and optional "- " or "-"
    if not re.match(r'^\s*-?\s*$', between_content):
        return False

    # 5) box non-empty and contains exactly "Correct" or "Wrong"
    boxed_content = boxed_matches[0].group(1).strip()
    if boxed_content not in ["Correct", "Wrong"]:
        return False

    # 6) Nothing should appear after \boxed{...}
    # after_boxed = solution_str[boxed_matches[0].end():].strip()
    # if after_boxed:
    #     return False

    return True


def _extract_single_score(solution_str: str):
    """Try to extract 'Correct' or 'Wrong' from \\boxed{...}. Return the string or None."""
    boxed_pattern = r"\\boxed\{([^}]*)\}"
    boxed_matches = list(re.finditer(boxed_pattern, solution_str))
    if len(boxed_matches) != 1:
        return None
    m = re.search(boxed_pattern, solution_str)
    if not m:
        return None
    content = m.group(1).strip()
    if content in ["Correct", "Wrong"]:
        return content
    return None


def compute_score(solution_str, ground_truth, **kwargs):
    """
    Strict format reward (all-or-nothing) and independent accuracy reward.
    Also tracks fine-grained metrics for Correct vs Wrong predictions.

    - format_reward: 1.0 iff all four rules pass; else 0.0
    - accuracy_reward: 1.0 if prediction matches ground_truth; else 0.0
    - correct_accuracy: True Positive rate (TP) - correctly predicted "Correct" when ground_truth is "Correct"
    - wrong_accuracy: True Negative rate (TN) - correctly predicted "Wrong" when ground_truth is "Wrong"
    - false_positive: False Positive rate (FP) - predicted "Correct" when ground_truth is "Wrong"
    - false_negative: False Negative rate (FN) - predicted "Wrong" when ground_truth is "Correct"
    """
    # --- format (independent) ---
    format_ok = _validate_format(solution_str)
    format_reward = 1.0 if format_ok else 0.0

    # --- accuracy (independent) ---
    predicted_score = _extract_single_score(solution_str)
    accuracy_reward = 0.0
    correct_accuracy = 0.0  # TP: correctly predicted "Correct" when ground_truth is "Correct"
    wrong_accuracy = 0.0    # TN: correctly predicted "Wrong" when ground_truth is "Wrong"
    false_positive = 0.0    # FP: predicted "Correct" when ground_truth is "Wrong"
    false_negative = 0.0    # FN: predicted "Wrong" when ground_truth is "Correct"

    if predicted_score is not None:
        # Overall accuracy
        accuracy_reward = 1.0 if predicted_score == ground_truth else 0.0

        # Fine-grained accuracy tracking
        if ground_truth == "Correct":
            if predicted_score == "Correct":
                correct_accuracy = 1.0  # True Positive
            else:
                false_negative = 1.0    # False Negative
        elif ground_truth == "Wrong":
            if predicted_score == "Wrong":
                wrong_accuracy = 1.0    # True Negative
            else:
                false_positive = 1.0    # False Positive

    # final weighted score (unchanged weights)
    score = 0.15 * format_reward + 0.85 * accuracy_reward

    return {
        "score": score,
        "format_reward": format_reward,
        "accuracy_reward": accuracy_reward,
        "TP": correct_accuracy,  # TP rate: correct when ground_truth="Correct"
        "TN": wrong_accuracy,      # TN rate: correct when ground_truth="Wrong"
        "FP": false_positive,      # FP rate: predicted "Correct" when ground_truth="Wrong"
        "FN": false_negative,      # FN rate: predicted "Wrong" when ground_truth="Correct"
    }


if __name__ == "__main__":
    # Example usage
    # solution = "The final answer is \\boxed{9, 7}."
    print(compute_score(example, ground_truth=None))  # Output: "9, 7"