"""
This script is used to implement parallel sampling of API models on AIME2025 dataset
"""

import os
import re
import json
import time
import random
import argparse
import datasets
import numpy as np
from google import genai
from google.genai import types
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def grade_answer(model_answer: str, answer: str) -> bool:
    """Grade the answer."""
    if answer == model_answer:
        return True
    try:
        model_answer = float(model_answer)
        answer = float(answer)
        if model_answer == answer:
            return True
        return False
    except:
        return False


def last_boxed_only_string(string):
  r"""Extract the content of the last \boxed{...}."""
  idx = string.rfind("\\boxed")
  if idx < 0:
    idx = string.rfind("\\fbox")
    if idx < 0:
      return None

  i = idx
  right_brace_idx = None
  num_left_braces_open = 0
  while i < len(string):
    if string[i] == "{":
      num_left_braces_open += 1
    if string[i] == "}":
      num_left_braces_open -= 1
      if num_left_braces_open == 0:
        right_brace_idx = i
        break
    i += 1

  if right_brace_idx == None:
    retval = None
  else:
    retval = string[idx : right_brace_idx + 1]

  return retval


def remove_boxed(s):
  left = "\\boxed{"
  try:
    assert s[: len(left)] == left
    assert s[-1] == "}"
    return s[len(left) : -1]
  except:
    return None


def extract_boxed_answer(solution: str) -> str | None:
  r"""Extract the answer from inside a LaTeX \\boxed{} command."""
  solution = last_boxed_only_string(solution)
  solution = remove_boxed(solution)
  return solution


def extract_model_final_output(api_candidate):
    """Extract the model final output."""
    model_response = None
    for part in api_candidate.content.parts:
        if not part.text:
            continue
        elif part.thought:
            continue
        else:  # final solution
            model_response = part.text
    return model_response


def get_re_answer_prompt(template_id):
    ALL_REANSWER_PROMPTS = [
        "Please re-answer.",
        "Please review your previous response and find problems with your answer. Based on the problems you found, improve your answer.",
    ]
    return ALL_REANSWER_PROMPTS[template_id]


def get_question_dict(question, answer, question_id):

    # build generation config
    generation_seed = args.exp_id
    generation_cfg = types.GenerateContentConfig(
        candidateCount=1,
        seed=generation_seed,
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
        ),
    )

    # build quetsion dict
    question_dict = {
        "question_id": question_id,
        "question_content": question,
        "answer": answer,
    }

    # for each question we create a new chat
    chat = client.chats.create(model=model_id)
    for attempt_id in range(1, args.n_rounds+1):
        if attempt_id == 1:
            if args.model_id == "gemini-2.5-pro":
                message = question + "Please reason step by step, and put your final answer within \\boxed{}."
            else:
                message = question
        else:
            message = get_re_answer_prompt(args.template_id)

        question_dict[f"attempt{attempt_id}"] = {}

        while True:
            try:
                response = chat.send_message(message=message, config=generation_cfg)
                model_final_output = extract_model_final_output(
                    response.candidates[0]
                )
                assert model_final_output is not None
                break
            except Exception as e:
                print("Failed to send message, retry in 30 seconds.")
                print(e)
                time.sleep(30)
            
        # save model output and accuracy
        question_dict[f"attempt{attempt_id}"]["generation_seed"] = generation_seed
        question_dict[f"attempt{attempt_id}"]["model_output"] = model_final_output
        question_dict[f"attempt{attempt_id}"]["message"] = message
        model_raw_answer = extract_boxed_answer(model_final_output)
        question_dict[f"attempt{attempt_id}"]["is_correct"] = grade_answer(model_raw_answer, answer)

        # extract model raw answer
        question_dict[f"attempt{attempt_id}"][
            "model_raw_answer"
        ] = model_raw_answer

        # statistics of token counts
        question_dict[f"attempt{attempt_id}"][
            "prompt_token_count"
        ] = response.usage_metadata.prompt_token_count

        question_dict[f"attempt{attempt_id}"][
            "thoughts_token_count"
        ] = response.usage_metadata.thoughts_token_count

        question_dict[f"attempt{attempt_id}"][
            "output_token_count"
        ] = response.usage_metadata.candidates_token_count

        question_dict[f"attempt{attempt_id}"][
            "total_token_count"
        ] = response.usage_metadata.total_token_count

    return question_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_rounds", type=int, required=True, help="number of repeats for each example")
    parser.add_argument("--exp_id", type=int, required=True, help="experiment id, better to start from 1, used as seed in sequential sampling")
    parser.add_argument("--template_id", type=int, required=True, help="template id")
    parser.add_argument("--model_id", type=str, help="model id used in api")
    parser.add_argument(
        "--num_process_generate",
        type=int,
        default=30,
        help="Number of processes to use for response generation",
    )
    args = parser.parse_args()

    # load API models
    # api_key = os.environ.get("GEMINI_API_KEY")
    
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"), #http_options={"api_version": "v1alpha"}
        # vertexai=True,
        # project=os.getenv("VERTEX_GEMINI_PROJECT"),
        # location=os.getenv("VERTEX_GEMINI_LOCATION"),
    )
    model_id = args.model_id # "gemini-2.5-flash"  # "gemini-2.5-flash-preview-05-20"  # "gemini-2.5-flash"

    print(f"Building api client with model_id: {model_id} with thinking")

    # load the dataset
    data = datasets.load_dataset("yentinglin/aime_2025")["train"]

    # 30 questions in total
    questions = data["problem"]
    answers = data["answer"]
    question_ids = [f"id_{i:02}" for i in range(1, len(questions)+1)]

    print(f"Load {len(questions)} questions from the dataset AIME2025!")

    # first generate
    print("----------------- Start Answering -------------------")

    with ThreadPoolExecutor(max_workers=args.num_process_generate) as executor:
        results = list(tqdm(executor.map(get_question_dict, questions, answers, question_ids), total=len(questions)))

    # prepare for code evaluation
    all_dicts = []
    for result in results:
        all_dicts.append(result)
    all_dicts = sorted(all_dicts, key=lambda x: x["question_id"])


    # save the results
    os.makedirs("results", exist_ok=True)
    os.makedirs(f"results/aime_{args.model_id}", exist_ok=True)
    file_path = os.path.join(
        f"results/aime_{args.model_id}",
        f"aime2025_{model_id}_exp_id{args.exp_id}_n_rounds{args.n_rounds}_chat_template{args.template_id}.jsonl",
    )

    with open(file_path, "w") as f:
        json.dump(all_dicts, f, indent=4)
