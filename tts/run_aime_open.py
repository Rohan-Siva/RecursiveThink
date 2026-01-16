"""
This script is used to implement parallel sampling of open-sourced models on AIME2025 dataset
"""

import os
import json
import time

import fire
import vllm

from datasets import load_from_disk
from math_grader import extract_answer, grade


def boxed_reward_fn(model_answer, gt_answer, fast=False):
    if model_answer is None:
        return False  # Cannot even parse anything.
    if isinstance(gt_answer, float) or isinstance(gt_answer, int):
        gt_answer = str(gt_answer)
    if isinstance(gt_answer, str):
        is_correct = grade(model_answer, gt_answer, fast)
    elif isinstance(gt_answer, list):
        is_correct = False
        for gt in gt_answer:
            is_correct |= grade(model_answer, gt, fast)
    return is_correct


def get_re_answer_prompt(template_id):
    ALL_REANSWER_PROMPTS = [
        "Please re-answer.",
        "Please review your previous response and find problems with your answer. Based on the problems you found, improve your answer.",
        "Please solve the question in a different way.",
        "Try a new idea to solve the question.",
    ]
    return ALL_REANSWER_PROMPTS[template_id]


def main(
    model_name: str = "Qwen/Qwen2.5-Math-1.5B",
    tasks: list = ["aime2025"],#, "amc", "math", "minerva", "olympiad_bench"],
    exp_id: int = 1,
    template_id: int = 0,
    temperature: float = 0,
    top_p: float = 1,
    top_k: int = -1,
    min_p: float = 0,
    max_tokens: int = 3000,
    max_model_len: int = 4096,  # VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 for longer ones.
    n_rounds: int = 1,
    tensor_parallel_size: int = 1,
):
    
    dataset_name: str = "./datasets/evaluation_suite"

    model = vllm.LLM(
        model_name,
        swap_space=32,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        dtype="bfloat16",
        # enable_prefix_caching=True,
    )


    if "DeepSeek-R1-Distill" in model_name:  # r1-distill
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def apply_template(history):
            return tokenizer.apply_chat_template(
                # [{"content": question, "role": "user"}],
                history,
                tokenize=False,
                add_generation_prompt=True,
            )
        end_of_think_token_id = 151649
        end_of_sequence_token_id = 151643

    elif "Qwen3" in model_name:              # qwen 3
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def apply_template(history):
            return tokenizer.apply_chat_template(
                # [{"content": question, "role": "user"}],
                history,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True, # Switches between thinking and non-thinking modes. Default is True.
            )
        
        end_of_think_token_id = 151668
        end_of_sequence_token_id = 151645

    else:
        raise NotImplementedError
    
    split_think = "</think>\n\n"
    
    for task_name, dataset in load_from_disk(dataset_name).items():
        if task_name not in tasks:
            continue

        # load dataset
        questions = dataset["problem"]
        answers = dataset["answer"]
        question_ids = [f"id_{i:02}" for i in range(1, len(questions)+1)]
        print(f"Load {len(questions)} questions from the dataset {task_name}!")


        # first generate
        print("----------------- Start Answering -------------------")
        all_dicts = []
        for question, answer, question_id in zip(questions, answers, question_ids):
            question_dict = {
                "question_id": question_id,
                "question_content": question,
                "answer": answer,
            }
            all_dicts.append(question_dict)

        for attempt_id in range(1, n_rounds+1):
            print(f"Math solution generation in sequence, attempt {attempt_id}")
            
            sampling_params = vllm.SamplingParams(
                n=1,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_tokens=max_tokens,
                seed=exp_id,
                # logprobs=20,
                # seed=int(time.time_ns()),
            )
            
            # apply template for prompt
            if attempt_id == 1:
                histories = [[{"role": "user", "content": question}] for question in questions]
            else:
                assert histories is not None
                for question, question_dict, history in zip(questions, all_dicts, histories):
                    assert question == question_dict["question_content"]
                    last_response =  question_dict[f"attempt{attempt_id-1}"]["model_output"]
                    # new_question = question + f"\nHere is your last response:\n{last_response}\n\n" + get_re_answer_prompt(template_id)
                    history.extend([
                        {"role": "assistant", "content": last_response},
                        {"role": "user", "content": get_re_answer_prompt(template_id)},
                    ])
            assert len(histories[0]) == 2 * attempt_id - 1, "Number of histories should be 2*round-1"
            messages = list(map(apply_template, histories))
            # start generating responses
            responses = model.generate(messages, sampling_params)

            for question, answer, question_id, message, response, question_dict in zip(questions, answers, question_ids, messages, responses, all_dicts):
                assert question == question_dict["question_content"]
                assert answer == question_dict["answer"]
                assert question_id == question_dict["question_id"]
                # save model output and accuracy
                question_dict[f"attempt{attempt_id}"] = {}
                question_dict[f"attempt{attempt_id}"]["generation_seed"] = exp_id
                question_dict[f"attempt{attempt_id}"]["message"] = message
                
                model_all_output = response.outputs[0].text        # thinking + final solution
                model_token_ids = response.outputs[0].token_ids    # token_ids of thinking + final solution
                # question_dict[f"attempt{attempt_id}"]["model_output"] = model

                # split model output into thinking traces and model answers
                if split_think in model_all_output:
                    model_thinking_part = split_think.join(model_all_output.split(split_think)[:-1])
                    model_answer_part = model_all_output.split(split_think)[-1]
                    assert model_thinking_part + split_think + model_answer_part == model_all_output

                    # statistic of token numbers
                    assert end_of_think_token_id in model_token_ids
                    # find the final </think> id as there are maybe more than one
                    all_end_of_think_indices = [i for i, token_id in enumerate(model_token_ids) if token_id == end_of_think_token_id]
                    thinking_token_lens = len(model_token_ids[:all_end_of_think_indices[-1]+1])
                    answer_token_lens = len(model_token_ids[all_end_of_think_indices[-1]+1:])
                else:
                    model_thinking_part = model_all_output
                    model_answer_part = ""
                    thinking_token_lens = len(model_token_ids)
                    answer_token_lens = 0
                
                question_dict[f"attempt{attempt_id}"]["model_output"] = model_answer_part
                question_dict[f"attempt{attempt_id}"]["thinking_traces"] = model_thinking_part

                # correct or not
                model_raw_answer = extract_answer(model_all_output)
                question_dict[f"attempt{attempt_id}"]["model_raw_answer"] = model_raw_answer
                question_dict[f"attempt{attempt_id}"]["is_correct"] = boxed_reward_fn(model_raw_answer, answer)

                # save statistics of token counts
                question_dict[f"attempt{attempt_id}"]["prompt_token_count"] = len(response.prompt_token_ids)
                question_dict[f"attempt{attempt_id}"]["thinking_token_count"] = thinking_token_lens
                question_dict[f"attempt{attempt_id}"]["answer_token_count"] = answer_token_lens
                question_dict[f"attempt{attempt_id}"]["total_token_count"] = len(model_token_ids) + len(response.prompt_token_ids)


    model_save_name = model_name.replace("/", "_")
    save_folder = f"./results/aime_{model_save_name}"
    fn = f"aime2025_{model_save_name}_exp_id{exp_id}_n_rounds{n_rounds}_chat_template{template_id}.json"
    os.makedirs(save_folder, exist_ok=True)
    fn = os.path.join(save_folder, fn)
    print(f"saving model outputs at {fn}")
    json.dump(
        all_dicts,
        open(
            fn,
            "w",
        ),
        indent=4,
    )


fire.Fire(main)
