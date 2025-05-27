import json
import math
import os
from typing import List
import re

from jsonargparse import ArgumentParser

# PROMPTS_DIR = '/shared/hdd/andre/predictions/formatted/prompts'
PROMPTS_DIR = '/shared/hdd/andre/predictions/formatted/prompts_noexplain'

def clean(input: str) -> str:
    THINK_PATTERN = re.compile(r"<think>.*?</think>\n\n", re.DOTALL)
    return THINK_PATTERN.sub("", input).strip()

def convert(file_path: str, section_name: str, model_name: str, dataset_proportion: float, output_dir: str, err_file) -> List[dict]:
    prompt_path = f"{PROMPTS_DIR}/{section_name}.json"
    results = []
    try:
        with open(file_path, 'r') as processing_file, open(prompt_path, 'r') as prompt_file:
            prompts = json.load(prompt_file)
            
            for idx, line in enumerate(processing_file):
                try:
                    data = json.loads(line)
                    entry = {
                        "instruction": prompts[idx],
                        "input": "",
                        "generator": model_name,
                        "output": clean(data["predict"])
                    }
                    results.append(entry)
                except json.JSONDecodeError as e:
                    err_file.write(f"Skipping invalid JSON line at idx {idx} in file {file_path}: {line.strip()}\n")
                    raise
    except FileNotFoundError as e:
        err_file.write(f"File not found: {file_path}\n")
        raise
    except Exception as e:
        err_file.write(f"Unexpected error while processing {file_path}: {str(e)}\n")
        raise

    # assert math.ceil(len(prompts) * dataset_proportion) == len(results), "SOMETHING WENT HORRIBLY WRONG! NUMBER OF PROMPTS SHOULD MATCH NUMBER OF EXAMPLES!"

    output_file = f"{output_dir}/{section_name}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as write_file:
        json.dump(results, write_file, indent=4)


def process_jsonl_files(base_dir: str, output_dir: str, dataset_proportion:float = 1.0) -> None:
    """
    Takes in a base directory, checks for 6 files. Process each of the 6 files 
    into the desired JSON format before concatenating them and writing to an output file.
    """
    model_name = os.path.basename(os.path.normpath(base_dir))
    file_names = [
        "abstract_predictions.jsonl",
        "background_predictions.jsonl",
        "conclusion_predictions.jsonl",
        "evaluation_predictions.jsonl",
        "introduction_predictions.jsonl",
        "title_predictions.jsonl"
    ]

    section_names = [
        'abstract',
        'background',
        'conclusion',
        'evaluation',
        'introduction',
        'title'
    ]

    # error_file = "./error.txt"
    error_file = "./converter_log.txt"
    with open(error_file, 'a') as err_file:
        for section_name, file_name in zip(section_names, file_names):
            processing_file = os.path.join(base_dir, file_name)
            convert(processing_file, section_name, model_name, dataset_proportion, output_dir, err_file)
    

if __name__ == "__main__":
    parser = ArgumentParser(description="Intermediate conversion step for alapca_eval.")
    parser.add_argument("--input_base_directory", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Name of the output directory of JSON files.")
    parser.add_argument("--dataset_proportion", type=float, default=1.0, help="Proportion of dataset to use (default: 1.0)")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    process_jsonl_files(args.input_base_directory, args.output_dir, args.dataset_proportion)