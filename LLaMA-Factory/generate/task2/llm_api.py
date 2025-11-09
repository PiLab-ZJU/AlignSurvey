# nohup python -u task/interview_generate/update_eval/llm_api.py --model gpt-4o
from openai import OpenAI
import httpx
import json
import os
import random
import argparse
from tqdm import tqdm
import glob

def setup_client(model_provider):
    if model_provider == "gpt":
        return OpenAI(
            base_url="https://api.xty.app/v1",
            api_key="your key",
            http_client=httpx.Client(
                base_url="https://api.xty.app/v1",
                follow_redirects=True,
            ),
        )
    elif model_provider == "deepseek":
        return OpenAI(
            base_url="https://api.deepseek.com",
            api_key="your key"
        )
    else:
        raise ValueError(f"Unsupported model provider: {model_provider}")

def process_instruction(instruction, model_provider, model_name):
    client = setup_client(model_provider)

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": instruction}
        ],
        stream=False,
    )

    return completion.choices[0].message.content

def main():
    parser = argparse.ArgumentParser(description='Process instructions with different LLM models')
    parser.add_argument('--model', required=True,
                        choices=['gpt-4', 'gpt-4o', 'deepseek-chat', 'deepseek-reasoner','claude-3-7-sonnet','qwen-max'],
                        help='Model to use for processing')
    parser.add_argument('--input-dir', default="task/interview_generate/update_eval/prompt/",
                        help='Input directory containing JSON files')
    parser.add_argument('--output-dir', default="task/interview_generate/update_eval/result/",
                        help='Output directory for processed JSONL files')
    parser.add_argument('--test-output', default="test_results.jsonl",
                        help='Test output JSONL file path')

    args = parser.parse_args()

    if args.model.startswith('gpt') or args.model.startswith('claude') or args.model.startswith('qwen'):
        model_provider = 'gpt'
    elif args.model.startswith('deepseek'):
        model_provider = 'deepseek'
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    input_dir = args.input_dir
    output_dir = args.output_dir

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Find all test_*.json files in the input directory
    input_files = glob.glob(os.path.join(input_dir, "*.jsonl"))

    if not input_files:
        print(f"No test_*.json files found in {input_dir}.")
        return

    for input_file in input_files:
        # Generate corresponding output file path
        file_name = os.path.basename(input_file)
        output_file = os.path.join(output_dir, f"{args.model}_{file_name}")

        print(f"\nProcessing file: {input_file}")
        if not os.path.exists(input_file):
            print(f"Input file {input_file} not found.")
            continue

        with open(input_file, 'r', encoding='utf-8') as f:
            all_items = json.load(f)

        results = []
        for i, item in enumerate(tqdm(all_items, desc="Processing items")):
            instruction = item["instruction"]
            # print(f"Processing {i + 1}/{len(all_items)}: {instruction[:50]}...")
            try:
                output = process_instruction(instruction, model_provider, args.model)
                # print(output)
                item["predict"] = output
                results.append(item)
            except Exception as e:
                print(f"Error processing instruction: {e}")
                item["error"] = str(e)
                results.append(item)

        with open(output_file, 'w', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"Processing complete for {input_file}. {len(results)} results saved to {output_file}")

if __name__ == "__main__":
    main()