from openai import OpenAI
import httpx
import json
import os
import random
import argparse
from tqdm import tqdm

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
                        choices=['gpt-5','gpt-4', 'gpt-4o', 'deepseek-chat', 'deepseek-reasoner','claude-3-7-sonnet','qwen2.5-72b-instruct'],
                        help='Model to use for processing')
    parser.add_argument('--input', default="LLaMA-Factory/data/our_survey_test_12.json",
                        help='Input JSONL file path')
    parser.add_argument('--output', default="task/survey.json",
                        help='Output JSONL file path')
    parser.add_argument('--test-output', default="test_results.jsonl",
                        help='Test output JSONL file path')

    args = parser.parse_args()

    if args.model.startswith('gpt') or args.model.startswith('claude') or args.model.startswith('qwen'):
        model_provider = 'gpt'
    elif args.model.startswith('deepseek'):
        model_provider = 'deepseek'
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    input_file = args.input
    output_file = args.output

    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found.")
        return

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Create the output file if it doesn't exist
    open(output_file, 'a').close()

    with open(input_file, 'r', encoding='utf-8') as f:
        all_items = json.load(f)

    results = []
    for i, item in enumerate(tqdm(all_items, desc="Processing items")):
        instruction = item["instruction"]

        try:
            output = process_instruction(instruction, model_provider, args.model)
            item["predict"] = output
            print(output)
            item["label"] = item["output"]
            # item["model"] = args.model
            results.append(item)
        except Exception as e:
            print(f"Error processing instruction: {e}")
            item["error"] = str(e)
            # item["model"] = args.model
            results.append(item)

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\nFull processing complete. {len(results)} results saved to {output_file}")

if __name__ == "__main__":
    main()