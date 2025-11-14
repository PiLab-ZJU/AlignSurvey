# AlignSurvey

## Overview

AlignSurvey is a benchmark and toolkit for using LLMs in social surveys. AlignSurvey defines four tasks: social role modeling, semi-structured interviews, attitude stance, and survey response.

[📊 Datasets](https://huggingface.co/datasets/PiLabZJU/AlignSurvey_Datasets) |
[🤖 Models](https://huggingface.co/PiLabZJU) |
[📄 Paper] |
[📑 Full version](https://arxiv.org/abs/2511.07871) Continuously updated

This project is powered by [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). We gratefully acknowledge its maintainers and contributors for the codebase, documentation and tooling.

![Overview](images/aaai-benchmark.jpg)

## Directory Layout

You only need to focus on the following documents：

```text
data/
  foundation_qual/      # qualitative interview data
  foundation_quan/      # quantitative questionnaire data
  enitre_pipeline/      # four-task evaluation datasets

LLaMA-Factory/
  data/
    dataset_info.json   # update your dataset
  run/
    train/              # training scripts
    test/               # inference scripts
    api/                # API call
  generate/             # evaluation scripts
```

## Datasets

### Structure and directories
- Social Foundation Corpus (foundation layer)
  - `data/foundation_qual`: interview dialogues
  - `data/foundation_quan`: structured survey records
- Entire-Pipeline Survey Datasets (evaluation layer)
  - `data/enitre_pipeline`: data for the four tasks

### Sources
- [CGSS](http://cgss.ruc.edu.cn), [CSS](http://csqr.cass.cn/page/announcements/detail/61e122ec9f1fef6cdcfb951b), [ATP](https://ropercenter.cornell.edu/pew-research-center), [ESS](https://www.europeansocialsurvey.org), [GSS](https://gss.norc.org), [CHIP](https://bs.bnu.edu.cn/zgjmsrfpdcsjk/sjsq/index.html).
- **Please follow each source’s usage statements and licenses**.
- We release processed derivatives only; raw/original data should be obtained from the providers. Processed datasets are also available on [Datasets](https://huggingface.co/datasets/PiLabZJU/AlignSurvey_Datasets) and continuously expanded with additional processed data.

### Custom dataset
We encourage you to build task‑specific datasets tailored to your own social science research questions. All data are stored and processed locally, so you do not need to address privacy constraints for this workflow. Please convert your data to the required format, save it under your data directory, and update both `data_info.json` and the run scripts under `run/` to point to your files.

#### Required dataset format
- File type: JSON.
- Schema: Alpaca-style fields are expected by the trainer.
  - instruction: the full prompt shown to the model (you can embed demographics, question, and options here).
  - input: keep "" if unused.
  - output: the target response (e.g., "C. middle class" or just "C").
- Optional metadata fields (kept but not required by training): respondent_id, question_index, question, answer, choices, etc.

Example item (compatible with your current data):
```json
[
  {
    "instruction": "The survey is in 2024. You need to simulate a real survey respondent answering the following question.\nYour personal background:\nyour age: 64.0\nyour sex: male\nyour race: white\nyour ethnicity: \nyour marital status: never married\nyour education: graduate\nyour household income: \$25,000 or more\nyour employment status: retired\nyour place of residence: in a medium-size city (50,000-250,000)\nyour region: new england\n\nSurvey question:\nif you were asked to use one of four names for your social class, which would you say you belong in: the lower class, the working class, the middle class, or the upper class?\nA. lower class\nB. working class\nC. middle class\nD. upper class\n\nPlease provide your answer directly.",
    "input": "",
    "output": "C. middle class",
    "respondent_id": 2,
    "question_index": 0,
    "question": "if you were asked to use one of four names for your social class, which would you say you belong in: the lower class, the working class, the middle class, or the upper class?",
    "answer": "middle class"
  }
]
```

#### Where to place your files
- Put your JSON files under your chosen data directory, for example:
  ../../data/enitre_pipeline/task4/your_data/individual/your_train.json

#### Update data_info.json
- Add an entry pointing to your dataset file. Since your trainer expects Alpaca-style fields, explicitly mark the formatting as alpaca.
Example:
```json
{
  "your_key": {
    "file_name": "../../data/enitre_pipeline/task4/your_data/individual/your_train.json",
  }
}
```

#### Update run/base.sh
- Ensure the dataset variable matches the key in data_info.json, and paths are correct. Keep your current key train_ase: dataset="your_key"


### Usage and compliance
- Academic, non-commercial use only; do not attempt to re-identify individuals.
- Cite both this project and the original data sources; comply with the original licenses.
- We support fair and inclusive research; explicitly consider marginalized and intersectional groups in study design and interpretation.


## Models

### Already integrated
- SurveyLM-Llama-3.1-8B (based on Meta-Llama-3.1-8B-Instruct)
- SurveyLM-Qwen2.5-7B (based on Qwen2.5-7B-Instruct)
- SurveyLM-Mistral-7B (based on Mistral-7B-Instruct-v0.3)

### Other compatible models
- For available models and their corresponding templates, see: [supported-models](https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file#supported-models)
  
How to add a new model
1) Download the model locally and note its path.
2) Add the model to the models array and choose the matching chat template.

Example: extend models in `base.sh`
```
declare -A models=(
    ["Meta-Llama-3.1-8B-Instruct"]="/mnt/nvme1/hf-model/Meta-Llama-3.1-8B-Instruct"
    # Additions:
    ["Qwen2.5-14B-Instruct"]="/mnt/nvme1/hf-model/Qwen2.5-14B-Instruct"
)
```

### API usage

Replace `run/api` YOUR_API_KEY with your own key. If you serve your fine‑tuned model via an OpenAI‑compatible endpoint, set the base URL to your server.


## Usage Guide

### Installation: LLaMA-Factory
- Follow the official [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) installation guide.

### Data Preparation
- Place the datasets you want to evaluate under `data/enitre_pipeline` for the corresponding task.
- Ensure the format matches the JSON/JSONL schema used in this project (see example in the Datasets section).
- Recommended layout:
  - `data/enitre_pipeline/task1/` for social role modeling
  - `data/enitre_pipeline/task2/` for semi-structured interview modeling
  - `data/enitre_pipeline/task3/` for attitude stance modeling
  - `data/enitre_pipeline/task4/` for survey response modeling
- Modify the corresponding path in `LLaMA-Factory/data/dataset_info.json`.

### Training, Inference and Evaluation (example)
- Training:
  ```bash
  bash LLaMA-Factory/run/train/task1.sh
  ```
  - Customize in the script:
    - Dataset path(s) under `data/enitre_pipeline/...`
    - Base model name/path and temple
    - Output directory
    - Fine-tuning hyperparameters
- Inference:
  ```bash
  bash LLaMA-Factory/run/test/task1.sh
  ```
- Evaluation:
  ```bash
  python LLaMA-Factory/generate/task1/result.py
  ```

## Citation
```bibtex
@article{lin2025alignsurvey,
  title={AlignSurvey: A Comprehensive Benchmark for Human Preferences Alignment in Social Surveys},
  author={Lin, Chenxi and Yuan, Weikang and Jiang, Zhuoren and Huang, Biao and Zhang, Ruitao and Ge, Jianan and Xu, Yueqian and Yu, Jianxing},
  journal={arXiv preprint arXiv:2511.07871},
  year={2025}
}
```

## Acknowledgments
- Built on LLaMA-Factory—we thank the authors and community for their excellent open-source work.
- We thank public survey projects (CGSS, CSS, ATP, ESS, GSS, CHIP, and others) for providing open data and research resources.

## Roadmap
- Package the toolkit as an installable Python library with high-level APIs and documentation.
- Expand datasets and models, improve metrics and analytics, and support broader social science and cross-cultural applications.
