# AlignSurvey

## Overview

AlignSurvey is a benchmark and toolkit for using LLMs in social surveys. AlignSurvey defines four tasks: social role modeling, semi-structured interviews, attitude stance, and survey response.

[📊 Datasets](https://huggingface.co/datasets/PiLab-ZJU/AlignSurvey) |
[🤖 Models](https://huggingface.co/PiLab-ZJU/AlignSurvey) |
[📄 Paper] |
[📑 Full version]()

This project is powered by [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). We gratefully acknowledge its maintainers and contributors for the codebase, documentation and tooling.

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
- We release processed derivatives only; raw/original data should be obtained from the providers. Processed datasets are also available on [Datasets](https://huggingface.co/datasets/PiLab-ZJU/AlignSurvey) and continuously expanded with additional processed data.

### Usage and compliance
- Academic, non-commercial use only; do not attempt to re-identify individuals.
- Cite both this project and the original data sources; comply with the original licenses.
- We support fair and inclusive research; explicitly consider marginalized and intersectional groups in study design and interpretation.

### Data format example
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

## Models
- SurveyLM-Llama-3.1-8B (based on Meta-Llama-3.1-8B-Instruct)
- SurveyLM-Qwen2.5-7B (based onQwen2.5-7B-Instruct)
- SurveyLM-Mistral-7B (based onMistral-7B-Instruct-v0.3)

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
  python LLaMA-Factory/generate/task1/acc_result.py
  ```

## Directory Layout (core)
```text
data/
  foundation_qual/      # qualitative interview data
  foundation_quan/      # quantitative questionnaire data
  enitre_pipeline/      # four-task evaluation datasets

LLaMA-Factory/
  data/
    dataset_info.json
  run/
    train/              # training scripts (e.g., task1.sh)
    test/               # inference/generation scripts (e.g., task1.sh)
  generate/             # evaluation and report scripts (e.g., eval_task1.sh)
```

## Citation
```bibtex
```

## Acknowledgments
- Built on LLaMA-Factory—we thank the authors and community for their excellent open-source work.
- We thank public survey projects (CGSS, CSS, ATP, ESS, GSS, CHIP, and others) for providing open data and research resources.

## Roadmap
- Package the toolkit as an installable Python library with high-level APIs and documentation.
- Expand datasets and models, improve metrics and analytics, and support broader social science and cross-cultural applications.
