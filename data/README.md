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