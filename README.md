# Decomposing Unitization and Typing for Efficient and Consistent Span-Bound Concept Annotation

Code for the paper: **"Decomposing Unitization and Typing for Efficient and Consistent Span-Bound Concept Annotation"**  
[OpenReview](https://openreview.net/forum?id=afyrC4Bxb8)

## Overview

Standard NER annotation requires annotators to identify both the *exact span boundaries* and the *semantic type* of each entity in a single pass — a cognitively demanding task that can be slow and inconsistently performed.

This paper proposes a **decomposed annotation strategy**:
1. **Unitization**: Annotators mark a single character position (⧫ lozenge) somewhere inside each entity span — a much faster and more consistent task.
2. **Typing**: A model trained on just 100–200 examples infers the exact span boundaries (Point-to-Span, P2S) and assigns the semantic type.

The resulting silver labels can then train a full NER model, achieving competitive performance while requiring far less annotator effort.

```
Annotator marks positions:  "The ⧫protein PCNA is involved in DNA ⧫replication."
P2S model infers spans:      [(protein, PCNA), (biological_process, DNA replication)]
```

### Two experimental conditions

| Condition | Description | Script |
|---|---|---|
| Full concept annotation | Annotators label full entity spans + types directly | `scripts/run_ner_silver.sh` |
| Decomposed annotation | Annotators mark positions only; model infers spans | `scripts/run_p2s_silver.sh` |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download datasets

**GENIA** (biomedical NER):
```python
from datasets import load_dataset
ds = load_dataset("DFKI-SLT/few-nerd", "supervised")
```

**CRAFT** (biomedical NER with ontology-linked entities):  
Download from [https://github.com/UCDenver-ccp/CRAFT](https://github.com/UCDenver-ccp/CRAFT) and set `craft_dir` in your config.

**POLIANNA** (climate policy annotation):  
Download from [https://github.com/kueddelmaier/POLIANNA](https://github.com/kueddelmaier/POLIANNA) and set `polianna_pickle_path` in your config.

### 3. Configure paths

Copy the config template and fill in your paths:

```bash
cp configs/config.template.json configs/my_config.json
# Edit configs/my_config.json to set PREPROCESSED_DIR, OUTPUT_DIR, etc.
```

## Data Preparation

Generate conversation-format training data from raw datasets:

```bash
# Generate NER training data for GENIA
python -m src.write_conversations \
    --config configs/my_config.json \
    --domain genia \
    --task ner_efficiency

# Generate P2S + silver NER data
python -m src.write_conversations \
    --config configs/my_config.json \
    --domain genia \
    --task silver_sample_size

# Generate job lists for batch experiments
python -m src.setup_jobs \
    --config configs/my_config.json \
    --domain genia
```

## Training and Evaluation

### Full concept annotation (NER silver baseline)

Train a NER model using the two-stage silver pipeline:

```bash
CONFIG_FILE=configs/example_ner_silver_genia.json \
REPO_DIR=/path/to/ner-efficiency-public \
bash scripts/run_ner_silver.sh
```

### Decomposed annotation (P2S silver — proposed method)

Train using the three-stage decomposed pipeline:

```bash
CONFIG_FILE=configs/example_p2s_silver_genia.json \
REPO_DIR=/path/to/ner-efficiency-public \
bash scripts/run_p2s_silver.sh
```

### P2S data efficiency study

Study how NER performance scales with the number of P2S-annotated examples:

```bash
CONFIG_FILE=configs/p2s_sample=249_genia.json \
REPO_DIR=/path/to/ner-efficiency-public \
bash scripts/run_dataeff.sh
```

### Standalone evaluation

Run evaluation on any trained model:

```bash
python -m src.evaluate \
    --model_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --lora_path /path/to/lora/checkpoint \
    --test_data_path /path/to/test.json \
    --result_filepath results.csv

# For P2S model output (converts span positions back to entity spans):
python -m src.evaluate \
    --model_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --lora_path /path/to/p2s/checkpoint \
    --silver_data_path /path/to/unlabeled.json \
    --from_p2s \
    --result_filepath results.csv
```

## Repository Structure

```
src/
├── data/
│   ├── instances.py        # EntityInstance dataclass
│   ├── conversation.py     # Conversation format generation for train/test
│   └── evaluate_utils.py   # NER evaluation metrics (micro/macro F1, span matching)
├── point_to_span.py        # Point-to-span conversion utilities
├── load_data.py            # Dataset loaders: POLIANNA, CRAFT, GENIA, FewNERD
├── write_conversations.py  # Generate training data splits and silver pipelines
├── setup_jobs.py           # Generate config files and job lists for experiments
└── evaluate.py             # vLLM-based inference and evaluation

configs/
├── config.template.json           # Master config template
├── example_ner_silver_genia.json  # Example: NER silver experiment on GENIA
└── example_p2s_silver_genia.json  # Example: P2S silver experiment on GENIA

scripts/
├── run_ner_silver.sh   # Full concept annotation pipeline
├── run_p2s_silver.sh   # Decomposed annotation pipeline (proposed)
└── run_dataeff.sh      # Data efficiency study
```

## Acknowledgements

The LoRA fine-tuning code in `src/train/` is adapted from [FastChat](https://github.com/lm-sys/FastChat) (Zheng et al., 2023), which is licensed under the Apache 2.0 License.

## Citation

```bibtex
@inproceedings{gandhi2024decomposing,
  title={Decomposing Unitization and Typing for Efficient and Consistent Span-Bound Concept Annotation},
author = "Gandhi, Nupoor  and
      Bada, Michael  and
      Strubell, Emma",
    editor = "Jurgens, David  and
        Zhang, Jiajun and
      Liakata, Maria  and
      Moreira, Viviane",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2026",
    month = jul,
    year = "2026",
    address = "San Diego",
    publisher = "Association for Computational Linguistics",
    abstract = "In specialized domains that require expert annotators and high inter-annotator agreement, high-quality datasets with span-bound semantic concept annotations remain expensive to develop. Substantial resources are typically spent on \textit{unitization}, the task of identifying precise span boundaries for entity mentions. Unitizing is a significant source of inter-annotator disagreement, a poor use of expensive domain expertise, and very time-consuming. We propose a lighter annotation procedure that concentrates manual efforts on typed position annotations, marking positions in the text that overlap with mentions of each entity type, abstracting away span boundary decisions. With as few as 100-200 example sentences, we train span boundary detection models to unitize typed position annotations. Through evaluation over three datasets: CRAFT (biomedical), GENIA (molecular biology), and POLIANNA (climate/energy policy text), we demonstrate that (1) annotating typed positions in the text instead of full concept annotation is a more efficient use of time in low-resource settings, and (2) model-inferred span boundaries result in higher agreement at both the annotator training and corpus annotation phases, without sacrificing utility."
}
```
