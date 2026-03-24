# Antahkarana v11 — IEEE Evaluation Pipeline

**Model**: Meta-Llama-3.1-8B-Instruct  
**Hardware**: NVIDIA L4 24GB · 32 vCPUs · 128GB RAM  
**Engine**: vLLM (continuous batching · BFloat16 · CUDA graphs)

---

## Project Structure

```
antahkarana/
├── main.py                  # Full pipeline entry point
├── vllm_engine.py           # Singleton vLLM engine (loads model ONCE)
├── Antahkarana_v11.ipynb    # JupyterLab version (cell-by-cell)
│
├── datasets/
│   ├── __init__.py
│   └── loader.py            # HotpotQA, MMLU, TruthfulQA, FEVER, SVAMP
│
├── baselines/
│   ├── __init__.py
│   ├── prompts.py           # Prompt builders for all 4 baselines
│   └── runner.py            # Batch runners: direct, cot, sc, tot
│
├── antahkarana/
│   ├── __init__.py
│   └── system.py            # Manas-Chitta-Buddhi-Sakshi orchestrator
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py           # EM, F1, SF, bootstrap CI, t-test
│   ├── ablation.py          # 4 ablation configs, batched
│   └── visualize.py         # Bar charts, latency scatter, ablation plot
│
└── results/
    ├── raw/                 # Per-dataset JSON predictions
    ├── processed/           # metrics_summary.json, metrics_table.csv, final_report.txt
    ├── ablation/            # ablation_summary.json, ablation_table.csv
    ├── stats/               # significance.json
    └── plots/               # PNG charts
```

---

## Setup

### 1. Install dependencies (terminal or Cell 1 of notebook)

```bash
pip install vllm
pip install datasets transformers accelerate
pip install scipy matplotlib numpy sentence-transformers
```

### 2. Set HuggingFace token

```bash
export HF_TOKEN=hf_your_token_here
```

Or in Cell 2 of the notebook:
```python
os.environ['HF_TOKEN'] = 'hf_your_token_here'
```

---

## Execution

### Notebook (recommended)

Open `Antahkarana_v11.ipynb` and run cells in order:

| Cell | Action |
|------|--------|
| 1    | Install dependencies |
| 2    | Environment and GPU check |
| 3    | Load vLLM engine (one-time, ~60s) |
| 4    | Load all 5 datasets |
| 5    | Run 4 baselines (batched) |
| 6    | Run Antahkarana (batched) |
| 7    | Compute EM/F1/SF metrics |
| 8    | Ablation study (50 samples) |
| 9    | Statistical significance (t-test) |
| 10   | Generate plots (PNG) |
| 11   | Save CSV and final report |

### Script

```bash
cd antahkarana
python main.py --n-main 100 --n-ablation 50
```

---

## Architecture

### vLLM Integration
- Single LLM instance created once via VLLMEngine.get()
- gpu_memory_utilization=0.92 leaves ~22GB of 24GB for KV cache
- max_model_len=4096 optimal for 8B on L4
- max_num_batched_tokens=8192 for continuous batching
- All methods share the same engine — no model reloading

### Batching
- All prompts for a method+dataset batched in a single llm.generate() call
- Self-Consistency: n=5 samples generated in one call per prompt (vLLM native)
- Antahkarana: 3 passes (Tarka, Pramana, Samsaya), each fully batched

### Antahkarana Stages (v11)
1. Manas — Route to QType: simple / multihop / comparison / math / verification / mchoice
2. Chitta — Score context paragraphs by keyword + entity overlap; extract evidence spans
3. Buddhi/Tarka — Type-specific staged reasoning prompt
4. Buddhi/Pramana — Grounding verification (hotpotqa/fever only)
5. Samsaya — Self-consistency repair for uncertain answers (n=5)
6. Sakshi — Fallback repair for empty/apology answers

### Dataset Fallbacks
Each loader tries multiple HuggingFace paths and falls back gracefully:
- hotpot_qa distractor -> hotpotqa
- cais/mmlu subject -> lukaemon/mmlu -> cais/mmlu all
- truthful_qa multiple_choice -> truthful_qa generation
- fever v1.0 -> pietrolesci/fever
- ChilleD/SVAMP -> GitHub CSV

---

## Outputs

| File | Description |
|------|-------------|
| results/raw/<dataset>_results.json | Full per-sample predictions |
| results/processed/metrics_table.csv | EM/F1/SF/latency for all methods |
| results/processed/metrics_summary.json | Aggregated metrics with 95% CI |
| results/processed/final_report.txt | Publication-ready tables + conclusion |
| results/ablation/ablation_table.csv | Ablation EM/F1 comparison |
| results/stats/significance.json | p-values and significance markers |
| results/plots/*.png | Bar charts, scatter, ablation plots |

---

## Metrics (IEEE Standard)

- EM — Exact Match (normalized)
- F1 — Token-level F1
- SF-EM / SF-F1 — Supporting Fact EM/F1 (HotpotQA only)
- Joint EM / Joint F1 — Answer x Supporting Fact (HotpotQA)
- Latency — Mean seconds per sample
- Throughput — Samples per second
- 95% CI — Bootstrap (1000 resamples)
- Significance — Paired t-test: *** p<0.001, ** p<0.01, * p<0.05

---

## Expected Runtime (L4, 8B model)

| Method           | Throughput   |
|------------------|-------------|
| Direct           | ~12-15 samp/s |
| CoT              | ~6-8 samp/s |
| Self-Consistency | ~2-3 samp/s |
| ToT              | ~5-7 samp/s |
| Antahkarana      | ~3-4 samp/s |

Total: 100 samples x 5 datasets x 5 methods ~ 45-75 minutes
