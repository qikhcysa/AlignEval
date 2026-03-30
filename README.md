# AlignEval
**Knowledge Alignment Evaluation for Insightful Diagnosis & Optimization**

AlignEval is a system for precisely detecting, quantifying, and visualizing **knowledge defects** in fine-tuned large language models. It implements a complete _Detection → Evaluation → Optimization_ closed loop.

---

## System Architecture

```
Q&A Dataset
    │
    ▼
┌─────────────────────────────┐
│   KG Builder (NER + RE)     │  ← spaCy NER + dependency parsing
│   src/kg_builder/           │  ← domain-specific pattern matching
└────────────┬────────────────┘
             │ Source KG
             ▼
┌─────────────────────────────┐
│   Probing Module            │  ← 3-level prompt design
│   src/probing/              │    (Factual / Relational / Reverse)
└────────────┬────────────────┘  ← OpenAI-compatible LLM API
             │ LLM Responses
             ▼
┌─────────────────────────────┐
│   Response Processor        │  ← NER + RE on LLM responses
│                             │  → Learned KG
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   KG Aligner                │  ← similarity-based triple matching
│   src/alignment/            │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   Metrics                   │  ← Precision / Recall / F1
│   EvaluationMetrics         │  ← Missing & wrong triples
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   Web Visualization         │  ← D3.js interactive graph
│   app/                      │  ← Chart.js metrics dashboard
└─────────────────────────────┘  ← Optimization suggestions
```

---

## Features

| Feature | Description |
|---|---|
| **Source KG construction** | NER + relation extraction from Q&A datasets (spaCy + domain patterns) |
| **Multi-level probing** | Factual, relational, and reverse-reasoning prompts sent to LLMs |
| **Learned KG construction** | IE applied to LLM responses to build a "what the model knows" graph |
| **KG alignment** | Similarity-based matching of source vs learned triples |
| **Precision / Recall / F1** | Quantitative knowledge quality metrics |
| **Knowledge gap analysis** | Identifies missing triples (gaps) and wrong triples (hallucinations) |
| **D3.js graph visualization** | Interactive force-directed graph with status colour coding |
| **Optimization suggestions** | Data / Model / Reasoning layer recommendations |
| **Mock LLM mode** | Full demo without any API key |
| **REST API** | FastAPI backend for all pipeline steps |

---

## Quick Start

### Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Start the server
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** in your browser.

### Environment variables (optional)
| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | `""` | Your LLM API key (if empty, mock mode is used) |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | LLM API base URL |
| `LLM_MODEL` | `gpt-4o-mini` | Model to use |
| `LLM_MOCK_MODE` | `false` | Force mock mode even with an API key |
| `KG_SIMILARITY_THRESHOLD` | `0.85` | Minimum similarity for triple matching |
| `APP_PORT` | `8000` | Server port |

---

## API Reference

### Sessions
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/sessions/` | Create new evaluation session |
| `GET` | `/api/sessions/` | List all sessions |
| `GET` | `/api/sessions/{id}` | Get session details |
| `DELETE` | `/api/sessions/{id}` | Delete session |
| `POST` | `/api/sessions/{id}/upload-dataset` | Upload Q&A dataset (JSON) |

### Probing
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/probe/{id}` | Run multi-level LLM probing |
| `GET` | `/api/probe/{id}/prompts` | Get probe prompts & responses |

### Evaluation
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/evaluate/{id}` | Align KGs and compute metrics |
| `GET` | `/api/evaluate/{id}/metrics` | Get evaluation metrics |
| `GET` | `/api/evaluate/{id}/source-graph` | Source KG as D3 node-link JSON |
| `GET` | `/api/evaluate/{id}/learned-graph` | Learned KG as D3 node-link JSON |
| `GET` | `/api/evaluate/{id}/aligned-graph` | Combined aligned graph |
| `GET` | `/api/evaluate/{id}/missing-triples` | Knowledge gap triples |
| `GET` | `/api/evaluate/{id}/wrong-triples` | Incorrect knowledge triples |

---

## Dataset Format

Upload a JSON file with this structure:
```json
{
  "data": [
    {
      "question": "What is metformin?",
      "answer": "Metformin is a drug that treats type 2 diabetes.",
      "context": "",
      "domain": "biomedical"
    }
  ]
}
```
A plain JSON array of objects also works.

---

## Running Tests
```bash
python -m pytest tests/ -v
```

---

## Project Structure
```
AlignEval/
├── app/                     # FastAPI web application
│   ├── main.py              # App entry point
│   ├── routes/              # API route handlers
│   ├── static/              # CSS, JS, vendor assets
│   └── templates/           # Jinja2 HTML templates
├── src/
│   ├── kg_builder/          # NER + relation extraction + KG construction
│   ├── probing/             # Prompt design, LLM client, response processing
│   ├── alignment/           # KG alignment + precision/recall metrics
│   └── models/              # Pydantic data models
├── data/
│   └── sample_qa.json       # Sample biomedical Q&A dataset
├── tests/                   # Unit tests
├── config.py                # Configuration
└── requirements.txt
```

