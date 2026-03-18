# RagBlog: Local RAG over Personal Blog Archives

RagBlog builds a semantic search and question-answering pipeline over personal blog content using retrieval augmented generation with local open models. The blog is a philosophical-poetic Blogger archive spanning 2011--2026; the pipeline crawls, chunks, embeds and retrieves context so a modest local LLM can answer intimate questions about the author's life, fatherhood, and worldview -- without sending a single token to the cloud.

**Pipeline**  
Crawl HTML -> extract posts -> JSONL -> split (500 chars, 50 overlap) -> embed with `qwen3-embedding:8b` -> persist in ChromaDB -> retrieve top-k (k=10) -> prompt local LLM -> parse answer.

Migration to llama.cpp enables Qwen3.5-27B (Q3_K_S quantized to fit 16GB VRAM on RTX 5060 Ti). Ollama limited larger models; llama.cpp provides precise GGUF control and GPU offload. Smaller models updated to same backend for fair comparison.

**Augmentation models tested**: `unsloth/Qwen3.5-27B-GGUF:Q3_K_S`, `unsloth/Qwen3.5-9B-GGUF:Q8_0`, `unsloth/gpt-oss-20b-GGUF:Q8_0` via llama.cpp. Same retrieval, different generation -- model size and quantization trade faithfulness for synthesis.

**Dataset**: 52 blog URLs from `delightfulobservaciones.blogspot.com` -> 904 document chunks (JSONL).  
**Output**: Natural language answers grounded in retrieved context, saved to `data/output.md`.

## Quick Start

```bash
git clone <repo>
cd ragblog
uv sync
ollama pull qwen3-embedding:8b
# Start llama.cpp server for aug models (see Backend section)
uv run python run/query.py
```

## Pipeline

```python
from src.rag import Rag

rag = Rag(is_overwrite_index=False)
response = rag.query(question="Describe the relation between Helena and Alejandra.")
print(response)
```

## Installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync  # deps + editable src
```

**Ollama** (embeddings):
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3-embedding:8b
```

**llama.cpp** (augmentation, GPU offload):
Build from https://github.com/ggerganov/llama.cpp then:
```bash
./build/bin/llama-server -hf unsloth/Qwen3.5-27B-GGUF:Q3_K_S -c 32768 --port 8080
# or for smaller:
# ./build/bin/llama-server -hf unsloth/Qwen3.5-9B-GGUF:Q8_0 -c 32768 --port 8080
# ./build/bin/llama-server -hf unsloth/gpt-oss-20b-GGUF:Q8_0 -c 32768 --port 8080
```
Hybrid: LLM on 8080 (llama.cpp), embeddings on 11434 (Ollama).

**Dev**: `uv sync --group dev` (ruff, pytest).

## CI

```bash
sh ci/lint.sh   # ruff check && ruff format
sh ci/test.sh   # uv run pytest -vv
```

## Evaluation

Ragas-based framework measuring retrieval and generation quality on a synthetic testset generated from the blog corpus.

```bash
uv run python run/generate_testset.py               # creates data/eval/eval_set.jsonl (16 cases)
uv run python run/evaluate.py --name "QWEN_3_5_9B"  # results -> data/eval/results/
```

### Aggregate Scores (Qwen3.5-9B Q8 via llama.cpp)

| metric | score |
| --- | --- |
| context_precision | 0.778 |
| context_recall | 0.950 |
| faithfulness | 0.833 |
| answer_relevancy | 0.383 |

Context recall at 0.95 confirms the retriever surfaces nearly all relevant chunks. Faithfulness at 0.83 shows the model rarely hallucinates beyond the retrieved context. Answer relevancy at 0.38 reflects a known weakness: the 9B model sometimes drifts into tangential elaboration or hedges rather than directly addressing the question. Precision at 0.78 indicates reasonable but imperfect ranking of retrieved chunks.

Compare configurations by running with different `--name` values (e.g., "baseline", "chunk-1000", "embedding-model-x").

## Insight: Local RAG with llama.cpp Backend

Updated tests extend to Qwen3.5-27B (Q3_K_S) requiring quantization to fit 16GB VRAM on RTX 5060 Ti. Migration from Ollama to llama.cpp preserves fairness by updating all models to GGUF backend with consistent OpenAI-compatible server.

Larger model improves synthesis on metaphor and cross-post inference while maintaining high faithfulness. 27B Q3 balances capability and memory; 9B Q8 and 20B Q8 provide baselines.

Local RAG excels in privacy and grounding. Semantic retrieval surfaces buried content. Model scale narrows the gap to frontier synthesis within hardware limits.

### Limitations

- Answer relevancy (0.38) remains weakest. 9B model elaborates; 27B Q3 shows synthesis gains but quantization trades some precision.
- The Grok comparison is qualitative, not automated. Formalising it would require a shared evaluation harness and ground-truth annotations.
- The blog corpus is small (52 posts, 904 chunks). RAG advantages compound with larger, more heterogeneous corpora where frontier models cannot memorise the full text.
- The HiperNeolítico retrieval success depends on embedding model quality. A weaker embedding model might miss the semantic bridge between "information as a good" and a post about civilisational transitions.
- Metaphorical content remains systematically underweighted. The 2015 boxing post ("Inventa un hueco") demonstrates that when the author's deepest expression avoids naming its subject, both embedding-based retrieval and frontier reasoning prefer the analytical gloss over the visceral source. Improving this would likely require re-ranking strategies or multi-hop reasoning that first retrieves the metaphor, then connects it to the explicit frame.

## Code Style

- clean code
  - object orientation
  - functional programming
  - classes, dataclasses and pydantic when justified
  - use conf where needed
  - hierarchical conf is clearer than long flat param lists
  - avoid underscore as function prefix for private. instead handle complexity as class method
- pytest TDD
- naming
  - self tabulated
  - promote 3 char naming
  - lexicography sorteable names if conceptually related
    - conf_data, conf_model instead of data_conf, model_conf
  - avoid plural naming: element_list instead of elements
  - self documenting naming
  - any code needing comments for clarification should be refactored into a named entity
  - each class implemented in its own file, snake_cased
    - caseclass and pydantic could be bundled into a single file
- **Configs**: `@dataclass(frozen=True)` hierarchical
- **Logging**: `LOGGER = logging.getLogger(__name__)` INFO/ERROR
- **Errors**: Specific `except`, log + sentinel return (`[]`)
- **Docs**: Google-style (`Args:`, `Returns:`)
- **No comments**: Refactor to named funcs/classes
- **Tests**: TDD `TestPascalName`, fixtures, `tmp_path`, strict asserts

## Data Source

- **Blog**: `delightfulobservaciones.blogspot.com` -- personal philosophical-poetic archive (2011--2026), 52 posts crawled via paginated Blogger HTML.
- **Grok chats** (frontier model comparisons referenced in this README):
  - [Boxing as Life Philosophy Metaphor](https://grok.com/share/bGVnYWN5_5e826f7e-70ab-4997-8d47-75e770fdc757)
  - [Bitcoin Thesis](https://grok.com/c/10602ead-e939-42ef-9257-8b044990c541?rid=36533622-acde-4674-bedd-c4f0ff2f4a79)
  - [Information as a Distinctive Commodity](https://grok.com/share/bGVnYWN5_33695110-1c15-444e-b30a-81eabcec9efb)

## Key Citations

- LangChain RAG documentation -- https://python.langchain.com/docs/tutorials/rag/
- ChromaDB vector store -- https://docs.trychroma.com/
- Ollama local inference -- https://ollama.com/
- Ragas evaluation framework -- https://docs.ragas.io/
- Grok 4.2 (xAI) -- https://grok.com/
