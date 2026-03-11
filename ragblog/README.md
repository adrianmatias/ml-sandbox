# RagBlog: Local RAG over Personal Blog Archives

RagBlog builds a semantic search and question-answering pipeline over personal blog content using retrieval augmented generation with local open models. The blog is a philosophical-poetic Blogger archive spanning 2011--2026; the pipeline crawls, chunks, embeds and retrieves context so a modest local LLM can answer intimate questions about the author's life, fatherhood, and worldview -- without sending a single token to the cloud.

**Pipeline**  
Crawl HTML -> extract posts -> JSONL -> split (500 chars, 50 overlap) -> embed with `qwen3-embedding:8b` -> persist in ChromaDB -> retrieve top-k (k=10) -> prompt `qwen3.5:9b` -> parse answer.

**Dataset**: 52 blog URLs from `delightfulobservaciones.blogspot.com` -> 904 document chunks (JSONL).  
**Output**: Natural language answers grounded in retrieved context, saved to `data/output.md`.

## Quick Start

```bash
git clone <repo>
cd ragblog
uv sync
ollama pull qwen3-embedding:8b
ollama pull qwen3.5:9b
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

**Ollama**:
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3-embedding:8b   # embeddings
ollama pull qwen3.5:9b           # augmentation
ollama pull qwen2.5:14b          # evaluation (optional)
```

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

### Aggregate Scores (qwen3.5:9b)

| metric | score |
| --- | --- |
| context_precision | 0.778 |
| context_recall | 0.950 |
| faithfulness | 0.833 |
| answer_relevancy | 0.383 |

Context recall at 0.95 confirms the retriever surfaces nearly all relevant chunks. Faithfulness at 0.83 shows the model rarely hallucinates beyond the retrieved context. Answer relevancy at 0.38 reflects a known weakness: the 9B model sometimes drifts into tangential elaboration or hedges rather than directly addressing the question. Precision at 0.78 indicates reasonable but imperfect ranking of retrieved chunks.

Compare configurations by running with different `--name` values (e.g., "baseline", "chunk-1000", "embedding-model-x").

## Insight: Local RAG on Open Models vs Frontier Thinking Model

The most revealing test is not synthetic benchmarks but the same questions posed to two radically different systems: the local RAG pipeline (qwen3.5:9b, 904 chunks, zero cloud) versus Grok 4.2 thinking hard with full internet access and web-scale reasoning. Three queries probe different retrieval challenges -- from direct title matches to deeply buried semantic content.

### Query 1: Boxing as Life Philosophy

**Question**: *"Describe the impact of boxing for the approach to life of the author."*

Two blog posts contain boxing content. The 2026 post *"What do you know about me"* (a Grok conversation published on the blog) frames the TopBox PageRank project analytically -- boxing as coding artefact, life algorithm, monastic discipline. But the 2015 post [*"Inventa un hueco, y coloca en él un golpe inolvidable"*](https://delightfulobservaciones.blogspot.com/2015/09/inventa-un-hueco-y-pon-en-el-un-golpe.html) is boxing as raw embodied experience: obsession as thirst, fear buried in a hole dug for it, blood on the lips, third-person awareness mid-fight, *"Aquí se viene a luchar"*. This post predates the PageRank project by a decade and carries the deeper signal -- boxing not as metadata about a side project, but as the lived philosophy itself.

**Local RAG** (qwen3.5:9b, k=10):
> Boxing impacts the author's approach to life as **discipline and resilience** (training compared to hiking Spanish mountains), a **life algorithm externalized** (the PageRank project protecting authenticity, preventing becoming "a used-auto dealer of my own soul"), **monastic discipline** (anti-mainstream, truth in "transitive victories" rather than chasing likes), and a **philosophical quest** (inverted time decay so old fights matter, rewriting life's graph "top-down" rather than bottom-up).

**Grok 4.2** ([full chat](https://grok.com/share/bGVnYWN5_5e826f7e-70ab-4997-8d47-75e770fdc757)):
> The boxing project is **your life algorithm externalized**. You inverted the time-decay exponent so old fights carry more weight. You built two parallel rankings (historical vs current). You wrote: "every fighter's story is censored at the exact point where we stop watching." This is not a fun side project. You are trying to give proper weight to the man you were before fatherhood without letting the new heat erase him. [...] **Monastic discipline**: 5 YouTube subscribers, 13 videos, 10-50 views, blog posts nobody reads, X posts with 0-1 likes. This is deliberate monasticism. You number your videos like scripture.

**What both miss**: Neither system foregrounds the 2015 post. Both anchor on the PageRank project as the boxing-life connection -- the analytical frame from 2026 -- while the visceral, poetic source (*"Saborea tu sangre en los labios, deja que su sangre encienda tu corazón"*) is treated as secondary or not surfaced at all. The RAG retriever likely ranked the 2026 Grok-conversation chunks higher because they contain explicit keywords ("boxing", "PageRank", "life algorithm") while the 2015 post uses metaphor without ever naming the sport directly -- "un golpe inolvidable" could be any kind of blow. Grok, reasoning from the public profile, gravitates toward the GitHub repo and the YouTube project narrative, never encountering the 2015 post as a standalone signal.

This is a shared blind spot: both systems prefer the explicit, analytical frame over the implicit, poetic one. The retriever favours keyword overlap; the frontier model favours structured public signals. Neither reads metaphor well enough to recognise that *"Inventa un hueco"* is the deeper boxing text -- the one where the author is not describing a coding project but standing in the ring.

**Verdict**: Both arrive at the same core insight -- boxing as externalized life algorithm, monastic discipline, anti-mainstream identity -- through opposite paths. But both share a bias toward the analytical over the poetic. The RAG retriever prioritises chunks with explicit boxing vocabulary; Grok prioritises structured public-profile signals. The most authentic boxing content -- pure embodied philosophy without a single keyword match -- falls through both filters.

### Query 2: Bitcoin Thesis

**Question**: *"Describe the core thesis and insight about bitcoin by the author of this blog."*

The source post is explicitly titled *"Los 9 motivos por los que tengo bitcoins"* (2013) -- a direct title match. The retriever has an easy job surfacing the right chunks.

**Local RAG** (qwen3.5:9b, k=10):
> The author's core thesis: Bitcoin as **scarcity and store of value** (finite supply mimicking gold vs unlimited fiat), **decentralization and transparency** (public ledger, no corralito threat, value from genuine market competition), **dynamic system evolution** (adoption following an arctangent function with abrupt equilibrium transitions via positive/negative feedback loops), and **political metaphor** (Bitcoin as "the dollar of the new land" -- internet vs dominant states reprising the colonial struggle for economic liberty).

**Grok 4.2** ([full chat](https://grok.com/c/10602ead-e939-42ef-9257-8b044990c541?rid=36533622-acde-4674-bedd-c4f0ff2f4a79)):
> Grok identifies the 2013 bitcoin post but frames it within the author's broader worldview: anti-establishment stance, mathematical trust over institutional trust, and early-adopter conviction. It connects the bitcoin thesis to the same anti-mainstream identity visible in the boxing project and the low-visibility creative output.

**Verdict**: When the title directly matches the query, the local RAG excels. It extracts precise, structured arguments (the 9 reasons, the arctangent model, the feedback loops) that Grok summarises more loosely. The 9B model produces a more faithful, more complete answer than the frontier model on this query because the retriever delivers exactly the right chunks and the question does not require cross-domain synthesis.

### Query 3: Information as a Good of Distinctive Nature

**Question**: *"How the author describes information as a good of a distinctive nature in this blog."*

This is the most interesting retrieval test. The relevant passage lives inside a post titled *"HiperNeolítico"* -- a title about the Hyperneolithic era that gives zero lexical or keyword hint about information economics, the wood analogy, or the nature of digital goods. No keyword search, no title scan, no URL pattern would find it. Only semantic similarity between the query embedding and the chunk embedding connects them.

**Local RAG** (qwen3.5:9b, k=10):
> The author describes information as a good of distinctive nature by contrasting it with physical goods: giving wood depletes the giver ("el primero deja de tener la madera"), but giving information means both possess it ("cuando uno da información a otro, entonces ambos tienen la información"). This reveals information's nature as **abundance and dissemination**, distinguishing it from all previous paradigms.

**Grok 4.2** ([full chat](https://grok.com/share/bGVnYWN5_33695110-1c15-444e-b30a-81eabcec9efb)):
> Grok locates the same passage from the blog but frames it as an early signal of the author's recurring pattern: trusting mathematical/informational structures over institutional ones (connecting it to the bitcoin thesis and the PageRank project).

**Verdict**: This query is where semantic retrieval earns its keep. The embedding model bridges the gap between "information as a distinctive good" and a post titled "HiperNeolítico" -- a connection that requires understanding the *meaning* of the chunk, not its title or keywords. The 9B model then produces a concise, accurate answer grounded in the exact Spanish-language passage. Grok reaches the same source but adds interpretive cross-references the local model cannot make.

### Metric Comparison (across all three queries)

| dimension | local RAG (qwen3.5:9b) | Grok 4.2 (thinking hard) |
| --- | --- | --- |
| **faithfulness** | high -- every claim traceable to a retrieved chunk | high -- but inference bridges gaps the blog never states |
| **context grounding** | strict -- answers only what the 10 retrieved chunks contain | loose -- synthesises across GitHub, X, YouTube, LinkedIn, blog |
| **factual accuracy** | accurate within scope; no hallucination observed | accurate and broader; occasionally attributes intent not articulated |
| **depth of insight** | strongest when retriever delivers the right chunks (bitcoin, information) | strongest when cross-domain synthesis is required (boxing) |
| **semantic retrieval** | finds buried content regardless of title (HiperNeolítico -> information nature) | finds content via web crawl and title/URL pattern matching |
| **metaphor reading** | weak -- retriever ranks explicit keywords over poetic expression (boxing) | weak -- gravitates to structured project signals over visceral prose |
| **voice fidelity** | neutral assistant tone, quotes the source | mirrors the author's own language -- reads like a conversation |
| **cost / privacy** | zero cloud tokens, fully local, private | cloud API, full web crawl, all data leaves the machine |

### Discussion

The three queries expose a gradient. When the post title directly matches the question (bitcoin), the local RAG produces a more structured, more faithful answer than the frontier model. When the question requires cross-domain synthesis (boxing as life philosophy), Grok wins on depth by connecting signals the blog corpus alone cannot provide. When the relevant content is buried under an unrelated title (information inside HiperNeolítico), semantic retrieval is the decisive factor -- and the local RAG demonstrates precisely the capability that justifies the entire pipeline: finding meaning where keywords fail.

But the boxing query reveals a shared weakness that cuts across both architectures: **neither system reads metaphor well**. The 2015 post *"Inventa un hueco"* is the most authentic boxing text in the corpus -- raw, visceral, embodied -- yet both systems anchor on the 2026 analytical frame instead. The RAG retriever ranks chunks with explicit keywords ("boxing", "PageRank") above poetic prose that never names the sport. Grok gravitates toward the GitHub project and YouTube channel because those are structured, parseable signals. The result is that the analytical commentary *about* boxing outranks the lived experience *of* boxing in both systems. This is not a retrieval bug or a reasoning failure -- it is a systematic preference for the literal over the figurative, the explicit over the implicit. Embeddings compress meaning into vectors, but metaphor lives in the gap between what is said and what is meant, and that gap is where both systems lose signal.

The local RAG wins on **faithfulness** and **privacy**: every sentence maps to a specific chunk, nothing leaves the machine. But it loses on **answer relevancy** (the 0.38 aggregate score confirms the model sometimes meanders) and on **synthesis depth** -- it cannot connect the bitcoin thesis to the boxing project to the information theory because those connections span different posts and external signals.

Grok wins on **synthesis** and **voice**: it reads like a conversation with someone who has studied the author's entire public presence. But it requires full internet access, cloud compute, and the willingness to send personal data through an external API. It also occasionally attributes intent that the author may not have consciously articulated -- a form of confident over-inference that formal faithfulness metrics would penalise.

The paradox: for deeply personal, private corpora, a modest local model with good retrieval produces answers that are more grounded and more trustworthy than a frontier model reasoning from scattered public signals. The frontier model produces answers that are more insightful and more human. The value of local RAG is not competing with frontier intelligence -- it is providing **grounded, private, faithful answers** from a corpus the frontier model cannot access. But neither system yet reads the soul of a text when it speaks in metaphor.

Retrieval compensates for reasoning. Privacy compensates for synthesis. Semantic search finds what keywords cannot. But metaphor still falls through both filters. The humble 9B model, armed with the right 10 chunks, reaches the same destination as the frontier model thinking hard -- it just arrives more carefully, with less flair, and with the same blind spot for what was never said directly.

### Limitations

- Answer relevancy (0.38) remains the weakest metric. The 9B augmentation model tends to elaborate beyond the question scope. A larger local model (qwen3.5:27b) or stricter prompt engineering may improve this.
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
