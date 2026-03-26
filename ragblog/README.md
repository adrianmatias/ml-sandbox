# RagBlog: Local RAG over Personal Blog Archives

RagBlog builds a semantic search and question-answering pipeline over personal blog content using retrieval augmented generation with local open models. The blog is a philosophical-poetic Blogger archive spanning 2011--2026; the pipeline crawls, chunks, embeds and retrieves context so a modest local LLM can answer intimate questions about the author's life, fatherhood, and worldview -- without sending a single token to the cloud.

**Pipeline**  
1. Crawl HTML
2. extract posts
3. JSONL
4. split (500 chars, 50 overlap)
5. embed with `qwen3-embedding:8b`
6. persist in ChromaDB
7. retrieve top-k (k=10)
8. prompt local LLM
9. parse answer

**Augmentation models tested**:
- `qwen3.5:27bIQ2_M` "https://huggingface.co/unsloth/Qwen3.5-27B-GGUF"
- `qwen3.5:9b`
- `gpt-oss:20b`

**Dataset**: 52 blog URLs from `delightfulobservaciones.blogspot.com` -> 904 document chunks (JSONL).  
**Output**: Natural language answers grounded in retrieved context, saved to `data/output.md`.

## Quick Start

```bash
git clone <repo>
cd ragblog
uv sync
ollama pull qwen3-embedding:8b
ollama pull gpt-oss:20b
ollama pull qwen3.5:9b
sh run/wrap_gguf_ollama.sh qwen3.5:27bIQ2_M ~/Downloads/Qwen3.5-27B-UD-IQ2_M.ggu
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
ollama pull gpt-oss:20b          # augmentation (default)
ollama pull qwen3.5:9b           # augmentation (alternative, smaller)
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
notebooks
- [query_gpt_oss_20b.ipynb](run/query_gpt_oss_20b.ipynb)
- [compare.ipynb](run/compare.ipynb)


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

The most revealing test is not synthetic benchmarks but the same questions posed to four systems: three local RAG pipelines (`qwen3.5:27bIQ2_M`, `qwen3.5:9b`, and `gpt-oss:20b`, same retrieval, different generation) versus Grok 4.2 thinking hard with full internet access and web-scale reasoning. Three queries probe different retrieval challenges -- from direct title matches to deeply buried semantic content. The three local models isolate the effect of augmentation model size on the same retrieved chunks.

### Query 1: Boxing as Life Philosophy

**Question**: *"Describe the impact of boxing for the approach to life of the author."*

Two blog posts contain boxing content. The 2026 post *"What do you know about me"* (a Grok conversation published on the blog) frames the TopBox PageRank project analytically -- boxing as coding artefact, life algorithm, monastic discipline. But the 2015 post [*"Inventa un hueco, y coloca en él un golpe inolvidable"*](https://delightfulobservaciones.blogspot.com/2015/09/inventa-un-hueco-y-pon-en-el-un-golpe.html) is boxing as raw embodied experience: obsession as thirst, fear buried in a hole dug for it, blood on the lips, third-person awareness mid-fight, *"Aquí se viene a luchar"*. This post predates the PageRank project by a decade and carries the deeper signal -- boxing not as metadata about a side project, but as the lived philosophy itself.

**Local RAG** (qwen3.5:9b, k=10):
> Boxing impacts the author's approach to life as **discipline and resilience** (training compared to hiking Spanish mountains), a **life algorithm externalized** (the PageRank project protecting authenticity, preventing becoming "a used-auto dealer of my own soul"), **monastic discipline** (anti-mainstream, truth in "transitive victories" rather than chasing likes), and a **philosophical quest** (inverted time decay so old fights matter, rewriting life's graph "top-down" rather than bottom-up).

**Local RAG** (gpt-oss:20b, k=10, same retrieval):
> Boxing teaches **discipline and routine** -- training linked to the same "spiritual routine" applied to hiking, writing code, maintaining a blog. **Resilience and mindset** -- the fight inside a boxer is a constant battle against loss of identity, just as a new father battles the erosion of his former self. **Analytical rigor** -- the sport's emphasis on timing and strategy dovetails with the PageRank code. **Identity and authenticity** -- boxing keeps him true to the "quiet scripture" he writes, refusing to become "the used-auto dealer of my own soul". **Life as a series of bouts** -- projects as fights against "entropy" and "Extremistan", daily grind as a boxer's training regimen.

**Local RAG** (qwen3.5:27bIQ2_M, k=10, same retrieval):
> Boxing shapes the author's life as **discipline and resilience** -- training compared to hiking Spanish mountains, a "spiritual routine" applied across domains. **Life algorithm externalized** -- the PageRank project protects authenticity, inverting time decay so old fights carry weight, preventing becoming "a used-auto dealer of my own soul". **Monastic discipline** -- anti-mainstream stance, truth in "transitive victories" rather than chasing likes. The 27B version is the most coherent, grounding each theme in specific retrieved quotes while maintaining narrative flow. After ThinkingOutputParser cleanup, it produces the cleanest synthesis of the three local models.

**Grok 4.2** ([full chat](https://grok.com/share/bGVnYWN5_5e826f7e-70ab-4997-8d47-75e770fdc757)):
> The boxing project is **your life algorithm externalized**. You inverted the time-decay exponent so old fights carry more weight. You built two parallel rankings (historical vs current). You wrote: "every fighter's story is censored at the exact point where we stop watching." This is not a fun side project. You are trying to give proper weight to the man you were before fatherhood without letting the new heat erase him. [...] **Monastic discipline**: 5 YouTube subscribers, 13 videos, 10-50 views, blog posts nobody reads, X posts with 0-1 likes. This is deliberate monasticism. You number your videos like scripture.

**What all three miss -- and what the 20B partially recovers**: All three systems anchor on the PageRank project as the boxing-life connection -- the analytical frame from 2026. The 2015 post (*"Saborea tu sangre en los labios, deja que su sangre encienda tu corazón"*) is never foregrounded as the primary source. The RAG retriever likely ranked the 2026 Grok-conversation chunks higher because they contain explicit keywords ("boxing", "PageRank", "life algorithm") while the 2015 post uses metaphor without ever naming the sport -- "un golpe inolvidable" could be any kind of blow. Grok, reasoning from the public profile, gravitates toward the GitHub repo and YouTube project narrative, never encountering the 2015 post as a standalone signal.

However, the `gpt-oss:20b` model partially bridges the gap. Working from the same 10 retrieved chunks as the 9B, it generates language that echoes the embodied register of the 2015 post: "the fight inside a boxer", "life as a series of bouts", "daily grind as a boxer's training regimen". The 9B model stays strictly analytical; the 20B infers a physical dimension the chunks only hint at. This is not retrieval improvement -- both models receive the same context. It is the larger model's capacity to read between the lines of the retrieved text, inferring the embodied experience from analytical descriptions of it. The metaphor gap narrows with model size, even when retrieval does not change.

**Verdict**: The 9B and Grok converge on the same analytical insight through opposite paths. The 20B, reading the same chunks, adds an embodied layer that neither the 9B nor Grok produce. But none of the three systems foreground *"Inventa un hueco"* as the primary boxing text. The deeper source -- pure visceral philosophy without a single keyword match -- still falls through all three filters.

### Query 2: Bitcoin Thesis

**Question**: *"Describe the core thesis and insight about bitcoin by the author of this blog."*

The source post is explicitly titled *"Los 9 motivos por los que tengo bitcoins"* (2013) -- a direct title match. The retriever has an easy job surfacing the right chunks.

**Local RAG** (qwen3.5:9b, k=10):
> The author's core thesis: Bitcoin as **scarcity and store of value** (finite supply mimicking gold vs unlimited fiat), **decentralization and transparency** (public ledger, no corralito threat, value from genuine market competition), **dynamic system evolution** (adoption following an arctangent function with abrupt equilibrium transitions via positive/negative feedback loops), and **political metaphor** (Bitcoin as "the dollar of the new land" -- internet vs dominant states reprising the colonial struggle for economic liberty).

**Local RAG** (gpt-oss:20b, k=10, same retrieval):
> Bitcoin as a **shared ledger** with **scarcity built into the protocol** (capped supply, mining progressively harder, "like gold"). Immune to arbitrary corralitos or inflationary policies. Value derived solely from real-world competitive markets. Growth driven by **dynamic feedback loops** (positive: price rise attracts buyers; negative: high value discourages spending) that may push Bitcoin into stable equilibrium **abruptly** -- potentially becoming "the dollar of the new internet".

**Local RAG** (qwen3.5:27bIQ2_M, k=10, same retrieval):
> Bitcoin as **scarcity and store of value** (finite supply, mining progressively harder, "like gold"). **Decentralization and transparency** -- public ledger, no corralito threat, value from genuine market competition. **Dynamic system evolution** -- adoption following an arctangent function with abrupt equilibrium transitions via feedback loops. **Political metaphor** -- "the dollar of the new land", internet vs dominant states reprising the colonial struggle. The 27B version is the most structured and faithful to the original 2013 post, producing the cleanest enumeration of the core thesis.

**Grok 4.2** ([full chat](https://grok.com/c/10602ead-e939-42ef-9257-8b044990c541?rid=36533622-acde-4674-bedd-c4f0ff2f4a79)):
> Grok identifies the 2013 bitcoin post but frames it within the author's broader worldview: anti-establishment stance, mathematical trust over institutional trust, and early-adopter conviction. It connects the bitcoin thesis to the same anti-mainstream identity visible in the boxing project and the low-visibility creative output.

**Verdict**: When the title directly matches the query, both local models excel over Grok. The 9B produces a broader enumeration (four themes, close to the original 9 reasons). The 20B produces a tighter, more structured synthesis -- fewer themes, crisper language, "shared ledger" and "dynamic feedback loops" distilled into a coherent narrative. Both outperform Grok on faithfulness here because the retriever delivers exactly the right chunks and no cross-domain synthesis is needed. The 20B trades coverage for clarity; the 9B trades clarity for completeness.

### Query 3: Information as a Good of Distinctive Nature

**Question**: *"How the author describes information as a good of a distinctive nature in this blog."*

This is the most interesting retrieval test. The relevant passage lives inside a post titled *"HiperNeolítico"* -- a title about the Hyperneolithic era that gives zero lexical or keyword hint about information economics, the wood analogy, or the nature of digital goods. No keyword search, no title scan, no URL pattern would find it. Only semantic similarity between the query embedding and the chunk embedding connects them.

**Local RAG** (qwen3.5:9b, k=10):
> The author describes information as a good of distinctive nature by contrasting it with physical goods: giving wood depletes the giver ("el primero deja de tener la madera"), but giving information means both possess it ("cuando uno da información a otro, entonces ambos tienen la información"). This reveals information's nature as **abundance and dissemination**, distinguishing it from all previous paradigms.

**Local RAG** (gpt-oss:20b, k=10, same retrieval):
> Information is not a finite resource one can "own" like a physical good. When knowledge is shared, both giver and receiver keep it -- it is essentially **abundant and freely diffusing**. Yet its value lies in the **authenticity and personal insight** the author brings to it, making information uniquely valuable even when widely shared. An intangible, endlessly shareable good whose distinctive quality lies in its non-exhaustive, personal nature rather than in scarcity.

**Local RAG** (qwen3.5:27bIQ2_M, k=10, same retrieval):
> Information contrasts with physical goods: giving wood depletes the giver, but giving information means both possess it. This reveals information's nature as **abundance and dissemination**. The 27B version is the most concise and grounded, sticking strictly to the exact contrast in the HiperNeolítico post without adding interpretive layers. Highest faithfulness to the corpus among local models.

**Grok 4.2** ([full chat](https://grok.com/share/bGVnYWN5_33695110-1c15-444e-b30a-81eabcec9efb)):
> Grok locates the same passage from the blog but frames it as an early signal of the author's recurring pattern: trusting mathematical/informational structures over institutional ones (connecting it to the bitcoin thesis and the PageRank project).

**Verdict**: This query is where semantic retrieval earns its keep. The embedding model bridges the gap between "information as a distinctive good" and a post titled "HiperNeolítico" -- a connection that requires understanding the *meaning* of the chunk, not its title or keywords. The 9B model produces a concise, accurate answer grounded in the exact Spanish-language passage. The 20B adds an interpretive layer -- "authenticity and personal insight" -- that is not in the retrieved text but resonates with the author's broader philosophy. This is the faithfulness-synthesis trade-off in miniature: the 9B sticks to what the chunks say, the 20B infers what the chunks mean. Grok reaches the same source but adds cross-domain connections the local models cannot make.

### Metric Comparison (across all three queries)

| dimension | local RAG (qwen3.5:27bIQ2_M) | local RAG (qwen3.5:9b) | local RAG (gpt-oss:20b) | Grok 4.2 (thinking hard) |
| --- | --- | --- | --- | --- |
| **faithfulness** | highest -- every claim traceable, parser removes thinking traces | high -- every claim traceable to a retrieved chunk | moderate -- adds interpretive layers not in chunks | high -- but inference bridges gaps the blog never states |
| **context grounding** | strict -- answers only what the 10 retrieved chunks contain | strict -- answers only what the 10 retrieved chunks contain | moderate -- infers meaning beyond retrieved text | loose -- synthesises across GitHub, X, YouTube, LinkedIn, blog |
| **factual accuracy** | accurate within scope; no hallucination observed | accurate within scope; no hallucination observed | accurate but sometimes over-interprets | accurate and broader; occasionally attributes intent not articulated |
| **depth of insight** | strongest coherence and quote grounding | strongest when retriever delivers the right chunks (bitcoin, information) | good synthesis, trades coverage for clarity | strongest when cross-domain synthesis is required (boxing) |
| **semantic retrieval** | finds buried content regardless of title (HiperNeolítico -> information nature) | finds buried content regardless of title | same retrieval as other local models | finds content via web crawl and title/URL pattern matching |
| **metaphor reading** | weak -- retriever ranks explicit keywords over poetic expression | weak -- retriever ranks explicit keywords over poetic expression (boxing) | moderate -- infers embodied dimension from analytical chunks | weak -- gravitates to structured project signals over visceral prose |
| **voice fidelity** | neutral assistant tone, quotes the source | neutral assistant tone, quotes the source | neutral but more interpretive | mirrors the author's own language -- reads like a conversation |
| **cost / privacy** | zero cloud tokens, fully local, private | zero cloud tokens, fully local, private | zero cloud tokens, fully local, private | cloud API, full web crawl, all data leaves the machine |

### Discussion
**Insight: Local RAG on Open Models**

We re-ran the same five queries across three augmentation models using identical retrieval (`k=10`): `qwen3.5:27bIQ2_M` (Unsloth + ThinkingOutputParser), `qwen3.5:9b` (native Ollama), and `gpt-oss:20b`.

The 27B IQ2_M variant fits within the 16 GB VRAM limit when quantized. Ollama handles it cleanly via Modelfile. It consistently produces high-quality synthesis but requires the ThinkingOutputParser to remove persistent reasoning traces.

#### Query Results

**1. Relation between Helena and Alejandra**  
All models correctly identify them as sisters, with Helena as the elder. The 27B version provides the most layered reflection on privacy, legacy, and the author's fatherhood transformation.

**2. Transcendence of Lanzarote**  
All models describe the “I was dead. And then I was not” moment. The 27B version most clearly captures the non-linear time insight (“top-down script”) and its impact on weighting the past.

**3. Impact of boxing**  
All three link boxing to discipline, the PageRank project as a “life algorithm,” authenticity, and monastic discipline. The 27B version is the most coherent after parser cleanup.

**4. Bitcoin thesis**  
All models cover scarcity, transparency, and feedback loops. The 27B version is the most structured and faithful to the original 2013 post.

**5. Information as a good of distinctive nature**  
All models accurately contrast information with physical goods (wood analogy) and highlight abundance and diffusion. The 27B version is the most concise.

**Ragas Evaluation (16 test cases)**

| model                | context_precision | context_recall | faithfulness | answer_relevancy |
|----------------------|-------------------|----------------|--------------|------------------|
| qwen3.5:27bIQ2_M     | 0.559             | 0.729          | 0.897        | 0.674            |
| qwen3.5:9b           | 0.568             | 0.778          | 0.851        | 0.565            |
| gpt-oss:20b          | 0.515             | 0.679          | 0.560        | **0.714**        |

**Metric context**  
- **faithfulness**: Measures how grounded the answer is in the retrieved context (how little the model hallucinates or adds unsupported information).  
- **answer_relevancy**: Measures how directly the response addresses the actual question asked (how well it stays on-topic without drifting).
- **context_precision**: Measures how relevant the retrieved chunks were to the question.  
- **context_recall**: Measures how much of the truly relevant information in the corpus was surfaced.  

Even though the embedding model, Chroma index, and retrieval parameters are identical, small differences in precision and recall appear across runs due to stochastic ranking behaviour in vector search.

**Comparison with Grok 4.2**  
Grok 4.2 (with full web access and thinking mode) produces more synthetic and conversational answers on the same questions. It often connects themes across posts and mirrors the author’s voice more naturally. However, it sometimes introduces external context or inferred intent not present in the retrieved chunks.

Specific cases:

- **Boxing query**: Grok emphasised the monastic discipline and identity preservation with poetic language (“the fight inside a boxer is a constant battle against loss of identity”) and linked it to the author’s broader life experiment. The local 27B version was more concise and strictly grounded in the PageRank project and hiking analogy from the retrieved chunks.

- **Information as a good of distinctive nature**: Grok added cross-domain connections (linking the wood analogy to bitcoin and anti-establishment themes). The local models (especially 27B) stayed strictly to the exact contrast in the HiperNeolítico post, showing higher faithfulness to the corpus.

**Summary**  
Each model shows different strengths. The 27B IQ2_M + parser combination delivers the highest faithfulness. The 9B model offers the best recall and speed. The 20B stands out for answer relevancy. Grok wins on breadth and voice. Choice depends on the priority: strict grounding in context, efficiency, or direct engagement with the question.
The eye-test, reading the responses for inspection, supported by the domain knowledge of me as the author of the blog, makes me favour the qwen3.5:27bIQ2_M variant over the rest of local models and Grok 4.2. It shows accuracy, low hallucination, depth, consistency and relevant quotes to actual content.


### Limitations

- Answer relevancy varies by model: 27bIQ2_M (0.674), 9b (0.565), 20b (0.714). The 9B augmentation model tends to elaborate beyond the question scope. The 20B achieves highest relevancy but at the cost of faithfulness (0.560). The 27B offers the best balance of faithfulness (0.897) and relevancy (0.674).
- The Grok comparison is qualitative, not automated. Formalising it would require a shared evaluation harness and ground-truth annotations.
- The blog corpus is small (52 posts, 904 chunks). RAG advantages compound with larger, more heterogeneous corpora where frontier models cannot memorise the full text.
- The HiperNeolítico retrieval success depends on embedding model quality. A weaker embedding model might miss the semantic bridge between "information as a good" and a post about civilisational transitions.
- Metaphorical content remains systematically underweighted. The 2015 boxing post ("Inventa un hueco") demonstrates that when the author's deepest expression avoids naming its subject, both embedding-based retrieval and frontier reasoning prefer the analytical gloss over the visceral source. Improving this would likely require re-ranking strategies or multi-hop reasoning that first retrieves the metaphor, then connects it to the explicit frame.

### Constraints on hardware and software

The 27B Unsloth IQ2_M variant runs on the RTX 5060 Ti 16GB but with these trade-offs:

- Quantized model fits within VRAM constraints (≈13–14.5 GB loaded).
- llama.cpp was the initial choice but abandoned (no native embeddings + unmanageable dual-backend sequencing with Ollama).
- Ollama succeeded with a simple Modelfile wrapper — cleanest experience, no process management, no interference with the embedding model.
- Response quality is excellent and noticeably superior to 9B/14B models.
- Thinking traces (`<think>`, "Thinking Process:", numbered lists, runes) persist despite every flag, prompt tweak, and `options={"think": False}`.
- ThinkingOutputParser (regex-based) was implemented as the safety net but leaks occur in varied formats.
- Native Ollama qwen3.5:27b with reduced context (1024) still caused CPU offload and unacceptable latency.

**Conclusion**: The thinking parser is an acceptable solution. The 27B IQ2_M remains the highest-quality local model for this corpus when quality > speed.

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
