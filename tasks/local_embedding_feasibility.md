# Local Sanskrit Embedding Model Feasibility

## Model

- Hugging Face model: `sanganaka/bge-m3-sanskritFT`
- Status checked: public repository, not gated, not private.
- License metadata: no explicit license tag in the model card/API response.
- Base model: `BAAI/bge-m3`
- Library: `sentence-transformers`
- Expected dense dimension: 1024

## Local CPU Use

The model is public and downloadable through Hugging Face, but the repository does
not publish an explicit license tag, so this project should treat it as an
external local runtime dependency rather than vendoring weights in Git.

The model is feasible to run on a 16 GB RAM CPU machine for query embedding and
small-batch corpus embedding. Full corpus re-indexing can be slow on CPU, so the
local backend should use small batches and cache embeddings aggressively.

## Implementation Decision

Use `sanganaka/bge-m3-sanskritFT` as the default local dense embedding backend
when `EMBEDDING_BACKEND=local`. Keep NVIDIA embeddings available through
`EMBEDDING_BACKEND=nvidia` for comparison, but avoid requiring `NVIDIA_API_KEY`
for normal dense retrieval tests.
