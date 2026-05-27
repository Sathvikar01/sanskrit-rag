# Lessons Learned

## Project: SansRAG - Sanskrit Text Retrieval System

---

## Technical Lessons

### 1. Embedding API Integration
- **Lesson**: NVIDIA NIM API requires proper error handling and rate limiting
- **Implementation**: Added exponential backoff with `MAX_RETRIES=3`
- **Pattern**: Always implement caching to avoid re-embedding same texts

### 2. Vector Database Index Selection
- **Lesson**: Different indices suit different embedding types
  - HNSW for dense vectors (fast ANN, good recall)
  - Sparse Inverted Index for lexical features
  - IVF for multi-vector (Colbert) embeddings
- **Decision**: Use hybrid indices based on embedding type

### 3. Regularization in Retrieval
- **Lesson**: L1/L2 regularization prevents overfitting to any single retrieval method
  - L1 (Lasso): Drives less important weights toward zero (feature selection)
  - L2 (Ridge): Smooths score distribution
- **Implementation**: Combined regularization with adaptive lambda tuning
- **Default values**: L1=0.01, L2=0.001

### 4. RRF Fusion for Hybrid Search
- **Lesson**: Reciprocal Rank Fusion works better than weighted sum for combining
  different retrieval methods
- **Formula**: `score = Σ(weight / (k + rank))` where k=60 is a tuning parameter

### 5. Semantic Chunking for Sanskrit
- **Lesson**: Sanskrit texts need sentence-aware chunking (।, ॥ delimiters)
- **Implementation**: Custom sentence splitter in `SemanticChunker`
- **Pattern**: Use tiktoken for token counting, 512 tokens with 50 overlap

---

## Process Lessons

### 1. Plan Before Implementation
- Always create detailed specs in `tasks/todo.md`
- Break complex tasks into phases
- Verify each phase before moving to next

### 2. Test Coverage
- Unit tests for each component
- Integration tests for pipeline
- Mock external dependencies (Milvus, NVIDIA API)

### 3. Error Handling
- Graceful degradation when Milvus unavailable
- Cache embeddings to avoid API re-calls
- Retry logic for network failures

---

## Patterns to Reuse

### 1. Dataclass for Results
```python
@dataclass
class SearchResult:
    id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### 2. Configuration via Environment
```python
L1_REG_LAMBDA = float(os.getenv("L1_REG_LAMBDA", "0.01"))
```

### 3. Retry with Exponential Backoff
```python
retry_strategy = Retry(
    total=MAX_RETRIES,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504]
)
```

---

## Mistakes to Avoid

1. **Don't** use hardcoded API keys in source code
2. **Don't** skip error handling for external services
3. **Don't** forget to flush/commit after batch inserts to Milvus
4. **Don't** use same index type for all vector types

---

## Last Updated
2026-03-30
