import { useEffect, useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

const defaultQuery = "What does BG 2.47 teach about action and duty?";

async function apiFetch(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!response.ok) {
    let detail = "";
    try {
      const payload = await response.json();
      detail = payload?.detail || payload?.error || payload?.message || "";
    } catch {
      detail = "";
    }
    throw new Error(detail ? `Request failed with ${response.status}: ${detail}` : `Request failed with ${response.status}`);
  }

  return response.json();
}

function sourceLabel(status) {
  if (!status?.available) return "Unavailable";
  if (status?.contributed) return "Contributed";
  return "No match";
}

function formatSourceModes(result) {
  const sources = result?.metadata?.sources || {};
  const qdrantModes = result?.metadata?.qdrant_modes || [];
  const labels = [];

  if (sources.neo4j) {
    labels.push("Neo4j");
  }
  if (sources.qdrant) {
    labels.push(qdrantModes.length ? `Qdrant (${qdrantModes.join(", ")})` : "Qdrant");
  }

  return labels.length ? labels.join(" | ") : "No source metadata";
}

function SourceCard({ name, status }) {
  const state = status?.contributed ? "good" : status?.available ? "idle" : "down";
  return (
    <article className={`source-card ${state}`}>
      <span className="eyebrow">{name}</span>
      <strong>{sourceLabel(status)}</strong>
      <small>{status?.candidate_count ?? 0} reranked candidates</small>
    </article>
  );
}

function ConfidenceStrip({ result }) {
  if (!result) {
    return null;
  }

  const confidence = Number(result.confidence || 0);
  const intent = result.query_intent?.intent || "unknown";
  const abstention = result.abstention_reason;
  const explicitRefs = result.explicit_references || [];

  return (
    <div className="confidence-strip">
      <div>
        <span className="eyebrow">Confidence</span>
        <strong>{Math.round(confidence * 100)}%</strong>
      </div>
      <div>
        <span className="eyebrow">Intent</span>
        <strong>{intent.replaceAll("_", " ")}</strong>
      </div>
      <div>
        <span className="eyebrow">Explicit refs</span>
        <strong>{explicitRefs.length ? explicitRefs.join(", ") : "None"}</strong>
      </div>
      {abstention ? <span className="warning-chip">{abstention.replaceAll("_", " ")}</span> : null}
    </div>
  );
}

function Timeline({ result }) {
  const stats = result?.retrieval_stats || {};
  const steps = [
    ["Query", result?.query || "Waiting for a question"],
    ["Intent route", result?.query_intent?.intent?.replaceAll("_", " ") || "Not classified"],
    ["Dual retrieval", `${stats.rrf_results ?? 0} fused candidates`],
    ["Canonical verses", `${stats.unique_verses ?? 0} unique verse IDs`],
    ["Commentary", `${stats.commentary_candidates ?? 0} matches`],
    ["LLM synthesis", result ? `${Math.round(result.latency_ms || 0)} ms` : "Not run yet"],
  ];

  return (
    <section className="panel timeline">
      <div className="panel-heading">
        <span className="eyebrow">Pipeline</span>
        <h2>Evidence path</h2>
      </div>
      {steps.map(([title, detail], index) => (
        <div className="timeline-step" key={title}>
          <span>{index + 1}</span>
          <div>
            <strong>{title}</strong>
            <p>{detail}</p>
          </div>
        </div>
      ))}
    </section>
  );
}

function VerseCard({ verse }) {
  const speaker = verse?.metadata?.speaker;
  return (
    <article className="verse-card">
      <div className="card-title">
        <span>{verse.verse_id || "Unknown verse"}</span>
        <small>{verse.source || "Evidence"}</small>
      </div>
      {speaker ? <p className="speaker">{speaker}</p> : null}
      <pre>{verse.text || "No verse text available."}</pre>
    </article>
  );
}

function CommentaryCard({ match }) {
  return (
    <article className="commentary-card">
      <div className="card-title">
        <span>{match.author_display_name || match.author_key || "Commentary"}</span>
        <small>{match.verse_id}</small>
      </div>
      <p>{match.text || "No commentary text available."}</p>
      <small>Semantic score: {Number(match.score || 0).toFixed(4)}</small>
    </article>
  );
}

function JsonDrawer({ payload }) {
  const [open, setOpen] = useState(false);
  return (
    <section className="panel debug-panel">
      <button className="ghost-button" onClick={() => setOpen((value) => !value)}>
        {open ? "Hide" : "Show"} debug JSON
      </button>
      {open ? <pre>{JSON.stringify(payload || {}, null, 2)}</pre> : null}
    </section>
  );
}

function StatsPanel({ stats, refreshStats }) {
  const components = stats?.components || {};
  return (
    <section className="panel stats-panel">
      <div className="panel-heading row-heading">
        <div>
          <span className="eyebrow">System</span>
          <h2>Runtime status</h2>
        </div>
        <button className="ghost-button" onClick={refreshStats}>Refresh</button>
      </div>
      <div className="status-grid">
        {["qdrant", "neo4j", "sqlite", "llm"].map((name) => (
          <div className="status-pill" key={name}>
            <span className={components[name] ? "dot on" : "dot"} />
            <strong>{name.toUpperCase()}</strong>
            <small>{components[name] ? "online" : "offline"}</small>
          </div>
        ))}
      </div>
      <div className="stats-copy">
        <p>SQLite verses: {stats?.sqlite?.total_verses ?? 0}</p>
        <p>Qdrant points: {stats?.qdrant?.row_count ?? 0}</p>
        <p>Neo4j chunks: {stats?.neo4j?.chunk_count ?? 0}</p>
      </div>
    </section>
  );
}

export default function App() {
  const [query, setQuery] = useState(defaultQuery);
  const [topK, setTopK] = useState(10);
  const [regularization, setRegularization] = useState("combined");
  const [answerResult, setAnswerResult] = useState(null);
  const [searchResult, setSearchResult] = useState(null);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState("");

  const dbStatus = answerResult?.evidence?.db_status || answerResult?.sources?.db_status || {};
  const searchDbStatus = searchResult?.db_status || {};
  const canonicalVerses = answerResult?.evidence?.canonical_verses || [];
  const commentaryMatches = answerResult?.commentary_matches || [];
  const citations = answerResult?.citations || [];
  const sourceSummary = useMemo(() => {
    const qdrant = sourceLabel(dbStatus.qdrant);
    const neo4j = sourceLabel(dbStatus.neo4j);
    return `Qdrant: ${qdrant} | Neo4j: ${neo4j}`;
  }, [dbStatus]);

  async function refreshStats() {
    try {
      const data = await apiFetch("/api/stats");
      setStats(data);
    } catch (err) {
      setError(err.message);
    }
  }

  useEffect(() => {
    refreshStats();
  }, []);

  async function submitAsk(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      const data = await apiFetch("/api/ask", {
        method: "POST",
        body: JSON.stringify({ query, top_k: Number(topK), regularization }),
      });
      setAnswerResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function submitSearch() {
    setSearching(true);
    setError("");
    try {
      const data = await apiFetch("/api/search", {
        method: "POST",
        body: JSON.stringify({ query, top_k: Number(topK), regularization }),
      });
      setSearchResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setSearching(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="hero">
        <div>
          <span className="eyebrow">SansRAG Scholar Console</span>
          <h1>Ask across graph, vector, verse, and commentary evidence.</h1>
          <p>
            Every answer attempts Qdrant and Neo4j retrieval, re-ranks the combined evidence,
            resolves original verses from SQLite, and then synthesizes with commentary.
          </p>
        </div>
        <div className="hero-seal">
          <span>Dual DB</span>
          <strong>RRF</strong>
          <small>canonical verses first</small>
        </div>
      </section>

      {error ? <div className="error-banner">{error}</div> : null}

      <section className="dashboard">
        <aside className="left-rail">
          <form className="panel query-panel" onSubmit={submitAsk}>
            <div className="panel-heading">
              <span className="eyebrow">Question</span>
              <h2>Compose query</h2>
            </div>
            <textarea
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Ask about a verse, concept, speaker, or Sanskrit term..."
            />
            <label>
              Top-K evidence
              <input
                type="range"
                min="1"
                max="50"
                value={topK}
                onChange={(event) => setTopK(event.target.value)}
              />
              <span>{topK}</span>
            </label>
            <label>
              Regularization
              <select value={regularization} onChange={(event) => setRegularization(event.target.value)}>
                <option value="combined">Combined</option>
                <option value="l1">L1</option>
                <option value="l2">L2</option>
                <option value="none">None</option>
              </select>
            </label>
            <div className="button-row">
              <button className="primary-button" disabled={loading}>
                {loading ? "Generating..." : "Generate answer"}
              </button>
              <button type="button" className="ghost-button" disabled={searching} onClick={submitSearch}>
                {searching ? "Searching..." : "Inspect RRF"}
              </button>
            </div>
          </form>

          <StatsPanel stats={stats} refreshStats={refreshStats} />
          <Timeline result={answerResult} />
        </aside>

        <section className="main-stage">
          <section className="source-grid">
            <SourceCard name="Qdrant" status={dbStatus.qdrant} />
            <SourceCard name="Neo4j" status={dbStatus.neo4j} />
          </section>

          <section className="panel answer-panel">
            <div className="panel-heading row-heading">
              <div>
                <span className="eyebrow">Final answer</span>
                <h2>{answerResult ? sourceSummary : "Awaiting evidence"}</h2>
                </div>
              {answerResult?.has_evidence === false ? <span className="warning-chip">No evidence</span> : null}
            </div>
            <ConfidenceStrip result={answerResult} />
            <div className="answer-copy">
              {answerResult?.answer || "Run a query to generate a grounded answer."}
            </div>
          </section>

          <section className="evidence-grid">
            <div className="panel">
              <div className="panel-heading">
                <span className="eyebrow">Original verses</span>
                <h2>Canonical evidence</h2>
              </div>
              {canonicalVerses.length ? (
                canonicalVerses.map((verse) => <VerseCard verse={verse} key={verse.verse_id} />)
              ) : (
                <p className="empty-state">No canonical verses selected yet.</p>
              )}
            </div>

            <div className="panel">
              <div className="panel-heading">
                <span className="eyebrow">Commentary</span>
                <h2>Related interpretation</h2>
              </div>
              {commentaryMatches.length ? (
                commentaryMatches.map((match) => (
                  <CommentaryCard match={match} key={`${match.verse_id}-${match.commentary_id}`} />
                ))
              ) : (
                <p className="empty-state">No commentary match found for the current evidence.</p>
              )}
            </div>
          </section>

          <section className="panel citations-panel">
            <div className="panel-heading">
              <span className="eyebrow">Citations</span>
              <h2>Answer references</h2>
            </div>
            {citations.length ? (
              citations.map((citation, index) => (
                <div className="citation-row" key={`${citation.verse_id}-${index}`}>
                  <strong>[{index + 1}] {citation.verse_id}</strong>
                  <span>{citation.source || "Evidence"} | {Number(citation.score || 0).toFixed(4)}</span>
                  <p>{citation.text}</p>
                </div>
              ))
            ) : (
              <p className="empty-state">Citations appear after answer generation.</p>
            )}
          </section>

          {searchResult ? (
            <section className="panel search-panel">
              <div className="panel-heading">
                <span className="eyebrow">RRF inspection</span>
                <h2>{searchResult.total_results || 0} fused candidates</h2>
              </div>
              <p className="empty-state">
                Qdrant: {sourceLabel(searchDbStatus.qdrant)} | Neo4j: {sourceLabel(searchDbStatus.neo4j)}
              </p>
              {searchDbStatus.qdrant?.contributed === false && searchDbStatus.neo4j?.contributed ? (
                <p className="empty-state">
                  No Qdrant match for this query run. The candidates below came from Neo4j retrieval.
                </p>
              ) : null}
              {(searchResult.results || []).slice(0, 6).map((result, index) => (
                <div className="result-row" key={result.id}>
                  <strong>#{index + 1} {result.verse_id || "No verse"}</strong>
                  <span>{Number(result.final_score || 0).toFixed(4)}</span>
                  <small>{formatSourceModes(result)}</small>
                  <p>{result.text}</p>
                </div>
              ))}
            </section>
          ) : null}

          <JsonDrawer payload={{ answerResult, searchResult, stats }} />
        </section>
      </section>
    </main>
  );
}
