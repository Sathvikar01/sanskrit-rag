import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import './App.css'

const API_URL = 'http://localhost:8000'

/* ============ Quill SVG Icon ============ */
const QuillIcon = ({ className = '', style = {} }) => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.5"
    strokeLinecap="round"
    strokeLinejoin="round"
    className={className}
    style={{ width: 20, height: 20, ...style }}
  >
    <path d="M2 22c0-15 16-22 16-22s-2 10 0 22z" />
    <path d="M18 2c-2 0-3.5 1.5-5 4-1.5 2.5-1.5 6-1.5 6s3.5-1.5 6-4c1.8-1.8 3.5-4 3.5-7 0-1-1-1-3-1z" />
    <path d="M6.5 14c2.5 2.5 5 3 7.5 3" strokeOpacity="0.5" />
  </svg>
)

/* ============ Toggle Switch ============ */
const ToggleSwitch = ({ label, checked, onChange, color = '#4a3b32' }) => (
  <label className="toggle-switch">
    <input type="checkbox" checked={checked} onChange={onChange} />
    <span className="toggle-slider" style={{ '--toggle-color': color }} />
    <span className="toggle-label">{label}</span>
  </label>
)

/* ============ Collapsible Section ============ */
const CollapsibleSection = ({ title, count, icon, children, defaultOpen = false }) => {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="collapsible-section">
      <button className="collapsible-header" onClick={() => setOpen(!open)}>
        <span className="collapsible-icon">{icon || '>'}</span>
        <span className="collapsible-title">{title}</span>
        {count !== undefined && <span className="collapsible-count">{count}</span>}
        <span className={`collapsible-arrow ${open ? 'open' : ''}`}>&#9654;</span>
      </button>
      {open && <div className="collapsible-body">{children}</div>}
    </div>
  )
}

/* ============ Result Card ============ */
const ResultCard = ({ item, index }) => (
  <div className="result-card">
    <div className="result-card-header">
      <span className="result-ref">{item.verse_ref || item.chunk_id}</span>
      <span className="result-score">{(item.score || item.confidence || item.final_score || 0).toFixed(4)}</span>
    </div>
    {item.sources && (
      <div className="result-sources">
        {item.sources.map((s, i) => <span key={i} className="source-tag">{s}</span>)}
      </div>
    )}
    {item.vector_score !== undefined && (
      <div className="result-scores">
        <span>v:{item.vector_score.toFixed(3)}</span>
        <span>g:{item.graph_score?.toFixed(3) || '0.000'}</span>
        <span>b:{item.bm25_score?.toFixed(3) || '0.000'}</span>
      </div>
    )}
    {item.chunk_type && <span className="result-type">{item.chunk_type}</span>}
  </div>
)

/* ============ Typewriter Hook ============ */
function useTypewriter(text, speed = 28) {
  const [displayed, setDisplayed] = useState('')
  const [done, setDone] = useState(false)

  useEffect(() => {
    if (!text) { setDisplayed(''); setDone(true); return }
    setDisplayed('')
    setDone(false)
    let i = 0
    let timers = []
    const scheduleNext = () => {
      if (i > text.length) {
        setDone(true)
        return
      }
      setDisplayed(text.slice(0, i))
      i++
      if (i <= text.length) {
        const delay = Math.max(5, speed + (Math.random() * 20 - 10))
        timers.push(setTimeout(scheduleNext, delay))
      } else {
        setDone(true)
      }
    }
    const startTimer = setTimeout(scheduleNext, speed * 2)
    timers.push(startTimer)
    return () => { timers.forEach(clearTimeout) }
  }, [text, speed])

  return { displayed, done }
}

/* ============ App ============ */
function App() {
  const [query, setQuery] = useState('')
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [sidePanelOpen, setSidePanelOpen] = useState(false)
  const [toggles, setToggles] = useState({ vector: true, graph: true, bm25: true })
  const [normalize, setNormalize] = useState('minmax')
  const [lastIntermediate, setLastIntermediate] = useState({})
  const [lastCommentaries, setLastCommentaries] = useState({})
  const messagesEndRef = useRef(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!query.trim() || loading) return

    const userMsg = query.trim()
    setQuery('')
    setMessages(prev => [...prev, { role: 'user', content: userMsg }])
    setLoading(true)

    try {
      const res = await fetch(`${API_URL}/api/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userMsg, toggles, normalize }),
      })
      const data = await res.json()

      const newMsg = {
        role: 'assistant',
        content: data.answer,
        metadata: {
          concepts: data.concepts,
          verses: data.verses_cited,
          confidence: data.pipeline_confidence,
          topVerses: data.top_verses,
          queryType: data.query_type,
        },
      }

      setMessages(prev => [...prev, newMsg])
      setLastIntermediate(data.intermediate || {})
      setLastCommentaries(data.commentaries || {})
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Error: ${err.message}. Make sure the API server is running on port 8000.`,
      }])
    }
    setLoading(false)
  }

  const toggleMethod = (method) => {
    setToggles(prev => {
      const next = { ...prev, [method]: !prev[method] }
      const activeCount = Object.values(next).filter(Boolean).length
      if (activeCount === 0) return prev
      return next
    })
  }

  const latestAssistantIndex = messages.length - 1
  const latestContent = messages[messages.length - 1]?.content || ''
  const isNewAssistant = messages[messages.length - 1]?.role === 'assistant'

  const { displayed: typedDisplay, done: typedDone } = useTypewriter(
    isNewAssistant ? latestContent : '',
    28,
    false
  )

  const exampleQueries = [
    "What is nishkama karma?",
    "Explain the concept of dharma in the Bhagavad Gita",
    "What does Krishna say about death?",
    "How does jnana yoga differ from bhakti yoga?",
    "What are the three gunas?",
  ]

  const hasIntermediate = Object.keys(lastIntermediate).length > 0

  return (
    <div className={`app ${sidePanelOpen ? 'panel-open' : ''}`}>
      {/* Side Panel */}
      <aside className={`side-panel ${sidePanelOpen ? 'open' : ''}`}>
        <div className="side-panel-header">
          <h3>Pipeline Stages</h3>
          <button className="panel-close" onClick={() => setSidePanelOpen(false)}>&times;</button>
        </div>

        <div className="side-panel-body">
          {/* Toggles */}
          <div className="toggle-group">
            <h4>Retrieval Methods</h4>
            <ToggleSwitch label="Vector (FAISS)" checked={toggles.vector}
              onChange={() => toggleMethod('vector')} color="#7B1FA2" />
            <ToggleSwitch label="Graph (Neo4j)" checked={toggles.graph}
              onChange={() => toggleMethod('graph')} color="#C62828" />
            <ToggleSwitch label="BM25 Lexical" checked={toggles.bm25}
              onChange={() => toggleMethod('bm25')} color="#00838F" />
          </div>

          <div className="panel-divider" />

          {/* Normalization */}
          <div className="toggle-group">
            <h4>Feature Normalization</h4>
            <select
              className="normalize-select"
              value={normalize}
              onChange={(e) => setNormalize(e.target.value)}
            >
              <option value="none">None (raw scores)</option>
              <option value="minmax">Min-Max [0, 1]</option>
              <option value="l2">L2 (unit vector)</option>
              <option value="zscore">Z-score (std)</option>
            </select>
            <p className="normalize-hint">
              {normalize === 'none' && 'Raw feature scores, no normalization'}
              {normalize === 'minmax' && 'Scales each feature to [0, 1] across candidates'}
              {normalize === 'l2' && 'Normalizes feature vectors to unit length'}
              {normalize === 'zscore' && 'Standardizes to zero mean, unit variance'}
            </p>
          </div>

          <div className="panel-divider" />

          {/* Intermediate Results */}
          {hasIntermediate ? (
            <>
              <CollapsibleSection title="Vector Results" count={lastIntermediate.vector_results?.length}
                icon={<span style={{color:'#7B1FA2'}}>V</span>}>
                {lastIntermediate.vector_results?.map((r, i) => <ResultCard key={i} item={r} index={i} />)}
              </CollapsibleSection>

              <CollapsibleSection title="Graph Results" count={lastIntermediate.graph_results?.length}
                icon={<span style={{color:'#C62828'}}>G</span>}>
                {lastIntermediate.graph_results?.map((r, i) => <ResultCard key={i} item={r} index={i} />)}
              </CollapsibleSection>

              <CollapsibleSection title="BM25 Results" count={lastIntermediate.bm25_results?.length}
                icon={<span style={{color:'#00838F'}}>B</span>}>
                {lastIntermediate.bm25_results?.map((r, i) => <ResultCard key={i} item={r} index={i} />)}
              </CollapsibleSection>

              <CollapsibleSection title="Fused Results" count={lastIntermediate.fused_results?.length}
                icon={<span style={{color:'#F57F17'}}>F</span>}>
                {lastIntermediate.fused_results?.map((r, i) => <ResultCard key={i} item={r} index={i} />)}
              </CollapsibleSection>

              <CollapsibleSection title="Re-ranked Results" count={lastIntermediate.reranked_results?.length}
                icon={<span style={{color:'#AD1457'}}>R</span>}>
                {lastIntermediate.reranked_results?.map((r, i) => <ResultCard key={i} item={r} index={i} />)}
              </CollapsibleSection>

              {Object.keys(lastCommentaries).length > 0 && (
                <CollapsibleSection title="Commentaries" count={Object.keys(lastCommentaries).length}
                  icon={<span style={{color:'#5D4037'}}>C</span>}>
                  {Object.entries(lastCommentaries).map(([ref, comms]) => (
                    <div key={ref} className="commentary-group">
                      <div className="commentary-verse-ref">{ref}</div>
                      {comms.map((c, i) => (
                        <div key={i} className="commentary-card">
                          <span className="commentary-author">{c.commentator}</span>
                          <p className="commentary-text">{c.text}</p>
                        </div>
                      ))}
                    </div>
                  ))}
                </CollapsibleSection>
              )}
            </>
          ) : (
            <p className="panel-empty">No pipeline data yet. Ask a question to see intermediate results.</p>
          )}
        </div>
      </aside>

      {/* Main Content */}
      <div className="main-content">
        <header className="header">
          <div className="header-content">
            <button className="panel-toggle" onClick={() => setSidePanelOpen(!sidePanelOpen)} title="Pipeline Inspector">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="3" y1="6" x2="21" y2="6" /><line x1="3" y1="12" x2="21" y2="12" /><line x1="3" y1="18" x2="21" y2="18" />
              </svg>
            </button>
            <div className="logo">
              <span className="logo-sanskrit">||</span>
              <h1>SansRAG</h1>
              <span className="logo-sanskrit">||</span>
            </div>
            <p className="subtitle">Sanskrit RAG with Graph-Enhanced Linguistic Re-ranking</p>
            <p className="description">Bhagavad Gita Knowledge Engine &mdash; Vector + Neo4j Graph + BM25 Hybrid Retrieval</p>
          </div>
        </header>

        <main className="chat-container">
          {messages.length === 0 && (
            <div className="welcome">
              <div className="welcome-icon"><QuillIcon /></div>
              <h2>Ask the Bhagavad Gita</h2>
              <p>Query the Bhagavad Gita with all 18 chapters, 700 verses, and 3 traditional commentaries from Sridhara Swamin, Visvanatha Chakravarti, and Baladeva Vidyabhushana.</p>
              <div className="examples">
                {exampleQueries.map((q, i) => (
                  <button key={i} className="example-btn" onClick={() => setQuery(q)}>
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="messages">
            {messages.map((msg, i) => (
              <div key={i} className={`message ${msg.role}`}>
                <div className="message-avatar">
                  {msg.role === 'user' ? 'You' : '||'}
                </div>
                <div className="message-body">
                  {msg.role === 'assistant' ? (
                    <>
                      <div className="answer-content">
                        {i === latestAssistantIndex && !typedDone ? (
                          <div className="typing-text">
                            <ReactMarkdown>{typedDisplay || ' '}</ReactMarkdown>
                          </div>
                        ) : (
                          <ReactMarkdown>{msg.content}</ReactMarkdown>
                        )}
                      </div>
                      {msg.metadata && (
                        <div className="metadata">
                          {msg.metadata.queryType && (
                            <div className="metadata-row">
                              <span className="meta-label">Query Type:</span>
                              <span className="concept-tag">{msg.metadata.queryType}</span>
                            </div>
                          )}
                          <div className="metadata-row">
                            <span className="meta-label">Concepts:</span>
                            {msg.metadata.concepts?.map((c, j) => (
                              <span key={j} className="concept-tag">{c}</span>
                            ))}
                          </div>
                          <div className="metadata-row">
                            <span className="meta-label">Verses Cited:</span>
                            {msg.metadata.verses?.map((v, j) => (
                              <span key={j} className="verse-tag">{v}</span>
                            ))}
                          </div>
                          {msg.metadata.confidence && (
                            <div className="confidence-bar">
                              <span className="meta-label">Confidence:</span>
                              <div className="bar-container">
                                <div
                                  className="bar-fill"
                                  style={{ width: `${(msg.metadata.confidence.overall_confidence || 0) * 100}%` }}
                                />
                              </div>
                              <span className="confidence-value">
                                {((msg.metadata.confidence.overall_confidence || 0) * 100).toFixed(0)}%
                              </span>
                            </div>
                          )}
                        </div>
                      )}
                    </>
                  ) : (
                    <p>{msg.content}</p>
                  )}
                </div>
              </div>
            ))}

            {loading && (
              <div className="message assistant">
                <div className="message-avatar"><QuillIcon /></div>
                <div className="message-body">
                  <div className="quill-ink-container">
                    <QuillIcon className="quill-svg" />
                    <div className="quill-text-area">
                      <div className="quill-ink-writing"></div>
                      <div className="quill-writing-line" />
                    </div>
                  </div>
                  <p className="thinking-text">The quill is dipping in ink... composing wisdom...</p>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </main>

        <form className="input-form" onSubmit={handleSubmit}>
          <div className="input-wrapper">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask about the Bhagavad Gita..."
              disabled={loading}
            />
            <button type="submit" disabled={loading || !query.trim()} aria-label="Ask">
              <QuillIcon />
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default App
