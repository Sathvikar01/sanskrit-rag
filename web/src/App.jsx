import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import './App.css'

const API_URL = 'http://localhost:8000'

function App() {
  const [query, setQuery] = useState('')
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
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
        body: JSON.stringify({ query: userMsg }),
      })
      const data = await res.json()
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.answer,
        metadata: {
          concepts: data.concepts,
          verses: data.verses_cited,
          confidence: data.pipeline_confidence,
          topVerses: data.top_verses,
        },
      }])
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Error: ${err.message}. Make sure the API server is running on port 8000.`,
      }])
    }
    setLoading(false)
  }

  const exampleQueries = [
    "What is nishkama karma?",
    "Explain the concept of dharma in the Bhagavad Gita",
    "What does Krishna say about death?",
    "How does jnana yoga differ from bhakti yoga?",
    "What are the three gunas?",
  ]

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <span className="logo-sanskrit">||</span>
            <h1>SRAG</h1>
            <span className="logo-sanskrit">||</span>
          </div>
          <p className="subtitle">Sanskrit RAG with Graph-Enhanced Linguistic Re-ranking</p>
          <p className="description">Bhagavad Gita Knowledge Engine &mdash; Vector + Neo4j Graph + BM25 Hybrid Retrieval</p>
        </div>
      </header>

      <main className="chat-container">
        {messages.length === 0 && (
          <div className="welcome">
            <div className="welcome-icon">||</div>
            <h2>Ask the Bhagavad Gita</h2>
            <p>Query the Bhagavad Gita with all 18 chapters, 700 verses, and 9 traditional commentaries.</p>
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
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                    </div>
                    {msg.metadata && (
                      <div className="metadata">
                        <div className="metadata-row">
                          <span className="meta-label">Concepts:</span>
                          {msg.metadata.concepts.map((c, j) => (
                            <span key={j} className="concept-tag">{c}</span>
                          ))}
                        </div>
                        <div className="metadata-row">
                          <span className="meta-label">Verses Cited:</span>
                          {msg.metadata.verses.map((v, j) => (
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
              <div className="message-avatar">||</div>
              <div className="message-body">
                <div className="thinking">
                  <span></span><span></span><span></span>
                </div>
                <p className="thinking-text">Searching 3,507 chunks across vector, graph, and BM25 indices...</p>
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
          <button type="submit" disabled={loading || !query.trim()}>
            {loading ? '...' : 'Ask'}
          </button>
        </div>
      </form>
    </div>
  )
}

export default App
