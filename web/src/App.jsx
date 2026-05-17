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
    {/* feather */}
    <path d="M2 22c0-15 16-22 16-22s-2 10 0 22z" />
    <path d="M18 2c-2 0-3.5 1.5-5 4-1.5 2.5-1.5 6-1.5 6s3.5-1.5 6-4c1.8-1.8 3.5-4 3.5-7 0-1-1-1-3-1z" />
    <path d="M6.5 14c2.5 2.5 5 3 7.5 3" strokeOpacity="0.5" />
  </svg>
);

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
  const [streamingMessages, setStreamingMessages] = useState({}) // id -> typed text
  const messagesEndRef = useRef(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamingMessages])

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

      // Build the message with full text hidden for typewriter effect
      const newMsg = {
        role: 'assistant',
        content: data.answer,
        metadata: {
          concepts: data.concepts,
          verses: data.verses_cited,
          confidence: data.pipeline_confidence,
          topVerses: data.top_verses,
        },
      }

      // Add message to array
      setMessages(prev => [...prev, newMsg])

      // Trigger typewriter stream per message (keyed by index once known)
      // We'll handle streaming in useEffect after message is added
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Error: ${err.message}. Make sure the API server is running on port 8000.`,
      }])
    }
    setLoading(false)
  }

  // Stream typing effect for the last assistant message when content changes
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
            <div className="welcome-icon"><QuillIcon /></div>
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
                      {/* Show typing text only for latest assistant message, else full text */}
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

          {/* Quill Loading Animation */}
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
  )
}

export default App
