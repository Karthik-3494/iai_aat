import { useState } from 'react'
import './App.css'

function App() {
  const [input, setInput] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);

  const askQuestion = async () => {
    if (!input) return;
    setLoading(true);
    try {
      const response = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: input }),
      });
      const data = await response.json();
      setAnswer(data.answer);
    } catch (error) {
      setAnswer("Error connecting to the backend.");
    }
    setLoading(false);
  };

  return (
    <section id="center">
      <h1>Campus Helper</h1>
      <div style={{ marginBottom: '20px' }}>
        <input 
          type="text" 
          value={input} 
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about exams, food, or holidays..."
          style={{ padding: '10px', width: '300px', borderRadius: '5px' }}
        />
        <button onClick={askQuestion} className="counter" style={{ marginLeft: '10px' }}>
          {loading ? "Thinking..." : "Ask Bot"}
        </button>
      </div>
      {answer && (
        <div style={{ background: 'var(--code-bg)', padding: '20px', borderRadius: '10px', textAlign: 'left', maxWidth: '600px' }}>
          <strong>Answer:</strong>
          <p>{answer}</p>
        </div>
      )}
    </section>
  );
}

export default App;
