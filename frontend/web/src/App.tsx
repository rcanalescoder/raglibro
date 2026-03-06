import { useMemo, useState } from "react";

type SourceChunk = {
    page?: number | null;
    snippet: string;
};

type AskResponse = {
    answer: string;
    sources: SourceChunk[];
};

export default function App() {
    const [question, setQuestion] = useState("");
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<AskResponse | null>(null);
    const [error, setError] = useState<string | null>(null);

    // Backend FastAPI en dev:
    const apiBase = useMemo(() => "http://127.0.0.1:8000", []);

    async function ask() {
        const q = question.trim();
        if (!q) return;

        setLoading(true);
        setError(null);
        setResult(null);

        try {
            const res = await fetch(`${apiBase}/ask`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question: q }),
            });

            if (!res.ok) {
                const text = await res.text();
                throw new Error(`HTTP ${res.status}: ${text}`);
            }

            const data = (await res.json()) as AskResponse;
            setResult(data);
        } catch (e: any) {
            setError(e?.message ?? "Error desconocido");
        } finally {
            setLoading(false);
        }
    }

    return (
        <div style={{ maxWidth: 860, margin: "40px auto", fontFamily: "system-ui, Arial" }}>
            <h1>RAG sobre libro.pdf (Ollama + FastAPI)</h1>

            <p style={{ color: "#444" }}>
                Este asistente responde usando únicamente el contenido indexado desde <b>libro.pdf</b>.
                Si la pregunta no está cubierta por el libro, debería rechazarla.
            </p>

            <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Escribe una pregunta o pide un resumen, por ejemplo: 'Resume el capítulo 2' o '¿Qué dice sobre X?'"
                rows={5}
                style={{ width: "100%", padding: 12, fontSize: 16 }}
            />

            <div style={{ marginTop: 12, display: "flex", gap: 12 }}>
                <button
                    onClick={ask}
                    disabled={loading}
                    style={{ padding: "10px 16px", fontSize: 16, cursor: "pointer" }}
                >
                    {loading ? "Consultando..." : "Preguntar"}
                </button>

                <button
                    onClick={() => {
                        setQuestion("");
                        setResult(null);
                        setError(null);
                    }}
                    disabled={loading}
                    style={{ padding: "10px 16px", fontSize: 16, cursor: "pointer" }}
                >
                    Limpiar
                </button>
            </div>

            {error && (
                <div style={{ marginTop: 18, padding: 12, background: "#ffe9e9", border: "1px solid #ffb3b3" }}>
                    <b>Error:</b> {error}
                </div>
            )}

            {result && (
                <div style={{ marginTop: 18, padding: 16, background: "#f6f6f6", border: "1px solid #ddd" }}>
                    <h2>Respuesta</h2>
                    <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.5 }}>{result.answer}</div>

                    <h3 style={{ marginTop: 18 }}>Fuentes (chunks recuperados)</h3>
                    {result.sources.length === 0 ? (
                        <p>No se devolvieron fuentes.</p>
                    ) : (
                        <ul>
                            {result.sources.map((s, i) => (
                                <li key={i} style={{ marginBottom: 10 }}>
                                    <b>{s.page !== null && s.page !== undefined ? `pág. ${s.page}` : "pág. ?"}</b>:{" "}
                                    <span>{s.snippet}</span>
                                </li>
                            ))}
                        </ul>
                    )}
                </div>
            )}
        </div>
    );
}