import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { Sparkles, FileText, Download, Send, Bot, BookOpen, ChevronDown, ChevronUp, CheckCircle, MessageSquare } from 'lucide-react';

export default function RagConsultation({
  results,
  patientInfo,
  availableModels,
  selectedModel,
  onSelectModel,
  onGenerateReport,
  reportData,
  loadingReport,
  onSendChat,
  chatLoading
}) {
  const [showGuidelines, setShowGuidelines] = useState(false);
  const [chatMessages, setChatMessages] = useState([]);
  const [userQuery, setUserQuery] = useState('');

  const handleSendQuestion = async (e) => {
    e.preventDefault();
    if (!userQuery.trim() || chatLoading) return;

    const q = userQuery.trim();
    setUserQuery('');

    // Append user question
    const updatedHistory = [...chatMessages, { sender: 'user', text: q }];
    setChatMessages(updatedHistory);

    try {
      const caseCtx = `Patient: ${JSON.stringify(patientInfo)}\nDiagnosis: ${results?.predicted_class} (${results?.predicted_grade})\nConfidence: ${(results?.confidence * 100).toFixed(1)}%`;
      const ans = await onSendChat(caseCtx, q);
      setChatMessages([...updatedHistory, { sender: 'bot', text: ans }]);
    } catch (err) {
      setChatMessages([...updatedHistory, { sender: 'bot', text: `Error querying Ollama: ${err.message}` }]);
    }
  };

  const handleDownload = () => {
    if (!reportData?.report) return;
    const blob = new Blob([reportData.report], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `NeuroOncology_Consultation_${results?.predicted_class || 'Report'}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-6">
      {/* Header & Model Selector */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center space-x-2">
            <Sparkles className="w-5 h-5 text-amber-400" />
            <span>RAG Clinical Decision Support</span>
          </h2>
          <p className="text-xs text-slate-400">Grounded in WHO CNS5 guidelines using local Ollama models.</p>
        </div>

        <div className="flex items-center space-x-2">
          <label className="text-xs text-slate-400">Ollama LLM:</label>
          <select
            value={selectedModel}
            onChange={(e) => onSelectModel(e.target.value)}
            className="bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-200 font-medium focus:outline-none focus:border-indigo-500"
          >
            {availableModels.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Generate Button Banner */}
      {!reportData?.report && (
        <div className="p-6 rounded-xl bg-slate-950 border border-slate-800/80 text-center space-y-3">
          <BookOpen className="w-8 h-8 mx-auto text-amber-400 opacity-80" />
          <h4 className="text-sm font-semibold text-white">Synthesize Neuro-Oncology Consultation Report</h4>
          <p className="text-xs text-slate-400 max-w-md mx-auto">
            Retrieves authoritative clinical protocols from ChromaDB and synthesizes an evidence-based neuropathology consultation using <code className="text-amber-400">{selectedModel}</code>.
          </p>
          <button
            onClick={onGenerateReport}
            disabled={!results || loadingReport}
            className="py-2.5 px-6 rounded-xl bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-500 hover:to-orange-500 text-white font-semibold text-xs shadow-lg shadow-amber-600/20 transition cursor-pointer disabled:opacity-50 inline-flex items-center space-x-2"
          >
            {loadingReport ? (
              <>
                <Sparkles className="w-4 h-4 animate-spin text-white" />
                <span>Retrieving Guidelines & Synthesizing...</span>
              </>
            ) : (
              <>
                <FileText className="w-4 h-4 text-white" />
                <span>Generate Clinical Report via Ollama</span>
              </>
            )}
          </button>
        </div>
      )}

      {/* Report Display */}
      {reportData?.report && (
        <div className="space-y-4">
          {/* Action Header */}
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center space-x-1.5">
              <CheckCircle className="w-4 h-4 text-emerald-400" />
              <span>WHO CNS5 Consultation Report</span>
            </span>
            <div className="flex items-center space-x-2">
              <button
                onClick={handleDownload}
                className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs transition cursor-pointer"
              >
                <Download className="w-3.5 h-3.5" />
                <span>Export Report (.md)</span>
              </button>
              <button
                onClick={onGenerateReport}
                disabled={loadingReport}
                className="px-3 py-1.5 rounded-lg bg-amber-600/20 hover:bg-amber-600/30 border border-amber-500/40 text-amber-300 text-xs font-medium transition cursor-pointer"
              >
                {loadingReport ? 'Regenerating...' : 'Regenerate'}
              </button>
            </div>
          </div>

          {/* Guidelines Drawer */}
          {reportData.retrieved_guidelines && reportData.retrieved_guidelines.length > 0 && (
            <div className="border border-slate-800 rounded-xl overflow-hidden bg-slate-950">
              <button
                onClick={() => setShowGuidelines(!showGuidelines)}
                className="w-full px-4 py-2.5 flex items-center justify-between text-xs text-slate-300 hover:bg-slate-900/50 transition cursor-pointer"
              >
                <div className="flex items-center space-x-2">
                  <BookOpen className="w-3.5 h-3.5 text-cyan-400" />
                  <span className="font-semibold">
                    Retrieved WHO CNS5 Evidence Passages ({reportData.retrieved_guidelines.length})
                  </span>
                </div>
                {showGuidelines ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              </button>

              {showGuidelines && (
                <div className="p-3 border-t border-slate-800 space-y-2 text-xs">
                  {reportData.retrieved_guidelines.map((doc, idx) => (
                    <div key={idx} className="p-2.5 rounded-lg bg-slate-900 border border-slate-800/80">
                      <div className="flex justify-between items-center mb-1">
                        <span className="font-bold text-indigo-400">
                          [{idx + 1}] {doc.title} &gt; {doc.header}
                        </span>
                        <span className="text-[10px] px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 font-mono">
                          Relevance: {doc.relevance_score}
                        </span>
                      </div>
                      <p className="text-slate-400 text-[11px] font-mono whitespace-pre-wrap line-clamp-3">
                        {doc.content}
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Formatted Markdown Container */}
          <div className="p-6 rounded-xl bg-slate-950 border border-slate-800 prose prose-invert prose-sm max-w-none prose-headings:text-indigo-300 prose-a:text-cyan-400 prose-strong:text-slate-200">
            <ReactMarkdown>{reportData.report}</ReactMarkdown>
          </div>
        </div>
      )}

      {/* Interactive Clinical Q&A Chat */}
      {results && (
        <div className="border-t border-slate-800/80 pt-4 space-y-3">
          <div className="flex items-center space-x-2">
            <MessageSquare className="w-4 h-4 text-indigo-400" />
            <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider">
              Interactive Clinical Q&amp;A (RAG Assistant)
            </h4>
          </div>

          {/* Chat Message Stream */}
          {chatMessages.length > 0 && (
            <div className="max-h-60 overflow-y-auto space-y-2 p-3 rounded-xl bg-slate-950 border border-slate-800 text-xs">
              {chatMessages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[85%] p-2.5 rounded-xl ${
                      msg.sender === 'user'
                        ? 'bg-indigo-600 text-white rounded-br-none'
                        : 'bg-slate-900 border border-slate-800 text-slate-200 rounded-bl-none'
                    }`}
                  >
                    <p className="whitespace-pre-wrap">{msg.text}</p>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Question Input Form */}
          <form onSubmit={handleSendQuestion} className="flex gap-2">
            <input
              type="text"
              value={userQuery}
              onChange={(e) => setUserQuery(e.target.value)}
              placeholder="Ask a clinical question (e.g. Stupp TMZ dosing, surgery margins, molecular impact)..."
              className="flex-1 bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
            />
            <button
              type="submit"
              disabled={chatLoading || !userQuery.trim()}
              className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-xs font-semibold flex items-center space-x-1.5 transition cursor-pointer disabled:opacity-50"
            >
              {chatLoading ? (
                <Bot className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <Send className="w-3.5 h-3.5" />
              )}
              <span>Ask</span>
            </button>
          </form>
        </div>
      )}
    </div>
  );
}
