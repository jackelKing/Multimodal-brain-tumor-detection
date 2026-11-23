import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import {
  Sparkles,
  FileText,
  Download,
  Send,
  Bot,
  BookOpen,
  ChevronDown,
  ChevronUp,
  CheckCircle,
  MessageSquare,
  Search,
  ExternalLink,
  HelpCircle
} from 'lucide-react';

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
  const [activeTab, setActiveTab] = useState('report'); // 'report', 'guidelines', 'chat'
  const [showGuidelines, setShowGuidelines] = useState(false);
  const [chatMessages, setChatMessages] = useState([]);
  const [userQuery, setUserQuery] = useState('');

  // Semantic Guidelines Search state
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [searchingGuidelines, setSearchingGuidelines] = useState(false);

  const handleSendQuestion = async (qToSend) => {
    const q = (typeof qToSend === 'string' ? qToSend : userQuery).trim();
    if (!q || chatLoading) return;

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

  const handleSearchGuidelines = async (e) => {
    e?.preventDefault();
    if (!searchQuery.trim() || searchingGuidelines) return;

    setSearchingGuidelines(true);
    try {
      const res = await fetch(`/api/guidelines/search?q=${encodeURIComponent(searchQuery.trim())}`);
      if (!res.ok) throw new Error('Search failed');
      const data = await res.json();
      setSearchResults(data.results || []);
    } catch (err) {
      alert(`Guideline Search Error: ${err.message}`);
    } finally {
      setSearchingGuidelines(false);
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

  const QUICK_QUESTIONS = [
    'What is the Stupp Protocol Temozolomide dosing & schedule?',
    'How does IDH1 wildtype vs mutation impact survival?',
    'What are surgical recommendations near eloquent speech cortex?',
    'What is the dexamethasone tapering schedule for vasogenic edema?'
  ];

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-5">
      {/* Header & Model Selector */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center space-x-2">
            <Sparkles className="w-5 h-5 text-amber-400" />
            <span>RAG Clinical Decision Support Suite</span>
          </h2>
          <p className="text-xs text-slate-400">
            WHO CNS5 guidelines powered by local Ollama ({selectedModel}) &amp; ChromaDB.
          </p>
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

      {/* Feature Navigation Tabs */}
      <div className="flex items-center space-x-1 border-b border-slate-800 pb-2 text-xs font-semibold">
        <button
          onClick={() => setActiveTab('report')}
          className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg transition ${
            activeTab === 'report'
              ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <FileText className="w-3.5 h-3.5" />
          <span>Consultation Report</span>
        </button>

        <button
          onClick={() => setActiveTab('guidelines')}
          className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg transition ${
            activeTab === 'guidelines'
              ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <Search className="w-3.5 h-3.5" />
          <span>Semantic Guidelines Search</span>
        </button>

        <button
          onClick={() => setActiveTab('chat')}
          className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg transition ${
            activeTab === 'chat'
              ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/40'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <MessageSquare className="w-3.5 h-3.5" />
          <span>Tumor Board Q&amp;A Chat</span>
        </button>
      </div>

      {/* TAB 1: CONSULTATION REPORT */}
      {activeTab === 'report' && (
        <div className="space-y-4">
          {!reportData?.report ? (
            <div className="p-6 rounded-xl bg-slate-950 border border-slate-800/80 text-center space-y-3">
              <BookOpen className="w-8 h-8 mx-auto text-amber-400 opacity-80" />
              <h4 className="text-sm font-semibold text-white">Synthesize Neuro-Oncology Consultation Report</h4>
              <p className="text-xs text-slate-400 max-w-md mx-auto">
                Retrieves authoritative WHO CNS5 protocols from ChromaDB and synthesizes an evidence-based clinical neuropathology consultation using <code className="text-amber-400">{selectedModel}</code>.
              </p>
              <button
                onClick={onGenerateReport}
                disabled={!results || loadingReport}
                className="py-2.5 px-6 rounded-xl bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-500 hover:to-orange-500 text-white font-semibold text-xs shadow-lg shadow-amber-600/20 transition cursor-pointer disabled:opacity-50 inline-flex items-center space-x-2"
              >
                {loadingReport ? (
                  <>
                    <Sparkles className="w-4 h-4 animate-spin text-white" />
                    <span>Retrieving Guidelines &amp; Synthesizing...</span>
                  </>
                ) : (
                  <>
                    <FileText className="w-4 h-4 text-white" />
                    <span>Generate Consultation Report via Ollama</span>
                  </>
                )}
              </button>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center space-x-1.5">
                  <CheckCircle className="w-4 h-4 text-emerald-400" />
                  <span>WHO CNS5 Multi-Disciplinary Report</span>
                </span>
                <div className="flex items-center space-x-2">
                  <button
                    onClick={handleDownload}
                    className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs transition cursor-pointer"
                  >
                    <Download className="w-3.5 h-3.5" />
                    <span>Export (.md)</span>
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
                        Retrieved Evidence Grounding ({reportData.retrieved_guidelines.length} Guidelines)
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
                              Similarity: {doc.relevance_score}
                            </span>
                          </div>
                          <p className="text-slate-400 text-[11px] font-mono whitespace-pre-wrap">
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
        </div>
      )}

      {/* TAB 2: SEMANTIC GUIDELINES SEARCH */}
      {activeTab === 'guidelines' && (
        <div className="space-y-4">
          <form onSubmit={handleSearchGuidelines} className="flex gap-2">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search WHO CNS5 database (e.g. MGMT methylation, Stupp RT doses, Dexamethasone)..."
              className="flex-1 bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2 text-xs text-white focus:outline-none focus:border-cyan-500"
            />
            <button
              type="submit"
              disabled={searchingGuidelines || !searchQuery.trim()}
              className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-xl text-xs font-semibold flex items-center space-x-1.5 transition cursor-pointer disabled:opacity-50"
            >
              <Search className="w-3.5 h-3.5" />
              <span>{searchingGuidelines ? 'Searching...' : 'Search'}</span>
            </button>
          </form>

          {/* Quick preset chips */}
          <div className="flex flex-wrap gap-1.5 text-[11px]">
            <span className="text-slate-500 py-0.5">Suggested:</span>
            {['IDH1 Mutation Impact', 'Stupp Protocol Radiation', 'Pituitary Macroadenoma Apoplexy', 'Meningioma Simpson Grade'].map((term) => (
              <button
                key={term}
                onClick={() => {
                  setSearchQuery(term);
                  fetch(`/api/guidelines/search?q=${encodeURIComponent(term)}`)
                    .then(r => r.json())
                    .then(d => setSearchResults(d.results || []));
                }}
                className="px-2 py-0.5 rounded-full bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700/60"
              >
                {term}
              </button>
            ))}
          </div>

          {/* Search Results Display */}
          <div className="space-y-2.5 max-h-80 overflow-y-auto">
            {searchResults.length > 0 ? (
              searchResults.map((res, i) => (
                <div key={i} className="p-3 rounded-xl bg-slate-950 border border-slate-800 space-y-1 text-xs">
                  <div className="flex justify-between items-center">
                    <span className="font-bold text-cyan-400">
                      {res.title} — {res.header}
                    </span>
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                      Score: {res.relevance_score}
                    </span>
                  </div>
                  <p className="text-slate-300 text-[11px] leading-relaxed whitespace-pre-wrap font-mono">
                    {res.content}
                  </p>
                </div>
              ))
            ) : (
              <div className="text-center py-8 text-slate-500 text-xs">
                Enter a query above to query the vector database of WHO CNS5 neuropathology guidelines.
              </div>
            )}
          </div>
        </div>
      )}

      {/* TAB 3: TUMOR BOARD Q&A CHAT */}
      {activeTab === 'chat' && (
        <div className="space-y-3">
          {/* Quick clinical questions */}
          <div className="space-y-1.5">
            <span className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">
              Clinical Inquiries:
            </span>
            <div className="flex flex-wrap gap-1.5">
              {QUICK_QUESTIONS.map((q, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSendQuestion(q)}
                  disabled={chatLoading}
                  className="text-[11px] px-2.5 py-1 rounded-lg bg-slate-950 hover:bg-slate-800 border border-slate-800 text-slate-300 transition text-left"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>

          {/* Chat Message Stream */}
          <div className="h-64 overflow-y-auto space-y-2.5 p-3 rounded-xl bg-slate-950 border border-slate-800 text-xs">
            {chatMessages.length === 0 ? (
              <div className="h-full flex items-center justify-center text-slate-500 text-center">
                Ask any question regarding this patient case, surgical margins, radiation volumes, or chemotherapy regimens.
              </div>
            ) : (
              chatMessages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[85%] p-3 rounded-xl ${
                      msg.sender === 'user'
                        ? 'bg-indigo-600 text-white rounded-br-none'
                        : 'bg-slate-900 border border-slate-800 text-slate-200 rounded-bl-none'
                    }`}
                  >
                    <p className="whitespace-pre-wrap leading-relaxed">{msg.text}</p>
                  </div>
                </div>
              ))
            )}
          </div>

          {/* Input Bar */}
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleSendQuestion();
            }}
            className="flex gap-2"
          >
            <input
              type="text"
              value={userQuery}
              onChange={(e) => setUserQuery(e.target.value)}
              placeholder="Ask the clinical AI advisor..."
              className="flex-1 bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
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
