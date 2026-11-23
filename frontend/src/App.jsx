import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import PatientCaseSelector from './components/PatientCaseSelector';
import MriViewer from './components/MriViewer';
import DiagnosticResults from './components/DiagnosticResults';
import RagConsultation from './components/RagConsultation';
import AnalyticsModal from './components/AnalyticsModal';

export default function App() {
  const [systemStatus, setSystemStatus] = useState(null);
  const [cases, setCases] = useState([]);
  const [selectedCaseId, setSelectedCaseId] = useState(null);
  const [patientInfo, setPatientInfo] = useState({
    age: 58,
    gender: 'Male',
    symptoms: 'New-onset focal motor seizures, worsening headaches, left-sided hemiparesis',
    kps: 75,
    idh1: 'Wildtype',
    mgmt: 'Unmethylated',
    codeletion_1p19q: 'Non-codeleted'
  });

  const [uploadedFile, setUploadedFile] = useState(null);
  const [originalImage, setOriginalImage] = useState(null);
  const [camOverlay, setCamOverlay] = useState(null);
  const [gnnOverlay, setGnnOverlay] = useState(null);

  const [results, setResults] = useState(null);
  const [reportData, setReportData] = useState(null);
  const [selectedModel, setSelectedModel] = useState('llama3.2:latest');

  const [loadingAnalysis, setLoadingAnalysis] = useState(false);
  const [loadingReport, setLoadingReport] = useState(false);
  const [chatLoading, setChatLoading] = useState(false);
  const [analyticsOpen, setAnalyticsOpen] = useState(false);
  const [metricsData, setMetricsData] = useState(null);

  // 1. Fetch initial status, cases, and metrics
  useEffect(() => {
    fetch('/api/status')
      .then((res) => res.json())
      .then((data) => {
        setSystemStatus(data);
        if (data.active_model) setSelectedModel(data.active_model);
      })
      .catch((err) => console.error('Failed to fetch status:', err));

    fetch('/api/cases')
      .then((res) => res.json())
      .then((data) => {
        if (data.cases && data.cases.length > 0) {
          setCases(data.cases);
          // Set first case as default
          const defaultCase = data.cases[0];
          setSelectedCaseId(defaultCase.id);
          setPatientInfo(defaultCase.patient_info);
          if (defaultCase.image_base64) {
            setOriginalImage(`data:image/jpeg;base64,${defaultCase.image_base64}`);
          }
        }
      })
      .catch((err) => console.error('Failed to fetch cases:', err));

    fetch('/api/metrics')
      .then((res) => res.json())
      .then((data) => setMetricsData(data))
      .catch((err) => console.error('Failed to fetch metrics:', err));
  }, []);

  // Handle preset case selection
  const handleSelectCase = (c) => {
    setSelectedCaseId(c.id);
    setPatientInfo(c.patient_info);
    setUploadedFile(null);
    if (c.image_base64) {
      setOriginalImage(`data:image/jpeg;base64,${c.image_base64}`);
    }
    // Clear old predictions for fresh run
    setCamOverlay(null);
    setGnnOverlay(null);
    setResults(null);
    setReportData(null);
  };

  // Handle custom image upload
  const handleCustomImageUpload = (file) => {
    setUploadedFile(file);
    setSelectedCaseId(null);
    const reader = new FileReader();
    reader.onload = (e) => {
      setOriginalImage(e.target.result);
      setCamOverlay(null);
      setGnnOverlay(null);
      setResults(null);
      setReportData(null);
    };
    reader.readAsDataURL(file);
  };

  // Run Multimodal Inference
  const handleRunAnalysis = async () => {
    setLoadingAnalysis(true);
    setReportData(null);

    try {
      const formData = new FormData();
      if (uploadedFile) {
        formData.append('file', uploadedFile);
      } else if (selectedCaseId) {
        formData.append('case_id', selectedCaseId);
      } else if (originalImage) {
        formData.append('image_base64', originalImage);
      }

      formData.append('patient_info', JSON.stringify(patientInfo));

      const res = await fetch('/api/predict', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Prediction failed');
      }

      const data = await res.json();
      setResults(data);

      if (data.cam_overlay_base64) {
        setCamOverlay(`data:image/jpeg;base64,${data.cam_overlay_base64}`);
      }
      if (data.gnn_overlay_base64) {
        setGnnOverlay(`data:image/jpeg;base64,${data.gnn_overlay_base64}`);
      }
      if (data.original_image_base64 && !originalImage) {
        setOriginalImage(`data:image/jpeg;base64,${data.original_image_base64}`);
      }
    } catch (err) {
      alert(`Inference Error: ${err.message}`);
    } finally {
      setLoadingAnalysis(false);
    }
  };

  // Generate RAG Clinical Report
  const handleGenerateReport = async () => {
    if (!results) return;
    setLoadingReport(true);

    try {
      const res = await fetch('/api/rag/report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          patient_info: patientInfo,
          multimodal_results: results,
          model_name: selectedModel
        })
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Failed to generate report');
      }

      const data = await res.json();
      setReportData(data);
    } catch (err) {
      alert(`RAG Report Error: ${err.message}`);
    } finally {
      setLoadingReport(false);
    }
  };

  // Interactive Clinical Chat
  const handleSendChat = async (caseContext, question) => {
    setChatLoading(true);
    try {
      const res = await fetch('/api/rag/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          case_context: caseContext,
          question: question,
          model_name: selectedModel
        })
      });

      if (!res.ok) throw new Error('Chat failed');
      const data = await res.json();
      return data.answer;
    } finally {
      setChatLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Top Navigation */}
      <Navbar
        systemStatus={systemStatus}
        onOpenAnalytics={() => setAnalyticsOpen(true)}
      />

      {/* Main Workspace */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 space-y-6">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Left Column: Patient Case & Biomarker Selector (5 Cols) */}
          <div className="lg:col-span-5 space-y-6">
            <PatientCaseSelector
              cases={cases}
              selectedCaseId={selectedCaseId}
              onSelectCase={handleSelectCase}
              patientInfo={patientInfo}
              onUpdatePatientInfo={setPatientInfo}
              onCustomImageUpload={handleCustomImageUpload}
              onRunAnalysis={handleRunAnalysis}
              loading={loadingAnalysis}
            />
          </div>

          {/* Right Column: Visualizer, Diagnostics & RAG Suite (7 Cols) */}
          <div className="lg:col-span-7 space-y-6">
            {/* Interactive MRI Visualizer */}
            <MriViewer
              originalImage={originalImage}
              camOverlay={camOverlay}
              gnnOverlay={gnnOverlay}
              results={results}
            />

            {/* Diagnostic Classification & Multi-Task Results */}
            <DiagnosticResults results={results} />

            {/* RAG Clinical Consultation & Evidence Drawer */}
            <RagConsultation
              results={results}
              patientInfo={patientInfo}
              availableModels={systemStatus?.models || [selectedModel]}
              selectedModel={selectedModel}
              onSelectModel={setSelectedModel}
              onGenerateReport={handleGenerateReport}
              reportData={reportData}
              loadingReport={loadingReport}
              onSendChat={handleSendChat}
              chatLoading={chatLoading}
            />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950/80 py-4 text-center text-xs text-slate-500">
        NeuroMultimodal AI • Deep CNN (ResNet18) + Graph Neural Network (GAT) + Clinical RAG with Ollama ({selectedModel})
      </footer>

      {/* Analytics Modal */}
      <AnalyticsModal
        isOpen={analyticsOpen}
        onClose={() => setAnalyticsOpen(false)}
        metricsData={metricsData}
      />
    </div>
  );
}
