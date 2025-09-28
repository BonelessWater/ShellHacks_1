// frontend/src/components/upload/InvoiceUploader.jsx
import React, { useState } from 'react';
import axios from 'axios';
import InvoiceResponse from './InvoiceResponse';
import { mapBackendToUI } from '../../utils/mapAnalysisResult';

const API_BASE = process.env.REACT_APP_API_URL || '';

const InvoiceUploader = () => {
  const [analysisResult, setAnalysisResult] = useState(null);
  const [invoiceData, setInvoiceData] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const handleUpload = async (file) => {
    try {
      setBusy(true);
      setError(null);

      const form = new FormData();
      form.append('file', file);

      const res = await axios.post(`${API_BASE}/api/upload`, form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      console.log('UPLOAD RESPONSE:', res.data);

      setInvoiceData(res.data?.data || { filename: file.name });

      const ui = mapBackendToUI(res.data);
      console.log('MAPPED UI RESULT:', ui);

      if (!ui) {
        setError('Could not map backend response to UI format.');
      } else {
        setAnalysisResult(ui);
      }
    } catch (e) {
      console.error('UPLOAD ERROR:', e?.response?.data || e.message);
      setError(e?.response?.data?.detail || e.message || 'Upload failed.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <input
        type="file"
        onChange={e => e.target.files?.[0] && handleUpload(e.target.files[0])}
      />

      {busy && <div className="mt-2 text-sm text-blue-600">Analyzing…</div>}
      {error && <div className="mt-2 text-sm text-red-600">{error}</div>}

      <div className="mt-6">
        <InvoiceResponse invoiceData={invoiceData} analysisResult={analysisResult} />
      </div>
    </div>
  );
};

export default InvoiceUploader;
