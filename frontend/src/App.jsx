// frontend/src/App.jsx
import React, { useState, useEffect } from 'react';
import { apiService } from './services/api';
import './App.css';

function App() {
  const [backendStatus, setBackendStatus] = useState(null);
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [invoiceData, setInvoiceData] = useState('');
  const [results, setResults] = useState(null);

  // Check backend connection on component mount
  useEffect(() => {
    checkBackendConnection();
  }, []);

  const checkBackendConnection = async () => {
    try {
      setLoading(true);
      const healthData = await apiService.checkHealth();
      const messageData = await apiService.getMessage();
      
      setBackendStatus(healthData);
      setMessage(messageData.message);
      setError(null);
    } catch (err) {
      setError(`Backend connection failed: ${err.message}`);
      setBackendStatus({ status: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const handleInvoiceSubmit = async (e) => {
    e.preventDefault();
    if (!invoiceData.trim()) {
      setError('Please enter invoice data');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      
      // Parse invoice data (assuming JSON format)
      const parsedData = JSON.parse(invoiceData);
      const result = await apiService.uploadInvoice(parsedData);
      
      setResults(result);
    } catch (err) {
      setError(`Invoice processing failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🧾 ShellHacks Invoice System</h1>
        
        {/* Backend Status Indicator */}
        <div className={`status-indicator ${backendStatus?.status}`}>
          <h3>Backend Status: {backendStatus?.status || 'Unknown'}</h3>
          {message && <p>📡 {message}</p>}
          {error && <p className="error">❌ {error}</p>}
          <button onClick={checkBackendConnection} disabled={loading}>
            {loading ? '🔄 Checking...' : '🔄 Check Connection'}
          </button>
        </div>

        {/* Invoice Processing Form */}
        <div className="invoice-form">
          <h3>📄 Process Invoice</h3>
          <form onSubmit={handleInvoiceSubmit}>
            <textarea
              value={invoiceData}
              onChange={(e) => setInvoiceData(e.target.value)}
              placeholder='Enter invoice data as JSON:
{
  "vendor": "Example Corp",
  "amount": 1500.00,
  "date": "2024-09-28",
  "invoice_number": "INV-001"
}'
              rows={8}
              cols={60}
              disabled={loading}
            />
            <br />
            <button type="submit" disabled={loading || !invoiceData.trim()}>
              {loading ? '⏳ Processing...' : '🔍 Analyze Invoice'}
            </button>
          </form>
        </div>

        {/* Results Display */}
        {results && (
          <div className="results">
            <h3>📊 Analysis Results</h3>
            <pre>{JSON.stringify(results, null, 2)}</pre>
          </div>
        )}

        {/* Development Info */}
        <div className="dev-info">
          <h4>🛠️ Development Info</h4>
          <p>Frontend: React on port 3000</p>
          <p>Backend: FastAPI on port 8000</p>
          <p>Environment: {process.env.NODE_ENV}</p>
        </div>
      </header>
    </div>
  );
}

export default App;