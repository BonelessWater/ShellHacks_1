import React, { useState, useEffect } from 'react';
import { bigQueryService } from '../services/bigQueryService';

const BigQueryTest = () => {
  const [status, setStatus] = useState('checking');
  const [data, setData] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    testConnection();
  }, []);

  const testConnection = async () => {
    try {
      setStatus('connecting');
      const response = await bigQueryService.healthCheck();
      if (response && response.bigquery === 'connected') {
        setStatus('connected');
      } else if (response && response.status === 'healthy') {
        setStatus('connected');
      } else {
        setStatus('error');
        setError('BigQuery connection failed');
      }
    } catch (err) {
      setStatus('error');
      setError(err.message);
    }
  };

  const fetchFraudData = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await bigQueryService.getFraudTransactions(10);
      if (response.success) setData(response.data || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h2 className="text-2xl font-bold mb-6">BigQuery Connection Test</h2>

      <div className="bg-white rounded-lg shadow p-4 mb-6">
        <h3 className="font-semibold mb-2">Connection Status</h3>
        <div className="flex items-center gap-2">
          <span className={`w-3 h-3 rounded-full ${
            status === 'connected' ? 'bg-green-500' :
            status === 'error' ? 'bg-red-500' :
            status === 'connecting' ? 'bg-yellow-500 animate-pulse' :
            'bg-gray-500'
          }`} />
          <span className="capitalize">{status}</span>
          {status === 'connected' && <span className="text-green-600">✓ BigQuery Connected</span>}
        </div>
        {error && (
          <div className="mt-2 p-2 bg-red-50 border border-red-200 rounded text-red-700 text-sm">{error}</div>
        )}
      </div>

      <div className="bg-white rounded-lg shadow p-4">
        <div className="flex justify-between items-center mb-4">
          <h3 className="font-semibold">Fraud Transactions</h3>
          <button onClick={fetchFraudData} disabled={loading || status !== 'connected'} className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:bg-gray-400">
            {loading ? 'Loading...' : 'Fetch Data'}
          </button>
        </div>

        {data.length > 0 && (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Transaction ID</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Amount</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Product</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Card Type</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {data.map((row) => (
                  <tr key={row.TransactionID}>
                    <td className="px-4 py-2 text-sm">{row.TransactionID}</td>
                    <td className="px-4 py-2 text-sm">${row.TransactionAmt}</td>
                    <td className="px-4 py-2 text-sm">{row.ProductCD}</td>
                    <td className="px-4 py-2 text-sm">{row.CardType}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export default BigQueryTest;
