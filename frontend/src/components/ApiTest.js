// frontend/src/components/ApiTest.js
import React, { useState, useEffect } from 'react';

const ApiTest = () => {
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('checking');

  const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  const fetchMessage = async () => {
    try {
      setLoading(true);
      setError(null);
      setConnectionStatus('connecting');

      const response = await fetch(`${apiUrl}/api/message`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      setMessage(data.message);
      setConnectionStatus('connected');
    } catch (err) {
      setError(`Failed to connect to backend: ${err.message}`);
      setConnectionStatus('failed');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMessage();
  }, []);

  const getStatusColor = () => {
    switch (connectionStatus) {
      case 'connected': return 'text-green-600';
      case 'failed': return 'text-red-600';
      case 'connecting': return 'text-yellow-600';
      default: return 'text-gray-600';
    }
  };

  const getStatusIcon = () => {
    switch (connectionStatus) {
      case 'connected': return '✅';
      case 'failed': return '❌';
      case 'connecting': return '🔄';
      default: return '⏳';
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-6 bg-white rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold text-center mb-6 text-gray-800">
        🐳 Docker Setup Test
      </h2>
      
      <div className="space-y-4">
        {/* Connection Status */}
        <div className="p-4 border rounded-lg bg-gray-50">
          <h3 className="font-semibold text-gray-700 mb-2">Connection Status</h3>
          <div className={`flex items-center space-x-2 ${getStatusColor()}`}>
            <span className="text-lg">{getStatusIcon()}</span>
            <span className="capitalize font-medium">{connectionStatus}</span>
          </div>
          <p className="text-sm text-gray-600 mt-1">
            API URL: <code className="bg-gray-200 px-1 rounded">{apiUrl}</code>
          </p>
        </div>

        {/* Message Display */}
        <div className="p-4 border rounded-lg">
          <h3 className="font-semibold text-gray-700 mb-2">Backend Message</h3>
          
          {loading && (
            <div className="flex items-center space-x-2 text-blue-600">
              <div className="animate-spin rounded-full h-4 w-4 border-2 border-blue-600 border-t-transparent"></div>
              <span>Loading...</span>
            </div>
          )}
          
          {error && (
            <div className="p-3 bg-red-100 border border-red-300 rounded text-red-700">
              <strong>Error:</strong> {error}
            </div>
          )}
          
          {message && !loading && (
            <div className="p-3 bg-green-100 border border-green-300 rounded text-green-700">
              <strong>Success:</strong> {message}
            </div>
          )}
        </div>

        {/* Retry Button */}
        <div className="text-center">
          <button
            onClick={fetchMessage}
            disabled={loading}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? 'Testing...' : 'Test Connection Again'}
          </button>
        </div>

        {/* Instructions */}
        <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <h3 className="font-semibold text-blue-800 mb-2">🚀 Setup Instructions</h3>
          <ol className="text-sm text-blue-700 space-y-1">
            <li>1. Make sure Docker is running</li>
            <li>2. Run: <code className="bg-blue-200 px-1 rounded">docker-compose up --build</code></li>
            <li>3. Frontend: <a href="http://localhost:3000" className="underline">http://localhost:3000</a></li>
            <li>4. Backend API: <a href="http://localhost:8000" className="underline">http://localhost:8000</a></li>
            <li>5. API Docs: <a href="http://localhost:8000/docs" className="underline">http://localhost:8000/docs</a></li>
          </ol>
        </div>
      </div>
    </div>
  );
};

export default ApiTest;