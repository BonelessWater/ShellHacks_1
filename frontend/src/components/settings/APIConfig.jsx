import React, { useState } from 'react';
import { AlertCircle, CheckCircle } from 'lucide-react';

const APIConfig = () => {
  const [apiSettings, setApiSettings] = useState({
    backendUrl: 'http://localhost:8000',
    websocketUrl: 'ws://localhost:8000/ws',
    apiKey: '',
    timeout: 30
  });

  const [connectionStatus, setConnectionStatus] = useState({
    backend: 'disconnected',
    websocket: 'disconnected',
    database: 'pending'
  });

  const handleInputChange = (field, value) => {
    setApiSettings(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const testConnection = async (service) => {
    setConnectionStatus(prev => ({
      ...prev,
      [service]: 'testing'
    }));

    // Simulate connection test
    setTimeout(() => {
      const isSuccess = Math.random() > 0.3; // 70% success rate for demo
      setConnectionStatus(prev => ({
        ...prev,
        [service]: isSuccess ? 'connected' : 'failed'
      }));
    }, 2000);
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'connected':
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'failed':
        return <AlertCircle className="h-5 w-5 text-red-500" />;
      case 'testing':
        return <div className="h-5 w-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />;
      default:
        return <AlertCircle className="h-5 w-5 text-yellow-500" />;
    }
  };

  const getStatusText = (status) => {
    switch (status) {
      case 'connected': return 'Connected';
      case 'failed': return 'Failed';
      case 'testing': return 'Testing...';
      case 'pending': return 'Pending';
      default: return 'Disconnected';
    }
  };

  return (
    <div className="space-y-6">
      {/* API Configuration */}
      <div className="bg-white shadow rounded-lg p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">API Configuration</h3>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Backend URL
            </label>
            <input
              type="url"
              value={apiSettings.backendUrl}
              onChange={(e) => handleInputChange('backendUrl', e.target.value)}
              className="block w-full border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
              placeholder="http://localhost:8000"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              WebSocket URL
            </label>
            <input
              type="url"
              value={apiSettings.websocketUrl}
              onChange={(e) => handleInputChange('websocketUrl', e.target.value)}
              className="block w-full border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
              placeholder="ws://localhost:8000/ws"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              API Key (Optional)
            </label>
            <input
              type="password"
              value={apiSettings.apiKey}
              onChange={(e) => handleInputChange('apiKey', e.target.value)}
              className="block w-full border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
              placeholder="Enter API key if required"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Request Timeout (seconds)
            </label>
            <input
              type="number"
              value={apiSettings.timeout}
              onChange={(e) => handleInputChange('timeout', parseInt(e.target.value))}
              className="block w-full border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
              min="5"
              max="120"
            />
          </div>
          
          <button className="w-full bg-indigo-600 text-white px-4 py-2 rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500">
            Save Configuration
          </button>
        </div>
      </div>

      {/* System Status */}
      <div className="bg-white shadow rounded-lg p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">System Status</h3>
        <div className="space-y-3">
          {Object.entries(connectionStatus).map(([service, status]) => (
            <div key={service} className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700 capitalize">
                {service === 'backend' ? 'Backend API' : 
                 service === 'websocket' ? 'WebSocket' : 
                 'Database'}
              </span>
              <div className="flex items-center space-x-3">
                <div className="flex items-center space-x-2">
                  {getStatusIcon(status)}
                  <span className={`text-sm font-medium ${
                    status === 'connected' ? 'text-green-600' :
                    status === 'failed' ? 'text-red-600' :
                    status === 'testing' ? 'text-blue-600' :
                    'text-yellow-600'
                  }`}>
                    {getStatusText(status)}
                  </span>
                </div>
                {status !== 'testing' && (
                  <button
                    onClick={() => testConnection(service)}
                    className="text-xs bg-gray-100 hover:bg-gray-200 text-gray-700 px-2 py-1 rounded"
                  >
                    Test
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Environment Variables */}
      <div className="bg-white shadow rounded-lg p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Environment Variables</h3>
        <div className="bg-gray-50 rounded-md p-4">
          <p className="text-sm text-gray-600 mb-2">
            Set these environment variables in your <code className="bg-gray-200 px-1 rounded">.env</code> file:
          </p>
          <div className="space-y-1 font-mono text-xs">
            <div>REACT_APP_API_URL={apiSettings.backendUrl}</div>
            <div>REACT_APP_WEBSOCKET_URL={apiSettings.websocketUrl}</div>
            {apiSettings.apiKey && <div>REACT_APP_API_KEY={apiSettings.apiKey}</div>}
          </div>
        </div>
      </div>
    </div>
  );
};

export default APIConfig;