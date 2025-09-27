import React, { useState } from 'react';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold text-indigo-600">
            🧾 ShellHacks Invoice Verification System
          </h1>
          <p className="mt-2 text-gray-600">AI-powered invoice processing with multi-agent architecture</p>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex space-x-8">
            {['dashboard', 'upload', 'analytics', 'settings'].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-3 py-4 text-sm font-medium capitalize border-b-2 transition-colors ${
                  activeTab === tab
                    ? 'border-indigo-500 text-indigo-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                {tab}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto py-6 px-4">
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-2xl font-bold mb-6 text-gray-900">
            {activeTab.charAt(0).toUpperCase() + activeTab.slice(1)} Overview
          </h2>
          
          {activeTab === 'dashboard' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                <div className="bg-blue-50 rounded-lg p-6">
                  <div className="flex items-center">
                    <div className="text-3xl mr-4">📄</div>
                    <div>
                      <p className="text-sm font-medium text-blue-600">Total Invoices</p>
                      <p className="text-2xl font-bold text-blue-900">156</p>
                    </div>
                  </div>
                </div>
                
                <div className="bg-green-50 rounded-lg p-6">
                  <div className="flex items-center">
                    <div className="text-3xl mr-4">✅</div>
                    <div>
                      <p className="text-sm font-medium text-green-600">Approved</p>
                      <p className="text-2xl font-bold text-green-900">142</p>
                    </div>
                  </div>
                </div>
                
                <div className="bg-yellow-50 rounded-lg p-6">
                  <div className="flex items-center">
                    <div className="text-3xl mr-4">⚠️</div>
                    <div>
                      <p className="text-sm font-medium text-yellow-600">Review Needed</p>
                      <p className="text-2xl font-bold text-yellow-900">8</p>
                    </div>
                  </div>
                </div>
                
                <div className="bg-red-50 rounded-lg p-6">
                  <div className="flex items-center">
                    <div className="text-3xl mr-4">❌</div>
                    <div>
                      <p className="text-sm font-medium text-red-600">Rejected</p>
                      <p className="text-2xl font-bold text-red-900">6</p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-gray-50 rounded-lg p-6">
                <h3 className="text-lg font-semibold mb-4">Recent Invoices</h3>
                <div className="space-y-3">
                  {[
                    { id: 'INV-2024-001', vendor: 'ABC Office Supplies', amount: '$1,287.00', status: 'Approved' },
                    { id: 'INV-2024-002', vendor: 'TechCorp Solutions', amount: '$8,500.00', status: 'Review Needed' },
                    { id: 'INV-2024-003', vendor: 'Global Services Inc', amount: '$2,340.00', status: 'Approved' }
                  ].map((invoice, index) => (
                    <div key={index} className="flex justify-between items-center bg-white p-4 rounded-md shadow-sm">
                      <div>
                        <p className="font-medium">{invoice.id}</p>
                        <p className="text-sm text-gray-600">{invoice.vendor}</p>
                      </div>
                      <div className="text-right">
                        <p className="font-bold">{invoice.amount}</p>
                        <span className={`text-xs px-2 py-1 rounded-full ${
                          invoice.status === 'Approved' ? 'bg-green-100 text-green-800' :
                          invoice.status === 'Review Needed' ? 'bg-yellow-100 text-yellow-800' :
                          'bg-red-100 text-red-800'
                        }`}>
                          {invoice.status}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {activeTab === 'upload' && (
            <div className="text-center">
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-12 hover:border-gray-400 transition-colors">
                <div className="text-6xl mb-4">📁</div>
                <h3 className="text-lg font-medium text-gray-900 mb-2">Upload Invoice Files</h3>
                <p className="text-gray-600 mb-6">Drag and drop your PDF, PNG, JPG, or Excel files here</p>
                <button className="bg-indigo-600 text-white px-6 py-3 rounded-md hover:bg-indigo-700 transition-colors">
                  Select Files
                </button>
              </div>
              
              <div className="mt-8 bg-blue-50 rounded-lg p-6">
                <h4 className="font-medium text-blue-900 mb-2">Supported Features:</h4>
                <div className="grid grid-cols-2 gap-4 text-sm text-blue-700">
                  <div>• Data extraction and validation</div>
                  <div>• Fraud pattern detection</div>
                  <div>• Compliance checking</div>
                  <div>• Duplicate detection</div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'analytics' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-gray-50 rounded-lg p-6">
                  <h3 className="text-lg font-semibold mb-4">System Performance</h3>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Average Processing Time</span>
                      <span className="font-bold">2.3 seconds</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Success Rate</span>
                      <span className="font-bold text-green-600">94.2%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Fraud Detection Rate</span>
                      <span className="font-bold text-blue-600">99.7%</span>
                    </div>
                  </div>
                </div>
                
                <div className="bg-gray-50 rounded-lg p-6">
                  <h3 className="text-lg font-semibold mb-4">Agent Status</h3>
                  <div className="space-y-3">
                    {[
                      { name: 'Orchestrator', load: 85 },
                      { name: 'Validation', load: 72 },
                      { name: 'Fraud Detection', load: 93 },
                      { name: 'Analytics', load: 68 }
                    ].map((agent, index) => (
                      <div key={index} className="flex items-center justify-between">
                        <span className="text-gray-600">{agent.name}</span>
                        <div className="flex items-center">
                          <div className="w-24 bg-gray-200 rounded-full h-2 mr-3">
                            <div 
                              className="bg-indigo-600 h-2 rounded-full" 
                              style={{ width: `${agent.load}%` }}
                            ></div>
                          </div>
                          <span className="text-sm font-medium">{agent.load}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="bg-red-50 border border-red-200 rounded-lg p-6">
                <h3 className="text-lg font-semibold text-red-900 mb-4">🚨 Recent Fraud Alerts</h3>
                <div className="space-y-3">
                  <div className="bg-white p-4 rounded-md border-l-4 border-red-400">
                    <p className="font-medium text-red-800">High Risk Invoice: INV-2024-003</p>
                    <p className="text-sm text-red-600">Amount just under $10,000 limit - suspicious patterns detected</p>
                  </div>
                  <div className="bg-white p-4 rounded-md border-l-4 border-yellow-400">
                    <p className="font-medium text-yellow-800">Medium Risk: INV-2024-002</p>
                    <p className="text-sm text-yellow-600">New vendor with high amount - requires verification</p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'settings' && (
            <div className="space-y-6">
              <div className="bg-gray-50 rounded-lg p-6">
                <h3 className="text-lg font-semibold mb-4">Agent Configuration</h3>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Fraud Detection Threshold
                    </label>
                    <input 
                      type="range" 
                      min="0" 
                      max="100" 
                      defaultValue="75" 
                      className="w-full"
                    />
                    <span className="text-sm text-gray-500">Current: 75% confidence required</span>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Auto-approval Limit
                    </label>
                    <input 
                      type="number" 
                      placeholder="$5,000" 
                      className="block w-full border border-gray-300 rounded-md px-3 py-2"
                    />
                  </div>
                  
                  <div className="flex items-center">
                    <input 
                      type="checkbox" 
                      id="duplicate-check" 
                      defaultChecked 
                      className="mr-2"
                    />
                    <label htmlFor="duplicate-check" className="text-sm text-gray-700">
                      Enable duplicate detection
                    </label>
                  </div>
                </div>
              </div>

              <div className="bg-gray-50 rounded-lg p-6">
                <h3 className="text-lg font-semibold mb-4">System Status</h3>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Frontend Status</span>
                    <span className="text-green-600 font-medium">✅ Running</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Backend Status</span>
                    <span className="text-red-600 font-medium">❌ Not Connected</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Database</span>
                    <span className="text-yellow-600 font-medium">⚠️ Pending</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;