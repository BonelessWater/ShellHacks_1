import React from 'react';

const SystemMetrics = ({ systemStatus }) => {
  return (
    <div className="bg-white shadow rounded-lg p-6">
      <h3 className="text-lg font-medium text-gray-900 mb-4">System Performance</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h4 className="text-sm font-medium text-gray-700 mb-3">Agent Status</h4>
          <div className="space-y-3">
            {Object.entries(systemStatus.agents).map(([name, agent]) => (
              <div key={name} className="flex items-center justify-between">
                <span className="text-sm text-gray-600 capitalize">{name.replace('_', ' ')}</span>
                <div className="flex items-center">
                  <div className="w-24 bg-gray-200 rounded-full h-2 mr-3">
                    <div
                      className="bg-blue-600 h-2 rounded-full"
                      style={{ width: `${agent.load}%` }}
                    ></div>
                  </div>
                  <span className="text-sm text-gray-900">{agent.load}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>
        <div>
          <h4 className="text-sm font-medium text-gray-700 mb-3">Processing Stats</h4>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-sm text-gray-600">Avg Processing Time</span>
              <span className="text-sm text-gray-900">2.3s</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-600">Success Rate</span>
              <span className="text-sm text-gray-900">94.2%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-600">Fraud Detection Rate</span>
              <span className="text-sm text-gray-900">99.7%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-600">System Uptime</span>
              <span className="text-sm text-gray-900">99.9%</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SystemMetrics;