import React from 'react';

const AgentConfig = () => {
  return (
    <div className="bg-white shadow rounded-lg p-6">
      <h3 className="text-lg font-medium text-gray-900 mb-4">Agent Configuration</h3>
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700">Fraud Detection Threshold</label>
          <input
            type="range"
            min="0"
            max="100"
            defaultValue="75"
            className="mt-1 w-full"
          />
          <span className="text-sm text-gray-500">Current: 75% confidence required</span>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700">Auto-approval Limit</label>
          <input
            type="number"
            placeholder="$5,000"
            className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
          />
        </div>
        <div className="flex items-center">
          <input
            id="duplicate-check"
            type="checkbox"
            defaultChecked
            className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
          />
          <label htmlFor="duplicate-check" className="ml-2 block text-sm text-gray-900">
            Enable duplicate detection
          </label>
        </div>
      </div>
    </div>
  );
};

export default AgentConfig;