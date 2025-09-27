import React from 'react';
import { FileText } from 'lucide-react';
import StatusIndicator from './StatusIndicator';

const Header = ({ systemStatus }) => {
  const allAgentsOnline = systemStatus?.agents && 
    Object.values(systemStatus.agents).every(agent => agent.status === 'active');

  return (
    <header className="bg-white shadow">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center py-6">
          <div className="flex items-center">
            <FileText className="h-8 w-8 text-indigo-600" />
            <h1 className="ml-3 text-2xl font-bold text-gray-900">
              Invoice Verification System
            </h1>
          </div>
          <div className="flex items-center space-x-4">
            <StatusIndicator 
              isOnline={allAgentsOnline}
              text={allAgentsOnline ? "All agents online" : "Some agents offline"}
            />
            <button className="bg-indigo-600 text-white px-4 py-2 rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500">
              View Logs
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;