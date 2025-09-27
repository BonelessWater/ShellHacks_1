import React from 'react';
import { Activity } from 'lucide-react';

const ProcessingStatus = () => {
  return (
    <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
      <div className="flex items-center">
        <Activity className="h-5 w-5 text-blue-600 animate-spin mr-3" />
        <span className="text-blue-800">Processing invoices through specialized agents...</span>
      </div>
      <div className="mt-3 space-y-2">
        <div className="text-sm text-blue-700">• Data extraction and validation</div>
        <div className="text-sm text-blue-700">• Fraud pattern detection</div>
        <div className="text-sm text-blue-700">• Compliance checking</div>
        <div className="text-sm text-blue-700">• Duplicate detection</div>
      </div>
    </div>
  );
};

export default ProcessingStatus;