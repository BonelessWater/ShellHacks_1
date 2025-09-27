import React from 'react';
import { AlertTriangle } from 'lucide-react';

const FraudAlerts = () => {
  const alerts = [
    {
      type: 'high',
      title: 'High Risk Invoice:',
      description: 'INV-2024-003 - Amount just under $10,000 limit',
      details: 'Detected: Suspicious vendor patterns, round amount'
    },
    {
      type: 'medium',
      title: 'Medium Risk:',
      description: 'INV-2024-002 - Vendor verification needed',
      details: 'Detected: New vendor, high amount'
    }
  ];

  return (
    <div className="bg-white shadow rounded-lg p-6">
      <h3 className="text-lg font-medium text-gray-900 mb-4">Recent Fraud Alerts</h3>
      <div className="space-y-3">
        {alerts.map((alert, index) => (
          <div key={index} className={`border-l-4 p-4 ${
            alert.type === 'high' 
              ? 'border-red-400 bg-red-50' 
              : 'border-yellow-400 bg-yellow-50'
          }`}>
            <div className="flex">
              <AlertTriangle className={`h-5 w-5 ${
                alert.type === 'high' ? 'text-red-400' : 'text-yellow-400'
              }`} />
              <div className="ml-3">
                <p className={`text-sm ${
                  alert.type === 'high' ? 'text-red-800' : 'text-yellow-800'
                }`}>
                  <strong>{alert.title}</strong> {alert.description}
                </p>
                <p className={`text-xs mt-1 ${
                  alert.type === 'high' ? 'text-red-600' : 'text-yellow-600'
                }`}>
                  {alert.details}
                </p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default FraudAlerts;