import React from 'react';
import SystemMetrics from './SystemMetrics';
import FraudAlerts from './FraudAlerts';

const Analytics = ({ systemStatus, invoices }) => {
  return (
    <div className="space-y-6">
      <SystemMetrics systemStatus={systemStatus} />
      <FraudAlerts />
    </div>
  );
};

export default Analytics;