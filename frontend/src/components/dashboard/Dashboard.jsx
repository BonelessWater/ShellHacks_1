import React from 'react';
import SummaryCards from './SummaryCards';
import InvoiceTable from './InvoiceTable';

const Dashboard = ({ invoices }) => {
  return (
    <div className="space-y-6">
      <SummaryCards invoices={invoices} />
      <InvoiceTable invoices={invoices} />
    </div>
  );
};

export default Dashboard;