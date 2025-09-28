import React, { useState } from 'react';
import SummaryCards from './SummaryCards';
import InvoiceTable from './InvoiceTable';
import InvoiceFilters from './InvoiceFilters';

const Dashboard = ({ invoices }) => {
  const [filteredInvoices, setFilteredInvoices] = useState(invoices);

  const handleFilterChange = (filtered) => {
    setFilteredInvoices(filtered);
  };

  return (
    <div className="space-y-6">
      <SummaryCards invoices={filteredInvoices} />
      <InvoiceFilters 
        invoices={invoices} 
        onFilterChange={handleFilterChange} 
      />
      <InvoiceTable 
        invoices={invoices} 
        filteredInvoices={filteredInvoices} 
      />
    </div>
  );
};

export default Dashboard;