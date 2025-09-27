import { useState, useEffect } from 'react';

const mockInvoices = [
  {
    id: 'INV-2024-001',
    vendor: 'ABC Office Supplies',
    amount: 1287.00,
    status: 'approved',
    confidence: 0.95,
    issues: 0,
    date: '2024-01-15'
  },
  {
    id: 'INV-2024-002',
    vendor: 'TechCorp Solutions',
    amount: 8500.00,
    status: 'review_required',
    confidence: 0.73,
    issues: 2,
    date: '2024-01-16'
  },
  {
    id: 'INV-2024-003',
    vendor: 'Suspicious Vendor Inc',
    amount: 9999.99,
    status: 'rejected',
    confidence: 0.34,
    issues: 5,
    date: '2024-01-17'
  }
];

export const useInvoices = () => {
  const [invoices, setInvoices] = useState([]);
  const [processing, setProcessing] = useState(false);

  useEffect(() => {
    setInvoices(mockInvoices);
  }, []);

  const handleFileUpload = (event) => {
    const files = Array.from(event.target.files);
    setProcessing(true);
    
    setTimeout(() => {
      const newInvoices = files.map((file, index) => ({
        id: `INV-2024-${String(invoices.length + index + 1).padStart(3, '0')}`,
        vendor: 'New Vendor',
        amount: Math.random() * 10000,
        status: 'processing',
        confidence: 0,
        issues: 0,
        date: new Date().toISOString().split('T')[0],
        fileName: file.name
      }));
      
      setInvoices([...invoices, ...newInvoices]);
      setProcessing(false);
    }, 2000);
  };

  return { invoices, processing, handleFileUpload };
};