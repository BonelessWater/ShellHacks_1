import { useState, useEffect } from 'react';

const mockInvoices = [
  {
    id: 'INV-2024-001',
    vendor: 'ABC Office Supplies',
    amount: 1287.00,
    status: 'approved',
    confidence: 0.95,
    issues: 0,
    date: '2024-01-15',
    description: 'Monthly office supplies - paper, pens, folders'
  },
  {
    id: 'INV-2024-002',
    vendor: 'TechCorp Solutions',
    amount: 8500.00,
    status: 'review_required',
    confidence: 0.73,
    issues: 2,
    date: '2024-01-16',
    description: 'Software licensing and technical support'
  },
  {
    id: 'INV-2024-003',
    vendor: 'Suspicious Vendor Inc',
    amount: 9999.99,
    status: 'rejected',
    confidence: 0.34,
    issues: 5,
    date: '2024-01-17',
    description: 'Questionable consulting services'
  },
  {
    id: 'INV-2024-004',
    vendor: 'CloudServices Pro',
    amount: 2500.00,
    status: 'approved',
    confidence: 0.88,
    issues: 1,
    date: '2024-01-12',
    description: 'Monthly cloud infrastructure hosting'
  },
  {
    id: 'INV-2024-005',
    vendor: 'Global Marketing Ltd',
    amount: 15750.00,
    status: 'review_required',
    confidence: 0.67,
    issues: 3,
    date: '2024-01-18',
    description: 'Q1 marketing campaign services'
  },
  {
    id: 'INV-2024-006',
    vendor: 'SecureIT Systems',
    amount: 750.50,
    status: 'approved',
    confidence: 0.92,
    issues: 0,
    date: '2024-01-19',
    description: 'Cybersecurity software license'
  },
  {
    id: 'INV-2024-007',
    vendor: 'FraudCorp LLC',
    amount: 25000.00,
    status: 'rejected',
    confidence: 0.28,
    issues: 7,
    date: '2024-01-20',
    description: 'Suspicious high-value transaction'
  },
  {
    id: 'INV-2024-008',
    vendor: 'Office Furniture Direct',
    amount: 4200.00,
    status: 'processing',
    confidence: 0.85,
    issues: 1,
    date: '2024-01-21',
    description: 'Ergonomic desk chairs and standing desks'
  },
  {
    id: 'INV-2024-009',
    vendor: 'Legal Advisors Group',
    amount: 12500.00,
    status: 'approved',
    confidence: 0.94,
    issues: 0,
    date: '2024-01-22',
    description: 'Legal consultation and contract review'
  },
  {
    id: 'INV-2024-010',
    vendor: 'QuickPay Services',
    amount: 350.00,
    status: 'review_required',
    confidence: 0.71,
    issues: 2,
    date: '2024-01-23',
    description: 'Payment processing fees'
  },
  {
    id: 'INV-2024-011',
    vendor: 'DataCenter Holdings',
    amount: 18900.00,
    status: 'approved',
    confidence: 0.89,
    issues: 1,
    date: '2024-01-24',
    description: 'Data storage and backup services'
  },
  {
    id: 'INV-2024-012',
    vendor: 'Maintenance Solutions',
    amount: 890.00,
    status: 'approved',
    confidence: 0.96,
    issues: 0,
    date: '2024-01-25',
    description: 'Building maintenance and cleaning'
  }
];

export const useInvoices = () => {
  const [invoices, setInvoices] = useState([]);
  const [processing, setProcessing] = useState(false);

  useEffect(() => {
    // Simulate loading delay
    setTimeout(() => {
      setInvoices(mockInvoices);
    }, 500);
  }, []);

  const handleFileUpload = (event) => {
    const files = Array.from(event.target.files);
    setProcessing(true);
    
    // Simulate processing time
    setTimeout(() => {
      const newInvoices = files.map((file, index) => ({
        id: `INV-2024-${String(invoices.length + index + 1).padStart(3, '0')}`,
        vendor: `New Vendor ${index + 1}`,
        amount: Math.round((Math.random() * 15000 + 500) * 100) / 100,
        status: 'processing',
        confidence: Math.round((Math.random() * 0.4 + 0.6) * 100) / 100, // 60-100%
        issues: Math.floor(Math.random() * 4), // 0-3 issues
        date: new Date().toISOString().split('T')[0],
        fileName: file.name,
        description: `Uploaded invoice from ${file.name}`
      }));
      
      setInvoices(prev => [...prev, ...newInvoices]);
      setProcessing(false);
      
      // Simulate status updates after processing
      setTimeout(() => {
        setInvoices(prev => prev.map(invoice => {
          if (invoice.status === 'processing') {
            const finalStatus = invoice.issues > 2 ? 'rejected' : 
                               invoice.issues > 0 ? 'review_required' : 'approved';
            return { ...invoice, status: finalStatus };
          }
          return invoice;
        }));
      }, 3000);
    }, 2000);
  };

  const updateInvoiceStatus = (invoiceId, newStatus) => {
    setInvoices(prev => prev.map(invoice => 
      invoice.id === invoiceId 
        ? { ...invoice, status: newStatus }
        : invoice
    ));
  };

  return { 
    invoices, 
    processing, 
    handleFileUpload, 
    updateInvoiceStatus 
  };
};