import { useState, useEffect, useCallback } from 'react';
import { apiService } from '../services/api';

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
  const [error, setError] = useState(null);
  const [backendConnected, setBackendConnected] = useState(false);

  const checkBackendConnection = async () => {
    try {
      const connected = await apiService.checkBackendConnection();
      setBackendConnected(connected);
      return connected;
    } catch (err) {
      setBackendConnected(false);
      return false;
    }
  };

  const loadInvoices = useCallback(async () => {
    try {
      setProcessing(true);
      
      // First check if backend is available
      const isConnected = await checkBackendConnection();
      
      if (isConnected) {
        const response = await apiService.getInvoices();
        if (response.invoices && response.invoices.length > 0) {
          setInvoices(response.invoices);
          setError(null);
        } else {
          // Backend connected but no data, keep mock data
          console.log('Backend connected but no invoices found, using mock data');
        }
      } else {
        // Backend not available, use mock data (already set)
        setError('Backend not available. Using offline mode with sample data.');
      }
    } catch (err) {
      console.error('Failed to load invoices:', err);
      setError('Unable to connect to backend. Using offline mode with sample data.');
      setBackendConnected(false);
    } finally {
      setProcessing(false);
    }
  }, []);

  useEffect(() => {
    // Start with mock data immediately for better UX
    setInvoices(mockInvoices);
    setBackendConnected(false);

    // In development prefer dev sample file (sanitized) to give realistic UI
    const doDevSample = async () => {
      if (process.env.NODE_ENV !== 'production') {
        try {
          const samples = await apiService.getSampleInvoices(50, false);
          if (samples && samples.length > 0) {
            // normalize sample rows to expected frontend shape if needed
            const normalized = samples.map(s => ({
              id: s.get?.('id') || s.id || s.invoice_number || s.invoice_id || s._raw?.invoice_id || s._raw?.id || (s.vendor && s.vendor.name ? s.vendor.name + '-' + Math.random().toString(36).slice(2,8) : Math.random().toString(36).slice(2,8)),
              vendor: typeof s.vendor === 'string' ? s.vendor : (s.vendor?.name || s.vendor_display || (s._raw && s._raw.vendor_name) || 'Unknown Vendor'),
              amount: s.total_amount || s.amount || s.total || 0,
              status: s.verification_status || s.status || 'processed',
              confidence: s.confidence_score || s.confidence || 0,
              issues: (s.verification_results && s.verification_results.length) || (s._raw && s._raw.verification_results && s._raw.verification_results.length) || 0,
              date: s.invoice_date || s.date || s.processed_date || new Date().toISOString().split('T')[0],
              description: s.notes || s.description || '' ,
              line_items: s.line_items || s.items || []
            }));
            setInvoices(normalized);
            return;
          }
        } catch (err) {
          // ignore and fall back to loadInvoices
          console.warn('Failed to load dev sample', err);
        }
      }

      // Then try to load from backend normally
      loadInvoices();
    };

    doDevSample();
  }, [loadInvoices]);

  const handleFileUpload = async (event) => {
    const files = Array.from(event.target.files);
    setProcessing(true);
    setError(null);
    
    try {
      if (backendConnected) {
        // Try to upload to backend
        const response = await apiService.uploadInvoices(files);
        
        if (response.success && response.results) {
          const newInvoices = response.results
            .filter(result => result.success && result.invoice)
            .map(result => ({
              id: result.invoice.id || result.invoice.invoice_number,
              vendor: result.invoice.vendor?.name || 'Unknown Vendor',
              amount: result.invoice.total_amount || 0,
              status: result.invoice.verification_status || 'processed',
              confidence: result.invoice.confidence_score || 0,
              riskLevel: result.invoice.risk_level || 'unknown',
              issues: result.invoice.verification_results?.length || 0,
              date: result.invoice.invoice_date || new Date().toISOString().split('T')[0],
              fileName: result.filename || 'unknown',
              description: `Processed invoice: ${result.invoice.vendor?.name || 'Unknown'}`
            }));
          
          setInvoices(prev => [...prev, ...newInvoices]);
        } else {
          throw new Error(response.message || 'Upload failed');
        }
      } else {
        // Fallback to mock processing
        const newInvoices = files.map((file, index) => ({
          id: `INV-2024-${String(invoices.length + index + 1).padStart(3, '0')}`,
          vendor: `New Vendor ${index + 1}`,
          amount: Math.round((Math.random() * 15000 + 500) * 100) / 100,
          status: 'processing',
          confidence: Math.round((Math.random() * 0.4 + 0.6) * 100) / 100,
          issues: Math.floor(Math.random() * 4),
          date: new Date().toISOString().split('T')[0],
          fileName: file.name,
          description: `Uploaded invoice from ${file.name}`
        }));
        
        setInvoices(prev => [...prev, ...newInvoices]);
        
        // Simulate status updates
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
      }
    } catch (err) {
      console.error('Upload failed:', err);
      setError(`Upload failed: ${err.message}`);
    } finally {
      setProcessing(false);
    }
  };

  const updateInvoiceStatus = async (invoiceId, newStatus) => {
    try {
      if (backendConnected) {
        await apiService.updateInvoiceStatus(invoiceId, newStatus);
      }
      
      setInvoices(prev => prev.map(invoice => 
        invoice.id === invoiceId 
          ? { ...invoice, status: newStatus }
          : invoice
      ));
    } catch (err) {
      console.error('Failed to update invoice status:', err);
      setError(`Failed to update status: ${err.message}`);
    }
  };

  return { 
    invoices, 
    processing, 
    error,
    backendConnected,
    handleFileUpload, 
    updateInvoiceStatus,
    refreshInvoices: loadInvoices
  };
};