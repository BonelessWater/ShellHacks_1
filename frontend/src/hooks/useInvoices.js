import { useState, useEffect, useCallback } from 'react';
import { apiService } from '../services/api';

export const useInvoices = () => {
  const [invoices, setInvoices] = useState([]);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [backendConnected, setBackendConnected] = useState(false);

  const loadInvoices = useCallback(async () => {
    try {
      setProcessing(true);
      setError(null);
      
      const response = await apiService.getInvoices();
      if (response.invoices) {
        setInvoices(response.invoices);
        setBackendConnected(true);
      } else {
        setInvoices([]);
        setBackendConnected(false);
        setError("Failed to fetch invoices.");
      }
    } catch (err) {
      console.error('Failed to load invoices:', err);
      setError('Unable to connect to backend.');
      setBackendConnected(false);
      setInvoices([]);
    } finally {
      setProcessing(false);
    }
  }, []);

  useEffect(() => {
    loadInvoices();
  }, [loadInvoices]);

  return { 
    invoices, 
    processing, 
    error,
    backendConnected,
    refreshInvoices: loadInvoices
  };
};