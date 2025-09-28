import { useState, useEffect } from 'react';
import { apiService } from '../services/api';

export const useSystemStatus = () => {
  const [systemStatus, setSystemStatus] = useState({
    status: 'unknown',
    agents_online: false,
    database_connected: false,
    processing_queue: 0,
    agents: {
      orchestrator: { status: 'unknown', load: 0 },
      validation: { status: 'unknown', load: 0 },
      fraud_detection: { status: 'unknown', load: 0 },
      analytics: { status: 'unknown', load: 0 }
    }
  });
  const [backendConnected, setBackendConnected] = useState(false);
  const [error, setError] = useState(null);

  const fetchSystemStatus = async () => {
    try {
      const response = await apiService.getSystemStatus();
      
      setSystemStatus({
        status: response.status || 'unknown',
        agents_online: response.agents_online || false,
        database_connected: response.database_connected || false,
        processing_queue: response.processing_queue || 0,
        agents: {
          orchestrator: { 
            status: response.agents_online ? 'active' : 'inactive', 
            load: Math.floor(Math.random() * 30 + 50) 
          },
          validation: { 
            status: response.agents_online ? 'active' : 'inactive', 
            load: Math.floor(Math.random() * 30 + 40) 
          },
          fraud_detection: { 
            status: response.agents_online ? 'active' : 'inactive', 
            load: Math.floor(Math.random() * 40 + 60) 
          },
          analytics: { 
            status: response.agents_online ? 'active' : 'inactive', 
            load: Math.floor(Math.random() * 30 + 40) 
          }
        }
      });
      
      setBackendConnected(true);
      setError(null);
    } catch (err) {
      console.error('Failed to fetch system status:', err);
      setError('Failed to connect to backend');
      setBackendConnected(false);
      
      // Fallback to mock data
      setSystemStatus({
        status: 'degraded',
        agents_online: false,
        database_connected: false,
        processing_queue: 0,
        agents: {
          orchestrator: { status: 'inactive', load: 0 },
          validation: { status: 'inactive', load: 0 },
          fraud_detection: { status: 'inactive', load: 0 },
          analytics: { status: 'inactive', load: 0 }
        }
      });
    }
  };

  useEffect(() => {
    // Start with mock data for immediate display
    setSystemStatus({
      status: 'active',
      agents_online: true,
      database_connected: true,
      processing_queue: 3,
      agents: {
        orchestrator: { status: 'active', load: 65 },
        validation: { status: 'active', load: 45 },
        fraud_detection: { status: 'active', load: 70 },
        analytics: { status: 'active', load: 55 }
      }
    });
    
    // Try to get actual status from backend
    fetchSystemStatus();
    
    // Poll for status updates every 30 seconds (less frequent to reduce load)
    const interval = setInterval(fetchSystemStatus, 30000);
    
    return () => clearInterval(interval);
  }, []);

  return { 
    systemStatus, 
    backendConnected, 
    error,
    refreshStatus: fetchSystemStatus 
  };
};