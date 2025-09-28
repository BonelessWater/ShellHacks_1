// frontend/src/services/api.js
import axios from 'axios';

// Configure API base URL
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'https://agentzero.azurewebsites.net'  // Your Azure backend URL
  : 'http://localhost:8000';  // Local development backend

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor for debugging
api.interceptors.request.use(
  (config) => {
    console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('❌ API Request Error:', error);
    return Promise.reject(error);
  }
);

// Add response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    console.log(`✅ API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('❌ API Response Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// API Methods
export const apiService = {
  // Health checks
  checkHealth: async () => {
    try {
      const response = await api.get('/health');
      return response.data;
    } catch (error) {
      throw new Error(`Health check failed: ${error.message}`);
    }
  },

  checkApiHealth: async () => {
    try {
      const response = await api.get('/api/health');
      return response.data;
    } catch (error) {
      throw new Error(`API health check failed: ${error.message}`);
    }
  },

  // Get message from backend
  getMessage: async () => {
    try {
      const response = await api.get('/api/message');
      return response.data;
    } catch (error) {
      throw new Error(`Failed to get message: ${error.message}`);
    }
  },

  // Invoice processing (placeholder for your fraud detection)
  uploadInvoice: async (invoiceData) => {
    try {
      const response = await api.post('/api/invoices/upload', {
        invoice_data: invoiceData
      });
      return response.data;
    } catch (error) {
      throw new Error(`Invoice upload failed: ${error.message}`);
    }
  },

  // Get system status
  getSystemStatus: async () => {
    try {
      const response = await api.get('/api/system/status');
      return response.data;
    } catch (error) {
      throw new Error(`System status check failed: ${error.message}`);
    }
  },

  // Agent configuration
  getAgentConfig: async () => {
    try {
      const response = await api.get('/api/agents/config');
      return response.data;
    } catch (error) {
      throw new Error(`Failed to get agent config: ${error.message}`);
    }
  },

  updateAgentConfig: async (config) => {
    try {
      const response = await api.put('/api/agents/config', { config });
      return response.data;
    } catch (error) {
      throw new Error(`Failed to update agent config: ${error.message}`);
    }
  },

  // File upload functionality
  uploadFile: async (file) => {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await api.post('/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 60000, // 60 seconds for file uploads
      });
      return response.data;
    } catch (error) {
      throw new Error(`File upload failed: ${error.message}`);
    }
  },

  uploadMultipleFiles: async (files) => {
    try {
      const formData = new FormData();
      files.forEach((file, index) => {
        formData.append(`files`, file);
      });

      const response = await api.post('/api/upload/multiple', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 120000, // 2 minutes for multiple files
      });
      return response.data;
    } catch (error) {
      throw new Error(`Multiple file upload failed: ${error.message}`);
    }
  },

  // Invoice analysis functionality
  analyzeInvoice: async (invoiceData) => {
    try {
      const response = await api.post('/api/invoices/analyze', invoiceData);
      return response.data;
    } catch (error) {
      throw new Error(`Invoice analysis failed: ${error.message}`);
    }
  },

  getInvoiceAnalysis: async (invoiceId) => {
    try {
      const response = await api.get(`/api/invoices/${invoiceId}`);
      return response.data;
    } catch (error) {
      throw new Error(`Failed to get invoice analysis: ${error.message}`);
    }
  },

  // Check backend connectivity (returns boolean)
  checkBackendConnection: async () => {
    try {
      const response = await api.get('/api/system/status');
      // Consider backend connected when it returns a healthy status
      return response && response.data && response.data.status && response.data.status === 'healthy';
    } catch (error) {
      return false;
    }
  },

  // Get invoices (used by frontend hooks)
  getInvoices: async (params = {}) => {
    try {
      const response = await api.get('/api/invoices', { params });
      return response.data;
    } catch (error) {
      throw new Error(`Failed to get invoices: ${error.message}`);
    }
  },

  // Get a small dev-only sample (obfuscated) for UI prototyping
  getSampleInvoices: async (limit = 25, dynamic = false) => {
    try {
      const response = await api.get('/api/invoices/sample', { params: { limit, dynamic } });
      return response.data;
    } catch (error) {
      console.warn('Failed to fetch sample invoices, falling back to /api/invoices', error.message);
      return apiService.getInvoices({ limit });
    }
  },

  // Score invoices using a safe heuristic endpoint (dev-only)
  scoreInvoices: async (invoices) => {
    try {
      const response = await api.post('/api/invoices/score', { invoices });
      return response.data;
    } catch (error) {
      throw new Error(`Failed to score invoices: ${error.message}`);
    }
  },

  // Upload multiple invoice files (used by frontend)
  uploadInvoices: async (files) => {
    try {
      const formData = new FormData();
      files.forEach((file) => formData.append('files', file));

      const response = await api.post('/api/invoices/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 120000,
      });
      return response.data;
    } catch (error) {
      throw new Error(`Invoice upload failed: ${error.message}`);
    }
  },

  // Update invoice status
  updateInvoiceStatus: async (invoiceId, status) => {
    try {
      const response = await api.patch(`/api/invoices/${invoiceId}/status`, { status });
      return response.data;
    } catch (error) {
      throw new Error(`Failed to update invoice status: ${error.message}`);
    }
  },

  // Get uploaded files
  getUploadedFiles: async () => {
    try {
      const response = await api.get('/api/uploads');
      return response.data;
    } catch (error) {
      throw new Error(`Failed to get uploaded files: ${error.message}`);
    }
  }
};

export default api;