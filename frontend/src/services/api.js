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
  }
};

export default api;