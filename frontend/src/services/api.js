const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

class APIService {
  async request(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  }

  // Invoice methods
  async uploadInvoices(files) {
    const formData = new FormData();
    Array.from(files).forEach(file => {
      formData.append('files', file);
    });

    return this.request('/api/invoices/upload', {
      method: 'POST',
      body: formData,
      headers: {}, // Remove Content-Type to let browser set it for FormData
    });
  }

  async getInvoices() {
    return this.request('/api/invoices');
  }

  async getInvoice(id) {
    return this.request(`/api/invoices/${id}`);
  }

  async updateInvoiceStatus(id, status) {
    return this.request(`/api/invoices/${id}/status`, {
      method: 'PATCH',
      body: JSON.stringify({ status }),
    });
  }

  // System status methods
  async getSystemStatus() {
    return this.request('/api/system/status');
  }

  async getAnalytics() {
    return this.request('/api/analytics');
  }

  // Agent configuration methods
  async getAgentConfig() {
    return this.request('/api/agents/config');
  }

  async updateAgentConfig(config) {
    return this.request('/api/agents/config', {
      method: 'PUT',
      body: JSON.stringify(config),
    });
  }
}

export const apiService = new APIService();