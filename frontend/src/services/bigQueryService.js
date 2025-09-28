class BigQueryService {
  constructor() {
    this.baseURL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;

    try {
      const response = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        ...options,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('BigQuery API request failed:', error);
      throw error;
    }
  }

  // Get fraud transactions
  async getFraudTransactions(limit = 100) {
    return this.request(`/api/fraud-transactions?limit=${limit}`);
  }

  // Execute custom query
  async executeQuery(query) {
    return this.request('/api/query', {
      method: 'POST',
      body: JSON.stringify({ query }),
    });
  }

  // Health check
  async healthCheck() {
    return this.request('/health');
  }
}

export const bigQueryService = new BigQueryService();
