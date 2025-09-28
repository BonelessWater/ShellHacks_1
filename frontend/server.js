// frontend/server.js - Node.js Express server for production
const express = require('express');
const path = require('path');
const { createProxyMiddleware } = require('http-proxy-middleware');

const app = express();
const PORT = process.env.PORT || 3000;
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    message: 'Node.js frontend server is running',
    port: PORT,
    timestamp: new Date().toISOString()
  });
});

// Proxy API requests to FastAPI backend
app.use('/api', createProxyMiddleware({
  target: API_URL,
  changeOrigin: true,
  pathRewrite: {
    '^/api': '/api', // Keep /api prefix
  },
  onError: (err, req, res) => {
    console.error('Proxy error:', err.message);
    res.status(500).json({ 
      error: 'Backend service unavailable',
      message: err.message 
    });
  },
  onProxyReq: (proxyReq, req, res) => {
    console.log(`Proxying: ${req.method} ${req.url} -> ${API_URL}${req.url}`);
  }
}));

// Serve static files from React build
app.use(express.static(path.join(__dirname, 'build')));

// Serve React app for all non-API routes (SPA routing)
app.get('*', (req, res) => {
  // Don't serve React for API routes that failed proxy
  if (req.path.startsWith('/api/')) {
    return res.status(404).json({ error: 'API endpoint not found' });
  }
  
  res.sendFile(path.join(__dirname, 'build', 'index.html'));
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Server error:', err);
  res.status(500).json({ 
    error: 'Internal server error',
    message: process.env.NODE_ENV === 'development' ? err.message : 'Something went wrong'
  });
});

// Start server
app.listen(PORT, '0.0.0.0', () => {
  console.log(`🚀 Node.js server running on port ${PORT}`);
  console.log(`📡 Proxying API requests to: ${API_URL}`);
  console.log(`🏥 Health check: http://localhost:${PORT}/health`);
  console.log(`🌐 Environment: ${process.env.NODE_ENV || 'development'}`);
});