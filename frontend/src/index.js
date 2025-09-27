// frontend/index.js - Node.js server for Azure deployment
const express = require('express');
const path = require('path');
const compression = require('compression');

const app = express();
const port = process.env.PORT || 8080;

// Middleware
app.use(compression());
app.use(express.json());

// Security headers
app.use((req, res, next) => {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader('X-XSS-Protection', '1; mode=block');
  next();
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'healthy', timestamp: new Date().toISOString() });
});

// API proxy endpoints (optional - if you want to keep some backend functionality)
app.get('/api/health', (req, res) => {
  res.json({ status: 'frontend-healthy', message: 'Frontend server running' });
});

// Serve static files from build directory
app.use(express.static(path.join(__dirname, 'build'), {
  maxAge: '1y',
  etag: false
}));

// Handle React Router routes - serve index.html for all non-API routes
app.get('*', (req, res) => {
  // Don't serve index.html for API routes
  if (req.path.startsWith('/api/')) {
    return res.status(404).json({ error: 'API endpoint not found' });
  }
  
  res.sendFile(path.join(__dirname, 'build', 'index.html'), {
    headers: {
      'Cache-Control': 'no-cache'
    }
  });
});

// Error handling
app.use((err, req, res, next) => {
  console.error('Server error:', err);
  res.status(500).json({ error: 'Internal server error' });
});

// Start server
app.listen(port, () => {
  console.log(`🚀 Server running on port ${port}`);
  console.log(`📁 Serving static files from: ${path.join(__dirname, 'build')}`);
  console.log(`🌐 Environment: ${process.env.NODE_ENV || 'development'}`);
});

module.exports = app;