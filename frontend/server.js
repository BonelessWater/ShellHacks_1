// frontend/server.js - Express server for serving React build on port 3000
const express = require('express');
const path = require('path');
const { createProxyMiddleware } = require('http-proxy-middleware');

const app = express();
const PORT = process.env.PORT || 3000;

// Serve static files from the React app build
const buildPath = path.join(__dirname, 'build');
app.use(express.static(buildPath));

// API proxy to backend
app.use('/api', createProxyMiddleware({
  target: process.env.REACT_APP_API_URL || 'http://backend:8000',
  changeOrigin: true,
  logLevel: 'debug'
}));

// Health check proxy
app.use('/health', createProxyMiddleware({
  target: process.env.REACT_APP_API_URL || 'http://backend:8000',
  changeOrigin: true
}));

// Catch all handler: send back React's index.html file for any non-API routes
app.get('*', (req, res) => {
  res.sendFile(path.join(buildPath, 'index.html'));
});

app.listen(PORT, () => {
  console.log(`🚀 Frontend server running on port ${PORT}`);
  console.log(`📁 Serving static files from: ${buildPath}`);
  console.log(`🔗 API proxy target: ${process.env.REACT_APP_API_URL || 'http://backend:8000'}`);
});