// frontend/server.js - Express server for serving React build
const express = require('express');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 8080;

// Serve static files from the React app build
const buildPath = path.join(__dirname, 'build');
app.use(express.static(buildPath));

// API proxy to backend (if needed)
// In production, you might want to proxy API calls to your Python backend
// app.use('/api', createProxyMiddleware({
//   target: 'http://your-backend-url:8000',
//   changeOrigin: true
// }));

// Catch all handler: send back React's index.html file for any non-API routes
app.get('*', (req, res) => {
  res.sendFile(path.join(buildPath, 'index.html'));
});

app.listen(PORT, () => {
  console.log(`🚀 Frontend server running on port ${PORT}`);
  console.log(`📁 Serving static files from: ${buildPath}`);
});