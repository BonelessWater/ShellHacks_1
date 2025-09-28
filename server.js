// server.js - Fixed for Azure deployment
const express = require('express');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 8080;

// In Azure, files are deployed to /home/site/wwwroot
// Your deployment copies build files to /deploy/static, which becomes /home/site/wwwroot/static
const buildPath = path.join(__dirname, 'static');

// Serve static files from the React app build
app.use(express.static(buildPath));

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'OK', buildPath });
});

// Catch all handler: send back React's index.html file for any non-API routes
app.get('*', (req, res) => {
  const indexPath = path.join(buildPath, 'index.html');
  
  // Check if file exists before trying to serve it
  const fs = require('fs');
  if (fs.existsSync(indexPath)) {
    res.sendFile(indexPath);
  } else {
    res.status(404).json({ 
      error: 'index.html not found',
      buildPath,
      indexPath,
      files: fs.existsSync(buildPath) ? fs.readdirSync(buildPath) : 'buildPath does not exist'
    });
  }
});

app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
  console.log(`📁 Serving static files from: ${buildPath}`);
  console.log(`📄 Looking for index.html at: ${path.join(buildPath, 'index.html')}`);
});