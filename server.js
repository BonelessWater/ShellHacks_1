// server.js
const express = require('express');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 8080;

const buildPath = path.join(__dirname, 'frontend', 'build');
app.use(express.static(buildPath));

// ✅ Use a regex or '/*' instead of '*'
app.get(/.*/, (_req, res) => {
  res.sendFile(path.join(buildPath, 'index.html'));
});

app.listen(PORT, () => {
  console.log(`🚀 Server on ${PORT}, serving ${buildPath}`);
});
