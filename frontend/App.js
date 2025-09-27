// frontend/src/App.js
import React from 'react';
import ApiTest from './components/ApiTest';
import './App.css';

function App() {
  return (
    <div className="App">
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-8">
        <div className="container mx-auto px-4">
          <header className="text-center mb-8">
            <h1 className="text-4xl font-bold text-gray-800 mb-2">
              ShellHacks Invoice System
            </h1>
            <p className="text-lg text-gray-600">
              Docker + React + Python FastAPI
            </p>
          </header>
          
          <main>
            <ApiTest />
          </main>
          
          <footer className="text-center mt-8 text-gray-500">
            <p>Ready for Azure deployment! 🚀</p>
          </footer>
        </div>
      </div>
    </div>
  );
}

export default App;