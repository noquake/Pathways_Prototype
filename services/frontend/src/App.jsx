import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import Landing from './pages/Landing';        
import PublicChat from './pages/PublicChat'; 
import './App.css';

// This URL must match where your Python backend is running
const API_URL = 'http://localhost:8000'; 

function App() {
  return (
    <Router>
      <div className="App">
        {/* authenticated={false} ensures we see the simplified header */}
        <Header authenticated={false} /> 
        
        <main className="main-content">
          <Routes>
            {/* The Landing Page (Root) */}
            <Route path="/" element={<Landing />} />
            
            {/* The Chat Interface (No Login Required) */}
            <Route 
              path="/chat" 
              element={<PublicChat apiUrl={API_URL} />} 
            />
			
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;