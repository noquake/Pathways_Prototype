import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import Landing from './pages/Landing';        
import PublicChat from './pages/PublicChat'; 
import './App.css';
import Login from './pages/Login';
import './App.css';


const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const KEYCLOAK_URL = process.env.REACT_APP_KEYCLOAK_URL || 'http://localhost:8080';
const KEYCLOAK_REALM = process.env.REACT_APP_KEYCLOAK_REALM || 'pathways';
const KEYCLOAK_CLIENT_ID = process.env.REACT_APP_KEYCLOAK_CLIENT_ID || 'pathways-frontend';


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
            <Route path="/login" element={<Login />} />
			
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;