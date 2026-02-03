import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext'; // <--- IMPORT ADDED
import Header from './components/Header';
import Landing from './pages/Landing';        
import PublicChat from './pages/PublicChat'; 
import Login from './pages/Login';
import './App.css';
import Dashboard from './pages/Dashboard';
import AdminDashboard from './pages/AdminDashboard';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8080';

function App() {
  return (
    // 1. Wrap everything in AuthProvider so the "Tank" is available
    <AuthProvider>
      <Router>
        <div className="App">
          {/* 2. Removed 'authenticated={false}' so Header can check state itself */}
          <Header /> 
          
          <main className="main-content">
            <Routes>
              <Route path="/" element={<Landing />} />
              <Route path="/chat" element={<PublicChat apiUrl={API_URL} />} />
              <Route path="/login" element={<Login />} />

              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/admin-dashboard" element={<AdminDashboard apiUrl={API_URL} />} />
            </Routes>
          </main>
        </div>
      </Router>
    </AuthProvider>
  );
}

export default App;