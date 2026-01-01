import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Keycloak from 'keycloak-js';
import Header from './components/Header';
import PublicChat from './pages/PublicChat';
import PractitionerChat from './pages/PractitionerChat';
import HRDashboard from './pages/HRDashboard';
import AdminDashboard from './pages/AdminDashboard';
import Login from './pages/Login';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const KEYCLOAK_URL = process.env.REACT_APP_KEYCLOAK_URL || 'http://localhost:8080';
const KEYCLOAK_REALM = process.env.REACT_APP_KEYCLOAK_REALM || 'pathways';
const KEYCLOAK_CLIENT_ID = process.env.REACT_APP_KEYCLOAK_CLIENT_ID || 'pathways-frontend';

function App() {
  const [keycloak, setKeycloak] = useState(null);
  const [authenticated, setAuthenticated] = useState(false);
  const [userRole, setUserRole] = useState('public');

  useEffect(() => {
    const keycloakInstance = new Keycloak({
      url: KEYCLOAK_URL,
      realm: KEYCLOAK_REALM,
      clientId: KEYCLOAK_CLIENT_ID
    });

    keycloakInstance.init({ onLoad: 'check-sso' })
      .then((authenticated) => {
        setKeycloak(keycloakInstance);
        setAuthenticated(authenticated);
        if (authenticated) {
          // Extract role from token
          const token = keycloakInstance.tokenParsed;
          setUserRole(token?.realm_access?.roles?.[0] || 'public');
        }
      })
      .catch((error) => {
        console.error('Keycloak initialization failed:', error);
      });
  }, []);

  const handleLogin = () => {
    if (keycloak) {
      keycloak.login();
    }
  };

  const handleLogout = () => {
    if (keycloak) {
      keycloak.logout();
    }
  };

  return (
    <Router>
      <div className="App">
        <Header 
          authenticated={authenticated}
          userRole={userRole}
          onLogin={handleLogin}
          onLogout={handleLogout}
        />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<PublicChat apiUrl={API_URL} />} />
            <Route path="/login" element={<Login onLogin={handleLogin} />} />
            <Route 
              path="/practitioner" 
              element={
                authenticated && userRole === 'practitioner' ? (
                  <PractitionerChat apiUrl={API_URL} keycloak={keycloak} />
                ) : (
                  <Navigate to="/login" />
                )
              } 
            />
            <Route 
              path="/hr" 
              element={
                authenticated && (userRole === 'hr' || userRole === 'admin') ? (
                  <HRDashboard apiUrl={API_URL} keycloak={keycloak} />
                ) : (
                  <Navigate to="/login" />
                )
              } 
            />
            <Route 
              path="/admin" 
              element={
                authenticated && userRole === 'admin' ? (
                  <AdminDashboard apiUrl={API_URL} keycloak={keycloak} />
                ) : (
                  <Navigate to="/login" />
                )
              } 
            />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;

