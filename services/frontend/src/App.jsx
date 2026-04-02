import React from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext'; // <--- IMPORT ADDED
import Header from './components/Header';
import Landing from './pages/Landing';        
import PublicChat from './pages/PublicChat'; 
import Login from './pages/Login';
import './App.css';
import Dashboard from './pages/Dashboard';
import AdminDashboard from './pages/AdminDashboard';

const API_URL = process.env.REACT_APP_API_URL || '';

function AppLayout() {
  const location = useLocation();
  const isPublicChatRoute = location.pathname === '/chat';
  const showHeader = !isPublicChatRoute;
  const mainContentClassName = isPublicChatRoute
    ? 'main-content main-content-public-chat'
    : 'main-content';

  const appContent = (
    <div className="App">
      {showHeader && <Header />}

      <main className={mainContentClassName}>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/chat" element={<PublicChat apiUrl={API_URL} />} />
          <Route path="/login" element={<Login />} />

          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/admin-dashboard" element={<AdminDashboard apiUrl={API_URL} />} />
        </Routes>
      </main>
    </div>
  );

  if (isPublicChatRoute) {
    return appContent;
  }

  return <AuthProvider>{appContent}</AuthProvider>;
}

function App() {
  return (
    <Router>
      <AppLayout />
    </Router>
  );
}

export default App;
