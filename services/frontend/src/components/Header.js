import React from 'react';
import { Link } from 'react-router-dom';
import './Header.css';

function Header({ authenticated, userRole, onLogin, onLogout }) {
  return (
    <header className="app-header">
      <div className="header-content">
        <Link to="/" className="logo">
          <h1>Pathways Clinical Chat</h1>
        </Link>
        <nav className="header-nav">
          {authenticated && userRole === 'practitioner' && (
            <Link to="/practitioner" className="nav-link">Practitioner Chat</Link>
          )}
          {authenticated && (userRole === 'hr' || userRole === 'admin') && (
            <Link to="/hr" className="nav-link">HR Dashboard</Link>
          )}
          {authenticated && userRole === 'admin' && (
            <Link to="/admin" className="nav-link">Admin Dashboard</Link>
          )}
          {authenticated ? (
            <button onClick={onLogout} className="auth-button">Logout</button>
          ) : (
            <button onClick={onLogin} className="auth-button">Login</button>
          )}
        </nav>
      </div>
    </header>
  );
}

export default Header;

