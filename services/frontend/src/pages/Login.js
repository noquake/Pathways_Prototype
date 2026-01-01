import React from 'react';
import './Login.css';

function Login({ onLogin }) {
  return (
    <div className="login-page">
      <div className="login-container">
        <h2>Login Required</h2>
        <p>Please log in to access practitioner features, HR dashboard, or admin dashboard.</p>
        <button onClick={onLogin} className="login-button">
          Login with Keycloak
        </button>
      </div>
    </div>
  );
}

export default Login;

