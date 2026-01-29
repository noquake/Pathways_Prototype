import React from 'react';
import { Link } from 'react-router-dom';

function Header({ authenticated, onLogin, onLogout }) {
  // Styles consolidated into the component to keep it non-invasive
  const styles = {
    header: {
      backgroundColor: 'var(--accent-burgundy, #800020)', // Matches your theme
      color: 'white',
      padding: '8px 20px', // Slimmer padding for a less invasive feel
      boxShadow: '0 1px 3px rgba(0,0,0,0.2)',
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      height: '50px' // Fixed slim height
    },
    logoLink: {
      textDecoration: 'none',
      color: 'white',
      display: 'flex',
      alignItems: 'center'
    },
    logoText: {
      fontSize: '1.1rem', // Smaller, professional font size
      margin: 0,
      fontWeight: '600'
    },
    authButton: {
      backgroundColor: 'transparent',
      color: 'white',
      border: '1px solid rgba(255,255,255,0.5)', // Subtle outlined style
      padding: '5px 12px',
      borderRadius: '4px',
      cursor: 'pointer',
      fontSize: '0.85rem',
      transition: 'all 0.2s ease',
      fontWeight: '500'
    }
  };

  return (
    <header style={styles.header}>
      <Link to="/" style={styles.logoLink}>
        <h1 style={styles.logoText}>Pathways</h1>
      </Link>

      <nav>
        {authenticated ? (
          <button 
            onClick={onLogout} 
            style={styles.authButton}
            onMouseOver={(e) => e.target.style.backgroundColor = 'rgba(255,255,255,0.1)'}
            onMouseOut={(e) => e.target.style.backgroundColor = 'transparent'}
          >
            Logout
          </button>
        ) : (
          <button 
            onClick={onLogin} 
            style={styles.authButton}
            onMouseOver={(e) => e.target.style.backgroundColor = 'rgba(255,255,255,0.1)'}
            onMouseOut={(e) => e.target.style.backgroundColor = 'transparent'}
          >
            Login
          </button>
        )}
      </nav>
    </header>
  );
}

export default Header;