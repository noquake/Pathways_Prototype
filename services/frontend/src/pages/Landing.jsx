import React from 'react';
import '../index.css'; // Importing your theme variables

const Landing = () => {
  return (
    <div className="landing-container">
      <header className="hero-header">
        <h1>Pathways Clinical Chat</h1>
        <p>
          An evidence-driven clinical decision support platform focused on
          structured pathways, traceability, and safety-first AI assistance.
        </p>
      </header>

      <main className="landing-main">
        <section>
          <h2>About</h2>
          <p>
            Pathways Clinical Chat is an experimental system designed to assist
            clinicians by structuring, contextualizing, and surfacing established
            clinical pathways. The project emphasizes transparency, auditable
            reasoning, and secure infrastructure.
          </p>
        </section>

        <section>
          <h2>Application Areas</h2>
          <ul className="pathway-list">
            <li>
              <a href="/chat" className="pathway-list">Clinical Chat Interface</a>
              <span> — Ask questions about the Clinical Pathways without logging in.</span>
            </li>
            <li>
              <a href="/explorer" className="placeholder">Pathway Explorer</a>
              <span className="placeholder"> — coming soon</span>
            </li>
            <li>
              <a href="/evidence" className="placeholder">Evidence & References</a>
              <span className="placeholder"> — coming soon</span>
            </li>
            <li>
              <a href="/auth" className="placeholder">Authentication & User Access</a>
              <span className="placeholder"> — coming soon</span>
            </li>
          </ul>
        </section>
      </main>

      <footer>
        &copy; {new Date().getFullYear()} Pathways Clinical Chat. Secure Clinical Infrastructure.
      </footer>
    </div>
  );
};

export default Landing;