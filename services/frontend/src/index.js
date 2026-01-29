import React from "react";
import ReactDOM from "react-dom/client";
import "./index.css"; // This imports your Burgundy/Green theme
import App from "./App"; // This looks for App.jsx automatically

// 1. Find the "root" div in your index.html
const rootElement = document.getElementById("root");

// 2. Create the React engine "root"
const root = ReactDOM.createRoot(rootElement);

// 3. Render your app into that div
root.render(
	<React.StrictMode>
		<App />
	</React.StrictMode>
);
