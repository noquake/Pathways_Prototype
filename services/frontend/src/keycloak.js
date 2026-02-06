// src/keycloak.js
import Keycloak from "keycloak-js";

// Configure these details to match your Keycloak server
// Use environment variables if available, otherwise fall back to defaults
// NOTE: Default clientId changed to match docker-compose.yml
const keycloakConfig = {
	url: process.env.REACT_APP_KEYCLOAK_URL || "http://localhost:8080",
	realm: process.env.REACT_APP_KEYCLOAK_REALM || "pathways",
	clientId: process.env.REACT_APP_KEYCLOAK_CLIENT_ID || "pathways-frontend",
};

// Log the configuration to help debug
console.log("Keycloak Configuration:", {
	url: keycloakConfig.url,
	realm: keycloakConfig.realm,
	clientId: keycloakConfig.clientId,
	envVarsLoaded: {
		REACT_APP_KEYCLOAK_URL: !!process.env.REACT_APP_KEYCLOAK_URL,
		REACT_APP_KEYCLOAK_REALM: !!process.env.REACT_APP_KEYCLOAK_REALM,
		REACT_APP_KEYCLOAK_CLIENT_ID: !!process.env.REACT_APP_KEYCLOAK_CLIENT_ID,
	}
});

const keycloak = new Keycloak(keycloakConfig);

export default keycloak;
