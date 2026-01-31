// src/keycloak.js
import Keycloak from "keycloak-js";

// Configure these details to match your Keycloak server
const keycloakConfig = {
	url: "http://localhost:8080", // Your Keycloak URL
	realm: "pathways", // Your Realm Name
	clientId: "account-console", // Your Client ID
};

const keycloak = new Keycloak(keycloakConfig);

export default keycloak;
