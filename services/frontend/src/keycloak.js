// src/keycloak.js
import Keycloak from "keycloak-js";

// Configure these details to match your Keycloak server
const keycloak = new Keycloak({
	url: "http://localhost:8080", // Your Keycloak URL
	realm: "pathways", // Your Realm Name
	clientId: "pathways-keycloak", // Your Client ID
});

export default keycloak;
