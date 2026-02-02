import React, { createContext, useState, useEffect, useRef } from "react";
import keycloak from "../keycloak";

export const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
	const [isAuthenticated, setIsAuthenticated] = useState(false);
	const [userRole, setUserRole] = useState(null);
	const [isInitialized, setIsInitialized] = useState(false);

	// 1. Create a "ref" to track if we have already run the init
	const isRun = useRef(false);

	useEffect(() => {
		// 2. If we ran this already, STOP.
		if (isRun.current) return;
		isRun.current = true;

		console.log("Initializing Keycloak..."); // LOG 1: Prove code is running

		keycloak
			.init({
				// onLoad: "check-sso",
				onLoad: undefined,

				// 1. DISABLE the hidden iframe check (This is the #1 cause of hanging)
				checkLoginIframe: false,
				// 2. REMOVE the silentCheckSsoRedirectUri line entirely for now
				pkceMethod: "S256",
			})
			.then((authenticated) => {
				console.log("Keycloak Init Finished. Authenticated: " + authenticated); // LOG 2: The Verdict
				setIsAuthenticated(true);

				if (authenticated) {
					const roles = keycloak.tokenParsed?.realm_access?.roles || [];
					const groups = keycloak.tokenParsed?.groups || [];

					if (roles.includes("admin") || groups.includes("admin-group"))
						setUserRole("admin");
					else if (roles.includes("hr") || groups.includes("hr-group"))
						setUserRole("hr");
					else if (
						roles.includes("practitioner") ||
						groups.includes("practitioner-group")
					)
						setUserRole("practitioner");

					console.log("Keycloak Login Success!");
					console.log("Full Token Structure:", keycloak.tokenParsed);
					console.log(
						"Realm Roles:",
						keycloak.tokenParsed?.realm_access?.roles,
					);
					console.log(
						"Resource Access:",
						keycloak.tokenParsed?.resource_access,
					);
				} else {
					// ADDED THIS: Now we know if it failed silently
					console.warn("User is NOT authenticated. Redirecting to login...");
				}
				setIsInitialized(true);
			})
			.catch(console.error);
	}, []);

	const login = () => keycloak.login().catch(console.error);
	const logout = () => {
		// 1. Reset local state immediately
		setIsAuthenticated(false);
		setUserRole(null);

		// Call standard Keycloak logout
		keycloak.logout({
			redirectUri: window.location.origin, // Send us back to localhost:3000
			idTokenHint: keycloak.idToken, // CRITICAL: Tells Keycloak "Yes, it's really me, log me out"
		});
	};

	return (
		<AuthContext.Provider
			value={{ isAuthenticated, userRole, isInitialized, login, logout }}
		>
			{children}
		</AuthContext.Provider>
	);
};
