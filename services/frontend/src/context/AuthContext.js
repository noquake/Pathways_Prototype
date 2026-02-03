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

		console.log("--- STARTING KEYCLOAK INIT (SINGLE RUN) ---");

		keycloak
			.init({
				onLoad: "check-sso",
				checkLoginIframe: false,
				pkceMethod: "S256",
				responseMode: "query",
				enableLogging: true,
			})
			.then((isAuthenticated) => {
				console.log(
					"--- INIT COMPLETE. AUTHENTICATED:",
					isAuthenticated,
					"---",
				);

				if (isAuthenticated) {
					// 1. View the Raw Token (The confusing string)
					console.log("Raw Token:", keycloak.token);

					// 2. View the Decoded Data (The JSON payload)
					console.log("Decoded Token Data:", keycloak.tokenParsed);

					// Specific fields you might care about:
					console.log("User ID:", keycloak.tokenParsed.sub);
					console.log("Roles:", keycloak.tokenParsed.realm_access?.roles);
					console.log("Expires At:", new Date(keycloak.tokenParsed.exp * 1000));
				}
				setIsAuthenticated(isAuthenticated);

				if (isAuthenticated) {
					const realmRoles = keycloak.tokenParsed?.realm_access?.roles || [];
					const groups = keycloak.tokenParsed?.groups || [];

					if (realmRoles.includes("admin") || groups.includes("admin-group"))
						setUserRole("admin");
					else if (realmRoles.includes("hr") || groups.includes("hr-group"))
						setUserRole("hr");
					else if (
						realmRoles.includes("practitioner") ||
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
			.catch((err) => {
				console.error("KEYCLOAK INIT ERROR:", err);
				setIsInitialized(true);
			});
	}, []);

	const login = () => keycloak.login();
	const logout = () => {
		setIsAuthenticated(false);
		setUserRole(null);
		keycloak.logout({ redirectUri: window.location.origin });
	};

	return (
		<AuthContext.Provider
			value={{ isAuthenticated, userRole, isInitialized, login, logout }}
		>
			{children}
		</AuthContext.Provider>
	);
};
