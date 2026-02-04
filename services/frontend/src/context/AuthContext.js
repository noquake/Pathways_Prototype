import React, { createContext, useState, useEffect, useRef } from "react";
import keycloak from "../keycloak";

export const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
	const [isAuthenticated, setIsAuthenticated] = useState(false);
	const [userRole, setUserRole] = useState(null);
	const [isInitialized, setIsInitialized] = useState(false);

	// Helper function to update authentication state
	const updateAuthState = (authenticated) => {
		console.log("--- UPDATING AUTH STATE ---", authenticated);
		setIsAuthenticated(authenticated);

		if (authenticated && keycloak.tokenParsed) {
			// 1. View the Raw Token (The confusing string)
			console.log("Raw Token:", keycloak.token);

			// 2. View the Decoded Data (The JSON payload)
			console.log("Decoded Token Data:", keycloak.tokenParsed);

			// Specific fields you might care about:
			console.log("User ID:", keycloak.tokenParsed.sub);
			console.log("Roles:", keycloak.tokenParsed.realm_access?.roles);
			console.log("Expires At:", new Date(keycloak.tokenParsed.exp * 1000));

			const realmRoles = keycloak.tokenParsed?.realm_access?.roles || [];
			const clientRoles =
				keycloak.tokenParsed?.resource_access?.["pathways-keycloak"]
					?.roles || [];

			const allRoles = [...realmRoles, ...clientRoles];
			console.log("User Roles Found:", allRoles);

			if (realmRoles.includes("admin")) setUserRole("admin");
			else if (realmRoles.includes("hr")) setUserRole("hr");
			else if (realmRoles.includes("practitioner"))
				setUserRole("practitioner");
			else setUserRole("user");

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
			setUserRole(null);
		}
	};

	// 1. Create a "ref" to track if we have already run the init
	const isRun = useRef(false);

	useEffect(() => {
		// 2. If we ran this already, STOP.
		if (isRun.current) return;
		isRun.current = true;

		console.log("--- STARTING AUTH CHECK ---");

		const safetyTimer = setTimeout(() => {
			if (!isInitialized) {
				console.warn("Keycloak took too long. Forcing app initialization.");
				setIsInitialized(true);
			}
		}, 5000); // Increased timeout to 5 seconds

		// Set up token refresh handler
		keycloak.onTokenExpired = () => {
			console.log("Token expired, refreshing...");
			keycloak.updateToken(30)
				.then((refreshed) => {
					if (refreshed) {
						console.log("Token refreshed successfully");
						updateAuthState(keycloak.authenticated);
					}
				})
				.catch((error) => {
					console.error("Failed to refresh token:", error);
					updateAuthState(false);
				});
		};

		// Set up authentication success handler
		keycloak.onAuthSuccess = () => {
			console.log("Authentication success detected");
			updateAuthState(true);
		};

		// Set up authentication error handler
		keycloak.onAuthError = (error) => {
			console.error("Authentication error:", error);
			updateAuthState(false);
		};

		keycloak
			.init({
				onLoad: "check-sso",
				checkLoginIframe: false,
				pkceMethod: "S256",
				enableLogging: true,
			})
			.then((authenticated) => {
				clearTimeout(safetyTimer); // Clear the safety timer, we made it!
				console.log(
					"--- INIT COMPLETE. AUTHENTICATED:",
					authenticated,
					"---",
				);

				// CRITICAL FIX: Always set isInitialized to true, regardless of auth status
				updateAuthState(authenticated);
				setIsInitialized(true);

				// If we have a redirect URL with code/fragment, check auth again
				// This handles the case when user comes back from Keycloak login
				if (window.location.hash || window.location.search.includes('code=')) {
					console.log("Detected redirect from Keycloak, re-checking auth...");
					// Small delay to ensure Keycloak has processed the redirect
					setTimeout(() => {
						const currentAuth = keycloak.authenticated;
						console.log("Re-check auth status:", currentAuth);
						updateAuthState(currentAuth);
					}, 100);
				}
			})
			.catch((error) => {
				clearTimeout(safetyTimer);
				console.error("Keycloak initialization error:", error);
				// Even on error, set initialized so app doesn't hang
				setIsInitialized(true);
				setIsAuthenticated(false);
			});
	}, []);

	// Additional effect to check auth state after redirect from Keycloak
	useEffect(() => {
		// Only check if already initialized
		if (!isInitialized) return;

		// Check if we're coming back from Keycloak (has hash or code in URL)
		const hasKeycloakRedirect = window.location.hash || window.location.search.includes('code=');
		
		if (hasKeycloakRedirect) {
			console.log("Detected Keycloak redirect, checking authentication state...");
			// Wait a bit for Keycloak to process the redirect
			const checkAuth = setTimeout(() => {
				const currentAuth = keycloak.authenticated;
				console.log("Post-redirect auth check:", currentAuth);
				if (currentAuth !== isAuthenticated) {
					updateAuthState(currentAuth);
				}
				// Clean up URL
				if (window.location.hash) {
					window.history.replaceState(null, null, window.location.pathname + window.location.search);
				}
			}, 200);

			return () => clearTimeout(checkAuth);
		}
	}, [isInitialized, isAuthenticated]);

	const login = () => keycloak.login();

	const logout = () => {
		keycloak.logout({
			redirectUri: "http://localhost:3000/",
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
