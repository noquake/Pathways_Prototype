import React, { createContext, useState, useEffect, useRef } from "react";
import keycloak from "../keycloak";

export const AuthContext = createContext();

// Authentication Flow:
// 1. User clicks login -> keycloak.login() redirects to Keycloak
// 2. User authenticates -> Keycloak redirects back with authorization code (PKCE)
// 3. Keycloak JS library should automatically exchange code for tokens during init
// 4. We check keycloak.authenticated to determine if user is logged in
// 5. If authenticated, we extract roles from keycloak.tokenParsed

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

		// NOTE: Keycloak login/auth features temporarily disabled.
		// To re-enable, remove this block and uncomment the Keycloak init below.
		setIsAuthenticated(false);
		setIsInitialized(true);
		return;

		console.log("--- STARTING AUTH CHECK ---");
		console.log("Current URL:", window.location.href);
		console.log("Has code in query:", window.location.search.includes('code='));
		console.log("Has code in hash:", window.location.hash.includes('code='));
		console.log("Has hash in URL:", !!window.location.hash);
		console.log("Hash content:", window.location.hash);
		
		// Keycloak properties might not be available until after init
		// Log what we can access
		console.log("Keycloak instance:", keycloak);
		console.log("Keycloak config (before init):", {
			url: keycloak?.authServerUrl || keycloak?.url || "not set",
			realm: keycloak?.realm || "not set",
			clientId: keycloak?.clientId || "not set"
		});

		const safetyTimer = setTimeout(() => {
			if (!isInitialized) {
				console.warn("Keycloak took too long. Forcing app initialization.");
				setIsInitialized(true);
			}
		}, 5000);

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
			console.log("Authentication success detected via callback");
			updateAuthState(true);
		};

		// Set up authentication error handler
		keycloak.onAuthError = (error) => {
			console.error("Authentication error:", error);
			updateAuthState(false);
		};

		// Check if we're coming back from a login redirect
		// PKCE codes can be in either query string or hash
		const hasRedirectCode = window.location.search.includes('code=') || window.location.hash.includes('code=');
		
		// Use appropriate initialization mode
		const initOptions = {
			checkLoginIframe: false,
			pkceMethod: "S256",
			enableLogging: true,
		};

		// IMPORTANT: When we have a redirect code, we should let Keycloak process it
		// Using 'check-sso' should work, but if the code is in the hash, Keycloak JS
		// should automatically process it during init
		// However, if that's not working, we might need to use a different approach
		initOptions.onLoad = "check-sso";
		
		// If we have a code in the hash, ensure responseMode is set correctly
		if (window.location.hash.includes('code=')) {
			// Code is in hash, which is the default for PKCE
			console.log("Code detected in hash - Keycloak should process automatically");
		}

		console.log("Initializing Keycloak with options:", initOptions);

		keycloak
			.init(initOptions)
			.then((authenticated) => {
				clearTimeout(safetyTimer);
				console.log("--- INIT COMPLETE. AUTHENTICATED:", authenticated, "---");
				console.log("Keycloak.authenticated property:", keycloak.authenticated);
				console.log("Keycloak.token exists:", !!keycloak.token);
				console.log("Keycloak.tokenParsed exists:", !!keycloak.tokenParsed);
				console.log("Keycloak object:", {
					authenticated: keycloak.authenticated,
					hasToken: !!keycloak.token,
					hasTokenParsed: !!keycloak.tokenParsed,
					realm: keycloak.realm,
					clientId: keycloak.clientId
				});

				// CRITICAL FIX: Always set isInitialized to true, regardless of auth status
				const actualAuth = keycloak.authenticated || authenticated;
				console.log("Using actual auth state:", actualAuth);
				
				// Always set initialized, but if we have a redirect code, keep checking
				updateAuthState(actualAuth);
				setIsInitialized(true);

				// If we have a redirect code but aren't authenticated yet,
				// Keycloak might still be processing the callback
				if (hasRedirectCode && !actualAuth) {
					console.log("WARNING: Has redirect code but not authenticated. Waiting for callback processing...");
					// Give Keycloak more time to process the PKCE callback
					// Try multiple times with increasing delays
					const checkDelayedAuth = (attempt = 1) => {
						setTimeout(() => {
							const delayedAuth = keycloak.authenticated;
							const hasToken = !!keycloak.token;
							console.log(`Delayed auth check after redirect (attempt ${attempt}):`, {
								authenticated: delayedAuth,
								hasToken: hasToken,
								tokenPreview: keycloak.token ? keycloak.token.substring(0, 20) + "..." : "none"
							});
							if (delayedAuth && hasToken) {
								console.log("SUCCESS: Authentication detected after redirect!");
								updateAuthState(delayedAuth);
							} else if (attempt < 10) {
								// Try again (up to 10 times = 5 seconds)
								checkDelayedAuth(attempt + 1);
							} else {
								// Last attempt failed
								console.error("Failed to authenticate after redirect. Token exchange may have failed.");
								console.error("Keycloak state:", {
									authenticated: keycloak.authenticated,
									hasToken: !!keycloak.token,
									hasTokenParsed: !!keycloak.tokenParsed,
									realm: keycloak.realm,
									clientId: keycloak.clientId
								});
							}
						}, attempt * 500); // 500ms, 1000ms, 1500ms, etc.
					};
					checkDelayedAuth();
				}

				// Additional checks after initialization
				if (hasRedirectCode) {
					console.log("Detected redirect from Keycloak, performing additional checks...");
					// Multiple checks with increasing delays
					[200, 500, 1000, 2000, 3000].forEach((delay) => {
						setTimeout(() => {
							const currentAuth = keycloak.authenticated;
							const hasToken = !!keycloak.token;
							console.log(`Re-check auth status (${delay}ms):`, {
								authenticated: currentAuth,
								hasToken: hasToken
							});
							if (currentAuth && hasToken) {
								console.log("SUCCESS: Found authentication during re-check!");
								updateAuthState(currentAuth);
							}
						}, delay);
					});
				} else {
					// Double-check auth state after a short delay
					setTimeout(() => {
						const currentAuth = keycloak.authenticated;
						if (currentAuth !== actualAuth) {
							console.log("Auth state changed after init, updating...", currentAuth);
							updateAuthState(currentAuth);
						}
					}, 500);
				}
			})
			.catch((error) => {
				clearTimeout(safetyTimer);
				console.error("Keycloak initialization error:", error);
				// Safely access error properties
				console.error("Error details:", {
					message: error?.message || String(error) || "Unknown error",
					stack: error?.stack || "No stack trace",
					error: error,
					errorType: typeof error
				});
				
				// IMPORTANT: Even if init fails, check if we actually have a token
				// Sometimes Keycloak JS processes the callback but the promise rejects
				// Check immediately and then again after delays
				const checkTokenAfterError = (attempt = 1) => {
					setTimeout(() => {
						const hasToken = !!keycloak.token;
						const isAuth = keycloak.authenticated;
						const hasTokenParsed = !!keycloak.tokenParsed;
						
						console.log(`Post-error auth check (attempt ${attempt}):`, {
							authenticated: isAuth,
							hasToken: hasToken,
							tokenParsed: hasTokenParsed,
							tokenPreview: keycloak.token ? keycloak.token.substring(0, 30) + "..." : "none",
							keycloakState: {
								realm: keycloak.realm,
								clientId: keycloak.clientId,
								url: keycloak.authServerUrl
							}
						});
						
						if (isAuth && hasToken) {
							console.log("SUCCESS: Found authentication despite init error!");
							updateAuthState(true);
							setIsInitialized(true);
						} else if (attempt < 5) {
							// Keep checking - token might still be processing
							checkTokenAfterError(attempt + 1);
						} else {
							// Final check - check if we have a redirect code
							const hasCode = window.location.hash.includes('code=') || window.location.search.includes('code=');
							if (hasCode) {
								console.error("=== TOKEN EXCHANGE FAILED ===");
								console.error("Has redirect code but no token after multiple attempts.");
								console.error("Token POST returned 200 OK, but token wasn't stored.");
								console.error("");
								console.error("Possible causes:");
								console.error("1. Client ID mismatch:");
								console.error("   - Docker-compose sets: pathways-frontend");
								console.error("   - Keycloak.js defaults to: pathways-keycloak");
								console.error("   - Check if REACT_APP_KEYCLOAK_CLIENT_ID env var is loaded");
								console.error("2. Redirect URI mismatch in Keycloak client config");
								console.error("   - Should match:", window.location.origin + window.location.pathname);
								console.error("3. CORS issue preventing response body from being read");
								console.error("4. Token response format doesn't match Keycloak JS expectations");
								console.error("");
								console.error("To fix:");
								console.error("- Verify client ID in Keycloak admin console matches the one used");
								console.error("- Check Network tab -> token request -> Response tab to see actual token");
								console.error("- Ensure redirect URI is configured in Keycloak client");
							}
							setIsInitialized(true);
							setIsAuthenticated(false);
						}
					}, attempt * 300); // 300ms, 600ms, 900ms, 1200ms, 1500ms
				};
				
				// Check if we have a redirect code - if so, try re-initializing
				const hasCode = window.location.hash.includes('code=') || window.location.search.includes('code=');
				
				if (hasCode) {
					console.log("=== ATTEMPTING RECOVERY ===");
					console.log("Init failed but we have a code. Trying alternative initialization...");
					
					// Try re-initializing with login-required mode to force processing the code
					// This sometimes works when check-sso fails
					setTimeout(() => {
						console.log("Attempting re-init with login-required mode...");
						keycloak.init({
							onLoad: "login-required",
							checkLoginIframe: false,
							pkceMethod: "S256",
							enableLogging: true,
						})
						.then((authenticated) => {
							console.log("Re-init successful! Authenticated:", authenticated);
							if (authenticated) {
								updateAuthState(true);
								setIsInitialized(true);
							} else {
								// Still not authenticated, proceed with normal error handling
								checkTokenAfterError();
							}
						})
						.catch((retryError) => {
							console.error("Re-init also failed:", retryError);
							// Proceed with normal error handling
							checkTokenAfterError();
						});
					}, 500);
				} else {
					// No code, just proceed with normal error handling
					checkTokenAfterError();
				}
			});
	}, []);

	// Additional effect to check auth state after redirect from Keycloak
	useEffect(() => {
		// Only check if already initialized
		if (!isInitialized) return;

		// Function to manually check authentication state with detailed logging
		const checkAuthState = (label = "Manual check") => {
			const currentAuth = keycloak.authenticated;
			const hasToken = !!keycloak.token;
			const hasTokenParsed = !!keycloak.tokenParsed;
			
			console.log(`${label}:`, {
				authenticated: currentAuth,
				hasToken: hasToken,
				hasTokenParsed: hasTokenParsed,
				currentState: isAuthenticated,
				tokenPreview: keycloak.token ? keycloak.token.substring(0, 20) + "..." : "none"
			});
			
			if (currentAuth !== isAuthenticated) {
				console.log("Auth state mismatch detected! Updating from", isAuthenticated, "to", currentAuth);
				updateAuthState(currentAuth);
			} else if (currentAuth && !hasTokenParsed) {
				console.log("WARNING: Authenticated but no token parsed. This might indicate an issue.");
			}
		};

		// Check if we're coming back from Keycloak (has hash or code in URL)
		// PKCE codes can be in either query string or hash
		const hasKeycloakRedirect = window.location.hash.includes('code=') || window.location.search.includes('code=');
		
		if (hasKeycloakRedirect) {
			console.log("=== DETECTED KEYCLOAK REDIRECT ===");
			console.log("URL:", window.location.href);
			console.log("Search:", window.location.search);
			console.log("Hash:", window.location.hash);
			
			// More aggressive checking with multiple attempts
			const delays = [100, 300, 500, 1000, 2000, 3000];
			const timeouts = delays.map(delay => 
				setTimeout(() => checkAuthState(`Post-redirect check (${delay}ms)`), delay)
			);

			// Clean up URL after checking (but keep code if still processing)
			setTimeout(() => {
				if (window.location.hash && keycloak.authenticated) {
					console.log("Cleaning up URL hash");
					window.history.replaceState(null, null, window.location.pathname + window.location.search);
				}
			}, 4000);

			return () => {
				timeouts.forEach(clearTimeout);
			};
		}

		// Also check on window focus (happens after redirect)
		const handleFocus = () => {
			console.log("Window focused, checking auth state...");
			setTimeout(() => checkAuthState("Window focus check"), 100);
		};

		window.addEventListener('focus', handleFocus);
		
		// Periodic check every 2 seconds for the first 10 seconds after initialization
		// This helps catch authentication state changes after redirect
		let checkCount = 0;
		const periodicCheck = setInterval(() => {
			if (checkCount < 5) {
				checkAuthState(`Periodic check #${checkCount + 1}`);
				checkCount++;
			} else {
				clearInterval(periodicCheck);
			}
		}, 2000);

		return () => {
			window.removeEventListener('focus', handleFocus);
			clearInterval(periodicCheck);
		};
	}, [isInitialized, isAuthenticated]);

	const login = () => {
		// Check if Keycloak object exists and has login method
		if (!keycloak) {
			console.error("Keycloak object not available");
			return;
		}
		
		if (typeof keycloak.login !== 'function') {
			console.error("Keycloak login method not available. Keycloak may not be initialized.");
			console.log("Keycloak state:", {
				hasLogin: typeof keycloak.login === 'function',
				hasInit: typeof keycloak.init === 'function',
				isInitialized: isInitialized
			});
			
			// If not initialized, try to initialize first
			if (!isInitialized) {
				console.log("Attempting to initialize Keycloak before login...");
				keycloak.init({
					onLoad: "check-sso",
					checkLoginIframe: false,
					pkceMethod: "S256",
					enableLogging: true,
				})
				.then(() => {
					console.log("Keycloak initialized, attempting login...");
					if (typeof keycloak.login === 'function') {
						keycloak.login();
					} else {
						console.error("Keycloak login still not available after init");
					}
				})
				.catch((error) => {
					console.error("Failed to initialize Keycloak:", error);
				});
			}
			return;
		}
		
		try {
			keycloak.login();
		} catch (error) {
			console.error("Error calling keycloak.login():", error);
			console.error("Error details:", {
				message: error?.message,
				stack: error?.stack,
				error: error
			});
		}
	};

	const logout = () => {
		// Check if Keycloak is initialized before calling logout
		if (!keycloak || !keycloak.logout) {
			console.error("Keycloak not initialized. Cannot logout.");
			return;
		}
		
		try {
			keycloak.logout({
				redirectUri: "http://localhost:3000/",
			});
		} catch (error) {
			console.error("Error calling keycloak.logout():", error);
		}
	};

	return (
		<AuthContext.Provider
			value={{ isAuthenticated, userRole, isInitialized, login, logout }}
		>
			{children}
		</AuthContext.Provider>
	);
};
