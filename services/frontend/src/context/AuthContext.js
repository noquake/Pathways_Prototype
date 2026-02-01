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

		keycloak
			.init({
				onLoad: "check-sso",
				silentCheckSsoRedirectUri:
					window.location.origin + "/silent-check-sso.html",
			})
			.then((authenticated) => {
				setIsAuthenticated(authenticated);

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
				}
				setIsInitialized(true);
			})
			.catch(console.error);
	}, []);

	const login = () => keycloak.login();
	const logout = () => keycloak.logout();

	return (
		<AuthContext.Provider
			value={{ isAuthenticated, userRole, isInitialized, login, logout }}
		>
			{children}
		</AuthContext.Provider>
	);
};
