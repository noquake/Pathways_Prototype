import { useContext } from "react";
import { AuthContext } from "../context/AuthContext";

export const useAuth = () => {
	const context = useContext(AuthContext);

	if (context === undefined) {
		throw new Error("useAuth must be used within an AuthProvider");
	}

	const { userRole, isAuthenticated, isInitialized, login, logout } = context;

	// Helper flags to make UI logic cleaner
	const isAdmin = userRole === "admin";
	const isHR = userRole === "hr";
	const isPractitioner = userRole === "practitioner";

	return {
		userRole,
		isAuthenticated,
		isInitialized,
		isAdmin,
		isHR,
		isPractitioner,
		login,
		logout,
	};
};
