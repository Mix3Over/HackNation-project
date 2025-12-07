/**
 * Log helpers
 */
const COLORS = {
	red: "\x1b[31m",
	green: "\x1b[32m",
	yellow: "\x1b[33m",
	blue: "\x1b[34m",
	magenta: "\x1b[35m",
	cyan: "\x1b[36m",
	reset: "\x1b[0m",
};

export const info = (message?: any, ...rest: any[]) =>
	console.log(message, ...rest);

export const log = (message?: any, ...rest: any[]) =>
	console.log(COLORS.yellow, message, ...rest, COLORS.reset);

export const error = (message?: any, ...rest: any[]) =>
	console.error(COLORS.red, message, ...rest, COLORS.reset);

export const getPackageManagerRun = () => {
	let pm = process.env.npm_config_user_agent || "";
	if (typeof pm === "string") {
		pm = pm.split("/")[0];
	} else {
		pm = "npm";
	}
	if (pm === "npm") pm += " run";
	return pm;
};
