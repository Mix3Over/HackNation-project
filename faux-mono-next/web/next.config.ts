import type { NextConfig } from "next";
import "./env";
// @ts-check
/**
 * Run `build` or `dev` with `SKIP_ENV_VALIDATION` to skip env validation. This is especially useful
 * for Docker builds.
 */

const PHASE_PRODUCTION_BUILD = "phase-production-build"; // see next/constants

function config(phase: string) {
	let nextConfig: NextConfig = {
		typedRoutes: true,
		reactStrictMode: true,
	};

	if (
		phase === PHASE_PRODUCTION_BUILD &&
		(process.platform === "freebsd" || process.platform === "openbsd")
	) {
		console.log("BUILDING PRODUCTION");
		nextConfig = {
			...nextConfig,
			// disable typecheck when building in prod to speed up deployments
			typescript: {
				ignoreBuildErrors: true,
			},
			experimental: {
				// HACK: limit CPU usage when building in prod(test server cannot handle load)
				// HACK: remove this if we upgrade from small.pl hosting
				workerThreads: false,
				cpus: 1,
			},
		};
	}
	return nextConfig;
}

export default config;
