import { createEnv } from "@t3-oss/env-nextjs";
import { z } from "zod";

export const env = createEnv({
	shared: {
		NODE_ENV: z
			.enum(["development", "production", "test"])
			.default("development"),
	},
	server: {
		DATABASE_URL: z.url(),
		DATABASE_PREFIX: z.string().default(""),
		VERCEL_URL: z.string().url().optional(),
		AUTH_SECRET:
			process.env.NODE_ENV === "production"
				? z.string()
				: z.string().optional(),
		AUTH_DISCORD_ID: z.string().optional(),
		AUTH_DISCORD_SECRET: z.string().optional(),
		EMAIL_SERVER_USER: z.string().optional(),
		EMAIL_SERVER_PASSWORD: z.string().optional(),
		EMAIL_SERVER_HOST: z.string().optional(),
		EMAIL_SERVER_PORT: z.string().optional(),
		EMAIL_FROM: z.string().optional(),
		PORT: z.string().optional(),
	},

	client: {
		NEXT_PUBLIC_TITLE: z.string().min(1).optional(),
	},

	/**
	 * What object holds the environment variables at runtime. This is usually
	 * `process.env` or `import.meta.env`.
	 */
	runtimeEnv: {
		AUTH_SECRET: process.env.AUTH_SECRET,
		AUTH_DISCORD_ID: process.env.AUTH_DISCORD_ID,
		AUTH_DISCORD_SECRET: process.env.AUTH_DISCORD_SECRET,
		DATABASE_URL: process.env.DATABASE_URL,
		DATABASE_PREFIX: process.env.DATABASE_PREFIX,
		NODE_ENV: process.env.NODE_ENV,
		VERCEL_URL: process.env.VERCEL_URL,
		EMAIL_SERVER_USER: process.env.EMAIL_SERVER_USER,
		EMAIL_SERVER_PASSWORD: process.env.EMAIL_SERVER_PASSWORD,
		EMAIL_SERVER_HOST: process.env.EMAIL_SERVER_HOST,
		EMAIL_SERVER_PORT: process.env.EMAIL_SERVER_PORT,
		EMAIL_FROM: process.env.EMAIL_FROM,
		PORT: process.env.PORT,
		NEXT_PUBLIC_TITLE: process.env.NEXT_PUBLIC_TITLE,
	},
	/**
	 * By default, this library will feed the environment variables directly to
	 * the Zod validator.
	 *
	 * This means that if you have an empty string for a value that is supposed
	 * to be a number (e.g. `PORT=` in a ".env" file), Zod will incorrectly flag
	 * it as a type mismatch violation. Additionally, if you have an empty string
	 * for a value that is supposed to be a string with a default value (e.g.
	 * `DOMAIN=` in an ".env" file), the default value will never be applied.
	 *
	 * In order to solve these issues, we recommend that all new projects
	 * explicitly specify this option as true.
	 */
	emptyStringAsUndefined: true,

	skipValidation:
		!!process.env.SKIP_ENV_VALIDATION ||
		!!process.env.CI ||
		process.env.npm_lifecycle_event === "lint",
});
