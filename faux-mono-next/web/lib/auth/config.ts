import { betterAuth } from "better-auth";
import { drizzleAdapter } from "better-auth/adapters/drizzle";
import { nextCookies } from "better-auth/next-js";
import { db } from "@/db";
import * as schema from "@/db/schema";
import { env } from "@/env";
import { getBaseUrl } from "@/utils/getBaseUrl";

const socialProviders: Parameters<typeof betterAuth>[0]["socialProviders"] = {};

if (
	env.AUTH_DISCORD_ID !== undefined &&
	env.AUTH_DISCORD_SECRET !== undefined
) {
	socialProviders["discord"] = {
		clientId: env.AUTH_DISCORD_ID,
		clientSecret: env.AUTH_DISCORD_SECRET,
		redirectURI: `${getBaseUrl()}/api/auth/callback/discord`,
	};
}

export const auth = betterAuth({
	plugins: [nextCookies()],
	database: drizzleAdapter(db, {
		provider: "sqlite", // or "pg" or "mysql"
		schema,
	}),
	emailAndPassword: {
		enabled: true,
	},
	socialProviders,
});

export type Session = typeof auth.$Infer.Session;
