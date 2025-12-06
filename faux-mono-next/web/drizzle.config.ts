import type { Config } from "drizzle-kit";

import { env } from "@/env";

export default {
	schema: "./db/schema.ts",
	dialect: "sqlite",
	dbCredentials: {
		url: env.DATABASE_URL,
	},
	tablesFilter: [`${env.DATABASE_PREFIX}*`],
} satisfies Config;
