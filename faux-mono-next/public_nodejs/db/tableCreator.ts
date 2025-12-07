import { sqliteTableCreator } from "drizzle-orm/sqlite-core";

import { env } from "@/env";

export const createTable = sqliteTableCreator(
	(name) => `${env.DATABASE_PREFIX}${name}`,
);
