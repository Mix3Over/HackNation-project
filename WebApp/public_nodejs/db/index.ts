// import Database from "better-sqlite3";

import Database from "better-sqlite3";
import { drizzle } from "drizzle-orm/better-sqlite3";
import { env } from "@/env";

import * as schema from "./schema";

// import { drizzle } from "drizzle-orm/better-sqlite3";

// You can specify any property from the better-sqlite3 connection options
const client = new Database(env.DATABASE_URL);
export const db = drizzle({ client, schema });
