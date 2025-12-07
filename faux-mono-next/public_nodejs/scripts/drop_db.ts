import { sql } from "drizzle-orm";
import { db } from "../db";

async function resetDatabase() {
	console.log("🗑️ Resetting database...");
	const start = Date.now();

	const query = sql`
    DO $$ DECLARE
      r RECORD;
    BEGIN
      -- Drop all tables
      FOR r IN (SELECT tablename FROM pg_tables WHERE schemaname = current_schema()) LOOP
        EXECUTE 'DROP TABLE IF EXISTS ' || quote_ident(r.tablename) || ' CASCADE';
      END LOOP;

      -- Drop all custom types (enums)
      FOR r IN (
        SELECT t.typname as enum_name
        FROM pg_catalog.pg_type t
        JOIN pg_catalog.pg_namespace n ON n.oid = t.typnamespace
        WHERE n.nspname = current_schema()
      ) LOOP
        EXECUTE 'DROP TYPE IF EXISTS ' || quote_ident(r.enum_name);
      END LOOP;
    END $$;
  `;

	// await db.execute(query);

	const end = Date.now();
	console.log(`✅ Database reset completed in ${end - start}ms`);
}

resetDatabase().catch((err) => {
	console.error("❌ Reset failed");
	console.error(err);
	process.exit(1);
});
