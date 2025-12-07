import { relations, sql } from "drizzle-orm";
import { index } from "drizzle-orm/sqlite-core";
import { createTable } from "@/db/tableCreator";
export const user = createTable("user", (d) => ({
	id: d
		.text({ length: 255 })
		.notNull()
		.primaryKey()
		.$defaultFn(() => crypto.randomUUID()),
	name: d.text({ length: 255 }),
	email: d.text({ length: 255 }).notNull().unique(),
	emailVerified: d.integer({ mode: "boolean" }).default(false),
	image: d.text({ length: 255 }),
	createdAt: d
		.integer({ mode: "timestamp" })
		.default(sql`(unixepoch())`)
		.notNull(),
	updatedAt: d.integer({ mode: "timestamp" }).$onUpdate(() => new Date()),
}));

export const userRelations = relations(user, ({ many }) => ({
	account: many(account),
	session: many(session),
}));

export const account = createTable(
	"account",
	(d) => ({
		id: d
			.text({ length: 255 })
			.notNull()
			.primaryKey()
			.$defaultFn(() => crypto.randomUUID()),
		userId: d
			.text({ length: 255 })
			.notNull()
			.references(() => user.id),
		accountId: d.text({ length: 255 }).notNull(),
		providerId: d.text({ length: 255 }).notNull(),
		accessToken: d.text(),
		refreshToken: d.text(),
		accessTokenExpiresAt: d.integer({ mode: "timestamp" }),
		refreshTokenExpiresAt: d.integer({ mode: "timestamp" }),
		scope: d.text({ length: 255 }),
		idToken: d.text(),
		password: d.text(),
		createdAt: d
			.integer({ mode: "timestamp" })
			.default(sql`(unixepoch())`)
			.notNull(),
		updatedAt: d.integer({ mode: "timestamp" }).$onUpdate(() => new Date()),
	}),
	(t) => [index("account_user_id_idx").on(t.userId)],
);

export const accountRelations = relations(account, ({ one }) => ({
	user: one(user, { fields: [account.userId], references: [user.id] }),
}));

export const session = createTable(
	"session",
	(d) => ({
		id: d
			.text({ length: 255 })
			.notNull()
			.primaryKey()
			.$defaultFn(() => crypto.randomUUID()),
		userId: d
			.text({ length: 255 })
			.notNull()
			.references(() => user.id),
		token: d.text({ length: 255 }).notNull().unique(),
		expiresAt: d.integer({ mode: "timestamp" }).notNull(),
		ipAddress: d.text({ length: 255 }),
		userAgent: d.text({ length: 255 }),
		createdAt: d
			.integer({ mode: "timestamp" })
			.default(sql`(unixepoch())`)
			.notNull(),
		updatedAt: d.integer({ mode: "timestamp" }).$onUpdate(() => new Date()),
	}),
	(t) => [index("session_user_id_idx").on(t.userId)],
);

export const sessionRelations = relations(session, ({ one }) => ({
	user: one(user, { fields: [session.userId], references: [user.id] }),
}));

export const verification = createTable(
	"verification",
	(d) => ({
		id: d
			.text({ length: 255 })
			.notNull()
			.primaryKey()
			.$defaultFn(() => crypto.randomUUID()),
		identifier: d.text({ length: 255 }).notNull(),
		value: d.text({ length: 255 }).notNull(),
		expiresAt: d.integer({ mode: "timestamp" }).notNull(),
		createdAt: d
			.integer({ mode: "timestamp" })
			.default(sql`(unixepoch())`)
			.notNull(),
		updatedAt: d.integer({ mode: "timestamp" }).$onUpdate(() => new Date()),
	}),
	(t) => [index("verification_identifier_idx").on(t.identifier)],
);
