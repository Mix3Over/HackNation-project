import { eq, inArray, sql } from "drizzle-orm";
import { db } from "@/db";
import { user } from "./schema";
import type { UpdatedUser, User } from "./validator";

const userPrepareGetById = db.query.user
	.findFirst({
		where: eq(user.id, sql.placeholder("id")),
	})
	.prepare();

async function getById(id: string) {
	const user = await userPrepareGetById.execute({ id });
	if (!user)
		throw new Error(`[UserService]: Could not find user with id ${id}`);
	return user;
}

async function create(userData: Partial<User>) {
	// const newUser = await authDBAdapter.createUser?.({
	//   emailVerified: new Date(),
	//   ...userData,
	// } as AdapterUser);
	// if (!newUser)
	//   throw new Error(
	//     `[UserService]: Could not create user with email ${userData.email}`,
	//   );
	// return newUser;
}

async function deleteById(id: string) {
	return await db.delete(user).where(eq(user.id, id));
}

async function update(userData: UpdatedUser) {
	const { id, ...dataToUpdate } = userData;
	const updatedUser = await db
		.update(user)
		.set(dataToUpdate)
		.where(eq(user.id, id))
		.returning();
	if (!updatedUser[0])
		throw new Error(`[UserService]: Could not update user with id ${id}`);
	return updatedUser[0];
}

// compile query ahead of time
const userPrepareGetByEmail = db.query.user
	.findFirst({
		where: eq(user.email, sql.placeholder("email")),
	})
	.prepare();

async function getByEmail(email: string): Promise<User> {
	const user = await userPrepareGetByEmail.execute({ email });
	if (!user)
		throw new Error(`[UserService]: Could not find user with email ${email}`);
	return user;
}

// compile query ahead of time
const userPrepareGetManyById = db
	.select()
	.from(user)
	.where(inArray(user.id, sql.placeholder("ids")))
	.prepare();

async function getManyByIds(ids: number[]): Promise<User[]> {
	const users = await userPrepareGetManyById.execute({ ids });
	if (users.length !== ids.length)
		throw Error(`[UserService]: Could not find users with ids ${ids}`);
	return users;
}

const userService = {
	getById,
	create,
	deleteById,
	update,
	getByEmail,
	getManyByIds,
};

export default userService;
