import { ORPCError, os } from "@orpc/server";
import { auth } from "../../lib/auth";

/**
 * Server-side oRPC client for direct invocation
 * Use this in Server Components or server actions
 */
export async function getAuthSession(request?: Request) {
	return await auth.api.getSession({ headers: request?.headers ?? [] });
}

type Session = NonNullable<
	Awaited<ReturnType<typeof auth.api.getSession>>
>["session"];
type User = NonNullable<
	Awaited<ReturnType<typeof auth.api.getSession>>
>["user"];
/**
 * Middleware for authenticated requests
 */

export const authMiddleware = os
	.$context<{ headers: Headers }>()
	.middleware(async ({ next, context }) => {
		const sessionData = await auth.api.getSession({ headers: context.headers });
		if (!sessionData) {
			throw new ORPCError("Unauthorized: Authentication required");
		}
		return next({
			context: {
				...context,
				user: sessionData?.user,
				session: sessionData?.session,
			},
		});
	});
