import z from "zod";
import { retry } from "@/api/middlewares/retry";
import { priv } from "../orpc";

export const getUserById = priv
	.use(retry({ times: 3 }))
	.route({
		method: "GET",
		path: "/users",
		summary: "get users ",
		tags: ["Users"],
	})
	.input(z.object({ message: z.string() }))
	.output(z.object({ echoedMessage: z.string() }))
	.handler(async ({ input, context }) => {
		return {
			echoedMessage: `Server received: ${input.message} ${JSON.stringify(context.user ?? {})}${context.user.id}`,
		};
	});
