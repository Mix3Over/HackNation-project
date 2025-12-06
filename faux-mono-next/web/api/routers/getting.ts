import _ from "lodash";
import { z } from "zod";
import { retry } from "@/api/middlewares/retry";
import { priv } from "@/api/orpc";

export const echoHandler = priv
	.use(retry({ times: 3 }))
	.route({
		method: "GET",
		path: "/planets",
		summary: "List all planets",
		tags: ["Planets"],
	})
	.input(z.object({ message: z.string() }))
	.output(z.object({ echoedMessage: z.string() }))
	.handler(async ({ input, context }) => {
		return {
			echoedMessage: `Server received: ${input.message} ${JSON.stringify(context.user ?? {})}${context.user.id}`,
		};
	});
