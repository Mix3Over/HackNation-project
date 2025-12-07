import { createORPCClient } from "@orpc/client";
import { RPCLink } from "@orpc/client/fetch";
import { BatchLinkPlugin } from "@orpc/client/plugins";
import type { RouterClient } from "@orpc/server";
import { createTanstackQueryUtils } from "@orpc/tanstack-query";
import type { router } from "@/api/root";

/**
 * This is part of the Optimize SSR setup.
 *
 * @see {@link https://orpc.unnoq.com/docs/adapters/next#optimize-ssr}
 */
declare global {
	var $client: RouterClient<typeof router> | undefined;
}

const link = new RPCLink({
	url: `${
		typeof window !== "undefined"
			? window.location.origin
			: "http://localhost:3000"
	}/api/rpc`,
	plugins: [
		new BatchLinkPlugin({
			groups: [
				{
					condition: () => true,
					context: {},
				},
			],
		}),
	],
});

export const client: RouterClient<typeof router> =
	globalThis.$client ?? createORPCClient(link);

export const orpc = createTanstackQueryUtils(client);
