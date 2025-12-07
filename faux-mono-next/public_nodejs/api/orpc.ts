import { os } from "@orpc/server";

import { dbProviderMiddleware } from "./middlewares/db";
import { authMiddleware } from "./middlewares/permissions";

export const pub = os.use(dbProviderMiddleware);
export const priv = os
	.$context<{ headers: Headers }>()
	.use(dbProviderMiddleware)
	.use(authMiddleware);
