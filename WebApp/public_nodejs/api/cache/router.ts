import z from "zod";
import { retry } from "../middlewares/retry";
import { pub } from "../orpc";
import { cacheService } from "./service";

export const cache_wskaznikow = pub
  .use(retry({ times: 3 }))
  .route({
    method: "GET",
    path: "/cache_wskaznikow",
    summary: "cache_wskaznikow",
    tags: ["cache_wskaznikow"],
  })
  .input(z.object({ pkds: z.string().array() }))
  .handler(async ({ context, input }) => {
    console.log(input.pkds);
    const data = await cacheService.getAllWithPKD(input.pkds);
    return data;
  });
