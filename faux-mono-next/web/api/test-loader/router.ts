import "server-only";
import * as csv from "csv";
import fs from "fs";
import fsp from "fs/promises";
import z from "zod";
import { retry } from "../middlewares/retry";
import { pub } from "../orpc";

// const __dirname = new URL(".", import.meta.url).pathname;

const processFile = async (fileName: string) => {
	const legendY: string[] = [];
	const records = [];
	const parser = fs.createReadStream(fileName).pipe(
		csv.parse({
			// CSV options if any
		}),
	);
	for await (const record of parser) {
		records.push(record);
	}
	const legendX = records[0];
	const width = records[0].length;
	const height = records.length;
	const data: Record<string, any>[] = [];
	for (let y = 1; y < height; y++) {
		legendY.push(records[y][0]);
	}
	let types: string[] = [];
	for (let y = 1; y < height; y++) {
		types.push(records[y][2]);
	}
	types = [...new Set(types)];
	console.log(types);
	for (const type of types) {
		for (let x = 3; x < width; x++) {
			const obj: Record<string, any> = {
				date: legendX[x],
				id: crypto.randomUUID(),
				type,
			};
			for (let y = 1; y < height; y++) {
				if (records[y][2] !== type) {
					continue;
				}
				const num = Number.parseFloat(
					(records[y][x] as string)
						.trim()
						.replace(/\s+/g, "")
						.replace(",", "."),
				);
				obj[records[y][0]] = Number.isNaN(num) ? null : num;
			}
			data.push(obj);
		}
	}
	// console.log(legendX);
	// const parser2 = fs.createReadStream(fileName).pipe(
	// 	csv.parse({
	// 		// CSV options if any
	// 	}),
	// );
	// let header = true;
	// for await (const record of parser2) {
	// 	if (header) {
	// 		header = false;
	// 		continue;
	// 	}
	// 	const [pkd, pkd_disc, pointer, ...rest] = record;
	// 	legendY.push(pkd);
	// 	const obj: Record<string, any> = record.reduce(
	// 		(prev: Record<string, any>, next: string, index: number) => {
	// 			prev[legendX[index]] = next;
	// 			return prev;
	// 		},
	// 		{},
	// 	);
	// 	obj.id = crypto.randomUUID();
	// 	records.push(obj);
	// }
	return { legendX, legendY, data };
};

export const testLoader = pub
	.use(retry({ times: 3 }))
	.route({
		method: "GET",
		path: "/test-loader",
		summary: "",
		tags: ["Test"],
	})
	.output(z.any())
	.handler(async () => {
		const data = await processFile("../testoweDane.csv");
		return data;
	});
