// Path paths to relative paths

import fsp from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import fg from "fast-glob";

const regexLibLib = /lib\/lib/g;
const regexLibComponentsUi = /lib\/components\/ui/g;

const uiComponents = fg.sync("lib/components/ui/**/*.tsx").reduce(
	(entries, file) => {
		const relativePath = path.relative("lib", file);
		const entryName = relativePath.replace(/\.[tj]sx$/, ""); // Remove file extension
		entries[entryName] = fileURLToPath(new URL(`../${file}`, import.meta.url));
		return entries;
	},
	{} as Record<string, string>,
);

for (const [_relPath, globalFilePath] of Object.entries(uiComponents)) {
	const raw = await fsp.readFile(globalFilePath, "utf-8");

	const newRaw = raw
		.replaceAll(regexLibLib, "../..")
		.replaceAll(regexLibComponentsUi, ".");
	await fsp.writeFile(globalFilePath, newRaw);
}
