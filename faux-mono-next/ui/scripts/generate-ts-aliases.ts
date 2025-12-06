import * as fs from "node:fs";
import * as path from "node:path";
import { entries, default as viteConfig } from "../vite.config";

// const helpExportsConfig = `
// type ExportsConfig = {
//   inDir?: string,
//   outDir?: string,
//   moreFiles?: string[],
//   moreExports?: Record<string, string>
//   moreTsPaths?: Record<string, string[]>
//   overrideModule?: string
//   overrideTypes?: string
// }
// `;

const entriesData = entries as Record<string, string>;
const packageName = viteConfig.build?.lib
	? viteConfig.build.lib.name
	: undefined;

if (!packageName) {
	console.error(`Package name is not present in vite.config
  Add lib config to your project: 
  export default defineConfig({
    build: {
      lib: {
        entry: entries,
        name: 'package-name',
        ...`);
	process.exit(-1);
}

const generateTsAliases = (entries: Record<string, string>) => {
	const tsConfigPath = path.resolve("tsconfig.app.json");
	const tsConfigRaw = fs.readFileSync(tsConfigPath, "utf-8");

	const pathsIndex = tsConfigRaw.indexOf("paths");

	if (pathsIndex === -1) {
		throw new Error("paths not found in tsconfig.app.json");
	}

	const pathsStart = tsConfigRaw.indexOf("{", pathsIndex);
	const pathsEnd = tsConfigRaw.indexOf("}", pathsIndex);

	if (pathsStart === -1 || pathsEnd === -1) {
		throw new Error("paths broken in tsconfig.app.json");
	}

	const raw1 = tsConfigRaw.slice(0, pathsStart + 1);
	const raw2 = tsConfigRaw.slice(pathsEnd);
	let pathsString = "\n";
	for (let [key, value] of Object.entries(entries)) {
		key = key.replace("components/ui/", "");
		value = path.relative(process.cwd(), value);
		if (key === "main") {
			const str = `      "@bluevoid/ui": ["./${value}"],\n`;
			pathsString += str;
			continue;
		}
		const str = `      "@bluevoid/ui/${key}": ["./${value}"],\n`;

		pathsString += str;
	}
	pathsString = `${pathsString}      "@/*": [ "./src/*" ]\n    `;
	// console.log(pathsString)

	fs.writeFileSync(`${tsConfigPath}`, `${raw1}${pathsString}${raw2}`);
};

generateTsAliases(entriesData);
