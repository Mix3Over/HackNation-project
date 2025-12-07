import path from "node:path";
import { defineConfig } from "vite";
import tsconfigPaths from "vite-tsconfig-paths";
import dts from "vite-plugin-dts";
import { builtinModules } from "node:module";
import { fileURLToPath } from "node:url";
import fg from "fast-glob";

const scripts = fg.sync("lib/scripts/**/*.ts").reduce(
	(entries, file) => {
		const relativePath = path.relative("lib", file);
		const entryName = relativePath.replace(/\.[tj]s$/, ""); // Remove file extension
		entries[entryName] = fileURLToPath(new URL(file, import.meta.url));
		return entries;
	},
	{} as Record<string, string>,
);

// Add main entry for the rest of the library
export const entries = {
	...scripts,
	main: path.resolve(__dirname, "lib/main.ts"),
};

// https://vite.dev/config/
export default defineConfig({
	plugins: [tsconfigPaths(), dts()],
	build: {
		lib: {
			entry: entries,
			name: "@bluevoid/cli",
			formats: ["es"],
			fileName: (_format, entryName) => `${entryName}.js`,
		},
		rollupOptions: {
			external: [
				"../dist/scripts/__cli-commands.js",
				...builtinModules.map((m) => `node:${m}`), // use node: prefix for all node modules
			],
			output: {
				chunkFileNames: "internal-chunk-[hash].js",
			},
		},
	},
});
