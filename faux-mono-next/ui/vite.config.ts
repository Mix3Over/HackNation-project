import path, { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { ExportsConfig } from "@bluevoid/cli";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import fg from "fast-glob";
import preserveDirectives from "rollup-preserve-directives";
import { defineConfig } from "vite";
import dts from "vite-plugin-dts";
import tsconfigPaths from "vite-tsconfig-paths";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// type BundleType = {
//   [key: string]: {
//     type: 'chunk' | string,
//     code: any,
//     fileName: string,
//     name: string,
//     modules: any,
//     imports: any,
//     dynamicImports: any,
//     exports: string[],
//     isEntry: boolean,
//     facadeModuleId: any,
//     isDynamicEntry: boolean,
//     moduleIds: any,
//     map: any,
//     sourcemapFileName: null,
//     preliminaryFileName: string,
//     viteMetadata: any
//   },
// }

// const defaultConfig = {
//   inDir: "lib",
//   outDir: "dist"
// }

// function modifyOnBuildPlugin() {
//   return {
//     name: 'modify-on-build-plugin',
//     // transform(_code: string, id: string, _moduleInfo: any) {
//     //   if (!id.includes("node_modules") && id.includes(defaultConfig.inDir)) {
//     //     console.log(path.relative(process.cwd(), id))
//     //   }
//     //   // if (id.endsWith('.ts')) {
//     //   //   // Modify JS files, example: replace console logs
//     //   //   return code.replace(/console\.log/g, 'console.debug');
//     //   // }
//     //   // return null;
//     // },
//     generateBundle(options: any, bundle: BundleType, t3: any) {
//       console.log(options, t3)
//       defaultConfig.inDir += " "
//       for (const key in bundle) {
//         const obj = bundle[key as keyof typeof bundle]
//         let str = ""
//         for (const key2 in obj) {
//           if (key2 === "code") continue
//           const val = obj[key2 as keyof typeof obj]
//           if (typeof val === "string") str += `${val}, `
//         }
//         console.log(str)
//       }
//       // for (const fileName in bundle) {
//       //   const file = bundle[fileName];
//       //   // Access or modify emitted files here (e.g., minify, inject code)
//       //   if (file.type === 'chunk') {
//       //     file.code = file.code.replace(/somePattern/g, 'replacement');
//       //   }
//       // }
//     }
//   };
// }

// Generate individual entries for each UI component
const uiComponents = fg.sync("lib/components/ui/**/*.tsx").reduce(
	(entries, file) => {
		const relativePath = path.relative("lib", file);
		const entryName = relativePath.replace(/\.[tj]sx$/, ""); // Remove file extension
		entries[entryName] = fileURLToPath(new URL(file, import.meta.url));
		return entries;
	},
	{} as Record<string, string>,
);

// Add main entry for the rest of the library
export const entries = {
	...uiComponents,
	main: resolve(__dirname, "lib/main.tsx"),
	utils: resolve(__dirname, "lib/utils.ts"),
	hooks: resolve(__dirname, "lib/hooks.ts"),
};

export const exportsConfig: ExportsConfig = {
	moreTsPaths: {
		"@/*": ["./src/*"],
	},
};

console.log(
	"Entries:",
	Object.keys(entries).map((val) => val.split("/").at(-1)),
);

// https://vite.dev/config/
export default defineConfig({
	plugins: [
		tailwindcss(),
		react(),
		tsconfigPaths(),
		dts({ tsconfigPath: "./tsconfig.app.json", exclude: ["src"] }),
	],
	build: {
		lib: {
			entry: entries,
			name: "@bluevoid/ui",
			formats: ["es"],
			fileName: (_format, entryName) => `${entryName}.js`,
		},
		rollupOptions: {
			external: [
				"@hookform/resolvers",
				"@mantine/hooks",
				"@tabler/icons-react",
				"date-fns",
				"embla-carousel-react",
				"react-day-picker",
				"react-dom",
				"react-hook-form",
				"react-resizable-panels",
				"react",
				"react/jsx-runtime",
				"recharts",
				"tinycolor2",
			],
			output: {
				chunkFileNames: "internal-chunk-[hash].js",
			},
			plugins: [preserveDirectives()],
		},
	},
	resolve: {
		alias: {
			"@tabler/icons-react": "@tabler/icons-react/dist/esm/icons/index.mjs",
		},
	},
});
