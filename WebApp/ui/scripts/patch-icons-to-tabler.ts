import { readdir, readFile, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import tablerModule from "@tabler/icons-react";

const lucideToTabler: Record<string, string> = {
	MoreHorizontal: "IconArrowsHorizontal",
	PanelLeft: "IconLayoutSidebar",
};

// Utility: get all files matching extensions (optionally recursive)
async function getSourceFiles(
	dir: string,
	exts = [".js", ".jsx", ".ts", ".tsx"],
	recursive = true,
): Promise<string[]> {
	const entries = await readdir(dir, { withFileTypes: true });
	const files: string[] = [];
	for (const entry of entries) {
		const fullPath = path.join(dir, entry.name);
		if (entry.isDirectory() && recursive) {
			files.push(...(await getSourceFiles(fullPath, exts, recursive)));
		} else if (entry.isFile() && exts.some((ext) => entry.name.endsWith(ext))) {
			files.push(fullPath);
		}
	}
	return files;
}

async function patchFile(filePath: string) {
	console.log(`Patching icons in ${filePath}`, path.resolve(filePath));
	const raw = await readFile(filePath, "utf-8");
	const lucideImportRegex =
		/import\s+\{([^}]+)\}\s+from\s+['"]lucide-react['"];{0,1}/;
	const match = raw.match(lucideImportRegex);

	if (!match) {
		console.error(`No lucide-react import found in ${filePath}`);
		return;
	}

	const iconsUsed = match[1]
		.split(",")
		.map((s) => s.trim())
		.filter(Boolean);
	const iconMap: Record<string, string> = {};
	const tablerNames: string[] = [];
	const missingIcons: string[] = [];

	for (const lucideIcon of iconsUsed) {
		const iconName = lucideIcon.replace(/Icon/, "");
		const tablerIconName = `Icon${iconName}`;
		// console.log(tablerIconName)
		if (tablerIconName in tablerModule) {
			iconMap[lucideIcon] = tablerIconName;
			tablerNames.push(tablerIconName);
			continue;
		}
		if (lucideToTabler[iconName] && lucideToTabler[iconName] in tablerModule) {
			iconMap[lucideIcon] = lucideToTabler[iconName];
			tablerNames.push(lucideToTabler[iconName]);
			continue;
		}
		missingIcons.push(iconName);
	}

	if (missingIcons.length) {
		throw new Error(
			`File: ${filePath}\nIcons missing in Tabler or mapping: ${missingIcons.join(", ")}`,
		);
	}

	let patched = raw.replace(
		lucideImportRegex,
		`import { ${tablerNames.join(", ")} } from '@tabler/icons-react';`,
	);

	for (const [lucide, tabler] of Object.entries(iconMap)) {
		const usageRegex = new RegExp(`\\b${lucide}\\b`, "g");
		patched = patched.replace(usageRegex, tabler);
	}

	await writeFile(filePath, patched, "utf-8");
	console.log(`Patched ${filePath} to use @tabler/icons-react`);
}

async function main() {
	const inputPath = process.argv[2];
	if (!inputPath) {
		console.error("Usage: patch-icons.ts <ComponentFileOrFolder>");
		process.exit(1);
	}
	const stats = await stat(inputPath);
	let files: string[] = [];
	if (stats.isDirectory()) {
		files = await getSourceFiles(inputPath);
	} else {
		files = [inputPath];
	}
	for (const file of files) {
		try {
			await patchFile(file);
		} catch (err: unknown) {
			console.error((err as Error)?.message);
			process.exit(1);
		}
	}
}

main().catch((err) => {
	console.error(err);
	process.exit(1);
});
