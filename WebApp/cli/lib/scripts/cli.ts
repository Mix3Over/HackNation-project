import fsp from "node:fs/promises";
import path from "node:path";

import { info, log, error, getPackageManagerRun } from "./utils";

/**
 * Constants
 */
const argv = process.argv;
const INIT_CWD = process.env.INIT_CWD ?? process.env.PWD ?? process.cwd();
const isRunningInDev = process.env.INIT_CWD === process.cwd();
const SCRIPT_PATH = isRunningInDev
	? argv[1].slice(0, -6)
	: argv[1].slice(0, -12); // cut "cli.js" or "cli_entry.js"
const relativeRoot = "../../";
const MODULE_PATH = path.join(SCRIPT_PATH, relativeRoot);
const packageJsonFile = await fsp.readFile(`${MODULE_PATH}/package.json`);
const packageJson = JSON.parse(packageJsonFile.toString());
const originName: string | undefined = packageJson.name.split("/")[1];
const currentPackageManagerRun = getPackageManagerRun();
const bluevoidFolderPath = path.join(INIT_CWD, "node_modules", "@bluevoid");

if (isRunningInDev) log(`\nRunning in dev, origin lib is ${packageJson.name}`);

const help = `Version ${packageJson.version}
Usage:\t${currentPackageManagerRun} bluevoid [module] [command] [flags]
\t${currentPackageManagerRun} --help`;

if (argv.length < 3) {
	info(help);
	process.exit(0);
}

// get all installed bluevoid packages
let bluevoidPackageNames: string[] = [];
if (isRunningInDev) {
	bluevoidPackageNames = [originName ?? ""];
} else {
	try {
		const entries = await fsp.readdir(bluevoidFolderPath, {
			withFileTypes: true,
		});
		bluevoidPackageNames = await entries
			.filter((dirent) => dirent.isDirectory() || dirent.isSymbolicLink())
			.map((dirent) => dirent.name);
	} catch (_e) {
		throw new Error(
			"This Command cannot find any bluevoid packages installed.\nThis should not be possible.\nThis command must not be run from global script.",
		);
	}
}

// Load all commands
const commands = await bluevoidPackageNames.reduce(
	async (obj, dirName) => {
		try {
			const lib = await import(
				isRunningInDev && dirName === originName
					? "../dist/scripts/__cli-commands.js"
					: path.join(
							bluevoidFolderPath,
							dirName,
							`/dist/scripts/__cli-commands.js`,
						)
			);
			obj[dirName] = lib;
		} catch (_e) {
			obj[dirName] = {};
		}
		return obj;
	},
	{} as Record<string, any>,
);

async function executeCommand(args: string[]) {
	const [
		libName,
		commandName,
		//  ...restArgs
	] = args;
	try {
		const moreEnv = {
			INIT_CWD,
			isRunningInDev,
			SCRIPT_PATH,
			relativeRoot,
			MODULE_PATH,
			packageJson,
			originName,
			currentPackageManagerRun,
			bluevoidFolderPath,
		};
		const lib = commands[libName as keyof typeof commands];
		const command = lib[commandName as keyof typeof lib];
		command(args, moreEnv);
	} catch (e) {
		error(`Command not found in lib @bluevoid/${libName}`);
	}
}

async function generateHelp(args: string[]) {
	const [
		libName,
		//  commandName,
		//  ...restArgs
	] = args;
	try {
		const lib = commands[libName as keyof typeof commands];
		const helpStr = lib["help" as keyof typeof lib];
		if (helpStr !== undefined) {
			info(`${help}\n\n${helpStr}\n`);
		} else {
			error(`Help not found in lib @bluevoid/${libName}`);
		}
	} catch (e) {
		error(`Help not found in lib @bluevoid/${libName}`);
	}
}
const argvArgs = argv.slice(2);

if (
	argvArgs.includes("--help") ||
	argvArgs.includes("-h") ||
	argvArgs.length < 1
) {
	const argvFilter = argvArgs.filter((arg) => arg !== "--help");
	generateHelp(argvFilter);
} else {
	executeCommand(argvArgs);
}
