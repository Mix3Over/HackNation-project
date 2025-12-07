import * as fs from "node:fs";
import * as path from "node:path";
import { entries, default as viteConfig } from "../vite.config";

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
  const packageJsonPath = path.resolve("package.json");
  const packageJsonRaw = fs.readFileSync(packageJsonPath, "utf-8");
  const packageJson = JSON.parse(packageJsonRaw);
  packageJson.exports = {} as Record<string, { types: string, import: string }>
  packageJson.files = ["dist"]
  for (let [key, value] of Object.entries(entries)) {
    key = key.replace("components/ui/", "");
    value = path.relative(process.cwd(), value);
    const importVal = value.replace(/^lib\//, "./dist/").replace(/\.tsx*/, ".js")
    const typesVal = value.replace(/^lib\//, "./dist/").replace(/\.tsx*/, ".d.ts")
    if (key === "main") {
      packageJson.types = typesVal
      packageJson.module = importVal
      packageJson.exports['.'] = {
        import: importVal,
        types: typesVal
      }
      continue
    }

    packageJson.exports[`./${key}`] = {
      import: importVal,
      types: typesVal
    }

  }
  console.log("package.json updated")
  fs.writeFileSync(`${packageJsonPath}`, JSON.stringify(packageJson, null, 2));
};

generateTsAliases(entriesData);
