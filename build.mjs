import { build } from "esbuild";
import { copyFileSync, mkdirSync } from "node:fs";

await build({
  entryPoints: ["src/main.js"],
  bundle: true,
  minify: false,
  sourcemap: true,
  outfile: "dist/app.js",
  format: "esm",
  target: ["es2020"],
  loader: { ".js": "jsx" },
  jsx: "automatic",
});

mkdirSync("dist", { recursive: true });
copyFileSync("dist/app.js", "app.js");
copyFileSync("dist/app.js.map", "app.js.map");

console.log("Built dist/app.js and root app.js fallback");
