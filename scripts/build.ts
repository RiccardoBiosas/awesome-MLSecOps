import { spawn } from "node:child_process";

const buildDate = process.env.BUILD_DATE ?? new Date().toISOString();
const astroCli = new URL("../node_modules/astro/bin/astro.mjs", import.meta.url);
const child = spawn(process.execPath, [astroCli.pathname, "build"], {
  env: { ...process.env, BUILD_DATE: buildDate },
  stdio: "inherit",
});

child.on("exit", (code) => process.exit(code ?? 1));