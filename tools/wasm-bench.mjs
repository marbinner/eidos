// Run the training benchmark inside the WebAssembly build, in Node (the same engine
// as Chrome), to compare with `eidos bench-training` run natively.
//
//   node tools/wasm-bench.mjs        (after ./build-web.sh)

import { copyFileSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const pkg = join(dirname(fileURLToPath(import.meta.url)), "..", "web", "pkg");
// the generated loader is an ES module with a .js name; give Node a copy it will read as one
const dir = mkdtempSync(join(tmpdir(), "eidos-bench-"));
try {
  copyFileSync(join(pkg, "eidos.js"), join(dir, "eidos.mjs"));
  const eidos = await import(pathToFileURL(join(dir, "eidos.mjs")).href);
  await eidos.default({ module_or_path: readFileSync(join(pkg, "eidos_bg.wasm")) });
  console.log(eidos.bench_training());
} finally {
  rmSync(dir, { recursive: true, force: true });
}
