// Drive a headless Chrome over the DevTools protocol: load a page, let it run, print
// its console output, optionally click/drag, and save a screenshot.
//
//   node tools/webshot.mjs --chrome <path> --url <url> --out shot.png
//        [--mode webgpu|webgl] [--wait 6000] [--size 1500x940]
//        [--click x,y]... [--drag x0,y0,x1,y1] [--scroll x,y,dy] [--key F1 [--settle ms]] [--eval 'script'] [--timing 1]
//        [--then scene] (after the wait, go on to this scene, as a click on its name does)
//        [--stats ms]   (print the program's own frame timings, taken ms after that)
//
// No dependencies beyond Node 22 (built-in fetch and WebSocket).

import { spawn } from "node:child_process";
import { writeFileSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const args = process.argv.slice(2);
const opt = { mode: "webgpu", wait: 6000, size: "1500x940", clicks: [] };
for (let i = 0; i < args.length; i += 2) {
  const [k, v] = [args[i].replace(/^--/, ""), args[i + 1]];
  if (k === "click") opt.clicks.push(v.split(",").map(Number));
  else opt[k] = v;
}
if (!opt.chrome || !opt.url || !opt.out) {
  console.error("usage: webshot.mjs --chrome <path> --url <url> --out <png> [--mode webgpu|webgl] [--wait ms]");
  process.exit(2);
}
const [width, height] = opt.size.split("x").map(Number);
const profile = mkdtempSync(join(tmpdir(), "eidos-chrome-"));
const gpuFlags =
  opt.mode === "webgl"
    ? ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--disable-features=WebGPU"]
    : ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-vulkan=swiftshader", "--use-webgpu-adapter=swiftshader",
       "--use-angle=swiftshader", "--enable-unsafe-swiftshader"];
const chrome = spawn(opt.chrome, [
  "--headless", "--no-sandbox", "--remote-debugging-port=0", `--user-data-dir=${profile}`,
  `--window-size=${width},${height}`, "--hide-scrollbars", "--mute-audio", "--no-first-run",
  ...gpuFlags, "about:blank",
]);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const port = await new Promise((resolve, reject) => {
  let buf = "";
  chrome.stderr.on("data", (d) => {
    buf += d;
    const m = buf.match(/DevTools listening on ws:\/\/127\.0\.0\.1:(\d+)\//);
    if (m) resolve(Number(m[1]));
  });
  chrome.on("exit", (c) => reject(new Error(`chrome exited (${c}): ${buf.slice(-400)}`)));
  setTimeout(() => reject(new Error("chrome did not start: " + buf.slice(-400))), 20000);
});

let exitCode = 0;
try {
  const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
  const page = targets.find((t) => t.type === "page");
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });
  let nextId = 1;
  const pending = new Map();
  const problems = [];
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.id && pending.has(msg.id)) {
      const { resolve, reject } = pending.get(msg.id);
      pending.delete(msg.id);
      msg.error ? reject(new Error(msg.error.message)) : resolve(msg.result);
    } else if (msg.method === "Runtime.consoleAPICalled") {
      const text = msg.params.args.map((a) => a.value ?? a.description ?? "").join(" ");
      const kind = msg.params.type;
      if (kind === "error" || kind === "warning") problems.push(`[${kind}] ${text}`);
      console.log(`[console.${kind}] ${text.slice(0, 600)}`);
    } else if (msg.method === "Runtime.exceptionThrown") {
      const d = msg.params.exceptionDetails;
      const text = d.exception?.description ?? d.text;
      problems.push(`[exception] ${text}`);
      console.log(`[exception] ${String(text).slice(0, 800)}`);
    } else if (msg.method === "Log.entryAdded") {
      const e = msg.params.entry;
      if (e.level === "error") problems.push(`[log] ${e.text}`);
      console.log(`[log.${e.level}] ${e.text.slice(0, 600)}`);
    }
  };
  const send = (method, params = {}) =>
    new Promise((resolve, reject) => {
      const id = nextId++;
      pending.set(id, { resolve, reject });
      ws.send(JSON.stringify({ id, method, params }));
    });
  const mouse = (type, x, y, extra = {}) => send("Input.dispatchMouseEvent", { type, x, y, button: "left", clickCount: 1, ...extra });

  await send("Page.enable");
  await send("Runtime.enable");
  await send("Log.enable");
  await send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 1, mobile: false });
  await send("Page.navigate", { url: opt.url });
  if (opt.timing) {
    // how long from asking for the page until the program has started (the
    // "loading" notice is taken away once the first frame is on its way)
    const began = Date.now();
    for (;;) {
      const r = await send("Runtime.evaluate", { expression: `!document.getElementById("loading") && !!document.getElementById("eidos")`, returnByValue: true });
      if (r.result.value || Date.now() - began > 60000) break;
      await sleep(15);
    }
    console.log(`started after ${Date.now() - began} ms`);
  }
  await sleep(Number(opt.wait));

  for (const [x, y] of opt.clicks) {
    await mouse("mouseMoved", x, y, { button: "none" });
    await sleep(120);
    await mouse("mousePressed", x, y);
    await mouse("mouseReleased", x, y);
    await sleep(1500);
  }
  if (opt.drag) {
    const [x0, y0, x1, y1] = opt.drag.split(",").map(Number);
    await mouse("mouseMoved", x0, y0, { button: "none" });
    await sleep(150);
    await mouse("mousePressed", x0, y0);
    for (let i = 1; i <= 24; i++) {
      await mouse("mouseMoved", x0 + ((x1 - x0) * i) / 24, y0 + ((y1 - y0) * i) / 24);
      await sleep(30);
    }
    await mouse("mouseReleased", x1, y1);
    await sleep(900);
  }
  if (opt.scroll) {
    const [x, y, dy] = opt.scroll.split(",").map(Number);
    await send("Input.dispatchMouseEvent", { type: "mouseWheel", x, y, deltaX: 0, deltaY: dy });
    await sleep(900);
  }

  if (opt.key) {
    // press a key (by its name, as in KeyboardEvent.key: "F1", " ", "h"), then let it settle
    const code = opt.key.length === 1 ? `Key${opt.key.toUpperCase()}` : opt.key;
    const vk = opt.key === "F1" ? 112 : opt.key.toUpperCase().charCodeAt(0);
    for (const type of ["keyDown", "keyUp"]) {
      await send("Input.dispatchKeyEvent", { type, key: opt.key, code, windowsVirtualKeyCode: vk, nativeVirtualKeyCode: vk });
    }
    await sleep(Number(opt.settle ?? 2500));
  }
  if (opt.eval) {
    // run a line of script in the page (to provoke something), then let it settle
    await send("Runtime.evaluate", { expression: opt.eval, userGesture: true, awaitPromise: true });
    await sleep(700);
  }

  if (opt.then) {
    // the address names the scene; the program looks at it in its next frame, which a
    // movement of the pointer asks for
    await send("Runtime.evaluate", { expression: `location.hash = ${JSON.stringify("#" + opt.then)}` });
    await mouse("mouseMoved", 700, 500, { button: "none" });
    await sleep(60);
    await mouse("mouseMoved", 702, 501, { button: "none" });
  }
  if (opt.stats) {
    // the program's own account of its frames: asked for once (which starts it keeping
    // one), and read after the scene has run for a while
    const ask = `import("./pkg/eidos.js").then((m) => m.stats())`;
    await send("Runtime.evaluate", { expression: ask, awaitPromise: true, returnByValue: true });
    await sleep(Number(opt.stats));
    const r = await send("Runtime.evaluate", { expression: ask, awaitPromise: true, returnByValue: true });
    console.log("stats:", r.result?.value ?? "");
  }

  const info = await send("Runtime.evaluate", {
    expression: `JSON.stringify({ webgpu: !!navigator.gpu, loading: !!document.getElementById("loading"),
                  fullscreen: !!(document.fullscreenElement || document.webkitFullscreenElement),
                  stopped: !!document.getElementById("stopped"), history: history.length,
                  text: (document.getElementById("loading")?.innerText ?? "").slice(0, 300) })`,
    returnByValue: true,
  });
  console.log("page state:", info.result.value);
  const shot = await send("Page.captureScreenshot", { format: "png" });
  writeFileSync(opt.out, Buffer.from(shot.data, "base64"));
  console.log(`saved ${opt.out}; ${problems.length} console problem(s)`);
  if (problems.length) exitCode = 1;
  ws.close();
} catch (e) {
  console.error("webshot failed:", e.message);
  exitCode = 2;
} finally {
  chrome.kill();
  await sleep(300);
  try { rmSync(profile, { recursive: true, force: true }); } catch {}
}
process.exit(exitCode);
