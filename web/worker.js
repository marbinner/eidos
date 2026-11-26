// The background half of eidos: a second copy of the program, started as a service
// instead of a window. Training runs here so that it never holds up a frame.
import init, { worker_start } from "./pkg/eidos.js";

// The page's first message is its own compiled program (see spawn in
// src/background.rs): the worker starts from that, so it is the same build as the
// page and nothing is downloaded or compiled twice. If it cannot start, the error is
// thrown where the page hears of it, and the page does the work itself.
self.onmessage = (first) => {
  self.onmessage = null;
  init({ module_or_path: first.data })
    .then(() => worker_start())
    .catch((e) => setTimeout(() => { throw e; }));
};
