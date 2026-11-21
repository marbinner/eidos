// The background half of eidos: a second copy of the program, started as a service
// instead of a window. Training runs here so that it never holds up a frame.
import init, { worker_start } from "./pkg/eidos.js";

init().then(() => worker_start());
