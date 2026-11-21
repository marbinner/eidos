# eidos

Interactive, GPU-rendered explorations of linear algebra, calculus, statistics, dynamics
and neural networks. Written in Rust on wgpu, compiled to WebAssembly; it runs entirely
in the browser (WebGPU, falling back to WebGL2).

**Open it: https://marbinner.github.io/eidos/**

Every scene has its own address, for example
https://marbinner.github.io/eidos/#mlp (the network playground) or
https://marbinner.github.io/eidos/#hermite_orth. Press F1 for frame timings and to see
whether a GPU is being used; `diag.html` reports the graphics hardware.

This branch holds the built site only.
