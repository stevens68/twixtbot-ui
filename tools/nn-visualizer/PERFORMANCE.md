# NN visualizer performance notes

This document tracks instrumentation, the original baseline, and subsequent optimizations. Rendering now separates persistent board and tensor geometry from move-dependent values: board backing/holes/finish lines/coordinate labels persist; the locations tensor and zero-cell markers for peg/link tensors are built once; occupied peg/link tensor blades update their existing InstancedMeshes; convolution filter geometry is cached while navigating game positions. Dynamic board pieces, labels, and the output feature tensor are still updated or rebuilt as appropriate.

## What is measured

Open the browser developer tools (F12) and select **Console**. The visualizer records structured `[NN perf]` samples:

- `requestSnapshot`: response-header wait, response-body read, JSON parsing, move count, decoded JSON bytes, on-wire bytes, and content encoding for `/api/trace`. The server gzip-compresses JSON snapshots when the browser advertises gzip support. The backend also caches reconstructed game states by move prefix, so navigation can resume from the closest cached position.
- `renderStep`: scene clearing time, scene construction/draw time, total synchronous render time, object/mesh/InstancedMesh counts, unique geometry/material counts, renderer draw calls, triangles, geometries and textures.
- `loadGamePosition`: complete time to fetch the snapshot, replace the data, and rebuild the selected view.
- Label rendering reuses sprite objects and cached canvas textures across renders. Static sample/effective board backing planes, holes, finish lines, and coordinate labels persist. Input tensor zero-cell markers for pegs and links and the full locations tensor are also persistent; occupied peg/link cells update instance matrices and instance counts rather than rebuilding tensor meshes. Convolution filter kernels and bias cubes persist while the weight arrays remain unchanged. The move list is drawn as one canvas-backed sprite per 20-move column, reducing move-list draw calls from O(plies) to O(plies / 20). Dynamic board pegs/links, labels, and the output feature tensor still update or rebuild as appropriate. Board links and effective-board directional pointers use instanced batches so draw-call count depends on player/direction groups rather than the number of links.

The browser console API is available for repeatable runs:

```js
nnPerf.reset()   // start a fresh sample set
nnPerf.report()  // print samples as a table and return them
nnPerf.samples   // inspect/copy raw sample objects
```

These timings cover synchronous JavaScript scene construction and the request/response path. They do not represent GPU completion time; use Chrome DevTools Performance for frame-time profiling if needed.

## Baseline procedure

1. Start the backend and visualizer from the repository root:

   ```powershell
   python tools/nn-visualizer/server.py
   ```

   Open `http://127.0.0.1:8765`.

2. Use a **valid game trajectory** of at least 300 plies, ideally loaded from a real `.T1` file. Do not invent arbitrary coordinates: Twixt legality and the swap rule affect whether the backend can replay a prefix.

3. Measure each target length separately: **10, 50, 100 and 300 moves**. Set the trajectory editor to the first N moves of the same valid game, click **Apply changes**, and wait for the view to finish rendering. The end of that trajectory is rendered automatically. For each length:
   - Run `nnPerf.reset()` in the console before applying the trajectory.
   - Apply the trajectory and wait for rendering to finish.
   - Run `nnPerf.report()` and save/copy the output.
   - Repeat at least three times; treat the first run as warm-up and compare the later runs.

4. Repeat navigation checks on the longest trajectory:
   - Move forward repeatedly through at least 20 plies.
   - Move backward through the same range.
   - Jump to the start and then to the end.
   - Repeat on **board & input tensors**, **conv filters & feature maps**, and **residual filters** tabs.

5. Record the browser, operating system, selected view, game length, and whether the run was warm-up. Compare median `requestSnapshot.totalMs`, `bodyMs`, `jsonParseMs`, `renderStep.totalMs`, `renderStep.drawMs`, and draw calls. Do not compare a run with DevTools closed to one with CPU profiling enabled.

## Current interpretation

The backend now caches up to 2,048 game states, keyed by the normalized move prefix and `allow_scl` setting. The response omits the raw 24×24×11 NAF intermediate because the browser only consumes the three tensors returned by `to_input_arrays()`. The server gzip-compresses API JSON when supported; browser metrics report both decoded `responseBytes` and compressed `wireResponseBytes` (the `Content-Length` header). It looks for the longest cached prefix, clones that state, and applies only the remaining moves. Cached states are cloned before use, so later requests cannot mutate the saved state. The cache is process-local and is cleared when the visualizer server restarts. The adapter log includes `cached_prefix`; compare that with the requested move count to see how much replay was skipped.

The browser now keeps a bounded 128-entry LRU cache of exact `/api/trace` responses keyed by the full move prefix and request options. This lets trajectory validation share the end-position snapshot with the initial end render, and makes revisiting cached positions skip the request and JSON parsing entirely. Cache hits are reported as `snapshotCacheHit`. The cache lives in the browser page and evicts the least-recently-used snapshot when full. The browser still clears and rebuilds the transient scene group after each position change, including dynamic board pieces and labels. Persistent input tensor groups keep zero-cell markers and locations in place; the occupied peg/link instances are updated in place. The filter group is rebuilt only if the weights object changes. `renderStep` now also reports `persistentInputStatic`, `persistentInputDynamic`, and `persistentFilterGeometry` object counts; compare those with `drawMs`, sprite counts, and draw calls.
