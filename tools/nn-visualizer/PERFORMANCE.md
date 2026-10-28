# NN visualizer performance notes

This document tracks instrumentation, the original baseline, and subsequent optimizations. The rendering strategy is still unchanged; backend game replay now reuses cached move-prefix positions.

## What is measured

Open the browser developer tools (F12) and select **Console**. The visualizer records structured `[NN perf]` samples:

- `requestSnapshot`: response-header wait, response-body read, JSON parsing, and move count sent to `/api/trace`. The backend now caches reconstructed game states by move prefix, so navigation can resume from the closest cached position.
- `renderStep`: scene clearing time, scene construction/draw time, total synchronous render time, object/mesh/InstancedMesh counts, unique geometry/material counts, renderer draw calls, triangles, geometries and textures.
- `loadGamePosition`: complete time to fetch the snapshot, replace the data, and rebuild the selected view.

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

The backend now caches up to 2,048 game states, keyed by the normalized move prefix and `allow_scl` setting. The response omits the raw 24×24×11 NAF intermediate because the browser only consumes the three tensors returned by `to_input_arrays()`. It looks for the longest cached prefix, clones that state, and applies only the remaining moves. Cached states are cloned before use, so later requests cannot mutate the saved state. The cache is process-local and is cleared when the visualizer server restarts. The adapter log includes `cached_prefix`; compare that with the requested move count to see how much replay was skipped.

The browser still clears and rebuilds the selected scene after each position change. Measure `requestSnapshot.totalMs` separately from `renderStep.totalMs` to determine how much of the remaining delay is rendering rather than backend replay.
