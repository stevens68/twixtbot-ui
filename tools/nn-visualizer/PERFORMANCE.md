# NN visualizer performance baseline

This is the **measurement-only** first phase of the performance work. It adds console instrumentation without changing the rendering strategy.

## What is measured

Open the browser developer tools (F12) and select **Console**. The visualizer records structured `[NN perf]` samples:

- `requestSnapshot`: request/response time, JSON parsing time, and move count sent to `/api/trace`. The browser currently replays the supplied move prefix on every navigation.
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

5. Record the browser, operating system, selected view, game length, and whether the run was warm-up. Compare median `requestSnapshot.totalMs`, `renderStep.totalMs`, `renderStep.drawMs`, and draw calls. Do not compare a run with DevTools closed to one with CPU profiling enabled.

## Current interpretation

The existing implementation clears the whole scene group and reconstructs the selected view after each position change. The snapshot request also replays the full move prefix. The samples distinguish the network/backend round trip from the browser-side scene rebuild so we can prioritize the next step based on evidence.

No optimization or expected speed-up is assumed by this baseline. Keep the raw samples so they can be compared with later changes.
