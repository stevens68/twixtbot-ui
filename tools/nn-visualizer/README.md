# Twixtbot 3D neural-network visualizer

A local browser-based Three.js tool for inspecting Twixt game positions, model input tensors, convolution filters, and feature-map values. It is part of `twixtbot-ui`; the Python backend is the source of truth for game rules and input encoding.

The visualizer loads the trained `links` convolution weights from `model/torch.pt` when it starts. It does **not** run a complete PyTorch model forward pass: the feature/output view remains illustrative and is not a numerical-parity result.

## Run locally

From the repository root, activate the project's virtual environment and start the server:

```powershell
python tools/nn-visualizer/server.py
```

Then open <http://127.0.0.1:8765>.

The server binds to loopback (`127.0.0.1`) and is intended for local development. Three.js is loaded from jsDelivr, so the browser needs internet access.

## Load and navigate a game

- **Load demo game** loads the built-in 13-move trajectory: `L12 L8 O9 L17 H16 I14 J13 N9 Q10 K6 H12 O11 F7`.
- **Load game...** loads a local `.T1` game file. The server parses it using the repository's game-file parser.
- Edit **Current game** and click **Apply changes** to replay a space- or comma-separated move sequence. Trajectories are limited to 500 moves.
- Use the navigation buttons to jump to the start, step backward or forward, or jump to the end.
- **Reset camera** restores the default view. Drag to orbit, scroll to zoom, and right-drag to pan.
- Select **board & input tensors**, **conv filters & feature maps**, or **residual filters** to change the visualization. The explanation panel can be collapsed to give the scene more room.

Game positions are reconstructed by the Python backend. The browser does not independently re-implement the authoritative game rules when using this server.

## What is displayed

- **Board & input tensors:** sample and effective boards alongside the `pegs`, `links`, and `locations` tensors.
- **Input tensor shapes:** pegs `24 × 24 × 2`, links `24 × 24 × 8`, and locations `24 × 24 × 2`. Tensor coordinates use `[row][column][channel]` order.
- **Convolution filters:** the trained `links.weight` tensor is loaded from `model/torch.pt` through `/api/weights`. The bank uses the checkpoint's actual filter count and kernel dimensions, arranged in up to 8 columns. A bias is shown only if the loaded layer has one.
- **Feature values:** signed values are represented by cube thickness along X; negative values extend toward −X and positive values toward +X.

The browser uses the exact three arrays returned by `NetInputs.to_input_arrays()` for backend-loaded positions. It does not return or send the intermediate `NetInputs.naf` array because the visualizer does not consume it.

## Performance work

The visualizer includes several optimizations for navigating longer games:

- The backend caches up to 2,048 game states by normalized move prefix and relevant game-rule setting. Cached states are cloned before continuing replay.
- The browser keeps a bounded 128-entry least-recently-used cache of exact API snapshots.
- API responses are gzip-compressed when the browser supports it.
- Static board geometry, tensor markers, location tensors, labels, and filter geometry are reused where possible. Occupied peg/link instances are updated instead of recreating their meshes.
- Board links and directional indicators are rendered in instanced batches.

These optimizations are described in more detail in [PERFORMANCE.md](./PERFORMANCE.md).

### Inspect performance

Open browser developer tools (F12) and use the Console:

```js
nnPerf.reset()    // clear collected samples
nnPerf.report()   // print and return the collected samples
nnPerf.samples    // inspect or copy the raw samples
```

The samples include API request/body/JSON parsing timings, snapshot-cache hits, scene construction timings, object counts, and renderer draw-call/frame statistics. JavaScript construction timings are not GPU completion timings; use Chrome DevTools Performance when profiling actual frame time.

## Backend and API

The local server exposes two POST endpoints used by the page:

- `/api/trace`: accepts a JSON object with a `moves` array and optional `allow_scl` / `allow_swap` settings. It returns the reconstructed board, next player, tensor arrays, tensor shapes, and source metadata. Interactive trace requests skip recursive draw detection to keep navigation responsive.
- `/api/t1`: accepts the text contents of a `.T1` file and returns the parsed move sequence.
- `GET /api/weights`: reads `model/torch.pt` and returns the trained `links.weight` values in browser-friendly `[filter][row][column][channel]` order, plus layer-shape metadata.

The adapter can also be invoked from the repository root to inspect a trajectory:

```powershell
python tools/nn_visualizer_adapter.py --moves L12 L8 O9 L17 H16 I14 J13 N9 Q10 K6 H12 O11 F7
```

The adapter CLI performs draw checking by default. Interactive snapshots omit the raw 24 × 24 × 11 NAF intermediate to reduce response size.

## Tests

From the repository root, run the adapter tests with the project environment:

```powershell
python -m unittest tools.test_nn_visualizer_adapter
```

The tests check that the serialized peg, link, and location tensors match the backend's `NetInputs.to_input_arrays()` results and that the API's compact snapshot schema stays consistent.

## Scope and limitations

- The input tensors for loaded games come from the Python backend.
- The displayed link convolution weights are loaded from the trained PyTorch checkpoint. The other branch weights and feature-map/output calculations are still demo data; the visualizer does not yet run full model inference.
- The visualizer is for exploration and debugging. Use framework-level tests to establish numerical parity with the PyTorch model.
