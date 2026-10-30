# Twixtbot 3D neural-network visualizer

A local browser-based Three.js tool for exploring a Twixt position and the first stage of the trained neural network in **one 3D scene**. The scene combines the sample and effective boards, the `pegs`, `links`, and `locations` input tensors, the three trained convolution filter banks, and the resulting `24 × 24 × 48` feature tensor.

The Python backend is the source of truth for game rules and input encoding. The visualizer loads the trained peg, link, and location convolution weights from `model/torch.pt` through its local API.

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
- **Reset camera** restores the overview framing. Drag to orbit, scroll to zoom, and right-drag to pan. The explanation panel can be collapsed to give the scene more room.

Game positions are reconstructed by the Python backend. The browser does not independently re-implement the authoritative game rules when using this server.

## What is displayed

- **Boards:** the sample board and the effective board used for model input.
- **Input tensors:** `pegs` `24 × 24 × 2`, `links` `24 × 24 × 8`, and `locations` `24 × 24 × 2`. Tensor coordinates use `[row][column][channel]` order.
- **Convolution filters:** trained peg, link, and location filter banks loaded from `model/torch.pt`. The banks use the checkpoint's actual filter and kernel dimensions. Bias cubes are shown only for layers that have a bias.
- **Feature tensor:** the three first-layer branch convolutions are summed into a `24 × 24 × 48` tensor. The displayed values are the raw sum **before** batch normalization and the absolute-value activation. A signed heat map encodes the feature values.

The feature tensor is calculated in the browser from the backend's input arrays and the loaded checkpoint weights. The convolution uses TensorFlow-style `SAME` padding, including asymmetric padding for even-sized kernels.

**Scope:** this is a view of the input encoding and first convolution stage, not a complete forward pass through the network. Residual blocks, policy/value heads, batch-normalization transforms, and later activations are not currently evaluated or rendered. Use framework-level tests to establish numerical parity with the PyTorch model.

## Interaction and diagnostics

The board can be navigated with the game controls. In the 3D viewport:

- Drag to orbit.
- Scroll to zoom.
- Right-drag to pan.
- Click an empty intersection on the sample board to submit a move to the backend; invalid moves leave the current position unchanged.

For performance diagnostics, open browser developer tools (F12) and use the Console:

```js
nnPerf.reset()    // clear collected samples
nnPerf.report()   // print and return collected samples
nnPerf.samples    // inspect or copy the raw samples
```

Samples include API request/body/JSON parsing timings, snapshot-cache hits, scene construction timings, object counts, and renderer draw-call/frame statistics. These are synchronous JavaScript timings, not GPU completion timings; use Chrome DevTools Performance when profiling actual frame time.

The visualizer reuses static board and tensor geometry, updates occupied peg/link instances, caches filter geometry and API snapshots, and uses instanced batches for board links and directional indicators. The backend caches reconstructed game states by move prefix. The browser's snapshot cache is bounded to 128 entries.

## Backend API and tests

The local server exposes:

- `POST /api/trace`: accepts a JSON object with a `moves` array and optional `allow_scl` / `allow_swap` settings. It returns the reconstructed board, next player, input tensor arrays, tensor shapes, and source metadata. Interactive trace requests skip recursive draw detection to keep navigation responsive.
- `POST /api/t1`: accepts the text contents of a `.T1` file and returns the parsed move sequence.
- `GET /api/weights`: reads `model/torch.pt` and returns the trained peg, link, and location convolution weights in browser-friendly order, plus layer-shape and bias metadata.

The adapter can also be invoked from the repository root:

```powershell
python tools/nn_visualizer_adapter.py --moves L12 L8 O9 L17 H16 I14 J13 N9 Q10 K6 H12 O11 F7
```

The adapter CLI performs draw checking by default. Interactive snapshots omit the raw `24 × 24 × 11` NAF intermediate because the browser consumes the three arrays returned by `NetInputs.to_input_arrays()`.

Run the adapter tests from the repository root:

```powershell
python -m unittest tools.test_nn_visualizer_adapter
```

The tests check that serialized peg, link, and location tensors match the backend's `NetInputs.to_input_arrays()` results and that the compact snapshot schema stays consistent.
