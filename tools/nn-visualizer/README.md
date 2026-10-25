# Twixtbot 3D neural-network visualizer

A standalone browser-based Three.js inspector for the first convolution stage. It visualizes the three input tensors, the 48 filter banks, and the summed 24×24×48 feature tensor. Batch normalization and later residual blocks are outside the current scope.

## Run locally

Serve this directory over HTTP (ES modules generally do not work from `file://`):

```powershell
cd tools/nn-visualizer
python -m http.server 8000
```

Open http://localhost:8000. Three.js and OrbitControls are imported from jsDelivr, so an internet connection is needed.

## Views

- **Inputs:** a fixed sample Twixt board is rendered in front of the three input tensors. The board uses the move sequence from the root README's T1J example: `E15`, `swap`, `F17`, `H18`, `P12`. The pegs and link on the board are derived from that sequence; the tensor blades still use generated demo values and are not yet encoded from the same game. The three tensors are pegs `24×24×2`, links `24×24×8`, and fixed locations `24×24×2`.
- **Filters:** 48 filters in an 8×6 arrangement. Link kernels are `5×5×8`; peg and location kernels are `1×1×2`. Each filter group has a separate yellow bias cube.
- **Conv output:** the summed `24×24×48` feature tensor. Signed values are represented by cube thickness along X: negative toward −X, positive toward +X.

## JSON input

Use **Load JSON** to load a board and optionally real weights. Board tensors are numeric nested arrays in `[row][column][channel]` order. Weights are indexed as `[outputChannel][kernelRow][kernelColumn][inputChannel]`.

```json
{
  "pegs": [[[0, 1]]],
  "links": [[[0, 0, 0, 0, 0, 0, 0, 0]]],
  "locations": [[[1, 0]]],
  "weights": {
    "pegs": "shape [48][1][1][2]",
    "links": "shape [48][5][5][8]",
    "locations": "shape [48][1][1][2]"
  },
  "bias": [0]
}
```

The values above are schema illustrations only; real tensors must contain all 24×24 cells, all channels, all 48 filters, and 48 bias values. Replace the strings under `weights` with actual numeric nested arrays. If weights are omitted, reproducible demo weights are used. This browser page does not load `model/torch.pt` directly; a separate exporter or backend is needed to provide its weights.

The convolution implementation uses zero padding for the 5×5 link kernel and sums the link, peg, and location branches plus one bias per output channel. Confirm padding, bias placement, and weight layouts against the PyTorch model before using this for numerical parity checks.

Large tensors use Three.js instanced meshes to reduce draw calls. This is an exploratory visualization, not a substitute for framework-level numerical tests.
