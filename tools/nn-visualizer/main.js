import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/controls/OrbitControls.js';
import { createRandomGenerator, createTensor3, createTensor4, flattenTensor } from './tensor-utils.js';
import { createLabelHelpers } from './labels.js?v=20261028-move-columns';
import { createBoardRenderer } from './board-renderer.js?v=20261029-link-cube-depth';
import { createLinkEncodingDiagnostics } from './link-encoding-diagnostics.js';
import { createTensorRenderer } from './tensor-renderer.js?v=20261031-residual-layout-v5';

const BOARD_SIZE = 24;
const FILTER_COUNT = 48;
const DEMO_TRAJECTORY = 'L12 L8 O9 L17 H16 I14 J13 N9 Q10 K6 H12 O11 F7';

const COLORS = {
  input: 0x78c7f4,
  fixed: 0xc9cdd3,
  weight: 0x9b6848,
  bias: 0xf0c64b,
  output: 0x54d68b,
};

const viewport = document.querySelector('#viewport');
const loading = document.querySelector('#loading');
const status = document.querySelector('#status');

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b1018);
scene.fog = new THREE.Fog(0x0b1018, 75, 190);

const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 500);
// Frame the boards, inputs, filter banks, and feature tensor together. Keep
// this preset shared with Reset camera so both views show the complete scene.
const INITIAL_CAMERA_POSITION = new THREE.Vector3(-60, 17, -24);
const INITIAL_CAMERA_TARGET = new THREE.Vector3(1, -1, 0);
camera.position.copy(INITIAL_CAMERA_POSITION);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
viewport.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
// A larger damping factor dissipates OrbitControls inertia more quickly after
// the pointer is released, while keeping rotation smooth during the drag.
// Dissipate the post-drag inertia quickly so the detailed residual banks
// can return without a long period of invisible-but-moving camera updates.
controls.dampingFactor = 0.3;
controls.target.copy(INITIAL_CAMERA_TARGET);

scene.add(new THREE.HemisphereLight(0xcfe3ff, 0x253041, 2));

const keyLight = new THREE.DirectionalLight(0xffffff, 2.4);
keyLight.position.set(20, 40, 30);
scene.add(keyLight);

const activeStep = 'inputs';
let sceneGroup = new THREE.Group();
const staticSceneGroup = new THREE.Group();
staticSceneGroup.name = 'persistent-static-board';
const inputStaticTensorGroup = new THREE.Group();
inputStaticTensorGroup.name = 'persistent-input-tensor-static';
const inputDynamicTensorGroup = new THREE.Group();
inputDynamicTensorGroup.name = 'persistent-input-tensor-dynamic';
const filterStaticGroup = new THREE.Group();
filterStaticGroup.name = 'persistent-filter-geometry';
// Must be initialized before createDemoData(), which calls createSampleGame().
let backendSnapshot = null;
const demoModel = createDemoData();
let data = demoModel;
let modelWeightStatus = 'Using generated demo weights.';
const residualBlockWeightCache = new Map();
let selectedResidualBlockIndex = 0;
let residualSelectionPending = false;
let cachedOutput = null;
let gameMoves = [];
let gamePly = 0;
let gameLoaded = false;
let currentPlayerName = "Black";

scene.add(staticSceneGroup);
scene.add(inputStaticTensorGroup);
scene.add(inputDynamicTensorGroup);
scene.add(filterStaticGroup);
scene.add(sceneGroup);

// Temporary baseline instrumentation for the performance work. This is
// observational only: it does not alter scene construction or game state.
const nnPerf = {
  samples: [],
  record(label, metrics) {
    const sample = { at: new Date().toISOString(), label, ...metrics };
    this.samples.push(sample);
    console.debug('[NN perf]', sample);
    return sample;
  },
  report() {
    console.table(this.samples.map(({ at, ...sample }) => sample));
    return this.samples.slice();
  },
  reset() {
    this.samples = [];
    console.info('[NN perf] Samples cleared.');
  },
};
window.nnPerf = nnPerf;

// Cache recent backend snapshots by exact move prefix. This avoids repeated
// HTTP and JSON work when revisiting positions or rendering a validated endpoint.
const SNAPSHOT_CACHE_LIMIT = 128;
const snapshotCache = new Map();

function getCachedSnapshot(key) {
  if (!snapshotCache.has(key)) return null;
  const value = snapshotCache.get(key);
  snapshotCache.delete(key);
  snapshotCache.set(key, value);
  return value;
}

function cacheSnapshot(key, value) {
  snapshotCache.delete(key);
  snapshotCache.set(key, value);
  while (snapshotCache.size > SNAPSHOT_CACHE_LIMIT) {
    snapshotCache.delete(snapshotCache.keys().next().value);
  }
}

function countSceneObjects(root) {
  const counts = { objects: 0, meshes: 0, instancedMeshes: 0, sprites: 0, geometries: 0, materials: 0 };
  const geometries = new Set();
  const materials = new Set();
  root.traverse((object) => {
    counts.objects += 1;
    if (object.isMesh) {
      counts.meshes += 1;
      if (object.isInstancedMesh) counts.instancedMeshes += 1;
      if (object.geometry) geometries.add(object.geometry);
      for (const material of (Array.isArray(object.material) ? object.material : [object.material])) {
        if (material) materials.add(material);
      }
    }
    if (object.isSprite) {
      counts.sprites += 1;
      for (const material of (Array.isArray(object.material) ? object.material : [object.material])) {
        if (material) materials.add(material);
      }
    }
  });
  counts.geometries = geometries.size;
  counts.materials = materials.size;
  return counts;
}

const cubeGeometry = new THREE.BoxGeometry(1, 1, 1);
const { addLabel, addRichLabel, addMoveColumnLabel, beginFrame: beginLabelFrame } = createLabelHelpers(THREE, () => sceneGroup);
const { addLabel: addStaticLabel, addRichLabel: addStaticRichLabel } =
  createLabelHelpers(THREE, () => staticSceneGroup);

function createDemoData() {
  const random=createRandomGenerator(2026);
  const {pegs,links}=encodeGameInputs({pegs:[],links:[],nextPlayer:1});
  const locations=createTensor3(BOARD_SIZE,BOARD_SIZE,2);
  for(let row=0;row<BOARD_SIZE;row++) for(let col=0;col<BOARD_SIZE;col++) {
    locations[row][col][0]=col/BOARD_SIZE; locations[row][col][1]=row/BOARD_SIZE;
  }
  const weights={
    pegs:createTensor4(FILTER_COUNT,5,5,2,()=> (random()-0.5)*0.4),
    links:createTensor4(FILTER_COUNT,4,4,8,()=> (random()-0.5)*0.16),
    locations:createTensor4(FILTER_COUNT,1,1,2,()=> (random()-0.5)*0.3),
    residualBlock0: {
      conv1: createTensor4(FILTER_COUNT,5,5,FILTER_COUNT,()=> (random()-0.5)*0.08),
      conv2: createTensor4(FILTER_COUNT,5,5,FILTER_COUNT,()=> (random()-0.5)*0.08),
    },
  };
  const bias=Array.from({length:FILTER_COUNT},()=> (random()-0.5)*0.15);
  // Identity BN keeps the generated fallback numerically usable, but it is
  // explicitly a demo and is never confused with checkpoint inference.
  const primaryBn = {
    weight: Array(FILTER_COUNT).fill(1),
    bias: Array(FILTER_COUNT).fill(0),
    running_mean: Array(FILTER_COUNT).fill(0),
    running_var: Array(FILTER_COUNT).fill(1),
    eps: 1e-3,
  };
  return {pegs,links,locations,weights,bias,primaryBn,residualBlockCount:20,isDemo:true};
}

/**
 * Convert a [row][column][channel] tensor into channel-major order.
 * The cube renderer iterates channels first, then rows and columns.
 */

function calculateConvolutionOutput() {
  if (cachedOutput && cachedOutput.source === data) {
    return cachedOutput.tensor;
  }

  const output = createTensor3(BOARD_SIZE, BOARD_SIZE, FILTER_COUNT);
  const branches = [
    { name: 'pegs', input: data.pegs, weights: data.weights.pegs, metadata: data.pegConvMetadata },
    { name: 'links', input: data.links, weights: data.weights.links, metadata: data.linkConvMetadata },
    { name: 'locations', input: data.locations, weights: data.weights.locations, metadata: data.locationConvMetadata },
  ];

  // The checkpoint uses TensorFlow-style SAME padding. For even-sized kernels,
  // the extra padding cell belongs on the bottom/right (before = floor(total/2)).
  for (const branch of branches) {
    const kernelHeight = branch.weights[0].length;
    const kernelWidth = branch.weights[0][0].length;
    const channels = branch.weights[0][0][0].length;
    const padTop = Math.floor((kernelHeight - 1) / 2);
    const padLeft = Math.floor((kernelWidth - 1) / 2);

    for (let row = 0; row < BOARD_SIZE; row += 1) {
      for (let col = 0; col < BOARD_SIZE; col += 1) {
        for (let filter = 0; filter < FILTER_COUNT; filter += 1) {
          let sum = branch.metadata?.has_bias
            ? Number(branch.metadata.bias[filter] || 0)
            : 0;

          for (let kernelRow = 0; kernelRow < kernelHeight; kernelRow += 1) {
            const inputRow = row + kernelRow - padTop;
            if (inputRow < 0 || inputRow >= BOARD_SIZE) continue;

            for (let kernelCol = 0; kernelCol < kernelWidth; kernelCol += 1) {
              const inputCol = col + kernelCol - padLeft;
              if (inputCol < 0 || inputCol >= BOARD_SIZE) continue;

              for (let channel = 0; channel < channels; channel += 1) {
                sum += Number(branch.input[inputRow][inputCol][channel]) *
                  Number(branch.weights[filter][kernelRow][kernelCol][channel]);
              }
            }
          }

          output[row][col][filter] += sum;
        }
      }
    }
  }

  // Match TwixtNet.forward exactly: sum the three convolution branches,
  // apply primary BatchNorm2d in inference mode, then torch.abs().
  const bn = data.primaryBn;
  if (!bn) {
    throw new Error('Primary batch-normalization parameters are unavailable; refusing to label raw convolution sums as model features.');
  }
  const epsilon = Number(bn.eps ?? 1e-3);
  for (let row = 0; row < BOARD_SIZE; row += 1) {
    for (let col = 0; col < BOARD_SIZE; col += 1) {
      for (let channel = 0; channel < FILTER_COUNT; channel += 1) {
        const variance = Number(bn.running_var[channel]);
        const normalized =
          (output[row][col][channel] - Number(bn.running_mean[channel])) /
          Math.sqrt(variance + epsilon);
        const scaled = normalized * Number(bn.weight[channel]) +
          Number(bn.bias[channel]);
        output[row][col][channel] = Math.abs(scaled);
      }
    }
  }

  cachedOutput = { source: data, tensor: output };
  return output;
}

function clearSceneGroup() {
  scene.remove(sceneGroup);

  // Most board geometry is recreated for each position. Dispose each unique
  // geometry once so renderer.info.memory.geometries does not grow on every
  // navigation. The shared tensor cube geometry must remain alive.
  const geometriesToDispose = new Set();
  sceneGroup.traverse((object) => {
    if (object.isSprite) {
      // Label sprites/materials are pooled and reused by labels.js. Disposing
      // them here would invalidate the shared textures and defeat the pool.
      if (!object.userData.pooledLabel) {
        if (object.material.map) object.material.map.dispose();
        object.material.dispose();
      }
    } else if (object.isMesh) {
      const materials = Array.isArray(object.material) ? object.material : [object.material];
      for (const material of materials) {
        if (material?.map) material.map.dispose();
        material?.dispose();
      }

      if (object.geometry && object.geometry !== cubeGeometry) {
        geometriesToDispose.add(object.geometry);
      }
    }
  });

  geometriesToDispose.forEach((geometry) => geometry.dispose());

  sceneGroup = new THREE.Group();
  scene.add(sceneGroup);
}

function createSampleGame() {
  if (backendSnapshot) {
    return {
      moves: backendSnapshot.moves,
      pegs: backendSnapshot.board.pegs,
      links: backendSnapshot.board.links,
      nextPlayer: backendSnapshot.next_player,
    };
  }
  const moves = [];
  const pegs=[]; const links=[]; let turn=1;
  for (const move of moves) {
    const col=move.charCodeAt(0)-65, row=Number(move.slice(1))-1;
    const next={row,col,player:turn};
    for (const peg of pegs) {
      if (peg.player!==turn) continue;
      const dr=Math.abs(peg.row-row), dc=Math.abs(peg.col-col);
      if ((dr===1&&dc===2)||(dr===2&&dc===1)) links.push({from:peg,to:next,player:turn});
    }
    pegs.push(next); turn=1-turn;
  }
  return {moves,pegs,links,nextPlayer:turn};
}

function encodeGameInputs(game) {
  // Canonicalize every position to the network's Black-to-move viewpoint:
  // transpose x/y on White's turn, and swap the peg/link colour planes.
  // Game player IDs follow twixt.Game: BLACK=0, WHITE=1.
  const rawPegs = Array.from({ length: 2 }, () =>
    createTensor3(BOARD_SIZE, BOARD_SIZE, 1),
  );
  const rawLinks = Array.from({ length: 8 }, () =>
    createTensor3(BOARD_SIZE, BOARD_SIZE, 1),
  );

  // Backend game arrays are indexed [x][y] = [column][row].
  for (const peg of game.pegs) {
    rawPegs[peg.player][peg.col][peg.row][0] = 1;
  }

  // Link channel encoding: bit 0 = colour, bit 2 = LONGY, bit 1 = DIFFSIGN.
  for (const link of game.links) {
    const x1 = link.from.col;
    const y1 = link.from.row;
    const x2 = link.to.col;
    const y2 = link.to.row;
    const channel =
      link.player +
      (((x1 + x2) % 2 !== 0) ? 4 : 0) +
      (((y2 - y1) * (x2 - x1) < 0) ? 2 : 0);
    rawLinks[channel][Math.floor((x1 + x2) / 2)][Math.floor((y1 + y2) / 2)][0] = 1;
  }

  const pegs = createTensor3(BOARD_SIZE, BOARD_SIZE, 2);
  const midLinks = createTensor3(BOARD_SIZE, BOARD_SIZE, 8);
  const whiteToMove = game.nextPlayer === 1;

  for (let row = 0; row < BOARD_SIZE; row += 1) {
    for (let col = 0; col < BOARD_SIZE; col += 1) {
      // The canonical tensor is [row][column]. On White's turn, transpose
      // the raw [x][y] axes and exchange the player colour planes.
      const sourceX = whiteToMove ? row : col;
      const sourceY = whiteToMove ? col : row;
      const firstColor = whiteToMove ? 1 : 0;
      const secondColor = whiteToMove ? 0 : 1;

      pegs[row][col][0] = rawPegs[firstColor][sourceX][sourceY][0];
      pegs[row][col][1] = rawPegs[secondColor][sourceX][sourceY][0];

      for (let channel = 0; channel < 8; channel += 1) {
        const sourceChannel = whiteToMove
          ? (channel ^ 1)
          : channel;
        midLinks[row][col][channel] =
          rawLinks[sourceChannel][sourceX][sourceY][0];
      }
    }
  }

  // Match NetInputs.to_input_arrays: shift link channels after perspective
  // mapping. Peg tensors are not shifted.
  const links = createTensor3(BOARD_SIZE, BOARD_SIZE, 8);
  for (let row = 0; row < BOARD_SIZE; row += 1) {
    for (let col = 0; col < BOARD_SIZE; col += 1) {
      for (let channel = 0; channel < 8; channel += 1) {
        const vertical = (channel & 4) !== 0;
        const diffSign = (channel & 2) !== 0;
        const sourceRow = row - ((vertical || diffSign) ? 1 : 0);
        const sourceCol = col - ((!vertical || diffSign) ? 1 : 0);
        if (sourceRow >= 0 && sourceCol >= 0) {
          links[row][col][channel] = midLinks[sourceRow][sourceCol][channel];
        }
      }
    }
  }

  return { pegs, links };
}

const { drawSampleBoard, drawEffectiveBoard } = createBoardRenderer({
  THREE,
  getSceneGroup: () => sceneGroup,
  getStaticSceneGroup: () => staticSceneGroup,
  addStaticLabel,
  addStaticRichLabel,
  addMoveColumnLabel,
  BOARD_SIZE,
  addLabel,
  addRichLabel,
  createSampleGame,
  getData: () => data,
});

const { logLinkEncodingDiagnostic, logFullInputTensorValidation } =
  createLinkEncodingDiagnostics({ BOARD_SIZE, createTensor3 });


/**
 * Print a source-of-truth diagnostic for the link input blades.
 * The effective board is drawn from game.links; the blades are drawn from
 * data.links after canonicalization and the input-cell shift. Comparing both
 * lists makes channel-label, transpose, and one-cell-offset errors visible.
 */



const {
  drawInputs, drawFilters, drawOutput, setResidualFiltersVisible,
  setSelectedResidualBlock, pickResidualBlock,
} = createTensorRenderer({
    THREE, BOARD_SIZE, FILTER_COUNT, COLORS, cubeGeometry, flattenTensor,
    createTensor3, getSceneGroup: () => sceneGroup, getData: () => data,
    getInputStaticGroup: () => inputStaticTensorGroup,
    getInputDynamicGroup: () => inputDynamicTensorGroup,
    getFilterStaticGroup: () => filterStaticGroup,
    createSampleGame, getBackendSnapshot: () => backendSnapshot,
    drawSampleBoard, drawEffectiveBoard,
    logLinkEncodingDiagnostic, logFullInputTensorValidation,
    calculateConvolutionOutput, addLabel,
  });

function renderStep() {
  const renderStarted = performance.now();
  staticSceneGroup.visible = true;
  inputStaticTensorGroup.visible = true;
  inputDynamicTensorGroup.visible = true;
  filterStaticGroup.visible = true;
  // Reset the label pool before detaching the previous scene; labels are
  // reattached from the pool as the new scene is constructed.
  beginLabelFrame();
  const clearStarted = renderStarted;
  clearSceneGroup();
  const clearMs = performance.now() - clearStarted;

  const drawStarted = performance.now();
  const info = drawInputs();
  const drawMs = performance.now() - drawStarted;

  document.querySelector('#view-title').textContent = info.title;
  document.querySelector('#view-description').textContent = info.description;
  document.querySelector('#tensor-shape').textContent = info.shape;
  document.querySelector('#value-range').textContent = info.range;
  document.querySelector('#cube-count').textContent = info.count;


  loading.style.display = 'none';

  const objects = countSceneObjects(sceneGroup);
  nnPerf.record('renderStep', {
    step: activeStep,
    ply: gameLoaded ? gamePly : null,
    moves: gameMoves.length,
    clearMs: Number(clearMs.toFixed(2)),
    drawMs: Number(drawMs.toFixed(2)),
    totalMs: Number((performance.now() - renderStarted).toFixed(2)),
    ...objects,
    persistentInputStatic: countSceneObjects(inputStaticTensorGroup),
    persistentInputDynamic: countSceneObjects(inputDynamicTensorGroup),
    persistentFilterGeometry: countSceneObjects(filterStaticGroup),
  });

  // Read renderer.info after the next normal animation-frame render, not from
  // the previous frame (renderStep itself only constructs the new scene).
  requestAnimationFrame(() => {
    nnPerf.record('renderFrameStats', {
      step: activeStep,
      ply: gameLoaded ? gamePly : null,
      moves: gameMoves.length,
      drawCalls: renderer.info.render.calls,
      triangles: renderer.info.render.triangles,
      rendererGeometries: renderer.info.memory.geometries,
      rendererTextures: renderer.info.memory.textures,
    });
  });
}

function validateTensor(tensor, channels, name) {
  const valid =
    Array.isArray(tensor) &&
    tensor.length === BOARD_SIZE &&
    tensor.every(
      (row) =>
        Array.isArray(row) &&
        row.length === BOARD_SIZE &&
        row.every(
          (cell) =>
            Array.isArray(cell) &&
            cell.length === channels &&
            cell.every((value) => Number.isFinite(Number(value))),
        ),
    );

  if (!valid) {
    throw new Error(
      name +
        ' must be a numeric [' +
        BOARD_SIZE +
        '][' +
        BOARD_SIZE +
        '][' +
        channels +
        '] tensor.',
    );
  }
}

function validateWeights(weights, name, kernelHeight, kernelWidth, channels) {
  const valid =
    Array.isArray(weights) &&
    weights.length === FILTER_COUNT &&
    weights.every(
      (filter) =>
        Array.isArray(filter) &&
        filter.length === kernelHeight &&
        filter.every(
          (row) =>
            Array.isArray(row) &&
            row.length === kernelWidth &&
            row.every(
              (cell) =>
                Array.isArray(cell) &&
                cell.length === channels &&
                cell.every((value) => Number.isFinite(Number(value))),
            ),
        ),
    );

  if (!valid) {
    throw new Error(
      'weights.' +
        name +
        ' must have shape [' +
        FILTER_COUNT +
        '][' +
        kernelHeight +
        '][' +
        kernelWidth +
        '][' +
        channels +
        '].',
    );
  }
}

function validateData(input) {
  validateTensor(input.pegs, 2, 'pegs');
  validateTensor(input.links, 8, 'links');
  validateTensor(input.locations, 2, 'locations');

  if (!input.weights) {
    input.weights = createDemoData().weights;
    status.textContent = 'Weights omitted: using generated demo weights.';
  } else {
    validateWeights(input.weights.pegs, 'pegs', 5, 5, 2);
    validateWeights(input.weights.links, 'links', 4, 4, 8);
    validateWeights(input.weights.locations, 'locations', 1, 1, 2);
  }

  if (
    !Array.isArray(input.bias) ||
    input.bias.length !== FILTER_COUNT ||
    input.bias.some((value) => !Number.isFinite(Number(value)))
  ) {
    input.bias = Array(FILTER_COUNT).fill(0);
  }

  input.isDemo = false;
  return input;
}


async function loadTrainedLinkWeights() {
  try {
    const response = await fetch('/api/weights', { cache: 'no-store' });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || response.statusText);
    if (!result.layers || !result.weights ||
        !result.layers.links || !result.layers.pegs || !result.layers.locations ||
        !Array.isArray(result.weights.links) ||
        !Array.isArray(result.weights.pegs) ||
        !Array.isArray(result.weights.locations) ||
        result.weights.links.length !== result.layers.links.filter_count ||
        result.weights.pegs.length !== result.layers.pegs.filter_count ||
        result.weights.locations.length !== result.layers.locations.filter_count) {
      throw new Error('The model-weight endpoint returned an unexpected peg/link/location weight shape.');
    }

    for (const [name, layer] of Object.entries(result.layers)) {
      if (layer.has_bias &&
          (!Array.isArray(layer.bias) ||
           layer.bias.length !== layer.filter_count ||
           layer.bias.some((value) => !Number.isFinite(Number(value))))) {
        throw new Error('The model-weight endpoint returned invalid bias values for ' + name + '.');
      }
    }

    demoModel.weights.links = result.weights.links;
    demoModel.weights.pegs = result.weights.pegs;
    demoModel.weights.locations = result.weights.locations;
    const residualBlock0 = result.residual_block_0;
    if (!residualBlock0 || residualBlock0.config?.channels !== FILTER_COUNT) {
      throw new Error('The model-weight endpoint did not return a 48-channel first residual block.');
    }
    for (const name of ['conv1', 'conv2']) {
      const layer = residualBlock0.layers?.[name];
      const filters = residualBlock0.weights?.[name];
      if (!layer || layer.filter_count !== FILTER_COUNT || layer.kernel_height !== 5 ||
          layer.kernel_width !== 5 || layer.channels !== FILTER_COUNT ||
          !Array.isArray(filters) || filters.length !== FILTER_COUNT ||
          filters.some(filter => filter.length !== 5 || filter.some(row => row.length !== 5 ||
            row.some(cell => cell.length !== FILTER_COUNT || cell.some(value => !Number.isFinite(Number(value))))))) {
        throw new Error('The model-weight endpoint returned an invalid first residual ' + name + ' filter bank.');
      }
    }
    demoModel.weights.residualBlock0 = residualBlock0.weights;
    demoModel.residualBlock0Metadata = residualBlock0.layers;
    demoModel.residualBlockCount = Number(result.residual_block_count) || 1;
    residualBlockWeightCache.set(0, residualBlock0.weights);
    selectedResidualBlockIndex = 0;
    demoModel.linkConvMetadata = result.layers.links;
    demoModel.pegConvMetadata = result.layers.pegs;
    demoModel.locationConvMetadata = result.layers.locations;
    const primaryBn = result.primary_bn;
    for (const key of ['weight', 'bias', 'running_mean', 'running_var']) {
      if (!Array.isArray(primaryBn?.[key]) ||
          primaryBn[key].length !== FILTER_COUNT ||
          primaryBn[key].some((value) => !Number.isFinite(Number(value)))) {
        throw new Error('The model-weight endpoint returned invalid primary batch-normalization parameter: ' + key);
      }
    }
    demoModel.primaryBn = primaryBn;
    modelWeightStatus = 'Trained peg/link/location weights and primary batch-normalization parameters loaded from ' + result.layers.links.source +
      ' (pegs: ' + result.layers.pegs.filter_count + ' filters, bias=' +
      (result.layers.pegs.has_bias ? 'loaded' : 'none') + '; links: ' +
      result.layers.links.filter_count + ' filters, bias=' +
      (result.layers.links.has_bias ? 'loaded' : 'none') + '; locations: ' +
      result.layers.locations.filter_count + ' filters, bias=' +
      (result.layers.locations.has_bias ? 'loaded' : 'none') + ').';
    console.info('[NN visualizer] ' + modelWeightStatus, {
      pegs: result.layers.pegs,
      links: result.layers.links,
      locations: result.layers.locations,
    });
  } catch (error) {
    modelWeightStatus = 'Using generated demo weights; trained weights and biases could not be loaded: ' + error.message;
    console.warn('[NN visualizer] ' + modelWeightStatus);
  }
}

function validateResidualBlockPayload(payload, expectedIndex) {
  if (payload?.index !== expectedIndex || payload.config?.channels !== FILTER_COUNT) {
    throw new Error('The server returned unexpected metadata for residual block ' + expectedIndex + '.');
  }
  for (const name of ['conv1', 'conv2']) {
    const layer = payload.layers?.[name];
    const filters = payload.weights?.[name];
    if (!layer || layer.filter_count !== FILTER_COUNT || layer.kernel_height !== 5 ||
        layer.kernel_width !== 5 || layer.channels !== FILTER_COUNT ||
        !Array.isArray(filters) || filters.length !== FILTER_COUNT ||
        filters.some(filter => filter.length !== 5 || filter.some(row => row.length !== 5 ||
          row.some(cell => cell.length !== FILTER_COUNT ||
            cell.some(value => !Number.isFinite(Number(value))))))) {
      throw new Error('The server returned an invalid ' + name + ' filter bank for residual block ' + expectedIndex + '.');
    }
  }
  return payload.weights;
}

async function selectResidualBlock(blockIndex) {
  if (residualSelectionPending || blockIndex === selectedResidualBlockIndex) return;
  if (blockIndex < 0 || blockIndex >= Number(data.residualBlockCount || 1)) return;
  residualSelectionPending = true;
  status.textContent = 'Loading residual block ' + blockIndex + '…';
  try {
    let weights = residualBlockWeightCache.get(blockIndex);
    if (!weights) {
      const response = await fetch('/api/residual-block?index=' + encodeURIComponent(blockIndex), {
        cache: 'no-store',
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || response.statusText);
      weights = validateResidualBlockPayload(payload, blockIndex);
      residualBlockWeightCache.set(blockIndex, weights);
      // Bound memory usage while retaining the initial block and recent choices.
      while (residualBlockWeightCache.size > 4) {
        const oldest = residualBlockWeightCache.keys().next().value;
        if (oldest === 0 || oldest === blockIndex) {
          const next = [...residualBlockWeightCache.keys()].find((key) => key !== 0 && key !== blockIndex);
          if (next === undefined) break;
          residualBlockWeightCache.delete(next);
        } else {
          residualBlockWeightCache.delete(oldest);
        }
      }
    }
    selectedResidualBlockIndex = blockIndex;
    setSelectedResidualBlock(blockIndex, weights);
    renderStep();
    status.textContent = 'Showing residual block ' + blockIndex + ' (both convolutions).';
  } catch (error) {
    status.textContent = 'Could not load residual block ' + blockIndex + ': ' + error.message;
  } finally {
    residualSelectionPending = false;
  }
}

function updateCameraCoordinates() {
  const readout = document.querySelector('#camera-coordinates');
  if (!readout) return;
  const fmt = (v) => v.toFixed(2);
  readout.textContent =
    'Camera  x ' + fmt(camera.position.x) + '  y ' + fmt(camera.position.y) + '  z ' + fmt(camera.position.z) +
    '\nTarget  x ' + fmt(controls.target.x) + '  y ' + fmt(controls.target.y) + '  z ' + fmt(controls.target.z);
}

// Hide the dense residual weight cubes during camera interaction, then
// restore them after OrbitControls has finished its damping motion. The
// simpler first-layer banks and all tensor/board geometry remain visible.
let residualRestoreTimer = null;
function deferResidualFilterRestore() {
  if (residualRestoreTimer !== null) clearTimeout(residualRestoreTimer);
  residualRestoreTimer = setTimeout(() => {
    setResidualFiltersVisible(true);
    residualRestoreTimer = null;
  }, 180);
}
controls.addEventListener('start', () => {
  if (residualRestoreTimer !== null) {
    clearTimeout(residualRestoreTimer);
    residualRestoreTimer = null;
  }
  setResidualFiltersVisible(false);
});
controls.addEventListener('change', updateCameraCoordinates);
// OrbitControls emits "change" throughout damping, not just while the user is
// dragging. Do not restart the visibility timer on every such frame: that can
// postpone the residual banks for seconds after motion is no longer noticeable.
controls.addEventListener('end', deferResidualFilterRestore);

function resetCamera() {
  camera.position.copy(INITIAL_CAMERA_POSITION);
  controls.target.copy(INITIAL_CAMERA_TARGET);
  controls.update();
  updateCameraCoordinates();
}

function resizeRenderer() {
  const width = viewport.clientWidth;
  const height = viewport.clientHeight;

  if (!width || !height) return;

  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

// The sample board uses this fixed world-space layout in board-renderer.js.
// Hit-testing deliberately compares projected screen pixels, not 3D distance.
const SAMPLE_BOARD_SIZE = (BOARD_SIZE - 1) * 0.34;
const SAMPLE_BOARD_SPACING = 0.34;
const SAMPLE_BOARD_X = -26.675 + 0.026;
const SAMPLE_BOARD_START = -SAMPLE_BOARD_SIZE / 2;

function projectSampleIntersection(row, col, rect) {
  const point = new THREE.Vector3(
    SAMPLE_BOARD_X,
    SAMPLE_BOARD_START + (BOARD_SIZE - 1 - row) * SAMPLE_BOARD_SPACING,
    SAMPLE_BOARD_START + col * SAMPLE_BOARD_SPACING,
  );
  point.project(camera);
  if (point.z < -1 || point.z > 1) return null;
  return {
    x: rect.left + (point.x + 1) * rect.width / 2,
    y: rect.top + (1 - point.y) * rect.height / 2,
  };
}

function handleSampleBoardClick(event) {
  if (activeStep !== 'inputs' || !gameLoaded || interactiveMovePending) return;

  const rect = renderer.domElement.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  controls.update();
  camera.updateMatrixWorld(true);

  let closest = null;
  for (let row = 0; row < BOARD_SIZE; row += 1) {
    for (let col = 0; col < BOARD_SIZE; col += 1) {
      const screen = projectSampleIntersection(row, col, rect);
      if (!screen) continue;
      const dx = screen.x - event.clientX;
      const dy = screen.y - event.clientY;
      const distanceSquared = dx * dx + dy * dy;
      if (!closest || distanceSquared < closest.distanceSquared) {
        closest = { row, col, screen, distanceSquared };
      }
    }
  }
  if (!closest) return;

  // Base the click tolerance on the local projected grid spacing. This keeps
  // intersections selectable at different zoom levels and camera angles.
  let nearestGridSpacing = Infinity;
  for (const [dr, dc] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
    const row = closest.row + dr;
    const col = closest.col + dc;
    if (row < 0 || row >= BOARD_SIZE || col < 0 || col >= BOARD_SIZE) continue;
    const neighbor = projectSampleIntersection(row, col, rect);
    if (!neighbor) continue;
    nearestGridSpacing = Math.min(
      nearestGridSpacing,
      Math.hypot(neighbor.x - closest.screen.x, neighbor.y - closest.screen.y),
    );
  }
  const hitRadius = Number.isFinite(nearestGridSpacing)
    ? Math.max(3, Math.min(18, nearestGridSpacing * 0.45))
    : 8;
  if (closest.distanceSquared > hitRadius * hitRadius) return;

  const coordinate = String.fromCharCode(65 + closest.col) + (closest.row + 1);
  playInteractiveMove(coordinate);
}

let interactiveMovePending = false;

async function playInteractiveMove(coordinate) {
  if (!gameLoaded || interactiveMovePending) return;
  // Clicking after navigating backward branches from that position and drops
  // the old future, just as editing the trajectory would.
  const nextMoves = gameMoves.slice(0, gamePly).concat(coordinate);
  interactiveMovePending = true;
  try {
    // Ask the real Python backend to validate the move before mutating UI state.
    // On rejection, the currently displayed position and trajectory stay intact.
    await requestSnapshot(nextMoves);
    gameMoves = nextMoves;
    trajectoryEditor.value = gameMoves.join(' ');
    appliedTrajectory = trajectoryEditor.value.trim();
    updateApplyButtonState();
    await loadGamePosition(gameMoves.length);
    status.textContent = 'Played ' + coordinate + '. ' + currentPlayerName.toLowerCase() +
      ' to move; board and input tensors updated.';
  } catch (error) {
    status.textContent = 'Cannot play ' + coordinate + ': ' + error.message;
  } finally {
    interactiveMovePending = false;
    updateGameNavigation();
  }
}

// OrbitControls uses pointer drags to rotate/pan the scene. A drag that ends
// over a board intersection must never be interpreted as a move click.
let boardPointerDown = null;
let boardPointerDragged = false;
const BOARD_DRAG_THRESHOLD_PX = 5;

renderer.domElement.addEventListener('pointerdown', (event) => {
  if (event.button !== 0) return;
  boardPointerDown = { x: event.clientX, y: event.clientY };
  boardPointerDragged = false;
});

renderer.domElement.addEventListener('pointermove', (event) => {
  if (!boardPointerDown || boardPointerDragged) return;
  const dx = event.clientX - boardPointerDown.x;
  const dy = event.clientY - boardPointerDown.y;
  if (dx * dx + dy * dy > BOARD_DRAG_THRESHOLD_PX * BOARD_DRAG_THRESHOLD_PX) {
    boardPointerDragged = true;
  }
});

renderer.domElement.addEventListener('pointerup', () => {
  boardPointerDown = null;
});

renderer.domElement.addEventListener('pointercancel', () => {
  boardPointerDown = null;
  boardPointerDragged = false;
});

renderer.domElement.addEventListener('click', (event) => {
  if (boardPointerDragged) {
    boardPointerDragged = false;
    return;
  }
  const residualBlockIndex = pickResidualBlock(event, camera, renderer.domElement);
  if (residualBlockIndex !== null) {
    selectResidualBlock(residualBlockIndex);
    return;
  }
  handleSampleBoardClick(event);
});


document.querySelector('#demo-button').addEventListener('click', async () => {
  trajectoryEditor.value = DEMO_TRAJECTORY;
  await loadTrajectoryFromEditor(true);
  if (gameLoaded) {
    appliedTrajectory = trajectoryEditor.value.trim();
    updateApplyButtonState();
    // Loading a game changes the position, not the user's camera view.
    status.textContent = 'Demo game loaded at the end of the trajectory.';
  }
});
const trajectoryEditor = document.querySelector('#trajectory');
const applyChangesButton = document.querySelector('#apply-changes');
let appliedTrajectory = trajectoryEditor.value.trim();
function updateApplyButtonState() {
  const changed = trajectoryEditor.value.trim() !== appliedTrajectory;
  applyChangesButton.disabled = !changed;
  applyChangesButton.setAttribute('aria-disabled', String(!changed));
}
trajectoryEditor.addEventListener('input', updateApplyButtonState);
updateApplyButtonState();
applyChangesButton.addEventListener('click', async () => {
  if (applyChangesButton.disabled) return;
  await loadTrajectoryFromEditor(true);
  if (gameLoaded) {
    appliedTrajectory = trajectoryEditor.value.trim();
    updateApplyButtonState();
  }
});

async function requestSnapshot(moves) {
  const cacheKey = JSON.stringify({ moves, allow_scl: false, allow_swap: true });
  const cached = getCachedSnapshot(cacheKey);
  if (cached) {
    nnPerf.record('snapshotCacheHit', {
      moves: moves.length,
      responseBytes: cached.responseBytes,
      wireResponseBytes: cached.wireResponseBytes,
      cacheEntries: snapshotCache.size,
    });
    return cached.result;
  }

  const started = performance.now();
  const response = await fetch('/api/trace', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ moves, allow_scl: false, allow_swap: true }),
  });
  const headersMs = performance.now() - started;
  const bodyStarted = performance.now();
  // Measure when the first response bytes become readable separately from
  // the full body read. This distinguishes delivery latency from body draining.
  const reader = response.body.getReader();
  const chunks = [];
  let responseLength = 0;
  let firstChunkMs = null;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (firstChunkMs === null) firstChunkMs = performance.now() - bodyStarted;
    chunks.push(value);
    responseLength += value.length;
  }
  const responseBuffer = new Uint8Array(responseLength);
  let responseOffset = 0;
  for (const chunk of chunks) {
    responseBuffer.set(chunk, responseOffset);
    responseOffset += chunk.length;
  }
  const responseText = new TextDecoder().decode(responseBuffer);
  const bodyMs = performance.now() - bodyStarted;
  const parseStarted = performance.now();
  let result;
  try {
    result = JSON.parse(responseText);
  } catch (error) {
    throw new Error('The visualizer API returned invalid JSON: ' + error.message);
  }
  const jsonParseMs = performance.now() - parseStarted;
  if (!response.ok) throw new Error(result.error || response.statusText);
  // Fetch exposes the decompressed body text when Content-Encoding is gzip.
  // Keep decoded and on-wire sizes separate so the perf report stays accurate.
  const responseBytes = new TextEncoder().encode(responseText).length;
  const wireResponseBytes = Number(response.headers.get('content-length')) || responseBytes;
  nnPerf.record('requestSnapshot', {
    moves: moves.length,
    headersMs: Number(headersMs.toFixed(2)),
    firstChunkMs: firstChunkMs === null ? null : Number(firstChunkMs.toFixed(2)),
    bodyMs: Number(bodyMs.toFixed(2)),
    jsonParseMs: Number(jsonParseMs.toFixed(2)),
    totalMs: Number((performance.now() - started).toFixed(2)),
    responseBytes,
    wireResponseBytes,
    compression: response.headers.get('content-encoding') || 'none',
  });
  cacheSnapshot(cacheKey, { result, responseBytes, wireResponseBytes });
  return result;
}
async function loadGamePosition(ply) {
  const loadStarted = performance.now();
  // Rebuild the authoritative backend position from the move prefix. This
  // makes Back an undo and Forward a redo, including captures/links and the
  // model's perspective-dependent input encoding.
  const result = await requestSnapshot(gameMoves.slice(0, ply));
  backendSnapshot = result;
  currentPlayerName = result.next_player === 1 ? 'Red' : 'Black';

  data = {
    pegs: result.tensors.pegs,
    links: result.tensors.links,
    locations: result.tensors.locations,
    weights: demoModel.weights,
    bias: demoModel.bias,
    isDemo: false,
    linkConvMetadata: demoModel.linkConvMetadata,
    pegConvMetadata: demoModel.pegConvMetadata,
    locationConvMetadata: demoModel.locationConvMetadata,
    primaryBn: demoModel.primaryBn,
  };

  gamePly = ply;
  gameLoaded = true;
  cachedOutput = null;
  // Rebuild whichever visualization is currently selected, not just Inputs.
  // Keep the camera and selected tab stable while the board/tensors change.
  renderStep();
  updateGameNavigation();
  status.textContent = 'Game position ' + gamePly + '/' + gameMoves.length +
    '. Next player: ' + result.next_player_name +
    '. Board and input tensors updated. ' + modelWeightStatus;
  nnPerf.record('loadGamePosition', {
    ply,
    moves: gameMoves.length,
    totalMs: Number((performance.now() - loadStarted).toFixed(2)),
  });
}

function updateGameNavigation() {
  document.querySelector('#game-start').disabled = !gameLoaded || gamePly <= 0;
  document.querySelector('#game-back').disabled = !gameLoaded || gamePly <= 0;
  document.querySelector('#game-forward').disabled = !gameLoaded || gamePly >= gameMoves.length;
  document.querySelector('#game-end').disabled = !gameLoaded || gamePly >= gameMoves.length;
  const positionLabel = document.querySelector('#game-position');
  if (!gameLoaded) {
    positionLabel.textContent = 'No game loaded';
  } else {
    const player = currentPlayerName.toLowerCase() === 'red' ? 'red' : 'black';
    positionLabel.innerHTML =
      'Move ' + gamePly + ': ' +
      (gamePly ? gameMoves[gamePly - 1].toUpperCase() : '—') + '<br>' +
      (player === 'red'
        ? '<span style="color:#ff4545">red</span> to move'
        : 'black to move');
  }
}

let navigationRequest = 0;
async function navigateGame(delta) {
  if (!gameLoaded) return;
  const ply = Math.max(0, Math.min(gameMoves.length, gamePly + delta));
  if (ply === gamePly) return;

  const requestId = ++navigationRequest;
  document.querySelector('#game-start').disabled = true;
  document.querySelector('#game-back').disabled = true;
  document.querySelector('#game-forward').disabled = true;
  document.querySelector('#game-end').disabled = true;
  try {
    await loadGamePosition(ply);
  } catch (error) {
    status.textContent = 'Could not step through game: ' + error.message;
  } finally {
    // Ignore stale responses if the user clicked quickly in succession.
    if (requestId === navigationRequest) updateGameNavigation();
  }
}
async function loadTrajectoryFromEditor(showEnd = false) {
  const moves = document.querySelector('#trajectory').value
    .split(/[\s,]+/)
    .map((move) => move.trim())
    .filter(Boolean);
  if (moves.length > 500) {
    status.textContent = 'At most 500 moves are accepted.';
    return;
  }

  try {
    // Validate the entire trajectory before changing the active game.
    await requestSnapshot(moves);
    gameMoves = moves;
    gamePly = 0;
    gameLoaded = true;
    backendSnapshot = null;
    data = { ...createDemoData(), weights: demoModel.weights, bias: demoModel.bias, linkConvMetadata: demoModel.linkConvMetadata, pegConvMetadata: demoModel.pegConvMetadata, locationConvMetadata: demoModel.locationConvMetadata, primaryBn: demoModel.primaryBn };
    cachedOutput = null;
    updateGameNavigation();
    renderStep();
    if (showEnd) await loadGamePosition(gameMoves.length);
    status.textContent = showEnd
      ? 'Demo game loaded at the end of the trajectory. ' + modelWeightStatus
      : 'Trajectory loaded at the starting position. ' + modelWeightStatus;
  } catch (error) {
    gameLoaded = false;
    updateGameNavigation();
    status.textContent = 'Could not load trajectory: ' + error.message;
  }
}

document.querySelector('#game-start').addEventListener('click', () => navigateGame(-gamePly));
document.querySelector('#game-back').addEventListener('click', () => navigateGame(-1));
document.querySelector('#game-forward').addEventListener('click', () => navigateGame(1));
document.querySelector('#game-end').addEventListener('click', () => navigateGame(gameMoves.length - gamePly));
document.querySelector('#t1-file').addEventListener('change',async(event)=>{const file=event.target.files&&event.target.files[0];if(!file)return;try{const response=await fetch('/api/t1',{method:'POST',headers:{'Content-Type':'text/plain; charset=utf-8'},body:await file.text()});const result=await response.json();if(!response.ok)throw new Error(result.error||response.statusText);gameMoves=result.moves;document.querySelector('#trajectory').value=gameMoves.join(' ');appliedTrajectory=trajectoryEditor.value.trim();updateApplyButtonState();await loadGamePosition(gameMoves.length);status.textContent='Loaded '+file.name+'. '+status.textContent;}catch(error){status.textContent='Could not load T1 game: '+error.message;}finally{event.target.value='';}});
document.querySelector('#toggle-explanation').addEventListener('click', () => {
  const layout = document.querySelector('.app-layout');
  const collapsed = layout.classList.toggle('explanation-collapsed');
  document.querySelector('#toggle-explanation').textContent = collapsed ? 'Expand ›' : 'Collapse ‹';
  document.querySelector('#toggle-explanation').setAttribute('aria-label', collapsed ? 'Expand explanation panel' : 'Collapse explanation panel');
  document.querySelector('#toggle-explanation').title = collapsed ? 'Expand explanation panel' : 'Collapse explanation panel';
  // Resize the renderer after the grid has changed so the canvas matches its panel.
  requestAnimationFrame(resizeRenderer);
});
document.querySelector('#reset-camera').addEventListener('click', resetCamera);
window.addEventListener('resize', resizeRenderer);

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

async function initializeVisualizer() {
  // Load the trained peg, link, and location convolution weights before rendering.
  await loadTrainedLinkWeights();
  resizeRenderer();
  updateCameraCoordinates();
  renderStep();
  updateGameNavigation();
  animate();
  // Start with the demo game fully played out at the end of its trajectory.
  await loadTrajectoryFromEditor(true);
  if (gameLoaded) {
    appliedTrajectory = trajectoryEditor.value.trim();
    updateApplyButtonState();
  }
}
initializeVisualizer();
