import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/controls/OrbitControls.js';
import { createTensor3, flattenTensor } from './tensor-utils.js';
import { encodeGameInputs } from './game-encoding.js?v=20261031-game-encoding-v1';
import { createSnapshotClient } from './snapshot-client.js?v=20261031-snapshot-client-v2';
import { loadTrainedModelWeights } from './model-weight-client.js?v=20261031-policy-conv1-v2';
import { createBoardInteraction } from './board-interaction.js?v=20261031-grid-origin-v2';
import { createDemoData, calculateConvolutionOutput as calculateConvolution, validateResidualBlockPayload } from './model-data.js?v=20261031-policy-conv1-v2';
import { createLabelHelpers } from './labels.js?v=20261031-formula-upright-v6';
import { createBoardRenderer } from './renderer/board-renderer.js?v=20261031-axis-alignment-v2';
import { createLinkEncodingDiagnostics } from './link-encoding-diagnostics.js';
import { createTensorRenderer } from './renderer/tensor-renderer.js?v=20261031-policy-conv1-v2';

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
// Restore the night scene. Ambient illumination keeps standard-material
// objects equally bright throughout the deep network, regardless of camera
// distance or their orientation.
scene.background = new THREE.Color(0x0b1018);

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

// AmbientLight has no positional or directional falloff: it illuminates
// every standard-material object uniformly across the full network depth.
scene.add(new THREE.AmbientLight(0xffffff, 1.65));

const activeStep = 'inputs';
let sceneGroup = new THREE.Group();
const annotationGroup = new THREE.Group();
annotationGroup.name = 'persistent-static-annotations';
const staticSceneGroup = new THREE.Group();
staticSceneGroup.name = 'persistent-static-board';
const inputStaticTensorGroup = new THREE.Group();
inputStaticTensorGroup.name = 'persistent-input-tensor-static';
const inputDynamicTensorGroup = new THREE.Group();
inputDynamicTensorGroup.name = 'persistent-input-tensor-dynamic';
const filterStaticGroup = new THREE.Group();
filterStaticGroup.name = 'persistent-filter-geometry';
// Must be initialized before createDemoData({ BOARD_SIZE, FILTER_COUNT, encodeGameInputs }), which calls createSampleGame().
let backendSnapshot = null;
const demoModel = createDemoData({ BOARD_SIZE, FILTER_COUNT, encodeGameInputs });
let data = demoModel;
let modelWeightStatus = 'Using generated demo weights.';
const residualBlockWeightCache = new Map();
let selectedResidualBlockIndex = 0;
let residualSelectionPending = false;
let cachedOutput = null;
const convolutionOutputCache = new WeakMap();
const residualOutputCache = new WeakMap();
let gameMoves = [];
let gamePly = 0;
let gameLoaded = false;
let interactiveMovePending = false;
let currentPlayerName = "Black";
let residualBlock0Output = null;
let residualBlock0OutputError = null;
let residualBlockLastOutput = null;
let colorTanhA = 0;

scene.add(staticSceneGroup);
scene.add(inputStaticTensorGroup);
scene.add(inputDynamicTensorGroup);
scene.add(filterStaticGroup);
scene.add(sceneGroup);
scene.add(annotationGroup);

// Temporary baseline instrumentation for the performance work. This is
// observational only: it does not alter scene construction or game state.
const nnPerf = {
  enabled: false,
  samples: [],
  record(label, metrics) {
    if (!this.enabled) return null;
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

// Record browser main-thread tasks that can delay fetch promise callbacks even
// when Chrome's Network panel shows that the response has already arrived.
if (typeof PerformanceObserver !== 'undefined' &&
    PerformanceObserver.supportedEntryTypes?.includes('longtask')) {
  try {
    const longTaskObserver = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        nnPerf.record('mainThreadLongTask', {
          startMs: Number(entry.startTime.toFixed(2)),
          durationMs: Number(entry.duration.toFixed(2)),
          attribution: (entry.attribution || []).map(item => ({
            name: item.name,
            entryType: item.entryType,
            containerType: item.containerType,
          })),
        });
      }
    });
    longTaskObserver.observe({ type: 'longtask', buffered: true });
  } catch (error) {
    console.warn('[NN perf] Long-task observation unavailable:', error);
  }
}

const { requestSnapshot, requestPosition } = createSnapshotClient({
  recordPerf: (label, metrics) => nnPerf.record(label, metrics),
});

async function playInteractiveMove(coordinate) {
  if (!gameLoaded || interactiveMovePending) return;

  // Clicking after navigating backward branches from that position and drops
  // the old future, just as editing the trajectory would.
  const nextMoves = gameMoves.slice(0, gamePly).concat(coordinate);
  interactiveMovePending = true;
  try {
    // Let the backend validate the move before changing the displayed game.
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
const { addLabel, addRichLabel, addFormulaLabel, addMoveColumnLabel, beginFrame: beginLabelFrame, setBigLabelsVisible } = createLabelHelpers(THREE, () => sceneGroup);
const { addLabel: addStaticLabel, addRichLabel: addStaticRichLabel, setBigLabelsVisible: setStaticBigLabelsVisible } =
  createLabelHelpers(THREE, () => staticSceneGroup);
const {
  addLabel: addAnnotationLabel,
  addFormulaLabel: addAnnotationFormulaLabel,
  setBigLabelsVisible: setAnnotationBigLabelsVisible,
} = createLabelHelpers(THREE, () => annotationGroup);

/**
 * Convert a [row][column][channel] tensor into channel-major order.
 * The cube renderer iterates channels first, then rows and columns.
 */

function calculateConvolutionOutput() {
  const source = data.snapshot || data;
  if (data.snapshot && convolutionOutputCache.has(source)) return convolutionOutputCache.get(source);
  if (cachedOutput && cachedOutput.source === source) return cachedOutput.tensor;
  const tensor = calculateConvolution({ data, BOARD_SIZE, FILTER_COUNT, createTensor3 });
  if (data.snapshot) convolutionOutputCache.set(source, tensor);
  cachedOutput = { source, tensor };
  return tensor;
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
  drawInputs, drawFilters, drawOutput, refreshTensorColors, setResidualFiltersVisible,
  setShowIntermediateResidualBlocks, getBlock20CameraOffset,
  setSelectedResidualBlock, pickResidualBlock,
} = createTensorRenderer({
    THREE, BOARD_SIZE, FILTER_COUNT, COLORS, cubeGeometry, flattenTensor,
    createTensor3, getSceneGroup: () => sceneGroup, getData: () => data,
    getInputStaticGroup: () => inputStaticTensorGroup,
    getInputDynamicGroup: () => inputDynamicTensorGroup,
    getFilterStaticGroup: () => filterStaticGroup,
    getAnnotationGroup: () => annotationGroup,
    createSampleGame, getBackendSnapshot: () => backendSnapshot,
    drawSampleBoard, drawEffectiveBoard,
    logLinkEncodingDiagnostic, logFullInputTensorValidation,
    calculateConvolutionOutput, addLabel, addFormulaLabel,
    getColorTanhA: () => colorTanhA,
    addAnnotationLabel, addAnnotationFormulaLabel,
  });

// Render only when the scene or camera changes. OrbitControls emits "change"
// during damping as well, so scheduling another frame from that event keeps
// inertia smooth without a permanent requestAnimationFrame loop.
let renderFramePending = false;
function requestRender() {
  if (renderFramePending) return;
  renderFramePending = true;
  requestAnimationFrame(() => {
    renderFramePending = false;
    controls.update();
    renderer.render(scene, camera);
  });
}

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

  requestRender();

  if (nnPerf.enabled) {
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

  // Read renderer.info after the requested on-demand frame.
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
}

async function loadTrainedLinkWeights() {
  modelWeightStatus = await loadTrainedModelWeights({
    demoModel,
    filterCount: FILTER_COUNT,
    residualBlockWeightCache,
    onInitialResidualBlock: (weights) => {
      selectedResidualBlockIndex = 0;
      setSelectedResidualBlock(0, weights);
    },
    onInitialLastResidualBlock: (weights) => {
      setSelectedResidualBlock(data.residualBlockCount - 1, weights);
    },
  });
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
      weights = validateResidualBlockPayload(payload, blockIndex, FILTER_COUNT);
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
    requestRender();
  }, 180);
}
controls.addEventListener('start', () => {
  if (residualRestoreTimer !== null) {
    clearTimeout(residualRestoreTimer);
    residualRestoreTimer = null;
  }
  setResidualFiltersVisible(false);
  requestRender();
});
controls.addEventListener('change', () => {
  updateCameraCoordinates();
  requestRender();
});
// OrbitControls accumulates pointer/wheel input and applies it in update().
// Invalidate a frame on input so update() can consume that change; its "change"
// event then schedules further frames while damping is still moving the camera.
renderer.domElement.addEventListener('pointerdown', requestRender);
renderer.domElement.addEventListener('pointermove', requestRender);
renderer.domElement.addEventListener('wheel', requestRender, { passive: true });
// OrbitControls emits "change" throughout damping, not just while the user is
// dragging. Do not restart the visibility timer on every such frame: that can
// postpone the residual banks for seconds after motion is no longer noticeable.
controls.addEventListener('end', deferResidualFilterRestore);

let lastCameraPreset = null;
function setCameraPreset(position, target) {
  lastCameraPreset = null;
  camera.position.set(position[0], position[1], position[2]);
  controls.target.set(target[0], target[1], target[2]);
  controls.update();
  updateCameraCoordinates();
  requestRender();
}

function resetCamera() {
  lastCameraPreset = null;
  camera.position.copy(INITIAL_CAMERA_POSITION);
  controls.target.copy(INITIAL_CAMERA_TARGET);
  controls.update();
  updateCameraCoordinates();
  requestRender();
}

function resizeRenderer() {
  const width = viewport.clientWidth;
  const height = viewport.clientHeight;

  if (!width || !height) return;

  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  requestRender();
}

createBoardInteraction({
  THREE,
  boardSize: BOARD_SIZE,
  camera,
  controls,
  canvas: renderer.domElement,
  isEnabled: () => activeStep === 'inputs' && gameLoaded && !interactiveMovePending,
  onBoardMove: playInteractiveMove,
  onSceneClick: (event) => {
    const residualBlockIndex = pickResidualBlock(event, camera, renderer.domElement);
    if (residualBlockIndex === null) return false;
    selectResidualBlock(residualBlockIndex);
    return true;
  },
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

async function loadGamePosition(ply) {
  const loadStarted = performance.now();
  const requestStarted = performance.now();
  // Rebuild the authoritative backend position from the move prefix. This
  // makes Back an undo and Forward a redo, including captures/links and the
  // model's perspective-dependent input encoding.
  const position = demoModel.weights.residualBlock0
    ? await requestPosition(gameMoves.slice(0, ply))
    : { snapshot: await requestSnapshot(gameMoves.slice(0, ply)), residualBlock0Output: null };
  const requestWaitMs = performance.now() - requestStarted;
  const result = position.snapshot;
  const stateSetupStarted = performance.now();
  backendSnapshot = result;
  currentPlayerName = result.next_player === 1 ? 'Red' : 'Black';

  data = {
    snapshot: result,
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
    residualBlock0Output: null,
    residualBlockLastOutput: null,
    residualBlockCount: demoModel.residualBlockCount,
  };

  gamePly = ply;
  gameLoaded = true;
  const stateSetupMs = performance.now() - stateSetupStarted;
  const residualValidationStarted = performance.now();
  residualBlock0Output = null;
  residualBlockLastOutput = null;
  residualBlock0OutputError = null;
  if (demoModel.weights.residualBlock0) {
    const residualStarted = performance.now();
    try {
      const outputPayload = position.residualBlock0Output;
      const values = outputPayload.values;
      if (outputPayload.shape?.join('x') !== '24x24x48' ||
          !Array.isArray(values) || values.length !== 24 ||
          values.some(row => !Array.isArray(row) || row.length !== 24 ||
            row.some(cell => !Array.isArray(cell) || cell.length !== 48 ||
              cell.some(value => !Number.isFinite(Number(value)))))) {
        throw new Error('The backend returned an invalid 24×24×48 output tensor.');
      }
      residualBlock0Output = values;
      data.residualBlock0Output = values;
      const finalPayload = position.residualBlockLastOutput;
      if (finalPayload?.shape?.join('x') === '24x24x48' &&
          Array.isArray(finalPayload.values) && finalPayload.values.length === 24 &&
          finalPayload.values.every(row => Array.isArray(row) && row.length === 24 &&
            row.every(cell => Array.isArray(cell) && cell.length === 48 &&
              cell.every(value => Number.isFinite(Number(value)))))) {
        residualBlockLastOutput = finalPayload.values;
        data.residualBlockLastOutput = finalPayload.values;
      }
    } catch (error) {
      residualBlock0OutputError = error.message;
      console.error('[NN visualizer] Could not calculate residual block 0 output:', error);
    } finally {
      nnPerf.record('positionResidualValidation', {
        ply,
        cacheHit: false,
        fetchMs: 0,
        bodyMs: 0,
        jsonParseMs: 0,
        validationMs: Number((performance.now() - residualStarted).toFixed(2)),
        responseBytes: position.responseBytes || 0,
        wireResponseBytes: position.wireResponseBytes || 0,
        totalMs: Number((performance.now() - residualStarted).toFixed(2)),
        failed: Boolean(residualBlock0OutputError),
        includedInPositionRequest: true,
      });
    }
  }
  const residualValidationMs = performance.now() - residualValidationStarted;
  // Rebuild whichever visualization is currently selected, not just Inputs.
  // Keep the camera and selected tab stable while the board/tensors change.
  const renderStarted = performance.now();
  renderStep();
  const renderStepCallMs = performance.now() - renderStarted;
  const navigationStarted = performance.now();
  updateGameNavigation();
  status.textContent = 'Game position ' + gamePly + '/' + gameMoves.length +
    '. Next player: ' + result.next_player_name +
    (residualBlock0OutputError ? ' | Residual block 1 output unavailable: ' + residualBlock0OutputError : '') +
    '. Board and input tensors updated. ' + modelWeightStatus;
  const navigationAndStatusMs = performance.now() - navigationStarted;
  nnPerf.record('loadGamePosition', {
    ply,
    moves: gameMoves.length,
    requestWaitMs: Number(requestWaitMs.toFixed(2)),
    stateSetupMs: Number(stateSetupMs.toFixed(2)),
    residualValidationMs: Number(residualValidationMs.toFixed(2)),
    renderStepCallMs: Number(renderStepCallMs.toFixed(2)),
    navigationAndStatusMs: Number(navigationAndStatusMs.toFixed(2)),
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
    // Always load the backend's real position at the selected ply. In
    // particular, ply 0 must use the empty-board tensors from NetInputs,
    // not the generated demo tensors (which contain illustrative values).
    await loadGamePosition(showEnd ? gameMoves.length : 0);
    status.textContent = showEnd
      ? 'Demo game loaded at the end of the trajectory. ' + modelWeightStatus
      : 'Trajectory loaded at the starting position. ' + modelWeightStatus;
  } catch (error) {
    gameLoaded = false;
    updateGameNavigation();
    status.textContent = 'Could not load trajectory: ' + error.message;
  }
}

document.querySelector('#reset-game').addEventListener('click', async () => {
  // Invalidate any navigation response already in flight, then load the
  // authoritative empty backend position so all three input tensors refresh.
  navigationRequest += 1;
  gameMoves = [];
  trajectoryEditor.value = '';
  appliedTrajectory = '';
  updateApplyButtonState();
  document.querySelector('#reset-game').disabled = true;
  try {
    await loadGamePosition(0);
    status.textContent = 'Game reset. Empty-board inputs loaded. ' + modelWeightStatus;
  } catch (error) {
    status.textContent = 'Could not reset game: ' + error.message;
  } finally {
    document.querySelector('#reset-game').disabled = false;
    updateGameNavigation();
  }
});

document.querySelector('#game-start').addEventListener('click', () => navigateGame(-gamePly));
document.querySelector('#game-back').addEventListener('click', () => navigateGame(-1));
document.querySelector('#game-forward').addEventListener('click', () => navigateGame(1));
document.querySelector('#game-end').addEventListener('click', () => navigateGame(gameMoves.length - gamePly));
document.querySelector('#t1-file').addEventListener('change',async(event)=>{const file=event.target.files&&event.target.files[0];if(!file)return;try{const response=await fetch('/api/t1',{method:'POST',headers:{'Content-Type':'text/plain; charset=utf-8'},body:await file.text()});const result=await response.json();if(!response.ok)throw new Error(result.error||response.statusText);gameMoves=result.moves;document.querySelector('#trajectory').value=gameMoves.join(' ');appliedTrajectory=trajectoryEditor.value.trim();updateApplyButtonState();await loadGamePosition(gameMoves.length);status.textContent='Loaded '+file.name+'. '+status.textContent;}catch(error){status.textContent='Could not load T1 game: '+error.message;}finally{event.target.value='';}});
document.querySelector('#toggle-explanation').addEventListener('click', () => {
  const layout = document.querySelector('.app-layout');
  const collapsed = layout.classList.toggle('explanation-collapsed');
  document.querySelector('#toggle-explanation').textContent = collapsed ? '›' : '‹';
  document.querySelector('#toggle-explanation').setAttribute('aria-label', collapsed ? 'Expand explanation panel' : 'Collapse explanation panel');
  document.querySelector('#toggle-explanation').title = collapsed ? 'Expand explanation panel' : 'Collapse explanation panel';
  // Resize the renderer after the grid has changed so the canvas matches its panel.
  requestAnimationFrame(resizeRenderer);
});
document.querySelector('#reset-camera').addEventListener('click', resetCamera);
document.querySelector('#camera-input-board').addEventListener('click', () => setCameraPreset(
  [-39.64, 1.21, -0.69], [1.00, -1.00, 0.00],
));
document.querySelector('#camera-effective-board').addEventListener('click', () => setCameraPreset(
  [-25.33, 3.09, -1.30], [-2.14, -3.39, -0.86],
));
document.querySelector('#camera-input-front').addEventListener('click', () => setCameraPreset(
  [-25.10, 23.78, -3.62], [6.28, 1.20, -3.19],
));
document.querySelector('#camera-input-top').addEventListener('click', () => setCameraPreset(
  [-1.65, 28.72, -1.83], [4.81, -2.66, -1.77],
));
document.querySelector('#camera-residual-block').addEventListener('click', () => setCameraPreset(
  [47.69, 27.75, -41.17], [45.99, -0.37, 7.98],
));
function setResidualBlock20CameraPreset() {
  const offset = getBlock20CameraOffset();
  setCameraPreset(
    [901.83 - offset, 27.43, -43.67],
    [900.09 - offset, -14.86, 29.37],
  );
  lastCameraPreset = 'residual-block-20';
}
document.querySelector('#camera-residual-block-20').addEventListener('click', setResidualBlock20CameraPreset);
function setPolicyConv1CameraPreset() {
  // Keep the policy camera aligned with block 20 when intermediate blocks
  // are collapsed; both the output anchor and camera use the compact layout.
  const offset = getBlock20CameraOffset();
  setCameraPreset(
    [901.83 - offset, 27.43, -43.67],
    [900.09 - offset, -4.86, 21.7],
  );
  lastCameraPreset = 'policy-conv1';
}
document.querySelector('#camera-policy-conv1').addEventListener('click', setPolicyConv1CameraPreset);
document.querySelector('#toggle-residual-blocks-2-19').addEventListener('change', (event) => {
  setShowIntermediateResidualBlocks(event.target.checked);
  renderStep();
  // If the block-20 preset is active, keep it framed after the stack changes.
  if (lastCameraPreset === 'residual-block-20') setResidualBlock20CameraPreset();
});
const tanhSlider = document.querySelector('#tanh-a');
const tanhValue = document.querySelector('#tanh-a-value');
tanhSlider.addEventListener('input', () => {
  colorTanhA = Number(tanhSlider.value);
  tanhValue.value = colorTanhA.toFixed(2);
  tanhValue.textContent = colorTanhA.toFixed(2);
  refreshTensorColors();
  renderStep();
});
const bigLabelsToggle = document.querySelector('#toggle-big-labels');
bigLabelsToggle.addEventListener('change', () => {
  setBigLabelsVisible(bigLabelsToggle.checked);
  setStaticBigLabelsVisible(bigLabelsToggle.checked);
  setAnnotationBigLabelsVisible(bigLabelsToggle.checked);
  requestRender();
});
window.addEventListener('resize', resizeRenderer);

async function initializeVisualizer() {
  // Load the trained peg, link, and location convolution weights before rendering.
  await loadTrainedLinkWeights();
  resizeRenderer();
  updateCameraCoordinates();
  renderStep();
  updateGameNavigation();
  requestRender();
  // Start with the demo game fully played out at the end of its trajectory.
  await loadTrajectoryFromEditor(true);
  if (gameLoaded) {
    appliedTrajectory = trajectoryEditor.value.trim();
    updateApplyButtonState();
  }
}
initializeVisualizer();
