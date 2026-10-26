import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/controls/OrbitControls.js';
import { createRandomGenerator, createTensor3, createTensor4, flattenTensor } from './tensor-utils.js';
import { createLabelHelpers } from './labels.js';
import { createBoardRenderer } from './board-renderer.js';
import { createLinkEncodingDiagnostics } from './link-encoding-diagnostics.js';

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
camera.position.set(36, 28, 48);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
viewport.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 0, 0);

scene.add(new THREE.HemisphereLight(0xcfe3ff, 0x253041, 2));

const keyLight = new THREE.DirectionalLight(0xffffff, 2.4);
keyLight.position.set(20, 40, 30);
scene.add(keyLight);

let activeStep = 'inputs';
let sceneGroup = new THREE.Group();
// Must be initialized before createDemoData(), which calls createSampleGame().
let backendSnapshot = null;
let data = createDemoData();
let cachedOutput = null;
let gameMoves = [];
let gamePly = 0;
let gameLoaded = false;
let currentPlayerName = "Black";

scene.add(sceneGroup);

const cubeGeometry = new THREE.BoxGeometry(1, 1, 1);
const { addLabel, addRichLabel } = createLabelHelpers(THREE, () => sceneGroup);

function createDemoData() {
  const random=createRandomGenerator(2026);
  const {pegs,links}=encodeGameInputs({pegs:[],links:[],nextPlayer:1});
  const locations=createTensor3(BOARD_SIZE,BOARD_SIZE,2);
  for(let row=0;row<BOARD_SIZE;row++) for(let col=0;col<BOARD_SIZE;col++) {
    locations[row][col][0]=col/BOARD_SIZE; locations[row][col][1]=row/BOARD_SIZE;
  }
  const weights={
    pegs:createTensor4(FILTER_COUNT,1,1,2,()=> (random()-0.5)*0.4),
    links:createTensor4(FILTER_COUNT,5,5,8,()=> (random()-0.5)*0.16),
    locations:createTensor4(FILTER_COUNT,1,1,2,()=> (random()-0.5)*0.3),
  };
  const bias=Array.from({length:FILTER_COUNT},()=> (random()-0.5)*0.15);
  return {pegs,links,locations,weights,bias,isDemo:true};
}

/**
 * Convert a [row][column][channel] tensor into channel-major order.
 * The cube renderer iterates channels first, then rows and columns.
 */
function addValueCubes(values, rows, cols, channels, origin, color, options = {}) {
  const maxAbs = Math.max(1e-8, ...values.map(Math.abs));
  const spacing = options.spacing || 0.34;
  const cellSize = options.cellSize || 0.26;
  const channelGap = options.channelGap ?? 0.2;
  const depthStep = cellSize + channelGap;
  const maxThickness = options.maxThickness || 0.4;
  // Zero values must have exactly zero thickness and therefore be invisible.

  const mesh = new THREE.InstancedMesh(
    cubeGeometry,
    new THREE.MeshStandardMaterial({
      color,
      roughness: 0.6,
      transparent: true,
      opacity: 0.82,
    }),
    values.length,
  );

  const transform = new THREE.Object3D();
  let index = 0;

  for (let channel = 0; channel < channels; channel += 1) {
    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        const value = values[index];
        const isZero = value === 0;
        // Keep empty tensor cells visible as tiny dot-like cubes instead of
        // full-size blades. Nonzero values retain their signed blade shape.
        const dotSize = Math.min(cellSize, 0.055);
        const thickness = isZero
          ? dotSize
          : Math.max(0.025, (Math.abs(value) / maxAbs) * maxThickness);

        // Signed values extend away from the zero plane along the X axis.
        // Each channel is a separate blade. Stack blades along X, perpendicular
        // to the 24×24 board plane, rather than along the column axis (Z).
        // Zero values sit on the channel plane as small cube dots.
        transform.position.set(
          origin[0] + channel * depthStep +
            (isZero ? 0 : value < 0 ? -thickness / 2 : thickness / 2),
          origin[1] + (rows - 1 - row) * spacing,
          origin[2] + col * spacing,
        );
        transform.scale.set(
          thickness,
          isZero ? dotSize : cellSize,
          isZero ? dotSize : cellSize,
        );
        transform.updateMatrix();

        mesh.setMatrixAt(index, transform.matrix);
        index += 1;
      }
    }
  }

  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  sceneGroup.add(mesh);

  return {
    count: values.length,
    min: Math.min(...values),
    max: Math.max(...values),
  };
}

function calculateConvolutionOutput() {
  if (cachedOutput && cachedOutput.source === data) {
    return cachedOutput.tensor;
  }

  const output = createTensor3(BOARD_SIZE, BOARD_SIZE, FILTER_COUNT);
  const weights = data.weights;
  const bias = data.bias || Array(FILTER_COUNT).fill(0);

  // This is a demo implementation of the sum of the three convolution
  // branches. Confirm padding and bias placement against the PyTorch model.
  for (let row = 0; row < BOARD_SIZE; row += 1) {
    for (let col = 0; col < BOARD_SIZE; col += 1) {
      for (let filter = 0; filter < FILTER_COUNT; filter += 1) {
        let sum = Number(bias[filter] || 0);

        // 5x5 link convolution with zero padding.
        for (let kernelRow = 0; kernelRow < 5; kernelRow += 1) {
          for (let kernelCol = 0; kernelCol < 5; kernelCol += 1) {
            const inputRow = row + kernelRow - 2;
            const inputCol = col + kernelCol - 2;

            if (
              inputRow < 0 ||
              inputRow >= BOARD_SIZE ||
              inputCol < 0 ||
              inputCol >= BOARD_SIZE
            ) {
              continue;
            }

            for (let channel = 0; channel < 8; channel += 1) {
              sum +=
                Number(data.links[inputRow][inputCol][channel]) *
                Number(
                  weights.links[filter][kernelRow][kernelCol][channel],
                );
            }
          }
        }

        // 1x1 peg and location branches.
        for (let channel = 0; channel < 2; channel += 1) {
          sum +=
            Number(data.pegs[row][col][channel]) *
            Number(weights.pegs[filter][0][0][channel]);
          sum +=
            Number(data.locations[row][col][channel]) *
            Number(weights.locations[filter][0][0][channel]);
        }

        output[row][col][filter] = sum;
      }
    }
  }

  cachedOutput = { source: data, tensor: output };
  return output;
}

function drawFilterKernel(
  filterIndex,
  originX,
  originY,
  originZ,
  weights,
  kernelHeight,
  kernelWidth,
  channels,
  label,
  showBias = false,
) {
  const values = flattenTensor(weights[filterIndex]);
  const maxAbs = Math.max(1e-8, ...values.map(Math.abs));
  const spacing = 0.18;
  const cellSize = 0.135;
  const channelDepth = cellSize + 0.055;

  const mesh = new THREE.InstancedMesh(
    cubeGeometry,
    new THREE.MeshStandardMaterial({ color: COLORS.weight, roughness: 0.65 }),
    values.length,
  );

  const transform = new THREE.Object3D();
  let index = 0;

  for (let channel = 0; channel < channels; channel += 1) {
    for (let row = 0; row < kernelHeight; row += 1) {
      for (let col = 0; col < kernelWidth; col += 1) {
        const value = values[index];
        const thickness = Math.max(
          0.025,
          (Math.abs(value) / maxAbs) * 0.17,
        );

        transform.position.set(
          originX + (value < 0 ? -thickness / 2 : thickness / 2),
          originY + (kernelHeight - 1 - row) * spacing,
          originZ + col * spacing + channel * channelDepth,
        );
        transform.scale.set(thickness, cellSize, cellSize);
        transform.updateMatrix();

        mesh.setMatrixAt(index, transform.matrix);
        index += 1;
      }
    }
  }

  mesh.instanceMatrix.needsUpdate = true;
  sceneGroup.add(mesh);

  if (showBias) {
    const biasCube = new THREE.Mesh(
      cubeGeometry,
      new THREE.MeshStandardMaterial({ color: COLORS.bias }),
    );

    biasCube.position.set(
      originX + 0.42,
      originY + ((kernelHeight - 1) * spacing) / 2,
      originZ + kernelWidth * spacing + channels * channelDepth * 0.5,
    );
    biasCube.scale.setScalar(0.22);
    sceneGroup.add(biasCube);
  }

  addLabel(
    label,
    [
      originX,
      originY + kernelHeight * spacing + 0.2,
      originZ + channels * channelDepth * 0.3,
    ],
    17,
  );
}

function clearSceneGroup() {
  scene.remove(sceneGroup);

  sceneGroup.traverse((object) => {
    if (object.isSprite) {
      if (object.material.map) object.material.map.dispose();
      object.material.dispose();
    } else if (object.isMesh) {
      object.material.dispose();
    }
  });

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

function drawInputs() {
  const game = createSampleGame();
  drawSampleBoard();
  drawEffectiveBoard();
  if (!backendSnapshot) {
    logLinkEncodingDiagnostic(game, data.links);
    logFullInputTensorValidation(game, data);
  }

  const tensors = [
    {
      key: 'pegs',
      label: 'Pegs · 24×24×2',
      x: -1.8,
      z: -13,
      channels: 2,
      bladeLabels: ['Red', 'Black'],
      color: COLORS.input,
    },
    {
      key: 'links',
      label: 'Links · 24×24×8',
      x: -1.8,
      z: -3.5,
      channels: 8,
      // Channel bits follow the Python Game/NetInputs convention:
      // bit 0 is colour (even=Black, odd=White), bit 2 is long-y,
      // and bit 1 is different-sign slope. These describe the tensor channels.
      bladeLabels: [
        'Ch 0: red SSE/NNW',
        'Ch 1: black SSE/NNW',
        'Ch 2: red ENE/WSW',
        'Ch 3: black ENE/WSW',
        'Ch 4: red ESE',
        'Ch 5: black ESE',
        'Ch 6: red ENE/WSW',
        'Ch 7: black ENE/WSW',
      ],
      color: COLORS.input,
    },
    {
      key: 'locations',
      label: 'Locations · 24×24×2',
      x: -1.8,
      z: 6,
      channels: 2,
      bladeLabels: ['X', 'Y'],
      color: COLORS.fixed,
    },
  ];

  tensors.forEach((tensor) => {
    addValueCubes(
      flattenTensor(data[tensor.key]),
      BOARD_SIZE,
      BOARD_SIZE,
      tensor.channels,
      [
        tensor.x - ((tensor.channels - 1) * (0.26 + 0.28)) / 2,
        -4,
        tensor.z,
      ],
      tensor.color,
      {
        spacing: 0.34,
        cellSize: 0.26,
        maxThickness: 0.34,
        channelGap: 0.28,
      },
    );

    // Keep the tensor-level label and the per-blade labels.
    addLabel(
      tensor.label,
      [tensor.x, 7.0, tensor.z + ((BOARD_SIZE - 1) * 0.34) / 2],
      22,
      '#ffffff',
      70,
    );

    const firstBladeX =
      tensor.x - ((tensor.channels - 1) * (0.26 + 0.28)) / 2;
    const boardTopY = -4 + (BOARD_SIZE - 1) * 0.34;

    tensor.bladeLabels.forEach((bladeLabel, channel) => {
      addLabel(
        bladeLabel,
        [firstBladeX + channel * (0.26 + 0.28), 5.65, tensor.z - 0.08],
        20,
        '#e7edf6',
        115,
        'left',
      );
    });

    // Tensor coordinates are array indices: both axes are numbered 0–23.
    // Show the coordinate grid only for the links tensor, as requested.
    if (tensor.key === 'links') {
      for (let col = 0; col < BOARD_SIZE; col += 1) {
        addLabel(
          String(col),
          [firstBladeX, 4.6, tensor.z + col * 0.34],
          17,
          '#dce7f5',
          145,
          'center',
        );
      }

      for (let row = 0; row < BOARD_SIZE; row += 1) {
        addLabel(
          String(row),
          [firstBladeX - 0.12, boardTopY - row * 0.34, tensor.z - 0.48],
          17,
          '#dce7f5',
          145,
          'center',
        );
      }
    }
  });

  return {
    title: 'Input tensors',
    description:
      'The 13-move sample game is encoded into the peg and link tensors. Locations remain fixed coordinate inputs; convolution weights are reproducible demo values.',
    shape: '24 × 24 × (2 + 8 + 2)',
    range: 'Binary demo inputs',
    count: '3 tensors · 6,912 cells',
  };
}

function drawFilters() {
  for (let filter = 0; filter < FILTER_COUNT; filter += 1) {
    const column = filter % 8;
    const row = Math.floor(filter / 8);
    const x = -18.5 + column * 4.8;
    const y = -4 + row * 5.3;

    drawFilterKernel(
      filter,
      x,
      y,
      -1.5,
      data.weights.links,
      5,
      5,
      8,
      'F' + filter + ' · link',
      true,
    );

    drawFilterKernel(
      filter,
      x + 1.65,
      y,
      2.7,
      data.weights.pegs,
      1,
      1,
      2,
      'peg',
    );

    drawFilterKernel(
      filter,
      x + 2.75,
      y,
      2.7,
      data.weights.locations,
      1,
      1,
      2,
      'loc',
    );
  }

  return {
    title: '48 convolution filters',
    description:
      'Eight columns × six rows. Each filter contains link, peg, and location kernels plus a bias cube.',
    shape: '48 filters · 8 × 6 grid',
    range: 'Signed weights and bias',
    count: '48 filter groups',
  };
}

function drawOutput() {
  const values = flattenTensor(calculateConvolutionOutput());

  const stats = addValueCubes(
    values,
    BOARD_SIZE,
    BOARD_SIZE,
    FILTER_COUNT,
    [-4.5, -4, -((FILTER_COUNT - 1) * 0.105) / 2],
    COLORS.output,
    {
      spacing: 0.34,
      cellSize: 0.24,
      maxThickness: 0.42,
      minThickness: 0.025,
      channelGap: 0.105,
    },
  );

  addLabel('Summed convolution output · 24×24×48', [0, 5, 2], 28);

  return {
    title: 'Convolution output',
    description:
      'Three branch convolutions are summed into one feature tensor. Thickness is scaled for visibility.',
    shape: '24 × 24 × 48',
    range: stats.min.toFixed(4) + ' … ' + stats.max.toFixed(4),
    count: stats.count.toLocaleString() + ' cells',
  };
}

function renderStep() {
  clearSceneGroup();

  let info;

  if (activeStep === 'inputs') {
    info = drawInputs();
  } else if (activeStep === 'filters') {
    info = drawFilters();
  } else {
    info = drawOutput();
  }

  document.querySelector('#view-title').textContent = info.title;
  document.querySelector('#view-description').textContent = info.description;
  document.querySelector('#tensor-shape').textContent = info.shape;
  document.querySelector('#value-range').textContent = info.range;
  document.querySelector('#cube-count').textContent = info.count;

  document.querySelectorAll('.step-tab').forEach((button) => {
    const selected = button.dataset.step === activeStep;
    button.classList.toggle('active', selected);
    button.setAttribute('aria-selected', String(selected));
  });

  loading.style.display = 'none';
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
    validateWeights(input.weights.pegs, 'pegs', 1, 1, 2);
    validateWeights(input.weights.links, 'links', 5, 5, 8);
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

function resetCamera() {
  camera.position.set(36, 28, 48);
  controls.target.set(0, 0, 0);
  controls.update();
}

function resizeRenderer() {
  const width = viewport.clientWidth;
  const height = viewport.clientHeight;

  if (!width || !height) return;

  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

document.querySelectorAll('.step-tab').forEach((button) => {
  button.addEventListener('click', () => {
    activeStep = button.dataset.step;
    renderStep();
  });
});

document.querySelector('#demo-button').addEventListener('click', async () => {
  trajectoryEditor.value = DEMO_TRAJECTORY;
  await loadTrajectoryFromEditor(true);
  if (gameLoaded) {
    appliedTrajectory = trajectoryEditor.value.trim();
    updateApplyButtonState();
    resetCamera();
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
 const response=await fetch('/api/trace',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({moves,allow_scl:false,allow_swap:true})});
 const result=await response.json();if(!response.ok)throw new Error(result.error||response.statusText);return result;
}
async function loadGamePosition(ply) {
  // Rebuild the authoritative backend position from the move prefix. This
  // makes Back an undo and Forward a redo, including captures/links and the
  // model's perspective-dependent input encoding.
  const result = await requestSnapshot(gameMoves.slice(0, ply));
  backendSnapshot = result;
  currentPlayerName = result.next_player_name || (result.next_player === 0 ? 'Black' : 'Red');

  const demo = createDemoData();
  data = {
    pegs: result.tensors.pegs,
    links: result.tensors.links,
    locations: result.tensors.locations,
    weights: demo.weights,
    bias: demo.bias,
    isDemo: false,
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
    '. Board and input tensors updated; filters remain demo weights.';
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
    data = createDemoData();
    cachedOutput = null;
    updateGameNavigation();
    renderStep();
    if (showEnd) await loadGamePosition(gameMoves.length);
    status.textContent = showEnd
      ? 'Demo game loaded at the end of the trajectory.'
      : 'Trajectory loaded at the starting position.';
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

resizeRenderer();
renderStep();
updateGameNavigation();
animate();
// Start with the demo game fully played out at the end of its trajectory.
loadTrajectoryFromEditor(true).then(() => {
  if (gameLoaded) {
    appliedTrajectory = trajectoryEditor.value.trim();
    updateApplyButtonState();
  }
});
