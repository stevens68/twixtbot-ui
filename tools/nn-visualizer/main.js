import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/controls/OrbitControls.js';

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

function createRandomGenerator(seed) {
  return function random() {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };
}

function createTensor3(dim1, dim2, dim3) {
  return Array.from({ length: dim1 }, () =>
    Array.from({ length: dim2 }, () => Array(dim3).fill(0)),
  );
}

function createTensor4(dim1, dim2, dim3, dim4, valueFactory) {
  return Array.from({ length: dim1 }, () =>
    Array.from({ length: dim2 }, () =>
      Array.from({ length: dim3 }, () =>
        Array.from({ length: dim4 }, valueFactory),
      ),
    ),
  );
}

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
function flattenTensor(tensor) {
  const values = [];

  for (let channel = 0; channel < tensor[0][0].length; channel += 1) {
    for (let row = 0; row < tensor.length; row += 1) {
      for (let col = 0; col < tensor[row].length; col += 1) {
        values.push(Number(tensor[row][col][channel]));
      }
    }
  }

  return values;
}

function addLabel(
  text,
  position,
  fontSize = 30,
  color = '#e7edf6',
  scaleDivisor = 45,
  alignment = 'center',
) {
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');

  context.font = '600 ' + fontSize + 'px Arial';
  canvas.width = Math.max(128, context.measureText(text).width + 28);
  canvas.height = fontSize + 26;

  context.font = '600 ' + fontSize + 'px Arial';
  context.fillStyle = color;
  context.textAlign = alignment;
  context.textBaseline = 'middle';
  context.fillText(
    text,
    alignment === 'left' ? 12 : canvas.width / 2,
    canvas.height / 2,
  );

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;

  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      depthTest: false,
    }),
  );

  const spriteWidth = canvas.width / scaleDivisor;
  // Anchor left-aligned sprites at their left edge. Sprite.center controls
  // the texture anchor; shifting the position alone can still look centered.
  if (alignment === 'left') sprite.center.set(0, 0.5);
  sprite.position.set(position[0], position[1], position[2]);
  sprite.scale.set(spriteWidth, canvas.height / scaleDivisor, 1);
  sceneGroup.add(sprite);
}

function addRichLabel(parts, position, fontSize = 30, scaleDivisor = 45, alignment = 'left') {
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  context.font = '600 ' + fontSize + 'px Arial';
  const widths = parts.map((part) => context.measureText(part.text).width);
  canvas.width = Math.max(128, widths.reduce((sum, width) => sum + width, 0) + 28);
  canvas.height = fontSize + 26;
  context.font = '600 ' + fontSize + 'px Arial';
  context.textAlign = 'left';
  context.textBaseline = 'middle';
  let x = alignment === 'center' ? (canvas.width - widths.reduce((sum, width) => sum + width, 0)) / 2 : 12;
  parts.forEach((part, index) => {
    context.fillStyle = part.color;
    context.fillText(part.text, x, canvas.height / 2);
    x += widths[index];
  });
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map: texture, transparent: true, depthTest: false,
  }));
  sprite.center.set(alignment === 'center' ? 0.5 : 0, 0.5);
  sprite.position.set(position[0], position[1], position[2]);
  sprite.scale.set(canvas.width / scaleDivisor, canvas.height / scaleDivisor, 1);
  sceneGroup.add(sprite);
}

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

function drawSampleBoard() {
  const game = createSampleGame();
  const spacing = 0.34;
  const boardSize = (BOARD_SIZE - 1) * spacing;
  const boardCenterZ = 0;
  const boardStartZ = boardCenterZ - boardSize / 2;
  const boardStartY = -boardSize / 2;
  // Keep the sample board four times farther from the input tensor layer.
  const boardX = -38.384; // 20% closer to the effective board at x = -15.2
  const pointAt = (row, col, x = boardX) => [
    x,
    boardStartY + (BOARD_SIZE - 1 - row) * spacing,
    boardStartZ + col * spacing,
  ];

  // A subtle board backing separates the real game from the tensor blades.
  const backing = new THREE.Mesh(
    new THREE.PlaneGeometry(boardSize + 0.42, boardSize + 0.42),
    new THREE.MeshStandardMaterial({
      color: 0xc8cacc,
      roughness: 0.88,
      transparent: true,
      opacity: 1,
      side: THREE.DoubleSide,
    }),
  );
  backing.rotation.y = -Math.PI / 2;
  backing.position.set(boardX + 0.035, 0, boardCenterZ);
  sceneGroup.add(backing);

  // Flat 2D finish lines lie directly on the board face (no cylinders or gaps).
  const finishLineX = boardX + 0.025;
  const makeFinishLine = (start, end, material, thickness = 0.065) => {
    // Use a flat rectangular strip instead of THREE.Line: WebGL line width is
    // commonly fixed at one pixel. The strip stays flush with the board face.
    const dy = Math.abs(end[1] - start[1]);
    const dz = Math.abs(end[2] - start[2]);
    const geometry = new THREE.BoxGeometry(0.008, dy || thickness, dz || thickness);
    const strip = new THREE.Mesh(geometry, material);
    strip.position.set(
      (start[0] + end[0]) / 2,
      (start[1] + end[1]) / 2,
      (start[2] + end[2]) / 2,
    );
    sceneGroup.add(strip);
  };
  const redFinish = new THREE.MeshBasicMaterial({ color: 0xe0525d, side: THREE.DoubleSide });
  const blackFinish = new THREE.MeshBasicMaterial({ color: 0x20252d, side: THREE.DoubleSide });
  const rowY = (row) => boardStartY + (BOARD_SIZE - 1 - row) * spacing;
  const colZ = (col) => boardStartZ + col * spacing;

  // Red goals: between rows 1–2 and 23–24, from column B through W.
  for (const boundaryY of [
    (rowY(0) + rowY(1)) / 2,
    (rowY(22) + rowY(23)) / 2,
  ]) {
    makeFinishLine(
      [finishLineX, boundaryY, colZ(1)],
      [finishLineX, boundaryY, colZ(22)],
      redFinish,
    );
  }

  // Black goals: between columns A–B and W–X, from row 1 through row 23.
  for (const boundaryZ of [
    (colZ(0) + colZ(1)) / 2,
    (colZ(22) + colZ(23)) / 2,
  ]) {
    makeFinishLine(
      [finishLineX, rowY(1), boundaryZ],
      [finishLineX, rowY(22), boundaryZ],
      blackFinish,
    );
  }

  // Holes are flat dark circles flush with the board face, not raised spheres.
  const holeGeometry = new THREE.CircleGeometry(0.035, 12);
  const holes = new THREE.InstancedMesh(
    holeGeometry,
    new THREE.MeshBasicMaterial({ color: 0x080a0d, side: THREE.DoubleSide }),
    BOARD_SIZE * BOARD_SIZE,
  );
  const holeTransform = new THREE.Object3D();
  let holeIndex = 0;
  for (let row = 0; row < BOARD_SIZE; row += 1) {
    for (let col = 0; col < BOARD_SIZE; col += 1) {
      const [x, y, z] = pointAt(row, col, boardX + 0.026);
      holeTransform.position.set(x, y, z);
      holeTransform.rotation.set(0, Math.PI / 2, 0);
      holeTransform.scale.setScalar(1);
      holeTransform.updateMatrix();
      holes.setMatrixAt(holeIndex++, holeTransform.matrix);
    }
  }
  holes.instanceMatrix.needsUpdate = true;
  sceneGroup.add(holes);

  // Use short cylinders for links: WebGL line width is often fixed at one pixel.
  // Keep each player's links in the same color as that player's pegs.
  const linkGeometry = new THREE.CylinderGeometry(0.025, 0.025, 1, 8);
  const linkMaterials = {
    0: new THREE.MeshStandardMaterial({ color: 0x20252d, roughness: 0.55 }),
    1: new THREE.MeshStandardMaterial({ color: 0xe0525d, roughness: 0.55 }),
  };
  const linkMeshes = {
    0: new THREE.Group(),
    1: new THREE.Group(),
  };
  const linkTransform = new THREE.Object3D();
  const up = new THREE.Vector3(0, 1, 0);
  // The sample board always shows the physical game in its native orientation:
  // Red goals top/bottom, Black goals left/right, independent of whose turn it is.
  // Only the effective board below uses naf.py's canonical perspective.
  const boardPoint = (peg, x = boardX) => pointAt(peg.row, peg.col, x);
  for (const link of game.links) {
    const a = new THREE.Vector3(...boardPoint(link.from, boardX - 0.32));
    const b = new THREE.Vector3(...boardPoint(link.to, boardX - 0.32));
    const direction = new THREE.Vector3().subVectors(b, a);
    linkTransform.position.copy(a).add(b).multiplyScalar(0.5);
    linkTransform.quaternion.setFromUnitVectors(up, direction.clone().normalize());
    linkTransform.scale.set(1, direction.length(), 1);
    linkTransform.updateMatrix();
    const linkMesh = new THREE.Mesh(linkGeometry, linkMaterials[link.player]);
    linkMesh.position.copy(linkTransform.position);
    linkMesh.quaternion.copy(linkTransform.quaternion);
    linkMesh.scale.copy(linkTransform.scale);
    linkMeshes[link.player].add(linkMesh);
  }
  sceneGroup.add(linkMeshes[0], linkMeshes[1]);

  // Slimmer, slightly longer pegs, kept in the sample game's native coordinates.
  const pegGeometry = new THREE.CylinderGeometry(0.082, 0.082, 0.36, 16);
  for (const player of [0, 1]) {
    const playerPegs = game.pegs.filter((peg) => peg.player === player);
    if (!playerPegs.length) continue;
    const pegMaterial = new THREE.MeshStandardMaterial({
      color: player === 1 ? 0xe0525d : 0x20252d,
      roughness: 0.38,
      metalness: 0.08,
    });
    const pegMesh = new THREE.InstancedMesh(pegGeometry, pegMaterial, playerPegs.length);
    const transform = new THREE.Object3D();
    playerPegs.forEach((peg, index) => {
      const [x, y, z] = boardPoint(peg, boardX - 0.16);
      transform.position.set(x, y, z);
      transform.rotation.z = Math.PI / 2;
      transform.scale.set(1, 1, 1);
      transform.updateMatrix();
      pegMesh.setMatrixAt(index, transform.matrix);
    });
    pegMesh.instanceMatrix.needsUpdate = true;
    sceneGroup.add(pegMesh);
  }

  // Coordinate labels follow the same A–X / 1–24 convention as the tensors.
  for (let col = 0; col < BOARD_SIZE; col += 1) {
    addLabel(
      String.fromCharCode(65 + col),
      [boardX - 0.12, 4.42, boardStartZ + col * spacing],
      14,
      '#e0e8e2',
      150,
    );
  }
  for (let row = 0; row < BOARD_SIZE; row += 1) {
    addLabel(
      String(row + 1),
      [boardX - 0.12, boardStartY + (BOARD_SIZE - 1 - row) * spacing, boardStartZ - 0.34],
      14,
      '#e0e8e2',
      150,
    );
  }

  const lastMove = game.moves.length ? game.moves[game.moves.length - 1].toUpperCase() : '—';
  const nextPlayerLabel = game.nextPlayer === 1 ? 'red' : 'black';
  addRichLabel(
    [
      { text: 'Last Move: ' + lastMove + ' - ', color: '#ffffff' },
      { text: nextPlayerLabel, color: nextPlayerLabel === 'red' ? '#ff4545' : '#e7edf6' },
      { text: ' to move', color: '#ffffff' },
    ],
    [boardX, 5.25, boardCenterZ],
    19,
    90,
    'center',
  );

  // Place the move list to the right of the board, in columns of 20 moves.
  // Each new column starts after moves 20, 40, 60, etc.
  const moveLabels = game.moves.map((move, index) => (index + 1) + '. ' + move.toUpperCase());
  const listX = boardX - 0.95;
  const firstListZ = boardStartZ + boardSize + 1.35;
  const movesPerColumn = 20;
  const rowSpacing = 0.38;
  const columnSpacing = 2.35;
  const listTopY = 3.4;
  const columnCount = Math.max(1, Math.ceil(moveLabels.length / movesPerColumn));
  for (let column = 0; column < columnCount; column += 1) {
    const columnZ = firstListZ + column * columnSpacing;
    if (column === 0) {
      addLabel('Moves', [listX, 3.95, columnZ], 16, '#e7edf6', 90, 'left');
    }
    const start = column * movesPerColumn;
    const end = Math.min(start + movesPerColumn, moveLabels.length);
    for (let index = start; index < end; index += 1) {
      const isRedMove = index % 2 === 0;
      addLabel(
        moveLabels[index],
        [listX, listTopY - (index - start) * rowSpacing, columnZ],
        14,
        isRedMove ? '#ff4545' : '#e7edf6',
        90,
        'left',
      );
    }
  }

  return game;
}


/**
 * Draw the canonical/effective position used to explain the network inputs.
 * Displays the encoded input tensors directly, so the effective board and
 * input blades use identical coordinates and channel assignments.
 */
function drawEffectiveBoard() {
  const game = createSampleGame();
  const spacing = 0.34;
  const boardSize = (BOARD_SIZE - 1) * spacing;
  // Keep the effective board twice as far from the input tensor layer.
  const boardX = -15.2;
  const start = -boardSize / 2;
  // Board and tensor share the same orientation: [0][0] is A1 (top-left).
  // Perspective, player remapping and link shifts are applied by the encoder.
  const pointAt = (row, col, x = boardX) => [
    x,
    start + (BOARD_SIZE - 1 - row) * spacing,
    start + col * spacing,
  ];

  const backing = new THREE.Mesh(
    new THREE.PlaneGeometry(boardSize + 0.42, boardSize + 0.42),
    new THREE.MeshStandardMaterial({
      color: 0xc8cacc, roughness: 0.88, side: THREE.DoubleSide,
    }),
  );
  backing.rotation.y = -Math.PI / 2;
  backing.position.set(boardX + 0.035, 0, 0);
  sceneGroup.add(backing);

  // The board grid uses tensor coordinates. For a Black-to-move game those
  // coordinates are transposed and colours remapped by the encoder; finish
  // edges and diagonal are transformed too, so the board remains recognizable
  // as a Twixt board in the encoded perspective.
  const finishLineX = boardX + 0.026;
  const makeFinishLine = (startPoint, endPoint, material, thickness = 0.065) => {
    // Flat, visibly thick strips; unlike WebGL lines, their width is reliable.
    const dy = Math.abs(endPoint[1] - startPoint[1]);
    const dz = Math.abs(endPoint[2] - startPoint[2]);
    const geometry = new THREE.BoxGeometry(0.008, dy || thickness, dz || thickness);
    const strip = new THREE.Mesh(geometry, material);
    strip.position.set(
      (startPoint[0] + endPoint[0]) / 2,
      (startPoint[1] + endPoint[1]) / 2,
      (startPoint[2] + endPoint[2]) / 2,
    );
    sceneGroup.add(strip);
  };
  const redFinish = new THREE.MeshBasicMaterial({ color: 0xe0525d });
  const blackFinish = new THREE.MeshBasicMaterial({ color: 0x20252d });
  const rowY = (row) => start + (BOARD_SIZE - 1 - row) * spacing;
  const colZ = (col) => start + col * spacing;

  // Keep goal-line colours and positions fixed in the displayed coordinate
  // system. Only encoded peg/link cells and their player colours are transposed.
  const blackToMove = game.nextPlayer === 0;
  // Match the sample board exactly: red finish lines top/bottom,
  // black finish lines left/right, regardless of the encoded perspective.
  for (const boundaryY of [
    (rowY(0) + rowY(1)) / 2,
    (rowY(22) + rowY(23)) / 2,
  ]) {
    makeFinishLine([finishLineX, boundaryY, colZ(1)], [finishLineX, boundaryY, colZ(22)], redFinish);
  }
  for (const boundaryZ of [
    (colZ(0) + colZ(1)) / 2,
    (colZ(22) + colZ(23)) / 2,
  ]) {
    makeFinishLine([finishLineX, rowY(1), boundaryZ], [finishLineX, rowY(22), boundaryZ], blackFinish);
  }

  const holeGeometry = new THREE.CircleGeometry(0.035, 12);
  const holes = new THREE.InstancedMesh(
    holeGeometry,
    new THREE.MeshBasicMaterial({ color: 0x080a0d, side: THREE.DoubleSide }),
    BOARD_SIZE * BOARD_SIZE,
  );
  const holeTransform = new THREE.Object3D();
  let holeIndex = 0;
  for (let row = 0; row < BOARD_SIZE; row += 1) {
    for (let col = 0; col < BOARD_SIZE; col += 1) {
      holeTransform.position.set(...pointAt(row, col, boardX + 0.027));
      holeTransform.rotation.set(0, Math.PI / 2, 0);
      holeTransform.updateMatrix();
      holes.setMatrixAt(holeIndex++, holeTransform.matrix);
    }
  }
  holes.instanceMatrix.needsUpdate = true;
  sceneGroup.add(holes);

  // Render each active encoded link cell as a cube standing out from the board.
  // Extend links clearly out from the board: X is the board-normal axis.
  // Their visible projection is 0.36, matching the peg height; board-plane
  // dimensions match the peg diameter.
  const linkGeometry = new THREE.BoxGeometry(0.36, 0.164, 0.164);
  const linkMaterials = [
    new THREE.MeshStandardMaterial({ color: 0x20252d, roughness: 0.55 }),
    new THREE.MeshStandardMaterial({ color: 0xe0525d, roughness: 0.55 }),
  ];
  const activeLinks = [];
  for (let row = 0; row < BOARD_SIZE; row += 1) {
    for (let col = 0; col < BOARD_SIZE; col += 1) {
      for (let channel = 0; channel < 8; channel += 1) {
        if (Number(data.links[row][col][channel]) !== 1) continue;

        // Mark the exact encoded tensor cell; do not reconstruct endpoints.
        const encodedPlayer = channel & 1;
        // Inputs are canonicalized to the Black-to-move viewpoint:
        // channel 0 is the opponent (red), channel 1 is side-to-move (black).
        const displayedPlayer = 1 - encodedPlayer;
        const mesh = new THREE.Mesh(linkGeometry, linkMaterials[displayedPlayer]);
        // The cube's back face touches the board face; the whole cube projects
        // outward, avoiding coplanar overlap with the backing and holes.
        mesh.position.set(...pointAt(row, col, boardX + 0.035 - 0.18));
        sceneGroup.add(mesh);

        // Two short pointers start at the cube's center when viewed from above.
        // Board north is +Y; west is -Z. Directions are (row/Y, col/Z).
        // Channel pairs: 0/1=SSE, 2/3=NNE, 4/5=ESE, 6/7=ENE.
        const pointerDirections = [
          [[1, -1], [-1, 0]],    // SSE: NW and south
          [[0, -1], [1, 0]],    // NNE: north and west
          [[1, -1], [0, 1]],    // ESE: NW and east
          [[0, -1], [1, 0]],    // ENE: north and west
        ];
        const dirs = pointerDirections[Math.floor(channel / 2)];
        const pointerLength = 0.14;
        // Use narrow cylinders instead of WebGL lines: LineBasicMaterial linewidth
        // is effectively fixed to one pixel in most browsers.
        const pointerRadius = 0.018;
        const pointerGeometry = new THREE.CylinderGeometry(
          pointerRadius, pointerRadius, pointerLength, 8,
        );
        // Flip pointer colors to match the opposite-color convention requested.
        const pointerMaterial = new THREE.MeshStandardMaterial({
          color: displayedPlayer === 1 ? 0xe0525d : 0x20252d,
          roughness: 0.7,
        });
        for (const [dy, dz] of dirs) {
          const direction = new THREE.Vector3(0, dy, dz).normalize();
          const pointer = new THREE.Mesh(pointerGeometry, pointerMaterial);
          pointer.position.set(
            mesh.position.x,
            mesh.position.y + dy * pointerLength / 2,
            mesh.position.z + dz * pointerLength / 2,
          );
          pointer.quaternion.setFromUnitVectors(
            new THREE.Vector3(0, 1, 0), direction,
          );
          sceneGroup.add(pointer);
        }
        activeLinks.push({ row, col, channel });
      }
    }
  }
  console.debug('[Effective board] active encoded link cells:', activeLinks);

  // Render pegs from the same encoded tensor used by the input blades.
  // Tensor channel 0 is the opponent (red); channel 1 is the side to move (black).
  // Encoder tensor coordinates already reflect the A1/X24 diagonal transpose.
  const pegGeometry = new THREE.CylinderGeometry(0.082, 0.082, 0.36, 16);
  for (let channel = 0; channel < 2; channel += 1) {
    const positions = [];
    for (let row = 0; row < BOARD_SIZE; row += 1) {
      for (let col = 0; col < BOARD_SIZE; col += 1) {
        if (Number(data.pegs[row][col][channel]) === 1) {
          positions.push({ row, col });
        }
      }
    }
    if (!positions.length) continue;
    // The encoded board always uses the Black-to-move viewpoint:
    // channel 0 is the opponent (red), channel 1 is side-to-move (black).
    // Coordinates and colors are already canonicalized in encodeGameInputs().
    const isBlack = channel === 1;
    const material = new THREE.MeshStandardMaterial({
      color: isBlack ? 0x20252d : 0xe0525d,
      roughness: 0.38,
      metalness: 0.08,
    });
    const mesh = new THREE.InstancedMesh(pegGeometry, material, positions.length);
    const transform = new THREE.Object3D();
    positions.forEach(({ row, col }, index) => {
      // Render the encoded tensor cell directly; the renderer does not transpose.
      transform.position.set(...pointAt(row, col, boardX - 0.16));
      transform.rotation.z = Math.PI / 2;
      transform.updateMatrix();
      mesh.setMatrixAt(index, transform.matrix);
    });
    mesh.instanceMatrix.needsUpdate = true;
    sceneGroup.add(mesh);
  }

  // Keep the familiar Twixt coordinates fixed, like the finish lines:
  // A–X run left-to-right across the top; 1–24 run top-to-bottom on the left.
  for (let col = 0; col < BOARD_SIZE; col += 1) {
    addLabel(
      String.fromCharCode(65 + col),
      [boardX - 0.12, start + (BOARD_SIZE - 1) * spacing + 0.34, start + col * spacing],
      14, '#e0e8e2', 150,
    );
  }
  for (let row = 0; row < BOARD_SIZE; row += 1) {
    addLabel(
      String(row + 1),
      [boardX - 0.12, start + (BOARD_SIZE - 1 - row) * spacing, start - 0.34],
      14, '#e0e8e2', 150,
    );
  }
  addRichLabel(
    [
      { text: 'Effective board - black to move', color: '#ffffff' },
    ],
    [boardX, 5.25, 0],
    19,
    90,
    'center',
  );
}

/**
 * Print a source-of-truth diagnostic for the link input blades.
 * The effective board is drawn from game.links; the blades are drawn from
 * data.links after canonicalization and the input-cell shift. Comparing both
 * lists makes channel-label, transpose, and one-cell-offset errors visible.
 */
function logLinkEncodingDiagnostic(game, linkTensor) {
  const transpose = game.nextPlayer === 0;
  // The upstream Game.links arrays are indexed [x][y]. On Black's
  // turn naf.py transposes that array, producing [row=y][col=x], which
  // matches the visualizer's { row, col } board coordinates. Do not swap
  // row/col again here.
  const canonicalPoint = (peg) => ({
    row: peg.row,
    col: peg.col,
  });
  const canonicalPlayer = (player) => transpose ? 1 - player : player;
  const channelNames = [
    'index 0', 'index 1', 'index 2', 'index 3',
    'index 4', 'index 5', 'index 6', 'index 7',
  ];

  // Reproduce the raw-link index calculation and the upstream naf.py
  // transpose/color swap and to_input_arrays shift independently of the
  // rendered tensor. This lets the log compare expected cells to actual data.
  const trace = game.links.map((link, index) => {
    const x1 = link.from.col;
    const y1 = link.from.row;
    const x2 = link.to.col;
    const y2 = link.to.row;
    const rawChannel = link.player
      + (((x1 + x2) % 2 !== 0) ? 4 : 0)
      + (((y2 - y1) * (x2 - x1) < 0) ? 2 : 0);
    const rawRow = Math.floor((x1 + x2) / 2);
    const rawCol = Math.floor((y1 + y2) / 2);

    // In the original encoder, game.links are indexed [x][y], then
    // transposed into [row][col]. Canonical row/col therefore start as y/x.
    const swappedColor = 1 - link.player;
    // Match NetInputs.init_from_game_black() in upstream naf.py:
    // raw 0-1 -> canonical 4-5; raw 2-3 -> 6-7;
    // raw 4-5 -> 0-1; raw 6-7 -> 2-3.
    // The colour within each pair is swapped.
    const midChannel =
      rawChannel < 2 ? 4 + swappedColor :
      rawChannel < 4 ? 6 + swappedColor :
      rawChannel < 6 ? swappedColor : 2 + swappedColor;

    // Canonical board coordinates use row=x, col=y when transposed.
    const canonicalA = canonicalPoint(link.from);
    const canonicalB = canonicalPoint(link.to);
    const canonicalRow = Math.floor((canonicalA.row + canonicalB.row) / 2);
    const canonicalCol = Math.floor((canonicalA.col + canonicalB.col) / 2);

    // to_input_arrays shifts the midpoint cell according to the channel bits.
    const vertical = (midChannel & 4) !== 0;
    const diff = (midChannel & 2) !== 0;
    const expectedRow = canonicalRow + ((vertical || diff) ? 1 : 0);
    const expectedCol = canonicalCol + ((!vertical || diff) ? 1 : 0);
    const inBounds = expectedRow >= 0 && expectedRow < BOARD_SIZE &&
      expectedCol >= 0 && expectedCol < BOARD_SIZE;
    const actualValue = inBounds ? linkTensor[expectedRow][expectedCol][midChannel] : null;

    return {
      link: index,
      originalPlayer: link.player === 1 ? 'Black' : 'White',
      originalFrom: String.fromCharCode(65 + link.from.col) + (link.from.row + 1),
      originalTo: String.fromCharCode(65 + link.to.col) + (link.to.row + 1),
      transpose,
      canonicalPlayer: canonicalPlayer(link.player) === 1 ? 'Black' : 'White',
      canonicalFrom: String.fromCharCode(65 + canonicalA.col) + (canonicalA.row + 1),
      canonicalTo: String.fromCharCode(65 + canonicalB.col) + (canonicalB.row + 1),
      rawChannel,
      rawMidpoint: '(' + rawRow + ',' + rawCol + ')',
      expectedChannel: midChannel,
      expectedChannelName: channelNames[midChannel],
      expectedCell: inBounds
        ? String.fromCharCode(65 + expectedCol) + (expectedRow + 1)
        : 'out of bounds',
      expectedCellValue: actualValue,
      foundAtExpectedCell: actualValue === 1,
    };
  });

  const actualCells = [];
  for (let channel = 0; channel < 8; channel += 1) {
    for (let row = 0; row < BOARD_SIZE; row += 1) {
      for (let col = 0; col < BOARD_SIZE; col += 1) {
        if (linkTensor[row][col][channel] !== 0) {
          actualCells.push({
            channel,
            cell: String.fromCharCode(65 + col) + (row + 1),
            row,
            col,
            value: linkTensor[row][col][channel],
          });
        }
      }
    }
  }

  console.groupCollapsed(
    '[Link encoding trace] transpose=' + transpose +
    ', nextPlayer=' + game.nextPlayer +
    ', board links=' + trace.length +
    ', nonzero tensor cells=' + actualCells.length,
  );
  console.log('Per-link trace: original -> canonical -> expected shifted tensor cell');
  console.table(trace);
  console.log('Actual nonzero tensor cells (channel labels intentionally omitted until verified):');
  console.table(actualCells);
  console.log('Loaded diagnostic build: link-trace-v4 (upstream channel + coordinate mapping). If this exact group title is absent, hard-reload before interpreting the output.');
  console.groupEnd();
}

/**
 * Independent full-tensor check against the upstream naf.py / NetInputs
 * transformations. It builds the raw [x][y][channel] link planes first,
 * applies the upstream transpose/color swap, then applies to_input_arrays'
 * per-channel shifts. This is a parity check of the JS encoder's complete
 * output, not a comparison against a saved TensorFlow runtime dump.
 */
function buildUpstreamReferenceInputs(game) {
  const rawPegs = Array.from({ length: 2 }, () => createTensor3(BOARD_SIZE, BOARD_SIZE, 1));
  const rawLinks = Array.from({ length: 8 }, () => createTensor3(BOARD_SIZE, BOARD_SIZE, 1));

  for (const peg of game.pegs) {
    rawPegs[peg.player][peg.col][peg.row][0] = 1;
  }
  for (const link of game.links) {
    const ax = link.from.col, ay = link.from.row;
    const bx = link.to.col, by = link.to.row;
    const channel = link.player
      + (((ax + bx) % 2 !== 0) ? 4 : 0)
      + (((by - ay) * (bx - ax) < 0) ? 2 : 0);
    const x = Math.floor((ax + bx) / 2);
    const y = Math.floor((ay + by) / 2);
    rawLinks[channel][x][y][0] = 1;
  }

  const blackToMove = game.nextPlayer === 0;
  const pegs = createTensor3(BOARD_SIZE, BOARD_SIZE, 2);
  const midpointLinks = createTensor3(BOARD_SIZE, BOARD_SIZE, 8);

  for (let x = 0; x < BOARD_SIZE; x += 1) {
    for (let y = 0; y < BOARD_SIZE; y += 1) {
      const row = blackToMove ? x : y;
      const col = blackToMove ? y : x;
      // naf.py changes perspective on Black's turn and swaps player planes.
      pegs[row][col][0] = rawPegs[blackToMove ? 1 : 0][blackToMove ? row : x][blackToMove ? col : y][0];
      pegs[row][col][1] = rawPegs[blackToMove ? 0 : 1][blackToMove ? row : x][blackToMove ? col : y][0];

      for (let rawChannel = 0; rawChannel < 8; rawChannel += 1) {
        const sourceChannel = blackToMove ? 1 - (rawChannel & 1) : rawChannel & 1;
        const group = rawChannel & 6;
        const sourceGroup = blackToMove
          ? (group === 0 ? 4 : group === 2 ? 6 : group === 4 ? 0 : 2)
          : group;
        const sourceIndex = sourceGroup + sourceChannel;
        midpointLinks[row][col][rawChannel] =
          rawLinks[sourceIndex][blackToMove ? row : x][blackToMove ? col : y][0];
      }
    }
  }

  const links = createTensor3(BOARD_SIZE, BOARD_SIZE, 8);
  for (let row = 0; row < BOARD_SIZE; row += 1) {
    for (let col = 0; col < BOARD_SIZE; col += 1) {
      for (let channel = 0; channel < 8; channel += 1) {
        const vertical = (channel & 4) !== 0;
        const diff = (channel & 2) !== 0;
        const sourceRow = row - ((vertical || diff) ? 1 : 0);
        const sourceCol = col - ((!vertical || diff) ? 1 : 0);
        if (sourceRow >= 0 && sourceCol >= 0) {
          links[row][col][channel] = midpointLinks[sourceRow][sourceCol][channel];
        }
      }
    }
  }
  return { pegs, links };
}

function compareTensors(label, actual, expected) {
  let mismatchCount = 0;
  let actualNonzero = 0;
  let expectedNonzero = 0;
  const examples = [];

  for (let row = 0; row < BOARD_SIZE; row += 1) {
    for (let col = 0; col < BOARD_SIZE; col += 1) {
      for (let channel = 0; channel < actual[row][col].length; channel += 1) {
        const a = actual[row][col][channel];
        const e = expected[row][col][channel];
        if (a !== 0) actualNonzero += 1;
        if (e !== 0) expectedNonzero += 1;
        if (a !== e) {
          mismatchCount += 1;
          if (examples.length < 12) {
            examples.push({
              cell: String.fromCharCode(65 + col) + (row + 1),
              row, col, channel, actual: a, expected: e,
            });
          }
        }
      }
    }
  }

  console.groupCollapsed(
    '[Full tensor validation] ' + label + ': ' +
    (mismatchCount === 0 ? 'PASS' : 'FAIL') +
    ', mismatches=' + mismatchCount,
  );
  console.table([{
    tensor: label, result: mismatchCount === 0 ? 'PASS' : 'FAIL',
    mismatches: mismatchCount, actualNonzero, expectedNonzero,
  }]);
  if (examples.length) console.table(examples);
  console.groupEnd();
  return { label, mismatchCount, actualNonzero, expectedNonzero };
}

function logFullInputTensorValidation(game, inputs) {
  const reference = buildUpstreamReferenceInputs(game);
  const results = [
    compareTensors('pegs 24×24×2', inputs.pegs, reference.pegs),
    compareTensors('links 24×24×8', inputs.links, reference.links),
  ];
  const totalMismatches = results.reduce((sum, result) => sum + result.mismatchCount, 0);
  console.log(
    '[Full tensor validation] ' +
    (totalMismatches === 0 ? 'PASS' : 'FAIL') +
    ': ' + totalMismatches + ' total cell mismatches. ' +
    'Reference reimplements upstream naf.py and NetInputs shifts; it is not yet a saved TF runtime comparison.',
  );
}

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
      'Move ' + gamePly + '/' + gameMoves.length + ': ' +
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
