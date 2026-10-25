import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/controls/OrbitControls.js';

const BOARD_SIZE = 24;
const FILTER_COUNT = 48;

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
let data = createDemoData();
let cachedOutput = null;

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
  const {pegs,links}=encodeGameInputs(createSampleGame());
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
        const thickness = isZero
          ? 0.018
          : (Math.abs(value) / maxAbs) * maxThickness;

        // Zero cells use a very thin blue sliver so they remain visible without
        // appearing as black edges. Nonzero signed values scale with magnitude.
        if (isZero && color === COLORS.input) {
          mesh.setColorAt(index, new THREE.Color(0x78c7f4));
        } else if (isZero && color === COLORS.fixed) {
          mesh.setColorAt(index, new THREE.Color(0xb6d8ed));
        }

        // Signed values extend away from the zero plane along the X axis.
        // Each channel is a separate blade. Stack blades along X, perpendicular
        // to the 24×24 board plane, rather than along the column axis (Z).
        // Signed value thickness still extends from each blade's X position.
        transform.position.set(
          origin[0] + channel * depthStep +
            (isZero ? 0 : value < 0 ? -thickness / 2 : thickness / 2),
          origin[1] + (rows - 1 - row) * spacing,
          origin[2] + col * spacing,
        );
        transform.scale.set(thickness, cellSize, cellSize);
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
  const moves = ['L12','L8','O9','L17','H16','I14','J13','N9','Q10','K6','H12','O11','F7'];
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
  const rawPegs=Array.from({length:2},()=>createTensor3(BOARD_SIZE,BOARD_SIZE,1));
  const rawLinks=Array.from({length:8},()=>createTensor3(BOARD_SIZE,BOARD_SIZE,1));
  for (const p of game.pegs) rawPegs[p.player][p.col][p.row][0]=1;
  for (const link of game.links) {
    const x1=link.from.col,y1=link.from.row,x2=link.to.col,y2=link.to.row;
    const ix=link.player+(((x1+x2)%2!==0)?4:0)+(((y2-y1)*(x2-x1)<0)?2:0);
    rawLinks[ix][Math.floor((x1+x2)/2)][Math.floor((y1+y2)/2)][0]=1;
  }
  const pegs=createTensor3(BOARD_SIZE,BOARD_SIZE,2), mid=createTensor3(BOARD_SIZE,BOARD_SIZE,8);
  // Thirteen moves means BLACK is next: channel 0 is WHITE/opponent, 1 BLACK/to-move.
  for(let r=0;r<BOARD_SIZE;r++) for(let col=0;col<BOARD_SIZE;col++) {
    pegs[r][col][0]=rawPegs[1][col][r][0]; pegs[r][col][1]=rawPegs[0][col][r][0];
    for(let color=0;color<2;color++) {
      const opp=1-color;
      mid[r][col][4+color]=rawLinks[opp][col][r][0];
      mid[r][col][6+color]=rawLinks[2+opp][col][r][0];
      mid[r][col][color]=rawLinks[4+opp][col][r][0];
      mid[r][col][2+color]=rawLinks[6+opp][col][r][0];
    }
  }
  const links=createTensor3(BOARD_SIZE,BOARD_SIZE,8);
  for(let r=0;r<BOARD_SIZE;r++) for(let col=0;col<BOARD_SIZE;col++) for(let ch=0;ch<8;ch++) {
    const vertical=(ch&4)!==0,diff=(ch&2)!==0;
    const sr=r-((vertical||diff)?1:0), sc=col-((!vertical||diff)?1:0);
    if(sr>=0&&sc>=0) links[r][col][ch]=mid[sr][sc][ch];
  }
  return {pegs,links};
}

function drawSampleBoard() {
  const game = createSampleGame();
  const spacing = 0.34;
  const boardSize = (BOARD_SIZE - 1) * spacing;
  const boardCenterZ = 0;
  const boardStartZ = boardCenterZ - boardSize / 2;
  const boardStartY = -boardSize / 2;
  const boardX = -15.5;
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
  for (const link of game.links) {
    const a = new THREE.Vector3(...pointAt(link.from.row, link.from.col, boardX - 0.32));
    const b = new THREE.Vector3(...pointAt(link.to.row, link.to.col, boardX - 0.32));
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

  // Slimmer, slightly longer pegs.
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
      const [x, y, z] = pointAt(peg.row, peg.col, boardX - 0.16);
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

  addLabel(
    'Sample Game',
    [boardX - 0.25, 5.25, boardCenterZ],
    19,
    '#ffffff',
    90,
  );

  // Move list is placed on the far side of the board, away from the tensors.
  const moveLabels = ['1. L12','2. L8','3. O9','4. L17','5. H16','6. I14','7. J13','8. N9','9. Q10','10. K6','11. H12','12. O11','13. F7'];
  const listX = boardX - 0.95;
  const listZ = boardStartZ - 1.35;
  addLabel('Moves', [listX, 3.95, listZ], 16, '#e7edf6', 90, 'left');
  moveLabels.forEach((label, index) => {
    addLabel(label, [listX, 3.4 - index * 0.38, listZ], 14, '#e7edf6', 90, 'left');
  });

  return game;
}


/**
 * Draw the canonical/effective position used to explain the network inputs.
 * White is always the side to move in this view. If Black is to move in the
 * sample game, transpose coordinates and swap the displayed player colours.
 */
function drawEffectiveBoard() {
  const game = createSampleGame();
  const spacing = 0.34;
  const boardSize = (BOARD_SIZE - 1) * spacing;
  const boardX = -8.5;
  const start = -boardSize / 2;
  const transpose = game.nextPlayer === 0;
  const effectivePlayer = (player) => transpose ? 1 - player : player;
  const pointAt = (row, col, x = boardX) => {
    // Match naf.py's matrix transpose: source (row, col) -> (col, row).
    const effectiveRow = transpose ? col : row;
    const effectiveCol = transpose ? row : col;
    return [
      x,
      start + (BOARD_SIZE - 1 - effectiveRow) * spacing,
      start + effectiveCol * spacing,
    ];
  };

  const backing = new THREE.Mesh(
    new THREE.PlaneGeometry(boardSize + 0.42, boardSize + 0.42),
    new THREE.MeshStandardMaterial({
      color: 0xc8cacc, roughness: 0.88, side: THREE.DoubleSide,
    }),
  );
  backing.rotation.y = -Math.PI / 2;
  backing.position.set(boardX + 0.035, 0, 0);
  sceneGroup.add(backing);

  // A thin 2D line marks the main-diagonal transpose axis directly on the board.
  if (transpose) {
    const diagonalGeometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(
        boardX + 0.026,
        start + (BOARD_SIZE - 1) * spacing,
        start,
      ),
      new THREE.Vector3(
        boardX + 0.026,
        start,
        start + (BOARD_SIZE - 1) * spacing,
      ),
    ]);
    // Dotted black transpose axis, thinner than the board grid.
    const diagonal = new THREE.Line(
      diagonalGeometry,
      new THREE.LineDashedMaterial({
        color: 0x080a0d,
        dashSize: 0.075,
        gapSize: 0.075,
        linewidth: 1,
      }),
    );
    diagonal.computeLineDistances();
    sceneGroup.add(diagonal);
  }

  // Finish lines are transformed with the board. Transposition swaps the
  // goal axes, and colour canonicalization swaps the players, so the effective
  // view retains the canonical red-horizontal / black-vertical appearance.
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

  const linkGeometry = new THREE.CylinderGeometry(0.025, 0.025, 1, 8);
  const linkMaterials = {
    0: new THREE.MeshStandardMaterial({ color: 0x20252d, roughness: 0.55 }),
    1: new THREE.MeshStandardMaterial({ color: 0xe0525d, roughness: 0.55 }),
  };
  const linkGroups = { 0: new THREE.Group(), 1: new THREE.Group() };
  const up = new THREE.Vector3(0, 1, 0);
  for (const link of game.links) {
    const player = effectivePlayer(link.player);
    const a = new THREE.Vector3(...pointAt(link.from.row, link.from.col, boardX - 0.32));
    const b = new THREE.Vector3(...pointAt(link.to.row, link.to.col, boardX - 0.32));
    const direction = new THREE.Vector3().subVectors(b, a);
    const mesh = new THREE.Mesh(linkGeometry, linkMaterials[player]);
    mesh.position.copy(a).add(b).multiplyScalar(0.5);
    mesh.quaternion.setFromUnitVectors(up, direction.clone().normalize());
    mesh.scale.set(1, direction.length(), 1);
    linkGroups[player].add(mesh);
  }
  sceneGroup.add(linkGroups[0], linkGroups[1]);

  const pegGeometry = new THREE.CylinderGeometry(0.082, 0.082, 0.36, 16);
  for (const player of [0, 1]) {
    const playerPegs = game.pegs.filter((peg) => effectivePlayer(peg.player) === player);
    if (!playerPegs.length) continue;
    const material = new THREE.MeshStandardMaterial({
      color: player === 1 ? 0xe0525d : 0x20252d,
      roughness: 0.38, metalness: 0.08,
    });
    const mesh = new THREE.InstancedMesh(pegGeometry, material, playerPegs.length);
    const transform = new THREE.Object3D();
    playerPegs.forEach((peg, index) => {
      transform.position.set(...pointAt(peg.row, peg.col, boardX - 0.16));
      transform.rotation.z = Math.PI / 2;
      transform.updateMatrix();
      mesh.setMatrixAt(index, transform.matrix);
    });
    mesh.instanceMatrix.needsUpdate = true;
    sceneGroup.add(mesh);
  }

  for (let col = 0; col < BOARD_SIZE; col += 1) {
    addLabel(String.fromCharCode(65 + col), [boardX - 0.12, 4.42, start + col * spacing], 14, '#e0e8e2', 150);
  }
  for (let row = 0; row < BOARD_SIZE; row += 1) {
    addLabel(String(row + 1), [boardX - 0.12, start + (BOARD_SIZE - 1 - row) * spacing, start - 0.34], 14, '#e0e8e2', 150);
  }
  addLabel(
    'Effective Board',
    [boardX - 0.25, 5.25, 0], 19, '#ffffff', 90,
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
  const canonicalPoint = (peg) => ({
    row: transpose ? peg.col : peg.row,
    col: transpose ? peg.row : peg.col,
  });
  const canonicalPlayer = (player) => transpose ? 1 - player : player;
  const channelNames = [
    'ENE · White', 'ENE · Black',
    'ESE · White', 'ESE · Black',
    'NNE · White', 'NNE · Black',
    'SSE · White', 'SSE · Black',
  ];
  const boardLinks = game.links.map((link) => {
    const a = canonicalPoint(link.from);
    const b = canonicalPoint(link.to);
    return {
      player: canonicalPlayer(link.player) === 1 ? 'White' : 'Black',
      from: String.fromCharCode(65 + a.col) + (a.row + 1),
      to: String.fromCharCode(65 + b.col) + (b.row + 1),
      midpointRow: (a.row + b.row) / 2,
      midpointCol: (a.col + b.col) / 2,
    };
  });
  const tensorCells = [];
  for (let channel = 0; channel < 8; channel += 1) {
    for (let row = 0; row < BOARD_SIZE; row += 1) {
      for (let col = 0; col < BOARD_SIZE; col += 1) {
        if (linkTensor[row][col][channel] !== 0) {
          tensorCells.push({
            channel,
            label: channelNames[channel],
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
    '[Link encoding diagnostic] ' + boardLinks.length +
    ' effective-board links; ' + tensorCells.length + ' nonzero tensor cells',
  );
  console.log('Effective-board links (canonical coordinates):');
  console.table(boardLinks);
  console.log('Nonzero link-tensor cells (the actual blade contents):');
  console.table(tensorCells);
  console.log('Compare each board link midpoint with tensor cells. The tensor cell may be shifted from the geometric midpoint by the input encoding.');
  console.groupEnd();
}

function drawInputs() {
  drawSampleBoard();
  drawEffectiveBoard();
  logLinkEncodingDiagnostic(createSampleGame(), data.links);

  const tensors = [
    {
      key: 'pegs',
      label: 'Pegs · 24×24×2',
      x: -1.8,
      z: -13,
      channels: 2,
      bladeLabels: ['White', 'Black'],
      color: COLORS.input,
    },
    {
      key: 'links',
      label: 'Links · 24×24×8',
      x: -1.8,
      z: -3.5,
      channels: 8,
      bladeLabels: [
        'ENE · White', 'ENE · Black',
        'ESE · White', 'ESE · Black',
        'NNE · White', 'NNE · Black',
        'SSE · White', 'SSE · Black',
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

    // Coordinate headers are shown on the first blade of every tensor.
    // There are 24 columns, so the correct spreadsheet-style labels are A–X.
    const columnNames = Array.from({ length: BOARD_SIZE }, (_, index) =>
      String.fromCharCode(65 + index),
    );
    columnNames.forEach((columnName, col) => {
      addLabel(
        columnName,
        [firstBladeX, 4.6, tensor.z + col * 0.34],
        17,
        '#dce7f5',
        145,
        'center',
      );
    });

    for (let row = 0; row < BOARD_SIZE; row += 1) {
      addLabel(
        String(row + 1),
        [firstBladeX - 0.12, boardTopY - row * 0.34, tensor.z - 0.48],
        17,
        '#dce7f5',
        145,
        'center',
      );
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

document.querySelector('#demo-button').addEventListener('click', () => {
  data = createDemoData();
  cachedOutput = null;
  activeStep = 'inputs';
  renderStep();
  resetCamera();
  status.textContent = 'Demo board and reproducible weights loaded.';
});

document.querySelector('#json-file').addEventListener('change', async (event) => {
  const file = event.target.files && event.target.files[0];
  if (!file) return;

  try {
    const parsed = JSON.parse(await file.text());
    data = validateData(parsed);
    cachedOutput = null;
    activeStep = 'inputs';
    renderStep();
    resetCamera();
    status.textContent = 'Board JSON loaded.';
  } catch (error) {
    status.textContent = 'Could not load JSON: ' + error.message;
  } finally {
    event.target.value = '';
  }
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
animate();
