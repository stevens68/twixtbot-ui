// Board geometry and labels for the Twixt neural-network visualizer.
// Scene state and board data are supplied by main.js to keep this module isolated.
export function createBoardRenderer({ THREE, getSceneGroup, getStaticSceneGroup, BOARD_SIZE, addLabel, addRichLabel, addStaticLabel, addStaticRichLabel, createSampleGame, getData }) {
  let sampleStaticBuilt = false;
  let effectiveStaticBuilt = false;
  function drawSampleBoard() {
  const game = createSampleGame();
  const spacing = 0.34;
  const boardSize = (BOARD_SIZE - 1) * spacing;
  const boardCenterZ = 0;
  const boardStartZ = boardCenterZ - boardSize / 2;
  const boardStartY = -boardSize / 2;
  const boardX = -26.675; // Match the gap from the effective board to the links tensor.
  const pointAt = (row, col, x = boardX) => [
    x,
    boardStartY + (BOARD_SIZE - 1 - row) * spacing,
    boardStartZ + col * spacing,
  ];

  if (!sampleStaticBuilt) {
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
  getStaticSceneGroup().add(backing);

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
    getStaticSceneGroup().add(strip);
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
  getStaticSceneGroup().add(holes);

  }

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
  getSceneGroup().add(linkMeshes[0], linkMeshes[1]);

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
    getSceneGroup().add(pegMesh);
  }

  if (!sampleStaticBuilt) {
  // Coordinate labels follow the same A–X / 1–24 convention as the tensors.
  for (let col = 0; col < BOARD_SIZE; col += 1) {
    addStaticLabel(
      String.fromCharCode(65 + col),
      [boardX - 0.12, 4.42, boardStartZ + col * spacing],
      14,
      '#e0e8e2',
      150,
    );
  }
  for (let row = 0; row < BOARD_SIZE; row += 1) {
    addStaticLabel(
      String(row + 1),
      [boardX - 0.12, boardStartY + (BOARD_SIZE - 1 - row) * spacing, boardStartZ - 0.34],
      14,
      '#e0e8e2',
      150,
    );
  }

  sampleStaticBuilt = true;
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

  if (!effectiveStaticBuilt) {
  const backing = new THREE.Mesh(
    new THREE.PlaneGeometry(boardSize + 0.42, boardSize + 0.42),
    new THREE.MeshStandardMaterial({
      color: 0xc8cacc, roughness: 0.88, side: THREE.DoubleSide,
    }),
  );
  backing.rotation.y = -Math.PI / 2;
  backing.position.set(boardX + 0.035, 0, 0);
  getStaticSceneGroup().add(backing);

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
    getStaticSceneGroup().add(strip);
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
  getStaticSceneGroup().add(holes);

  }

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
        if (Number(getData().links[row][col][channel]) !== 1) continue;

        // Mark the exact encoded tensor cell; do not reconstruct endpoints.
        const encodedPlayer = channel & 1;
        // Inputs are canonicalized to the Black-to-move viewpoint:
        // channel 0 is the opponent (red), channel 1 is side-to-move (black).
        const displayedPlayer = 1 - encodedPlayer;
        const mesh = new THREE.Mesh(linkGeometry, linkMaterials[displayedPlayer]);
        // The cube's back face touches the board face; the whole cube projects
        // outward, avoiding coplanar overlap with the backing and holes.
        mesh.position.set(...pointAt(row, col, boardX + 0.035 - 0.18));
        getSceneGroup().add(mesh);

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
          getSceneGroup().add(pointer);
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
        if (Number(getData().pegs[row][col][channel]) === 1) {
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
    getSceneGroup().add(mesh);
  }

  if (!effectiveStaticBuilt) {
  // Keep the familiar Twixt coordinates fixed, like the finish lines:
  // A–X run left-to-right across the top; 1–24 run top-to-bottom on the left.
  for (let col = 0; col < BOARD_SIZE; col += 1) {
    addStaticLabel(
      String.fromCharCode(65 + col),
      [boardX - 0.12, start + (BOARD_SIZE - 1) * spacing + 0.34, start + col * spacing],
      14, '#e0e8e2', 150,
    );
  }
  for (let row = 0; row < BOARD_SIZE; row += 1) {
    addStaticLabel(
      String(row + 1),
      [boardX - 0.12, start + (BOARD_SIZE - 1 - row) * spacing, start - 0.34],
      14, '#e0e8e2', 150,
    );
  }
  addStaticRichLabel(
    [
      { text: 'Effective board - black to move', color: '#ffffff' },
    ],
    [boardX, 5.25, 0],
    19,
    90,
    'center',
  );  effectiveStaticBuilt = true;
  }

}

  return { drawSampleBoard, drawEffectiveBoard };
}
