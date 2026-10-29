/** Rendering helpers for input tensors, convolution kernels, and feature outputs. */
export function createTensorRenderer({
  THREE, BOARD_SIZE, FILTER_COUNT, COLORS, cubeGeometry, flattenTensor,
  createTensor3, getSceneGroup, getData, createSampleGame,
  getBackendSnapshot, drawSampleBoard, drawEffectiveBoard,
  logLinkEncodingDiagnostic, logFullInputTensorValidation,
  calculateConvolutionOutput, addLabel,
  getInputStaticGroup, getInputDynamicGroup, getFilterStaticGroup,
}) {
  const inputStaticGroup = getInputStaticGroup();
  const inputDynamicGroup = getInputDynamicGroup();
  const filterStaticGroup = getFilterStaticGroup();
  const inputTensorSpecs = [
    { key: 'pegs', x: -1.8, z: -13, channels: 2 },
    { key: 'links', x: -1.8, z: -3.5, channels: 8 },
  ];
  const inputSpacing = 0.34;
  const inputCellSize = 0.26;
  const inputChannelGap = 0.28;
  const inputDepthStep = inputCellSize + inputChannelGap;
  const inputOriginY = -4;
  let inputGeometryBuilt = false;
  let cachedFilterWeights = null;
  const inputMeshes = new Map();

  function inputOrigin(spec) {
    return [
      spec.x - ((spec.channels - 1) * inputDepthStep) / 2,
      inputOriginY,
      spec.z,
    ];
  }

  function cellTransform(transform, origin, row, col, channel, thickness, cellSize, zero = false, value = 0) {
    transform.position.set(
      origin[0] + channel * inputDepthStep +
        (zero ? 0 : value < 0 ? -thickness / 2 : thickness / 2),
      origin[1] + (BOARD_SIZE - 1 - row) * inputSpacing,
      origin[2] + col * inputSpacing,
    );
    transform.scale.set(thickness, zero ? cellSize : cellSize, zero ? cellSize : cellSize);
    transform.updateMatrix();
  }

  function makeInputMaterial(color) {
    return new THREE.MeshStandardMaterial({
      color,
      roughness: 0.6,
      transparent: true,
      opacity: 0.82,
    });
  }

  function buildZeroCellBase(spec) {
    const capacity = BOARD_SIZE * BOARD_SIZE * spec.channels;
    const mesh = new THREE.InstancedMesh(
      cubeGeometry, makeInputMaterial(COLORS.input), capacity,
    );
    mesh.name = spec.key + '-persistent-zero-cells';
    const origin = inputOrigin(spec);
    const transform = new THREE.Object3D();
    const zeroCubeSize = inputCellSize * 0.9;
    let index = 0;
    for (let channel = 0; channel < spec.channels; channel += 1) {
      for (let row = 0; row < BOARD_SIZE; row += 1) {
        for (let col = 0; col < BOARD_SIZE; col += 1) {
          cellTransform(transform, origin, row, col, channel, zeroCubeSize, zeroCubeSize, true);
          mesh.setMatrixAt(index++, transform.matrix);
        }
      }
    }
    mesh.count = index;
    mesh.instanceMatrix.needsUpdate = true;
    inputStaticGroup.add(mesh);
  }

  function buildLocationTensor() {
    const spec = { x: -1.8, z: 6, channels: 2 };
    const tensor = getData().locations;
    const capacity = BOARD_SIZE * BOARD_SIZE * spec.channels;
    const mesh = new THREE.InstancedMesh(
      cubeGeometry, makeInputMaterial(COLORS.fixed), capacity,
    );
    mesh.name = 'locations-persistent-values';
    const origin = inputOrigin(spec);
    const transform = new THREE.Object3D();
    let maxAbs = 1e-8;
    for (let row = 0; row < BOARD_SIZE; row += 1) {
      for (let col = 0; col < BOARD_SIZE; col += 1) {
        for (let channel = 0; channel < spec.channels; channel += 1) {
          maxAbs = Math.max(maxAbs, Math.abs(Number(tensor[row][col][channel])));
        }
      }
    }
    let index = 0;
    for (let channel = 0; channel < spec.channels; channel += 1) {
      for (let row = 0; row < BOARD_SIZE; row += 1) {
        for (let col = 0; col < BOARD_SIZE; col += 1) {
          const value = Number(tensor[row][col][channel]);
          const isZero = value === 0;
          const cubeSize = isZero
            ? inputCellSize * 0.9
            : inputCellSize * (0.9 + 0.1 * Math.abs(value) / maxAbs);
          cellTransform(transform, origin, row, col, channel, cubeSize, cubeSize, isZero, value);
          mesh.setMatrixAt(index++, transform.matrix);
        }
      }
    }
    mesh.count = index;
    mesh.instanceMatrix.needsUpdate = true;
    inputStaticGroup.add(mesh);
  }

  function updateOccupiedTensor(spec, tensor) {
    const mesh = inputMeshes.get(spec.key);
    const origin = inputOrigin(spec);
    const transform = new THREE.Object3D();
    let maxAbs = 1e-8;
    for (let row = 0; row < BOARD_SIZE; row += 1) {
      for (let col = 0; col < BOARD_SIZE; col += 1) {
        for (let channel = 0; channel < spec.channels; channel += 1) {
          maxAbs = Math.max(maxAbs, Math.abs(Number(tensor[row][col][channel])));
        }
      }
    }

    let index = 0;
    for (let channel = 0; channel < spec.channels; channel += 1) {
      for (let row = 0; row < BOARD_SIZE; row += 1) {
        for (let col = 0; col < BOARD_SIZE; col += 1) {
          const value = Number(tensor[row][col][channel]);
          if (value === 0) continue;
          const cubeSize = inputCellSize;
          cellTransform(transform, origin, row, col, channel, cubeSize, cubeSize, false, value);
          mesh.setMatrixAt(index++, transform.matrix);
        }
      }
    }
    mesh.count = index;
    mesh.instanceMatrix.needsUpdate = true;
  }

  function ensureInputGeometry() {
    if (inputGeometryBuilt) return;
    inputTensorSpecs.forEach((spec) => {
      buildZeroCellBase(spec);
      const capacity = BOARD_SIZE * BOARD_SIZE * spec.channels;
      const mesh = new THREE.InstancedMesh(
        cubeGeometry, makeInputMaterial(COLORS.input), capacity,
      );
      mesh.name = spec.key + '-incremental-nonzero-cells';
      mesh.count = 0;
      inputMeshes.set(spec.key, mesh);
      inputDynamicGroup.add(mesh);
    });
    buildLocationTensor();
    inputGeometryBuilt = true;
  }

  function addValueCubes(values, rows, cols, channels, origin, color, options = {}) {
    const maxAbs = Math.max(1e-8, ...values.map(Math.abs));
    const spacing = options.spacing || 0.34;
    const cellSize = options.cellSize || 0.26;
    const channelGap = options.channelGap ?? 0.2;
    const depthStep = cellSize + channelGap;
    const maxThickness = options.maxThickness || 0.4;
    const mesh = new THREE.InstancedMesh(
      cubeGeometry,
      new THREE.MeshStandardMaterial({
        color: 0xffffff, roughness: 0.6, transparent: true, opacity: 0.9,
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
          const dotSize = Math.min(cellSize, 0.055);
          const thickness = isZero
            ? dotSize
            : Math.max(0.025, (Math.abs(value) / maxAbs) * maxThickness);
          transform.position.set(
            origin[0] + channel * depthStep +
              (isZero ? 0 : value < 0 ? -thickness / 2 : thickness / 2),
            origin[1] + (rows - 1 - row) * spacing,
            origin[2] + col * spacing,
          );
          transform.scale.set(thickness, isZero ? dotSize : cellSize, isZero ? dotSize : cellSize);
          transform.updateMatrix();
          mesh.setMatrixAt(index, transform.matrix);
          const signedColor = value < 0
            ? new THREE.Color(0x4da3ff)
            : value > 0
              ? new THREE.Color(0xff785a)
              : new THREE.Color(color);
          mesh.setColorAt(index, signedColor);
          index += 1;
        }
      }
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    getSceneGroup().add(mesh);
    return {
      count: values.length,
      min: Math.min(...values),
      max: Math.max(...values),
    };
  }

  function clearFilterGeometry() {
    for (const child of [...filterStaticGroup.children]) {
      filterStaticGroup.remove(child);
      if (child.material) {
        const materials = Array.isArray(child.material) ? child.material : [child.material];
        materials.forEach((material) => material?.dispose());
      }
    }
  }

  function drawFilterKernel(
    filterIndex, originX, originY, originZ, weights,
    kernelHeight, kernelWidth, channels, label, showBias = false, buildGeometry = true,
  ) {
    if (buildGeometry) {
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
      mesh.name = 'filter-' + filterIndex + '-' + label;
      const transform = new THREE.Object3D();
      let index = 0;
      for (let channel = 0; channel < channels; channel += 1) {
        for (let row = 0; row < kernelHeight; row += 1) {
          for (let col = 0; col < kernelWidth; col += 1) {
            const value = values[index];
            const thickness = Math.max(0.025, (Math.abs(value) / maxAbs) * 0.17);
            transform.position.set(
              originX + (value < 0 ? -thickness / 2 : thickness / 2),
              originY + (kernelHeight - 1 - row) * spacing,
              originZ + col * spacing + channel * channelDepth,
            );
            transform.scale.set(thickness, cellSize, cellSize);
            transform.updateMatrix();
            mesh.setMatrixAt(index++, transform.matrix);
          }
        }
      }
      mesh.instanceMatrix.needsUpdate = true;
      filterStaticGroup.add(mesh);

      if (showBias) {
        const biasCube = new THREE.Mesh(
          cubeGeometry, new THREE.MeshStandardMaterial({ color: COLORS.bias }),
        );
        biasCube.position.set(
          originX + 0.42,
          originY + ((kernelHeight - 1) * spacing) / 2,
          originZ + kernelWidth * spacing + channels * channelDepth * 0.5,
        );
        biasCube.scale.setScalar(0.22);
        filterStaticGroup.add(biasCube);
      }
    }

    addLabel(
      label,
      [originX, originY + kernelHeight * 0.18 + 0.2, originZ + channels * (0.135 + 0.055) * 0.3],
      17,
    );
  }

  function drawInputs() {
    const game = createSampleGame();
    drawSampleBoard();
    drawEffectiveBoard();
    ensureInputGeometry();
    updateOccupiedTensor(inputTensorSpecs[0], getData().pegs);
    updateOccupiedTensor(inputTensorSpecs[1], getData().links);

    if (!getBackendSnapshot()) {
      logLinkEncodingDiagnostic(game, getData().links);
      logFullInputTensorValidation(game, getData());
    }

    const tensors = [
      {
        key: 'pegs', label: 'Pegs · 24×24×2', x: -1.8, z: -13, channels: 2,
        bladeLabels: ['Red', 'Black'], color: COLORS.input,
      },
      {
        key: 'links', label: 'Links · 24×24×8', x: -1.8, z: -3.5, channels: 8,
        bladeLabels: [
          'Ch 0: SSE red',
          'Ch 1: SSE black',
          'Ch 2: NNE red',
          'Ch 3: NNE black',
          'Ch 4: ESE red',
          'Ch 5: ESE black',
          'Ch 6: ENE red',
          'Ch 7: ENE black',
        ],
        color: COLORS.input,
      },
      {
        key: 'locations', label: 'Locations · 24×24×2', x: -1.8, z: 6, channels: 2,
        bladeLabels: ['X', 'Y'], color: COLORS.fixed,
      },
    ];

    tensors.forEach((tensor) => {
      addLabel(
        tensor.label,
        [tensor.x, 7.0, tensor.z + ((BOARD_SIZE - 1) * inputSpacing) / 2],
        22, '#ffffff', 70,
      );
      const firstBladeX = tensor.x - ((tensor.channels - 1) * inputDepthStep) / 2;
      const boardTopY = -4 + (BOARD_SIZE - 1) * inputSpacing;
      tensor.bladeLabels.forEach((bladeLabel, channel) => {
        addLabel(
          bladeLabel,
          [firstBladeX + channel * inputDepthStep, 5.65, tensor.z - 0.08],
          20, '#e7edf6', 115, 'left',
        );
      });
      if (tensor.key === 'links') {
        for (let col = 0; col < BOARD_SIZE; col += 1) {
          addLabel(String(col), [firstBladeX, 4.6, tensor.z + col * inputSpacing], 17, '#dce7f5', 145, 'center');
        }
        for (let row = 0; row < BOARD_SIZE; row += 1) {
          addLabel(String(row), [firstBladeX - 0.12, boardTopY - row * inputSpacing, tensor.z - 0.48], 17, '#dce7f5', 145, 'center');
        }
      }
    });

    return {
      title: 'Input tensors',
      description: 'The game is encoded into peg and link tensors. Location coordinates and zero-cell markers persist between positions; only occupied peg/link tensor cells are updated.',
      shape: '24 × 24 × (2 + 8 + 2)',
      range: 'Binary game inputs and fixed coordinates',
      count: '3 tensors · 6,912 cells',
    };
  }

  function drawFilters() {
    const weights = getData().weights;
    const rebuild = cachedFilterWeights !== weights;
    if (rebuild) {
      clearFilterGeometry();
    }

    for (let filter = 0; filter < FILTER_COUNT; filter += 1) {
      const column = filter % 8;
      const row = Math.floor(filter / 8);
      const x = -18.5 + column * 4.8;
      const y = -4 + row * 5.3;
      drawFilterKernel(filter, x, y, -1.5, weights.links, 5, 5, 8, 'F' + filter + ' · link', true, rebuild);
      drawFilterKernel(filter, x + 1.65, y, 2.7, weights.pegs, 1, 1, 2, 'peg', false, rebuild);
      drawFilterKernel(filter, x + 2.75, y, 2.7, weights.locations, 1, 1, 2, 'loc', false, rebuild);
    }
    cachedFilterWeights = weights;

    return {
      title: '48 convolution filters',
      description: 'Eight columns × six rows. Each filter contains link, peg, and location kernels plus a bias cube. Filter geometry is reused while navigating game positions.',
      shape: '48 filters · 8 × 6 grid',
      range: 'Signed weights and bias',
      count: '48 filter groups',
    };
  }

  function drawOutput() {
    const values = flattenTensor(calculateConvolutionOutput());
    const stats = addValueCubes(
      values, BOARD_SIZE, BOARD_SIZE, FILTER_COUNT,
      [-4.5, -4, -((FILTER_COUNT - 1) * 0.105) / 2],
      COLORS.output,
      { spacing: 0.34, cellSize: 0.24, maxThickness: 0.42, minThickness: 0.025, channelGap: 0.105 },
    );
    addLabel('Summed convolution output · 24×24×48', [0, 5, 2], 28);
    return {
      title: 'Convolution output',
      description: 'Three branch convolutions are summed into one feature tensor. Thickness is scaled for visibility.',
      shape: '24 × 24 × 48',
      range: stats.min.toFixed(4) + ' … ' + stats.max.toFixed(4),
      count: stats.count.toLocaleString() + ' cells',
    };
  }

  return { addValueCubes, drawFilterKernel, drawInputs, drawFilters, drawOutput };
}
