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
  const inputSpacing = 0.33;
  const inputCellSize = 0.22;
  // Shared dimensions for every convolution filter and the feature tensor.
  const filterScale = 0.5832;
  const filterSpacing = inputSpacing * filterScale;
  const filterCubeSize = inputCellSize * filterScale * 0.72;
  // Keep zero-valued cells visible, but make them much smaller than active cells.
  const minimumCubeScale = 0.08;

  function heatMapColor(value, minValue, maxValue) {
    const range = maxValue - minValue;
    const normalized = range > 1e-8
      ? Math.max(0, Math.min(1, (value - minValue) / range))
      : 0.5;
    // Blue → cyan → green → yellow → red across the full value range.
    return new THREE.Color().setHSL((2 / 3) * (1 - normalized), 1, 0.5);
  }

  function locationHeatMapColor(value) {
    // Coordinates are nonnegative: start at green for 0, then progress
    // through yellow to red as the coordinate approaches 1.
    const normalized = Math.max(0, Math.min(1, Number(value)));
    return new THREE.Color().setHSL((1 / 3) * (1 - normalized), 1, 0.5);
  }
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

  function cellTransform(transform, origin, row, col, channel, thickness, cellSize, zero = false, value = 0, offsetBySign = true) {
    transform.position.set(
      origin[0] + channel * inputDepthStep +
        (zero || !offsetBySign ? 0 : value < 0 ? -thickness / 2 : thickness / 2),
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
    const zeroCubeSize = inputCellSize * minimumCubeScale;
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
      cubeGeometry, new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.6, transparent: true, opacity: 0.9 }), capacity,
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
          const cubeSize = inputCellSize * 0.9;
          cellTransform(transform, origin, row, col, channel, cubeSize, cubeSize, isZero, value, false);
          mesh.setMatrixAt(index, transform.matrix);
          mesh.setColorAt(index, locationHeatMapColor(value));
          index += 1;
        }
      }
    }
    mesh.count = index;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    inputStaticGroup.add(mesh);
  }

  function updateOccupiedTensor(spec, tensor) {
    const mesh = inputMeshes.get(spec.key);
    const origin = inputOrigin(spec);
    const transform = new THREE.Object3D();
    let index = 0;
    for (let channel = 0; channel < spec.channels; channel += 1) {
      for (let row = 0; row < BOARD_SIZE; row += 1) {
        for (let col = 0; col < BOARD_SIZE; col += 1) {
          const value = Number(tensor[row][col][channel]);
          if (value === 0) continue;
          const cubeSize = inputCellSize * 0.9;
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
          // Keep every marker a cube. Magnitude controls cube volume; the
          // largest magnitude matches the existing one-hot cell size.
          // Feature values retain their heat-map colors, but use a uniform size.
          const cubeSize = cellSize * 0.9;
          transform.position.set(
            origin[0] + channel * depthStep,
            origin[1] + (rows - 1 - row) * spacing,
            origin[2] + col * spacing,
          );
          transform.scale.set(cubeSize, cubeSize, cubeSize);
          transform.updateMatrix();
          mesh.setMatrixAt(index, transform.matrix);
          mesh.setColorAt(index, heatMapColor(value, -maxAbs, maxAbs));
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
    kernelHeight, kernelWidth, channels, label, showBias = false, buildGeometry = true, biasValues = null,
  ) {
    // Reduce convolution-filter dimensions and internal gaps by a further 20%.
    // Keep these dimensions consistent across peg/link filters.
    const spacing = filterSpacing;
    const cubeSize = filterCubeSize;
    const channelDepth = filterSpacing;
    if (buildGeometry) {
      // Weights are stored [row][column][channel], so preserve that layout
      // with channels along X, rows along Y, and columns along Z.
      const values = flattenTensor(weights[filterIndex]);
      const maxAbs = Math.max(1e-8, ...values.map(Math.abs));
      const mesh = new THREE.InstancedMesh(
        cubeGeometry,
        new THREE.MeshStandardMaterial({
          color: 0xffffff, roughness: 0.6, transparent: true, opacity: 0.9,
        }),
        values.length,
      );
      mesh.name = 'filter-' + filterIndex + '-' + label;
      const transform = new THREE.Object3D();
      let index = 0;
      for (let row = 0; row < kernelHeight; row += 1) {
        for (let col = 0; col < kernelWidth; col += 1) {
          for (let channel = 0; channel < channels; channel += 1) {
            const value = Number(weights[filterIndex][row][col][channel]);
            transform.position.set(
              originX + channel * channelDepth,
              originY + (kernelHeight - 1 - row) * spacing,
              originZ + col * spacing,
            );
            transform.scale.set(cubeSize, cubeSize, cubeSize);
            transform.updateMatrix();
            mesh.setMatrixAt(index, transform.matrix);
            mesh.setColorAt(index, heatMapColor(value, -maxAbs, maxAbs));
            index += 1;
          }
        }
      }
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
      filterStaticGroup.add(mesh);

      if (showBias) {
        const biasValue = Number(biasValues?.[filterIndex] ?? 0);
        const biasMaxAbs = Math.max(
          1e-8,
          ...(biasValues || []).map((value) => Math.abs(Number(value) || 0)),
        );
        const biasMagnitudeScale = 0.25 + 0.75 * Math.abs(biasValue) / biasMaxAbs;
        const biasCube = new THREE.Mesh(
          cubeGeometry,
          new THREE.MeshStandardMaterial({
            color: heatMapColor(biasValue, -biasMaxAbs, biasMaxAbs),
          }),
        );
        // Bias sits beside the channel stack, with twice the normal gap.
        const doubleGap = 2 * (inputSpacing - inputCellSize) * filterScale;
        biasCube.position.set(
          originX,
          originY + ((kernelHeight - 1) * spacing) / 2,
          originZ + kernelWidth * spacing + doubleGap,
        );
        // Color shows sign; side length also reflects absolute magnitude.
        biasCube.scale.setScalar(cubeSize * biasMagnitudeScale);
        biasCube.name = 'filter-' + filterIndex + '-bias';
        biasCube.userData.biasValue = biasValue;
        filterStaticGroup.add(biasCube);
      }
    }

    // A simple numeric filter label, centered above the kernel face.
    addLabel(
      String(filterIndex + 1),
      [
        originX + ((channels - 1) * channelDepth) / 2,
        originY + kernelHeight * spacing + 0.096,
        originZ + ((kernelWidth - 1) * spacing) / 2,
      ],
      10,
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

    // Show all trained first-layer filters and their summed feature tensor
    // alongside the game encoding, rather than switching between separate tabs.
    const filterInfo = drawFilters();
    const featureInfo = drawOutput();

    return {
      title: 'First-layer network view',
      description: 'Sample/effective boards, the three input tensors, trained peg/link/location convolution filters, and the summed 24×24×48 feature tensor are shown together. The feature tensor is the raw branch sum before batch normalization and the absolute-value activation.',
      shape: 'Inputs: 24 × 24 × (2 + 8 + 2) · Features: 24 × 24 × 48',
      range: 'Feature range ' + featureInfo.range,
      count: '3 input tensors · ' + filterInfo.count + ' · ' + featureInfo.count + ' feature values',
    };
  }

  function drawFilters() {
    const weights = getData().weights;
    const rebuild = cachedFilterWeights !== weights;
    if (rebuild) {
      clearFilterGeometry();
    }

    // Use the exact dimensions of the trained links convolution. The bank
    // stays 8 columns wide along Z and grows downward along Y as needed.
    const columns = 8;
    const filterCount = weights.links.length;
    const kernelHeight = weights.links[0].length;
    const kernelWidth = weights.links[0][0].length;
    const channels = weights.links[0][0][0].length;
    const rows = Math.ceil(filterCount / columns);
    const filterWidth = kernelWidth * filterSpacing;
    const filterHeight = kernelHeight * filterSpacing;
    const gapZ = 0.3;
    const gapY = 0.3;
    const bankHeight = rows * filterHeight + (rows - 1) * gapY;
    // Center the link filter's channel stack exactly on the links input's
    // X center (-1.8), using the same reduced channel spacing as the kernels.
    const startX = 12 + inputTensorSpecs[1].x - ((channels - 1) * filterSpacing) / 2;
    const startY = bankHeight / 2 - filterHeight;
    const startZ = -5 + kernelWidth * inputSpacing;
    const metadata = getData().linkConvMetadata;
    // Missing checkpoint metadata must not invent bias cubes.
    const showBias = Boolean(metadata?.has_bias);

    // Peg filters use their own 5×5×2 geometry in an 8-column × 6-row bank.
    const pegWeights = weights.pegs;
    const pegMetadata = getData().pegConvMetadata;
    const pegCount = pegWeights.length;
    const pegHeight = pegWeights[0].length;
    const pegWidth = pegWeights[0][0].length;
    const pegChannels = pegWeights[0][0][0].length;
    const pegGapY = gapY * 1.4;
    const pegGapZ = gapZ * 1.4;
    const pegFilterWidth = pegWidth * filterSpacing;
    const pegFilterHeight = pegHeight * filterSpacing;
    const pegStartX = startX;
    const pegStartY = rows * (pegFilterHeight + pegGapY) / 2 - pegFilterHeight;
    const pegStartZ = startZ - 12;
    const pegShowBias = Boolean(pegMetadata?.has_bias);
    for (let filter = 0; filter < pegCount; filter += 1) {
      const column = filter % columns;
      const row = Math.floor(filter / columns);
      drawFilterKernel(
        filter,
        pegStartX,
        pegStartY - row * (pegFilterHeight + pegGapY),
        pegStartZ + column * (pegFilterWidth + pegGapZ),
        pegWeights,
        pegHeight, pegWidth, pegChannels,
        String(filter + 1),
        pegShowBias,
        rebuild,
        pegMetadata?.bias || null,
      );
    }

    // Location filters are 1×1×2 and have no bias in the trained model.
    const locationWeights = weights.locations;
    const locationMetadata = getData().locationConvMetadata;
    const locationCount = locationWeights.length;
    const locationHeight = locationWeights[0].length;
    const locationWidth = locationWeights[0][0].length;
    const locationChannels = locationWeights[0][0][0].length;
    const locationGapY = gapY * 0.1;
    const locationGapZ = gapZ * 0.1;
    const locationSlotWidth = pegFilterWidth;
    const locationSlotHeight = pegFilterHeight;
    const locationStartX = startX;
    const locationStartY = rows * (locationSlotHeight + locationGapY) / 2 - locationSlotHeight;
    const locationStartZ = startZ + 10;
    const locationShowBias = Boolean(locationMetadata?.has_bias);
    for (let filter = 0; filter < locationCount; filter += 1) {
      const column = filter % columns;
      const row = Math.floor(filter / columns);
      drawFilterKernel(
        filter,
        locationStartX,
        locationStartY - row * (locationSlotHeight + locationGapY),
        locationStartZ + column * (locationSlotWidth + locationGapZ),
        locationWeights,
        locationHeight, locationWidth, locationChannels,
        String(filter + 1),
        locationShowBias,
        rebuild,
        locationMetadata?.bias || null,
      );
    }

    for (let filter = 0; filter < filterCount; filter += 1) {
      const column = filter % columns;
      const row = Math.floor(filter / columns);
      drawFilterKernel(
        filter,
        startX,
        startY - row * (filterHeight + gapY),
        startZ + column * (filterWidth + gapZ),
        weights.links,
        kernelHeight, kernelWidth, channels,
        String(filter + 1),
        showBias,
        rebuild,
        metadata?.bias || null,
      );
    }
    cachedFilterWeights = weights;

    return {
      title: 'Trained convolution filters',
      description: 'Trained location, peg, and link convolution kernels from the PyTorch checkpoint. Each bank uses 8 columns and 1-based filter labels; the location bank is 1×1×2, the peg bank 5×5×2, and the link bank 4×4×8.',
      shape: 'Locations ' + locationHeight + ' × ' + locationWidth + ' × ' + locationChannels +
        '; pegs ' + pegHeight + ' × ' + pegWidth + ' × ' + pegChannels +
        '; links ' + kernelHeight + ' × ' + kernelWidth + ' × ' + channels,
      range: 'Trained signed weights; all three convolutions have no bias in this checkpoint.',
      count: locationCount + ' location filters + ' + pegCount + ' peg filters + ' + filterCount + ' link filters',
    };
  }

  function drawOutput() {
    const values = flattenTensor(calculateConvolutionOutput());
    const stats = addValueCubes(
      values, BOARD_SIZE, BOARD_SIZE, FILTER_COUNT,
      [20, -2.5, -((FILTER_COUNT - 1) * 0.105) / 2],
      COLORS.output,
      {
        spacing: filterSpacing,
        cellSize: filterCubeSize / 0.9,
        channelGap: filterSpacing - filterCubeSize,
      },
    );
    addLabel('Post-BN + abs feature tensor · 24×24×48', [22.6, 5, 2], 28);
    return {
      title: 'Convolution output',
      description: 'The 48 channels show the trained peg + link + location convolution sum, followed by checkpoint primary batch normalization (inference running statistics, scale and offset) and absolute-value activation. Color encodes the resulting nonnegative magnitude.',
      shape: '24 × 24 × 48',
      range: stats.min.toFixed(4) + ' … ' + stats.max.toFixed(4),
      count: stats.count.toLocaleString() + ' cells',
    };
  }

  return { addValueCubes, drawFilterKernel, drawInputs, drawFilters, drawOutput };
}
