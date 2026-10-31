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
    const geometries = new Set();
    const materials = new Set();
    for (const child of [...filterStaticGroup.children]) {
      filterStaticGroup.remove(child);
      child.traverse((object) => {
        if (object.geometry && object.geometry !== cubeGeometry) geometries.add(object.geometry);
        if (object.material) {
          const assigned = Array.isArray(object.material) ? object.material : [object.material];
          assigned.forEach((material) => material && materials.add(material));
        }
      });
    }
    geometries.forEach((geometry) => geometry.dispose());
    materials.forEach((material) => material.dispose());
    residualFilterGroup = null;
    residualPlaceholderGroup = null;
    residualPlaceholderDirty = false;
    residualHitTargets = [];
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
      description: "Sample/effective boards, the three input tensors, trained first-layer filters, the selected residual block's two 48-channel convolution filter banks, and the post-BN + abs first-layer feature tensor are shown together.",
      shape: 'Inputs: 24 × 24 × (2 + 8 + 2) · Features: 24 × 24 × 48',
      range: 'Feature range ' + featureInfo.range,
      count: '3 input tensors · ' + filterInfo.count + ' · ' + filterInfo.residualBlockCount + ' residual blocks · selected block ' + filterInfo.selectedResidualBlockIndex + ' (' + featureInfo.residualCount + ' detailed filters) · ' + featureInfo.count + ' feature values',
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
    // Residual blocks start as glass volumes. Full trained filter weights
    // are fetched and rendered only after the user selects a block.
    const blockCount = Math.max(1, Number(getData().residualBlockCount) || 20);
    if (rebuild || residualPlaceholderDirty || !residualPlaceholderGroup) {
      if (residualPlaceholderGroup) {
        filterStaticGroup.remove(residualPlaceholderGroup);
        residualPlaceholderGroup.traverse((object) => {
          object.geometry?.dispose();
          if (object.material) {
            const materials = Array.isArray(object.material) ? object.material : [object.material];
            materials.forEach((material) => material?.dispose());
          }
        });
      }
      createResidualBlockPlaceholders(blockCount);
      residualPlaceholderDirty = false;
    }
    if (selectedResidualBlockIndex >= 0 && selectedResidualWeights) {
      if (rebuild || residualGeometryDirty) {
        disposeResidualDetail();
        residualGeometryDirty = false;
        drawSelectedResidualBlock(selectedResidualWeights, true, selectedResidualBlockIndex);
      } else {
        drawSelectedResidualBlock(selectedResidualWeights, false, selectedResidualBlockIndex);
      }
    } else {
      disposeResidualDetail();
      residualGeometryDirty = false;
    }
    drawResidualPlaceholderLabels(blockCount);

    cachedFilterWeights = weights;

    return {
      title: 'Trained convolution filters',
      description: 'Trained first-layer kernels plus an interactive residual stack. Each residual block contains two convolution banks aligned along X, followed by a glass 24×24×48 output tensor; select either convolution volume to inspect both 48-filter 5×5×48 banks.',
      shape: 'Locations ' + locationHeight + ' × ' + locationWidth + ' × ' + locationChannels +
        '; pegs ' + pegHeight + ' × ' + pegWidth + ' × ' + pegChannels +
        '; links ' + kernelHeight + ' × ' + kernelWidth + ' × ' + channels,
      range: 'Trained signed weights; all three convolutions have no bias in this checkpoint.',
      count: locationCount + ' location filters + ' + pegCount + ' peg filters + ' + filterCount + ' link filters',
      residualCount: 2 * FILTER_COUNT,
      residualBlockCount: blockCount,
      selectedResidualBlockIndex,
    };
  }

  let residualFilterGroup = null;
  let residualPlaceholderGroup = null;
  let residualPlaceholderDirty = false;
  let residualHitTargets = [];
  let selectedResidualBlockIndex = -1;
  let selectedResidualWeights = null;
  let residualGeometryDirty = false;
  let residualFiltersVisible = true;

  function disposeResidualDetail() {
    if (!residualFilterGroup) return;
    filterStaticGroup.remove(residualFilterGroup);
    const geometries = new Set();
    const materials = new Set();
    residualFilterGroup.traverse((object) => {
      // cubeGeometry is shared by the other tensor/filter renderers.
      if (object.geometry && object.geometry !== cubeGeometry) {
        geometries.add(object.geometry);
      }
      if (object.material) {
        const assigned = Array.isArray(object.material) ? object.material : [object.material];
        assigned.forEach((material) => material && materials.add(material));
      }
    });
    geometries.forEach((geometry) => geometry.dispose());
    materials.forEach((material) => material.dispose());
    residualFilterGroup = null;
  }


function residualLayout(blockIndex) {
  const channels = FILTER_COUNT;
  const kernelHeight = 5;
  const kernelWidth = 5;
  const columns = 8;
  const rows = Math.ceil(FILTER_COUNT / columns);
  const filterWidth = kernelWidth * filterSpacing;
  const filterHeight = kernelHeight * filterSpacing;
  const gapY = 0.3 * 1.4;
  const gapZ = 0.3 * 1.4;
  const bankSizeX = (channels - 1) * filterSpacing + filterCubeSize;
  const bankMinYOffset =
    -(rows - 1) * (filterHeight + gapY) - filterCubeSize / 2;
  const bankMaxYOffset =
    (kernelHeight - 1) * filterSpacing + filterCubeSize / 2;
  const bankSizeY = bankMaxYOffset - bankMinYOffset;
  const bankSizeZ =
    (columns - 1) * (filterWidth + gapZ) +
    (kernelWidth - 1) * filterSpacing + filterCubeSize;
  const outputChannelStep =
    filterCubeSize / 0.9 + filterSpacing - filterCubeSize;
  const outputCubeSize = filterCubeSize / 0.9;
  const outputSizeX =
    (channels - 1) * outputChannelStep + outputCubeSize;
  const outputSizeY =
    (BOARD_SIZE - 1) * filterSpacing + outputCubeSize;
  const outputSizeZ = outputSizeY;
  const outputCenterY = -2.5 + (BOARD_SIZE - 1) * filterSpacing / 2;
  const outputCenterZ =
    -((channels - 1) * 0.105) / 2 +
    (BOARD_SIZE - 1) * filterSpacing / 2;
  const bankStartY = outputCenterY -
    (bankMinYOffset + bankMaxYOffset) / 2;
  const bankStartZ = outputCenterZ - bankSizeZ / 2 + filterCubeSize / 2;
  const outputMaxX =
    20 + (channels - 1) * outputChannelStep + outputCubeSize / 2;
  const firstResidualGapX = 5.0;
  const convGapX = 1.25;
  const outputGapX = 1.25;
  const blockStartX =
    outputMaxX + firstResidualGapX + filterCubeSize / 2 +
    blockIndex * (2 * bankSizeX + convGapX + outputGapX +
      outputCubeSize / 2 + outputSizeX + 1.25 + filterCubeSize / 2);
  const conv2StartX = blockStartX + bankSizeX + convGapX;
  const outputStartX =
    conv2StartX + bankSizeX + outputGapX + outputCubeSize / 2;
  const outputCenterX =
    outputStartX + (channels - 1) * outputChannelStep / 2;
  return {
    channels, kernelHeight, kernelWidth, columns, rows,
    filterWidth, filterHeight, gapY, gapZ,
    bankSizeX, bankSizeY, bankSizeZ, bankStartY, bankStartZ,
    bankStartXs: [blockStartX, conv2StartX],
    outputStartX, outputCenterX,
    outputSizeX, outputSizeY, outputSizeZ,
    outputCenterY, outputCenterZ, outputCubeSize,
  };
}

function addResidualOutputGlass(group, layout, blockIndex) {
  const geometry = new THREE.BoxGeometry(
    layout.outputSizeX, layout.outputSizeY, layout.outputSizeZ,
  );
  const glass = new THREE.Mesh(
    geometry,
    new THREE.MeshStandardMaterial({
      color: 0x9bb4c8,
      roughness: 0.18,
      metalness: 0.05,
      transparent: true,
      opacity: 0.12,
      depthWrite: false,
    }),
  );
  glass.position.set(
    layout.outputCenterX, layout.outputCenterY, layout.outputCenterZ,
  );
  glass.name = 'residual-block-' + blockIndex + '-output-tensor-glass';
  group.add(glass);

  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(geometry),
    new THREE.LineBasicMaterial({
      color: 0xb9d5e8, transparent: true, opacity: 0.9,
    }),
  );
  edges.position.copy(glass.position);
  edges.name = 'residual-block-' + blockIndex + '-output-tensor-outline';
  group.add(edges);
}

function drawSelectedResidualBlock(residualWeights, buildGeometry, blockIndex) {
  if (!residualWeights?.conv1 || !residualWeights?.conv2) {
    throw new Error(
      'Residual block ' + blockIndex + ' convolution weights are unavailable.'
    );
  }

  const layout = residualLayout(blockIndex);
  if (buildGeometry) {
    residualFilterGroup = new THREE.Group();
    residualFilterGroup.name =
      'residual-block-' + blockIndex + '-filter-banks';
    residualFilterGroup.visible = residualFiltersVisible;

    for (const [bankIndex, name] of ['conv1', 'conv2'].entries()) {
      const bank = residualWeights[name];
      const instancesPerFilter =
        layout.kernelHeight * layout.kernelWidth * layout.channels;
      const mesh = new THREE.InstancedMesh(
        cubeGeometry,
        new THREE.MeshStandardMaterial({
          color: 0xffffff, roughness: 0.6, transparent: true, opacity: 0.9,
        }),
        FILTER_COUNT * instancesPerFilter,
      );
      mesh.name =
        'residual-block-' + blockIndex + '-' + name + '-all-filter-weights';
      const transform = new THREE.Object3D();
      let instance = 0;

      for (let filter = 0; filter < FILTER_COUNT; filter += 1) {
        const column = filter % layout.columns;
        const row = Math.floor(filter / layout.columns);
        const originY =
          layout.bankStartY - row * (layout.filterHeight + layout.gapY);
        const originZ =
          layout.bankStartZ + column * (layout.filterWidth + layout.gapZ);
        const values = flattenTensor(bank[filter]);
        const maxAbs = Math.max(1e-8, ...values.map(Math.abs));

        for (let kernelRow = 0; kernelRow < layout.kernelHeight; kernelRow += 1) {
          for (let kernelCol = 0; kernelCol < layout.kernelWidth; kernelCol += 1) {
            for (let channel = 0; channel < layout.channels; channel += 1) {
              const value = Number(
                bank[filter][kernelRow][kernelCol][channel]
              );
              transform.position.set(
                layout.bankStartXs[bankIndex] + channel * filterSpacing,
                originY + (layout.kernelHeight - 1 - kernelRow) * filterSpacing,
                originZ + kernelCol * filterSpacing,
              );
              transform.scale.setScalar(filterCubeSize);
              transform.updateMatrix();
              mesh.setMatrixAt(instance, transform.matrix);
              mesh.setColorAt(instance, heatMapColor(value, -maxAbs, maxAbs));
              instance += 1;
            }
          }
        }
      }
      mesh.count = instance;
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
      residualFilterGroup.add(mesh);
    }
    filterStaticGroup.add(residualFilterGroup);
  }

  for (const [index, name] of ['conv1', 'conv2'].entries()) {
    const startX = layout.bankStartXs[index];
    addLabel(
      'Residual block ' + blockIndex + ' · ' + name + ' · 48 × 5×5×48 filters',
      [
        startX + (layout.bankSizeX - filterCubeSize) / 2,
        layout.bankStartY + layout.filterHeight + 0.8,
        layout.bankStartZ + (layout.bankSizeZ - filterCubeSize) / 2,
      ],
      22,
    );
  }
}

  function createResidualBlockPlaceholders(blockCount) {
    residualPlaceholderGroup = new THREE.Group();
    residualPlaceholderGroup.name = 'residual-block-placeholders';
    residualHitTargets = [];

    const boxGeometry = new THREE.BoxGeometry(
      residualLayout(0).bankSizeX,
      residualLayout(0).bankSizeY,
      residualLayout(0).bankSizeZ,
    );
    const edgeGeometry = new THREE.EdgesGeometry(boxGeometry);
    boxGeometry.dispose();

    for (let index = 0; index < blockCount; index += 1) {
      const layout = residualLayout(index);
      if (index !== selectedResidualBlockIndex) {
        for (let bankIndex = 0; bankIndex < 2; bankIndex += 1) {
          const position = new THREE.Vector3(
            layout.bankStartXs[bankIndex] + (layout.bankSizeX - filterCubeSize) / 2,
            layout.bankStartY +
              (-(layout.rows - 1) * (layout.filterHeight + layout.gapY) +
                (layout.kernelHeight - 1) * filterSpacing) / 2,
            layout.bankStartZ + (layout.bankSizeZ - filterCubeSize) / 2,
          );
          const glassGeometry = new THREE.BoxGeometry(
            layout.bankSizeX, layout.bankSizeY, layout.bankSizeZ,
          );
          const glass = new THREE.Mesh(
            glassGeometry,
            new THREE.MeshStandardMaterial({
              color: 0x9bb4c8, roughness: 0.16, metalness: 0.04,
              transparent: true, opacity: 0.16, depthWrite: false,
            }),
          );
          glass.position.copy(position);
          glass.name =
            'residual-block-' + index + '-conv' + (bankIndex + 1) + '-glass';
          residualPlaceholderGroup.add(glass);

          const outline = new THREE.LineSegments(
            new THREE.EdgesGeometry(glassGeometry),
            new THREE.LineBasicMaterial({
              color: 0x6f879a, transparent: true, opacity: 0.95,
            }),
          );
          outline.position.copy(position);
          outline.name =
            'residual-block-' + index + '-conv' + (bankIndex + 1) + '-outline';
          residualPlaceholderGroup.add(outline);

          const hitTarget = new THREE.Mesh(
            new THREE.BoxGeometry(
              layout.bankSizeX, layout.bankSizeY, layout.bankSizeZ,
            ),
            new THREE.MeshBasicMaterial({ visible: false }),
          );
          hitTarget.position.copy(position);
          hitTarget.name =
            'residual-block-' + index + '-conv' + (bankIndex + 1) + '-hit-target';
          hitTarget.userData.residualBlockIndex = index;
          residualPlaceholderGroup.add(hitTarget);
          residualHitTargets.push(hitTarget);
        }
      }

      // Every block, including the selected one, has a glass output tensor
      // between it and the next residual block.
      addResidualOutputGlass(residualPlaceholderGroup, layout, index);

      if (index !== selectedResidualBlockIndex) {
        addLabel(
          'Residual block ' + index,
          [
            layout.bankStartXs[0] + layout.bankSizeX +
              (layout.bankStartXs[1] - layout.bankStartXs[0]) / 2,
            layout.outputCenterY + layout.bankSizeY / 2 + 1.0,
            layout.outputCenterZ,
          ],
          12, '#dce7f5', 100,
        );
      }
      addLabel(
        'Block ' + index + ' output · 24×24×48',
        [
          layout.outputCenterX,
          layout.outputCenterY + layout.outputSizeY / 2 + 0.45,
          layout.outputCenterZ,
        ],
        11, '#dce7f5', 100,
      );
    }
    filterStaticGroup.add(residualPlaceholderGroup);
  }

  function drawResidualPlaceholderLabels() {
    // Labels are created alongside their corresponding placeholder geometry.
  }

  function setSelectedResidualBlock(blockIndex, weights) {
    selectedResidualBlockIndex = blockIndex;
    selectedResidualWeights = weights;
    residualGeometryDirty = true;
    residualPlaceholderDirty = true;
  }

  function pickResidualBlock(event, camera, domElement) {
    if (!residualHitTargets.length) return null;
    const rect = domElement.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    const pointer = new THREE.Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1,
    );
    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObjects(residualHitTargets, false);
    return hits.length ? hits[0].object.userData.residualBlockIndex : null;
  }

  function setResidualFiltersVisible(visible) {
    residualFiltersVisible = Boolean(visible);
    if (residualFilterGroup) residualFilterGroup.visible = residualFiltersVisible;
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

  return {
    addValueCubes,
    drawFilterKernel,
    drawInputs,
    drawFilters,
    drawOutput,
    setResidualFiltersVisible,
    setSelectedResidualBlock,
    pickResidualBlock,
  };
}
