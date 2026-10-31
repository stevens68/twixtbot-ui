import { createFeatureOutputRenderer } from './feature-output-renderer.js?v=20261031-feature-output-v1';
import { INPUT_CHANNEL_GAP, createInputTensorRenderer } from './input-tensor-renderer.js?v=20261031-axis-alignment-v2';
import { createFilterBankRenderer } from './filter-bank-renderer.js?v=20261031-filter-bank-v1';
import { createResidualRenderer } from './residual-renderer.js?v=20261031-curved-arrows-v1';
/** Rendering helpers for input tensors, convolution kernels, and feature outputs. */
export function createTensorRenderer({
  THREE, BOARD_SIZE, FILTER_COUNT, COLORS, cubeGeometry, flattenTensor,
  getSceneGroup, getData, createSampleGame,
  getBackendSnapshot, drawSampleBoard, drawEffectiveBoard,
  logLinkEncodingDiagnostic, logFullInputTensorValidation,
  calculateConvolutionOutput, addLabel, addFormulaLabel,
  getInputStaticGroup, getInputDynamicGroup, getFilterStaticGroup,
}) {
  const filterStaticGroup = getFilterStaticGroup();
  const inputTensorSpecs = [
    { key: 'pegs', x: -1.8, z: -13, channels: 2 },
    { key: 'links', x: -1.8, z: -3.5, channels: 8 },
  ];
  const inputSpacing = 0.33;
  const inputCellSize = 0.22;
  const inputDepthStep = inputCellSize + INPUT_CHANNEL_GAP;
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
  const featureOutputRenderer = createFeatureOutputRenderer({
    THREE, BOARD_SIZE, FILTER_COUNT, COLORS, cubeGeometry, flattenTensor,
    getSceneGroup, calculateConvolutionOutput, addLabel, filterSpacing,
    filterCubeSize, heatMapColor,
  });
  const inputRenderer = createInputTensorRenderer({
    THREE, BOARD_SIZE, COLORS, cubeGeometry, getData,
    getInputStaticGroup, getInputDynamicGroup, inputTensorSpecs,
    inputCellSize, inputSpacing, minimumCubeScale, locationHeatMapColor,
  });
  const filterBankRenderer = createFilterBankRenderer({
    THREE, cubeGeometry, filterStaticGroup, flattenTensor, addLabel,
    heatMapColor, filterSpacing, filterCubeSize, inputSpacing, inputCellSize, filterScale,
  });
  const residualRenderer = createResidualRenderer({
    THREE, BOARD_SIZE, FILTER_COUNT, cubeGeometry, filterSpacing,
    filterCubeSize, filterStaticGroup, flattenTensor, addLabel, addFormulaLabel, getData,
    heatMapColor,
  });

  function addXArrow(group, startX, endX, y, z, label = null, labelOffset = 0.58, name = 'tensor-flow-arrow') {
    const fullLength = endX - startX;
    if (fullLength <= 0.15) return;
    const length = fullLength * 0.78;
    const arrow = new THREE.ArrowHelper(
      new THREE.Vector3(1, 0, 0),
      new THREE.Vector3(startX + (fullLength - length) / 2, y, z),
      length,
      0xaeb7c2,
      0.42,
      0.22,
    );
    arrow.name = name;
    group.add(arrow);
    if (label) {
      addFormulaLabel(label, [
        (startX + endX) / 2,
        y + labelOffset,
        z,
      ], 25, '#f2f4f8', 92);
    }
  }

  function addCenterlineCurveArrow(group, startX, endX, startY, startZ, endY, endZ, options = {}) {
    const { name = 'centerline-curve-arrow', label = null, labelOffset = 0.62 } = options;
    const dx = endX - startX;
    if (dx <= 0.2) return;
    // This is a shaft only. All three first-layer paths converge at the
    // feature axis; the single downstream arrow supplies their common head.
    const curve = new THREE.CubicBezierCurve3(
      new THREE.Vector3(startX, startY, startZ),
      new THREE.Vector3(startX + dx * 0.42, startY, startZ),
      new THREE.Vector3(endX - dx * 0.42, endY, endZ),
      new THREE.Vector3(endX, endY, endZ),
    );
    const shaft = new THREE.Mesh(
      new THREE.TubeGeometry(curve, 48, 0.06, 8, false),
      new THREE.MeshBasicMaterial({ color: 0xaeb7c2 }),
    );
    shaft.name = name + '-shaft';
    group.add(shaft);
    if (label) {
      const point = curve.getPoint(0.5);
      addFormulaLabel(label, [point.x, point.y + labelOffset, point.z], 25, '#f2f4f8', 92);
    }
  }

  let cachedFilterWeights = null;

  // Dispose the current filter scene before rebuilding it with new weights.
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
    residualRenderer.reset();
  }

  function drawInputs() {
    const game = createSampleGame();
    drawSampleBoard();
    drawEffectiveBoard();
    inputRenderer.ensureInputGeometry();
    inputRenderer.updateOccupiedTensor(inputTensorSpecs[0], getData().pegs);
    inputRenderer.updateOccupiedTensor(inputTensorSpecs[1], getData().links);

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
      const boardTopY = -4.082 + (BOARD_SIZE - 1) * inputSpacing;
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
    const featureInfo = featureOutputRenderer.drawOutput();

    // Connect each input/filter pair and the three filter banks to the
    // shared post-BN feature tensor. The bank layouts are calculated in
    // drawFilters(), where their actual geometry dimensions are available.
    const arrowGroup = getSceneGroup();
    const featureY = -2.5 + (BOARD_SIZE - 1) * filterSpacing / 2;
    const featureZ = -((FILTER_COUNT - 1) * 0.105) / 2 +
      (BOARD_SIZE - 1) * filterSpacing / 2;
    const featureLeftX = 20 - filterCubeSize / 2;
    for (const bank of filterInfo.arrowBanks) {
      const tensorRight = -1.8 +
        ((bank.tensorChannels - 1) * inputDepthStep) / 2 + inputCellSize / 2;
      const bankLeft = bank.startX - filterCubeSize / 2;
      const bankRight = bank.startX +
        (bank.channels - 1) * filterSpacing + filterCubeSize / 2;
      const bankBottomY = bank.startY -
        (bank.rows - 1) * (bank.slotHeight + bank.gapY);
      const bankCenterY = bank.startY - (bank.rows - 1) * (bank.slotHeight + bank.gapY) / 2 + (bank.slotHeight - filterSpacing) / 2;
      const bankCenterZ = bank.startZ + (8 - 1) * (bank.slotWidth + bank.gapZ) / 2 + (bank.slotWidth - filterSpacing) / 2;
      const tensorCenterY = -4 + (BOARD_SIZE - 1) * inputSpacing / 2;
      const tensorCenterZ = bank.tensorZ + (BOARD_SIZE - 1) * inputSpacing / 2;

      addXArrow(
        arrowGroup, tensorRight, bankLeft,
        (tensorCenterY + bankCenterY) / 2,
        (tensorCenterZ + bankCenterZ) / 2,
        null, 0.58, bank.key + '-input-to-filter-bank',
      );
      const mergeX = featureLeftX - 1.35;
      const isOuterBank = bank.key === 'pegs' || bank.key === 'locations';
      if (isOuterBank) {
        addCenterlineCurveArrow(
          arrowGroup, bankRight + 0.8, mergeX,
          bankCenterY, bankCenterZ, featureY, featureZ,
          {
            name: bank.key + '-filter-bank-to-merge',
            label: 'y = |BN(Conv(x))|',
          },
        );
      } else {
        const linkLine = new THREE.LineCurve3(
          new THREE.Vector3(bankRight, featureY, featureZ),
          new THREE.Vector3(mergeX, featureY, featureZ),
        );
        const linkShaft = new THREE.Mesh(
          new THREE.TubeGeometry(linkLine, 1, 0.06, 8, false),
          new THREE.MeshBasicMaterial({ color: 0xaeb7c2 }),
        );
        linkShaft.name = bank.key + '-filter-bank-to-merge-shaft';
        arrowGroup.add(linkShaft);
        addFormulaLabel(
          'y = |BN(Conv(x))|',
          [(bankRight + mergeX) / 2, featureY + 0.62, featureZ],
          25, '#f2f4f8', 92,
        );
      }
    }
    // The three convolution outputs meet at one merge point before the
    // shared post-BN feature tensor.
    const mergeX = featureLeftX - 1.35;
    addXArrow(
      arrowGroup, mergeX, featureLeftX,
      featureY, featureZ, null, 0.58, 'merged-convolutions-to-feature-output',
    );

    return {
      title: 'First-layer network view',
      description: "Sample/effective boards, the three input tensors, trained first-layer filters, the selected residual block's two 48-channel convolution filter banks, and the post-BN + abs first-layer feature tensor are shown together.",
      shape: 'Inputs: 24 × 24 × (2 + 8 + 2) · Features: 24 × 24 × 48',
      range: 'Feature range ' + featureInfo.range,
      count: '3 input tensors · ' + filterInfo.count + ' · ' + filterInfo.residualBlockCount + ' residual blocks · selected block ' + filterInfo.selectedResidualBlockIndex + ' (' + featureInfo.residualCount + ' detailed filters) · ' + filterInfo.count + ' feature values',
    };
  }

  function drawFilters() {
    const weights = getData().weights;
    const rebuild = cachedFilterWeights !== weights;
    if (rebuild) clearFilterGeometry();

    // Shared bank geometry; the individual banks retain their model-specific
    // spacing and placement while using one generic rendering implementation.
    const columns = 8;
    const gapY = 0.3;
    const gapZ = 0.3;
    const filterCount = weights.links.length;
    const kernelHeight = weights.links[0].length;
    const kernelWidth = weights.links[0][0].length;
    const channels = weights.links[0][0][0].length;
    const rows = Math.ceil(filterCount / columns);
    const filterWidth = kernelWidth * filterSpacing;
    const filterHeight = kernelHeight * filterSpacing;
    const bankHeight = rows * filterHeight + (rows - 1) * gapY;
    const startX = 12 + inputTensorSpecs[1].x - ((channels - 1) * filterSpacing) / 2;
    const startY = -0.287 + (rows - 1) * (filterHeight + gapY) / 2 - (kernelHeight - 1) * filterSpacing / 2;
    const startZ = -0.2545 - (columns - 1) * (filterWidth + gapZ) / 2 - (filterWidth - filterSpacing) / 2;

    const linkMetadata = getData().linkConvMetadata;
    const linkInfo = filterBankRenderer.drawBank({
      name: 'links', weights: weights.links,
      startX, startY, startZ, columns, gapY, gapZ,
      showBias: Boolean(linkMetadata?.has_bias),
      biasValues: linkMetadata?.bias || null,
      buildGeometry: rebuild,
      slotWidth: filterWidth, slotHeight: filterHeight,
      bankWidth: columns * filterWidth + (columns - 1) * gapZ,
      bankHeight,
    });

    // Peg kernels use a wider row/column separation than link kernels.
    const pegWeights = weights.pegs;
    const pegHeight = pegWeights[0].length;
    const pegWidth = pegWeights[0][0].length;
    const pegChannels = pegWeights[0][0][0].length;
    const pegGapY = gapY * 1.4;
    const pegGapZ = gapZ * 1.4;
    const pegFilterWidth = pegWidth * filterSpacing;
    const pegFilterHeight = pegHeight * filterSpacing;
    const pegRows = Math.ceil(pegWeights.length / columns);
    const pegStartY = -0.287 + (pegRows - 1) * (pegFilterHeight + pegGapY) / 2 - (pegHeight - 1) * filterSpacing / 2;
    const pegStartZ = -0.2545 - 11 - (columns - 1) * (pegFilterWidth + pegGapZ) / 2 - (pegFilterWidth - filterSpacing) / 2;
    const pegBankHeight = pegRows * pegFilterHeight + (pegRows - 1) * pegGapY;
    const pegMetadata = getData().pegConvMetadata;
    const pegInfo = filterBankRenderer.drawBank({
      name: 'pegs', weights: pegWeights,
      startX, startY: pegStartY, startZ: pegStartZ, columns,
      gapY: pegGapY, gapZ: pegGapZ,
      showBias: Boolean(pegMetadata?.has_bias),
      biasValues: pegMetadata?.bias || null,
      buildGeometry: rebuild,
      slotWidth: pegFilterWidth, slotHeight: pegFilterHeight,
      bankWidth: columns * pegFilterWidth + (columns - 1) * gapZ * 1.4,
      bankHeight: pegBankHeight,
    });

    // Location kernels share the peg-sized slots but are tightly packed.
    const locationWeights = weights.locations;
    const locationHeight = locationWeights[0].length;
    const locationWidth = locationWeights[0][0].length;
    const locationChannels = locationWeights[0][0][0].length;
    const locationGapY = gapY * 0.1;
    const locationGapZ = gapZ * 0.1;
    const locationRows = Math.ceil(locationWeights.length / columns);
    const locationStartY = -0.287 + (locationRows - 1) * (pegFilterHeight + locationGapY) / 2 - (locationHeight - 1) * filterSpacing / 2;
    const locationStartZ = -0.2545 + 11 - (columns - 1) * (pegFilterWidth + locationGapZ) / 2 - (pegFilterWidth - filterSpacing) / 2;
    const locationBankHeight = locationRows * pegFilterHeight + (locationRows - 1) * locationGapY;
    const locationMetadata = getData().locationConvMetadata;
    const locationInfo = filterBankRenderer.drawBank({
      name: 'locations', weights: locationWeights,
      startX, startY: locationStartY, startZ: locationStartZ, columns,
      gapY: locationGapY, gapZ: locationGapZ,
      showBias: Boolean(locationMetadata?.has_bias),
      biasValues: locationMetadata?.bias || null,
      buildGeometry: rebuild,
      slotWidth: pegFilterWidth, slotHeight: pegFilterHeight,
      bankWidth: columns * pegFilterWidth + (columns - 1) * gapZ * 0.1,
      bankHeight: locationBankHeight,
    });

    const blockCount = Math.max(1, Number(getData().residualBlockCount) || 20);
    const selectedResidualBlockIndex = residualRenderer.draw(rebuild, blockCount);
    cachedFilterWeights = weights;

    return {
      title: 'Trained convolution filters',
      description: 'Trained first-layer kernels plus an interactive residual stack. Residual block 0 shows both trained convolution banks inside glass enclosures and its actual 24×24×48 output tensor. The output includes batch normalization, absolute-value activations, and the skip connection.',
      shape: 'Locations ' + locationInfo.kernelHeight + ' × ' + locationInfo.kernelWidth + ' × ' + locationInfo.channels +
        '; pegs ' + pegInfo.kernelHeight + ' × ' + pegInfo.kernelWidth + ' × ' + pegInfo.channels +
        '; links ' + linkInfo.kernelHeight + ' × ' + linkInfo.kernelWidth + ' × ' + linkInfo.channels,
      range: 'Trained signed weights; all three convolutions have no bias in this checkpoint.',
      count: locationInfo.count + ' location filters + ' + pegInfo.count + ' peg filters + ' + linkInfo.count + ' link filters',
      residualCount: 2 * FILTER_COUNT,
      residualBlockCount: blockCount,
      selectedResidualBlockIndex,
      arrowBanks: [
        {
          key: 'pegs', tensorZ: -13, tensorChannels: 2,
          startX, startY: pegStartY, startZ: pegStartZ,
          rows: pegRows, slotHeight: pegFilterHeight, gapY: pegGapY, slotWidth: pegFilterWidth, gapZ: pegGapZ,
          bankHeight: pegBankHeight,
          bankWidth: columns * pegFilterWidth + (columns - 1) * pegGapZ,
          channels: pegChannels,
        },
        {
          key: 'links', tensorZ: -3.5, tensorChannels: 8,
          startX, startY, startZ, rows, slotHeight: filterHeight, gapY, slotWidth: filterWidth, gapZ,
          bankHeight,
          bankWidth: columns * filterWidth + (columns - 1) * gapZ,
          channels,
        },
        {
          key: 'locations', tensorZ: 6, tensorChannels: 2,
          startX, startY: locationStartY, startZ: locationStartZ,
          rows: locationRows, slotHeight: pegFilterHeight, gapY: locationGapY, slotWidth: pegFilterWidth, gapZ: locationGapZ,
          bankHeight: locationBankHeight,
          bankWidth: columns * pegFilterWidth + (columns - 1) * gapZ * 0.1,
          channels: locationChannels,
        },
      ],
    };
  }


  return {
    addValueCubes: featureOutputRenderer.addValueCubes,
    drawFilterKernel: filterBankRenderer.drawFilterKernel,
    drawInputs,
    drawFilters,
    drawOutput: featureOutputRenderer.drawOutput,
    setResidualFiltersVisible: residualRenderer.setResidualFiltersVisible,
    setSelectedResidualBlock: residualRenderer.setSelectedResidualBlock,
    pickResidualBlock: residualRenderer.pickResidualBlock,
  };
}
