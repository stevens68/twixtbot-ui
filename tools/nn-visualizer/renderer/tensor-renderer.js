import { createFeatureOutputRenderer } from './feature-output-renderer.js?v=20261031-feature-output-v1';
import { INPUT_CHANNEL_GAP, createInputTensorRenderer } from './input-tensor-renderer.js?v=20261031-input-tensor-v2';
import { createFilterBankRenderer } from './filter-bank-renderer.js?v=20261031-filter-bank-v1';
import { createResidualRenderer } from './residual-renderer.js?v=20261031-residual-renderer-v1';
/** Rendering helpers for input tensors, convolution kernels, and feature outputs. */
export function createTensorRenderer({
  THREE, BOARD_SIZE, FILTER_COUNT, COLORS, cubeGeometry, flattenTensor,
  getSceneGroup, getData, createSampleGame,
  getBackendSnapshot, drawSampleBoard, drawEffectiveBoard,
  logLinkEncodingDiagnostic, logFullInputTensorValidation,
  calculateConvolutionOutput, addLabel,
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
    filterCubeSize, filterStaticGroup, flattenTensor, addLabel, getData,
    heatMapColor,
  });

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
    const featureInfo = featureOutputRenderer.drawOutput();

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
    const startY = bankHeight / 2 - filterHeight;
    const startZ = -5 + kernelWidth * inputSpacing;

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
    const pegStartY = rows * (pegFilterHeight + pegGapY) / 2 - pegFilterHeight;
    const pegStartZ = startZ - 12;
    const pegRows = Math.ceil(pegWeights.length / columns);
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
    const locationStartY = rows * (pegFilterHeight + locationGapY) / 2 - pegFilterHeight;
    const locationStartZ = startZ + 10;
    const locationRows = Math.ceil(locationWeights.length / columns);
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
