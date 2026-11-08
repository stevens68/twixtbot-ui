import { createFeatureOutputRenderer } from './feature-output-renderer.js?v=20261101-policy-positive-colour-v1';
import { INPUT_CHANNEL_GAP, createInputTensorRenderer } from './input-tensor-renderer.js?v=20261031-input-tanh-v1';
import { createFilterBankRenderer } from './filter-bank-renderer.js?v=20261101-1x1-glass-alignment-v1';
import { createResidualRenderer } from './residual-renderer.js?v=20261108-block20-camera-offset-v1';
/** Rendering helpers for input tensors, convolution kernels, and feature outputs. */
export function createTensorRenderer({
  THREE, BOARD_SIZE, FILTER_COUNT, COLORS, cubeGeometry, flattenTensor,
  getSceneGroup, getData, createSampleGame,
  getBackendSnapshot, drawSampleBoard, drawEffectiveBoard,
  logLinkEncodingDiagnostic, logFullInputTensorValidation,
  calculateConvolutionOutput, addLabel, addFormulaLabel,
  getInputStaticGroup, getInputDynamicGroup, getFilterStaticGroup,
  getAnnotationGroup, addAnnotationLabel, addAnnotationFormulaLabel, getColorTanhA,
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
    // Green is the neutral zero point for every tensor. Negative values move
    // toward blue; positive values move toward yellow/red. Using a symmetric
    // magnitude scale also keeps zero green when the supplied range is 0…max.
    const magnitude = Math.max(Math.abs(minValue), Math.abs(maxValue), 1e-8);
    const normalized = Math.max(-1, Math.min(1, Number(value) / magnitude));
    const a = Math.max(0, Math.min(2, Number(getColorTanhA?.()) || 0));
    // Normalize tanh(a*x) by tanh(a), keeping endpoints fixed at -1/+1.
    // At a=0, use the original linear mapping.
    const scaled = a < 1e-8 ? normalized : Math.tanh(a * normalized) / Math.tanh(a);
    const hue = 1 / 3 - scaled / 3;
    return new THREE.Color().setHSL(hue, 1, 0.5);
  }

  function locationHeatMapColor(value) {
    // Location coordinates are nonnegative. Apply the same endpoint-preserving
    // tanh scaling as the convolution weights and feature tensors.
    const normalized = Math.max(0, Math.min(1, Number(value)));
    const a = Math.max(0, Math.min(2, Number(getColorTanhA?.()) || 0));
    const scaled = a < 1e-8 ? normalized : Math.tanh(a * normalized) / Math.tanh(a);
    return new THREE.Color().setHSL((1 / 3) * (1 - scaled), 1, 0.5);
  }
  const featureOutputRenderer = createFeatureOutputRenderer({
    THREE, BOARD_SIZE, FILTER_COUNT, COLORS, cubeGeometry, flattenTensor,
    getSceneGroup, getData, calculateConvolutionOutput, addLabel, filterSpacing,
    filterCubeSize, heatMapColor,
    addLabel: addAnnotationLabel,
  });
  const inputRenderer = createInputTensorRenderer({
    THREE, BOARD_SIZE, COLORS, cubeGeometry, getData,
    getInputStaticGroup, getInputDynamicGroup, inputTensorSpecs,
    inputCellSize, inputSpacing, minimumCubeScale, locationHeatMapColor,
  });
  const filterBankRenderer = createFilterBankRenderer({
    THREE, cubeGeometry, filterStaticGroup, flattenTensor, addLabel,
    heatMapColor, filterSpacing, filterCubeSize, inputSpacing, inputCellSize, filterScale,
    addAnnotationLabel,
  });
  const residualRenderer = createResidualRenderer({
    THREE, BOARD_SIZE, FILTER_COUNT, cubeGeometry, filterSpacing,
    filterCubeSize, filterStaticGroup, flattenTensor, addLabel, addFormulaLabel, getData,
    heatMapColor,
    addLabel: addAnnotationLabel,
    addFormulaLabel: addAnnotationFormulaLabel,
  });

  function addXArrow(group, startX, endX, y, z, label = null, labelOffset = 0.58, name = 'tensor-flow-arrow') {
    const fullLength = endX - startX;
    if (fullLength <= 0.15) return;
    // Double the clearance between the source object and the arrow shaft.
    const startClearance = fullLength * 0.22;
    const length = fullLength * 0.67;
    const arrow = new THREE.ArrowHelper(
      new THREE.Vector3(1, 0, 0),
      new THREE.Vector3(startX + startClearance, y, z),
      length,
      0x71879b,
      0.42,
      0.22,
    );
    arrow.name = name;
    group.add(arrow);
    if (label) {
      addAnnotationFormulaLabel(label, [
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
    const shaft = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(curve.getPoints(64)),
      new THREE.LineBasicMaterial({ color: 0x71879b }),
    );
    shaft.name = name + '-shaft';
    group.add(shaft);
    if (label) {
      const point = curve.getPoint(0.5);
      addAnnotationFormulaLabel(label, [point.x, point.y + labelOffset, point.z], 25, '#f2f4f8', 92);
    }
  }

  let cachedFilterWeights = null;
  let policyOutputAnnotationsBuilt = false;
  let policyConv2AnnotationsBuilt = false;
  let policySoftmaxAnnotationsBuilt = false;
  let policyBranchFilterGroup = null;
  let lastPolicyInputRight = null;
  const policyAnnotationSprites = [];

  function trackPolicyAnnotation(sprite) {
    if (sprite) policyAnnotationSprites.push(sprite);
    return sprite;
  }

  function addPolicyLabel(...args) {
    return trackPolicyAnnotation(addAnnotationLabel(...args));
  }

  function addPolicyFormulaLabel(...args) {
    return trackPolicyAnnotation(addAnnotationFormulaLabel(...args));
  }

  function updatePolicyBranchPosition(policyInput) {
    if (lastPolicyInputRight !== null && policyBranchFilterGroup) {
      const deltaX = policyInput.right - lastPolicyInputRight;
      if (Math.abs(deltaX) > 1e-8) {
        policyBranchFilterGroup.position.x += deltaX;
        for (const sprite of policyAnnotationSprites) {
          sprite.position.x += deltaX;
        }
      }
    }
    lastPolicyInputRight = policyInput.right;
  }

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
    policyBranchFilterGroup = null;
    lastPolicyInputRight = null;
    policyAnnotationSprites.length = 0;
    policyOutputAnnotationsBuilt = false;
    policyConv2AnnotationsBuilt = false;
    policySoftmaxAnnotationsBuilt = false;
    residualRenderer.reset();
  }

  let inputAnnotationsBuilt = false;
  let inputArrowsBuilt = false;

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

    if (!inputAnnotationsBuilt) {
    tensors.forEach((tensor) => {
      addAnnotationLabel(
        tensor.label,
        [tensor.x, 7.0, tensor.z + ((BOARD_SIZE - 1) * inputSpacing) / 2],
        24, '#ffffff', 70,
      );
      const firstBladeX = tensor.x - ((tensor.channels - 1) * inputDepthStep) / 2;
      const boardTopY = -4.082 + (BOARD_SIZE - 1) * inputSpacing;
      tensor.bladeLabels.forEach((bladeLabel, channel) => {
        addAnnotationLabel(
          bladeLabel,
          [firstBladeX + channel * inputDepthStep, 5.65, tensor.z - 0.08],
          20, '#e7edf6', 115, 'left',
        );
      });
      if (tensor.key === 'links') {
        for (let col = 0; col < BOARD_SIZE; col += 1) {
          addAnnotationLabel(String(col), [firstBladeX, 4.6, tensor.z + col * inputSpacing], 17, '#dce7f5', 145, 'center');
        }
        for (let row = 0; row < BOARD_SIZE; row += 1) {
          addAnnotationLabel(String(row), [firstBladeX - 0.12, boardTopY - row * inputSpacing, tensor.z - 0.48], 17, '#dce7f5', 145, 'center');
        }
      }
    });
    inputAnnotationsBuilt = true;
    }

    // Show all trained first-layer filters and their summed feature tensor
    // alongside the game encoding, rather than switching between separate tabs.
    const filterInfo = drawFilters();
    const policyOutputStats = filterInfo.policyOutputStats;
    const featureInfo = featureOutputRenderer.drawOutput();

    // Connect each input/filter pair and the three filter banks to the
    // shared feature tensor. The bank layouts are calculated in
    // drawFilters(), where their actual geometry dimensions are available.
    const arrowGroup = getAnnotationGroup();
    const featureY = -2.5 + (BOARD_SIZE - 1) * filterSpacing / 2;
    const featureZ = -((FILTER_COUNT - 1) * 0.105) / 2 +
      (BOARD_SIZE - 1) * filterSpacing / 2;
    const featureLeftX = 20 - filterCubeSize / 2;
    if (!inputArrowsBuilt) {
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
            label: bank.key === 'pegs' ? 'y₁ = |BN(Conv₁(x₁))|' : 'y₃ = |BN(Conv₃(x₃))|',
          },
        );
      } else {
        const linkLine = new THREE.LineCurve3(
          new THREE.Vector3(bankRight, featureY, featureZ),
          new THREE.Vector3(mergeX, featureY, featureZ),
        );
        const linkShaft = new THREE.Line(
          new THREE.BufferGeometry().setFromPoints(linkLine.getPoints(1)),
          new THREE.LineBasicMaterial({ color: 0x71879b }),
        );
        linkShaft.name = bank.key + '-filter-bank-to-merge-shaft';
        arrowGroup.add(linkShaft);
        addAnnotationFormulaLabel(
          bank.key === 'links' ? 'y₂ = |BN(Conv₂(x₂))|' : 'y₁ = |BN(Conv₁(x₁))|',
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
      featureY, featureZ, 'Σ', 0.58, 'merged-convolutions-to-feature-output',
    );
    inputArrowsBuilt = true;
    }

    return {
      title: 'First-layer network view',
      description: "Sample/effective boards, the three input tensors, trained first-layer filters, the selected residual block's two 48-channel convolution filter banks, the first-layer feature tensor, and both policy convolutions with their outputs are shown together.",
      shape: 'Inputs: 24 × 24 × (2 + 8 + 2) · Features: 24 × 24 × 48' +
        (policyOutputStats ? ' · Policy Conv₁: 24 × 24 × 2' +
          (policyOutputStats.policyConv2 ? ' · Policy Conv₂: 22 × 24 logits · Softmax: 22 × 24 probabilities' : '') : ''),
      range: 'Feature range ' + featureInfo.range +
        (policyOutputStats ? ' · Policy Conv₁ range ' + policyOutputStats.min.toFixed(4) +
          ' … ' + policyOutputStats.max.toFixed(4) +
          (policyOutputStats.policyConv2 ? ' · Policy Conv₂ range ' +
            policyOutputStats.policyConv2.min.toFixed(4) + ' … ' +
            policyOutputStats.policyConv2.max.toFixed(4) : '') +
          (policyOutputStats.policySoftmax ? ' · maximum move probability ' +
            policyOutputStats.policySoftmax.max.toFixed(4) : '') : ''),
      count: '3 input tensors · ' + filterInfo.count + ' · ' + filterInfo.residualBlockCount + ' residual blocks · selected block ' + filterInfo.selectedResidualBlockIndex + ' (' + featureInfo.residualCount + ' detailed filters) · ' + filterInfo.count + ' feature values' +
        (policyOutputStats ? ' · ' + policyOutputStats.count.toLocaleString() +
          ' Policy Conv₁ values' +
          (policyOutputStats.policyConv2 ? ' · ' +
            policyOutputStats.policyConv2.count.toLocaleString() +
            ' Policy Conv₂ logits' +
            (policyOutputStats.policySoftmax ? ' · ' +
              policyOutputStats.policySoftmax.count.toLocaleString() +
              ' move probabilities' : '') : '') : ''),
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
      glassPadding: [1.02, 1.0, 1.0],
    });

    // Policy Conv1 has two output filters, each with 48 input-channel
    // weights. Keep the model's [out_channels][1][1][in_channels] layout:
    // two filters side by side, with each filter's 48 blades along X.
    const policyInputChannels = weights.policyConv1?.[0]?.[0]?.[0]?.length ||
      weights.locations.length;
    // Keep the visualizer renderable if older cached/demo data is supplied.
    const policyWeights = Array.isArray(weights.policyConv1) &&
      weights.policyConv1.length === 2 &&
      Array.isArray(weights.policyConv1[0]?.[0]?.[0])
      ? weights.policyConv1
      : Array.from({ length: 2 }, () => [[Array(policyInputChannels).fill(0)]]);
    const policyColumns = 2;
    const policyRows = 1;
    const policyGapY = 0;
    const policyGapZ = 0.3;
    const policySlotWidth = filterSpacing;
    const policySlotHeight = filterSpacing;
    const policyBankHeight = policySlotHeight;
    // Policy Conv1 consumes the final residual-stack output, not the first-layer
    // feature tensor. Place its bank directly downstream of block 20's output.
    const policyInput = residualRenderer.getLastOutputAnchor();
    updatePolicyBranchPosition(policyInput);
    if (!policyBranchFilterGroup) {
      policyBranchFilterGroup = new THREE.Group();
      policyBranchFilterGroup.name = 'policy-branch-static-geometry';
      filterStaticGroup.add(policyBranchFilterGroup);
    }
    const policyInputRight = policyInput.right;
    // Leave a clearly visible processing gap between the residual output and
    // the policy bank; the arrow spans this full separation.
    const policyGapX = 6.0;
    const policyStartX = policyInputRight + policyGapX + filterCubeSize / 2;
    // Use the same row/column layout and vertical centering convention as Conv3.
    const policyStartY = policyInput.centerY +
      (policyRows - 1) * (policySlotHeight + policyGapY) / 2;
    // Center the two output filters around the residual-output centerline.
    const policyStartZ = policyInput.centerZ -
      (policyColumns - 1) * (policySlotWidth + policyGapZ) / 2 -
      (policySlotWidth - filterSpacing) / 2;
    const policyInfo = filterBankRenderer.drawBank({
      name: 'policy-conv1', weights: policyWeights,
      startX: policyStartX, startY: policyStartY, startZ: policyStartZ,
      columns: policyColumns, gapY: policyGapY, gapZ: policyGapZ,
      showBias: false, buildGeometry: rebuild,
      slotWidth: policySlotWidth, slotHeight: policySlotHeight,
      bankWidth: policyColumns * policySlotWidth + (policyColumns - 1) * policyGapZ,
      bankHeight: policyBankHeight,
      glassPadding: [1.04, 1.25, 1.25],
      targetGroup: policyBranchFilterGroup,
      targetLabel: addPolicyLabel,
    });

    if (rebuild) {
      const arrowStartX = policyInputRight + 0.04;
      const arrowEndX = policyStartX - 0.04;
      const arrowLength = arrowEndX - arrowStartX;
      const policyArrow = new THREE.ArrowHelper(
        new THREE.Vector3(1, 0, 0),
        new THREE.Vector3(arrowStartX, policyInput.centerY, policyInput.centerZ),
        arrowLength * 0.84,
        0x71879b,
        Math.min(0.14, arrowLength * 0.38),
        Math.min(0.10, arrowLength * 0.26),
      );
      policyArrow.name = 'final-residual-output-to-policy-conv1';
      policyBranchFilterGroup.add(policyArrow);
      addAnnotationLabel(
        'Pegs Conv₁ · 5×5×2×48',
        [startX + (pegChannels - 1) * filterSpacing / 2,
          pegStartY + pegHeight * filterSpacing + 0.75,
          pegStartZ + (columns * pegFilterWidth + (columns - 1) * pegGapZ) / 2],
        24, '#ffffff', 90,
      );
      addAnnotationLabel(
        'Links Conv₂ · 4×4×8×48',
        [startX + (channels - 1) * filterSpacing / 2,
          startY + kernelHeight * filterSpacing + 0.75,
          startZ + (columns * filterWidth + (columns - 1) * gapZ) / 2],
        24, '#ffffff', 90,
      );
      addAnnotationLabel(
        'Locations Conv₃ · 1×1×2×48',
        [startX + (locationChannels - 1) * filterSpacing / 2,
          locationStartY + locationHeight * filterSpacing + 0.75,
          locationStartZ + (columns * pegFilterWidth + (columns - 1) * locationGapZ) / 2],
        24, '#ffffff', 90,
      );
      addPolicyLabel(
        'Policy Conv₁ · 1×1×48×2',
        [policyStartX + (policyInputChannels - 1) * filterSpacing / 2,
          policyStartY + policySlotHeight + 0.75,
          policyStartZ + (policyColumns - 1) * (policySlotWidth + policyGapZ) / 2],
        24, '#ffffff', 90,
      );
    }

    // Policy Conv1 applies a 1×1 convolution to the final residual output,
    // followed by its own inference-mode BatchNorm and absolute activation.
    // Keep the displayed tensor at the full 24×24×2 shape (before policy Conv2
    // removes the first/last rows for the flattened move-policy output).
    let policyOutputStats = null;
    const policyInputTensor = getData().residualBlockLastOutput;
    const policyBn = getData().policyBn;
    if (policyInputTensor && policyBn &&
        Array.isArray(policyBn.weight) && policyBn.weight.length === 2 &&
        Array.isArray(weights.policyConv1) && weights.policyConv1.length === 2) {
      const policyValues = [];
      for (let outputChannel = 0; outputChannel < 2; outputChannel += 1) {
        for (let row = 0; row < BOARD_SIZE; row += 1) {
          for (let col = 0; col < BOARD_SIZE; col += 1) {
            let sum = 0;
            for (let inputChannel = 0; inputChannel < FILTER_COUNT; inputChannel += 1) {
              sum += Number(policyInputTensor[row][col][inputChannel]) *
                Number(weights.policyConv1[outputChannel][0][0][inputChannel]);
            }
            const normalized = (sum - Number(policyBn.running_mean[outputChannel])) /
              Math.sqrt(Number(policyBn.running_var[outputChannel]) + Number(policyBn.eps ?? 1e-3));
            policyValues.push(Math.abs(
              normalized * Number(policyBn.weight[outputChannel]) +
              Number(policyBn.bias[outputChannel])
            ));
          }
        }
      }
      const policyColorMax = Math.max(...policyBn.weight.map((weight, channel) =>
        Math.abs(Number(policyBn.bias[channel])) + 3 * Math.abs(Number(weight))));
      const policyBankRight = policyStartX +
        (policyInputChannels - 1) * filterSpacing + filterCubeSize / 2;
      // Reduce the Policy Conv1 → output separation by 20%.
      const policyOutputGap = 4.2 * 0.8;
      const policyOutputStartX = policyBankRight + policyOutputGap + filterCubeSize / 2;
      const outputChannelStep = filterCubeSize / 0.9 + filterSpacing - filterCubeSize;
      const outputOriginX = policyOutputStartX - outputChannelStep / 2;
      const outputOriginY = policyInput.centerY - (BOARD_SIZE - 1) * filterSpacing / 2;
      const outputOriginZ = policyInput.centerZ - (BOARD_SIZE - 1) * filterSpacing / 2;
      policyOutputStats = featureOutputRenderer.addValueCubes(
        policyValues, BOARD_SIZE, BOARD_SIZE, 2,
        [outputOriginX, outputOriginY, outputOriginZ],
        COLORS.output,
        {
          spacing: filterSpacing,
          cellSize: filterCubeSize / 0.9,
          channelGap: filterSpacing - filterCubeSize,
          colorMax: policyColorMax,
          positiveOnly: true,
        },
      );
      if (!policyOutputAnnotationsBuilt) {
        addPolicyLabel(
          'Policy Conv₁ output · 24×24×2',
          [outputOriginX + outputChannelStep / 2, 5, policyInput.centerZ],
          24, '#ffffff', 90,
        );
        const arrowLeft = policyBankRight + 0.08;
        const arrowRight = outputOriginX - filterCubeSize / 2;
        const arrowGap = arrowRight - arrowLeft;
        const centeredArrowLength = arrowGap * 0.58;
        const policyArrow = new THREE.ArrowHelper(
          new THREE.Vector3(1, 0, 0),
          new THREE.Vector3(
            (arrowLeft + arrowRight - centeredArrowLength) / 2,
            policyInput.centerY,
            policyInput.centerZ,
          ),
          centeredArrowLength,
          0x71879b,
          Math.min(0.14, centeredArrowLength * 0.38),
          Math.min(0.10, centeredArrowLength * 0.26),
        );
        policyArrow.name = 'policy-conv1-to-policy-output';
        policyBranchFilterGroup.add(policyArrow);
        addPolicyFormulaLabel(
          'y = |BN(Conv(x))|',
          [
            (arrowLeft + arrowRight) / 2,
            policyInput.centerY + 0.62,
            policyInput.centerZ,
          ],
          23, '#f2f4f8', 92,
        );
        policyOutputAnnotationsBuilt = true;
      }

      // Policy Conv2 is a 1×1 convolution from the two Policy Conv1
      // activation channels to one policy-logit channel. Since Black moves
      // and the first/last rows are illegal, retain only the 22×24 playable
      // board. Softmax below normalizes probabilities across those 528 cells.
      const policyConv2Weights = weights.policyConv2;
      if (Array.isArray(policyConv2Weights) &&
          policyConv2Weights.length === 1 &&
          policyConv2Weights[0]?.[0]?.[0]?.length === 2) {
        const policyConv2Gap = policyOutputGap;
        const policyOutputRight = outputOriginX + outputChannelStep + filterCubeSize / 2;
        const policyConv2StartX = policyOutputRight + policyConv2Gap +
          filterCubeSize / 2;
        const policyConv2StartY = policyInput.centerY;
        const policyConv2StartZ = policyInput.centerZ;
        // The policy weights can become available after the first filter-scene
        // build. In that case the output and arrows are drawn on later passes,
        // but a rebuild-only guard would leave the Conv2 bank (and glass) absent.
        const policyConv2GlassExists = policyBranchFilterGroup.children.some(
          (child) => child.name === 'policy-conv2-convolution-glass-outline',
        );
        const policyConv2Info = filterBankRenderer.drawBank({
          name: 'policy-conv2',
          weights: policyConv2Weights,
          startX: policyConv2StartX,
          startY: policyConv2StartY,
          startZ: policyConv2StartZ,
          columns: 1,
          gapY: 0,
          gapZ: 0,
          showBias: false,
          buildGeometry: rebuild || !policyConv2GlassExists,
          slotWidth: filterSpacing,
          slotHeight: filterSpacing,
          bankWidth: filterSpacing,
          bankHeight: filterSpacing,
          glassPadding: [1.32, 1.55, 1.55],
          cubeScale: 1.8,
          // Separate the two input-channel weights by a visible gap, matching
          // the spacing between the two Policy Conv1 filters.
          channelSpacing: filterSpacing + 0.3,
          targetGroup: policyBranchFilterGroup,
          targetLabel: addPolicyLabel,
        });
        const policyConv2BankRight = policyConv2StartX + filterSpacing + 0.3 +
          filterCubeSize / 2;
        const policyConv2OutputGap = policyOutputGap;
        const policyConv2OutputOriginX = policyConv2BankRight +
          policyConv2OutputGap + filterCubeSize / 2;
        const policyConv2Values = [];
        for (let row = 1; row < BOARD_SIZE - 1; row += 1) {
          for (let col = 0; col < BOARD_SIZE; col += 1) {
            let sum = 0;
            for (let channel = 0; channel < 2; channel += 1) {
              sum += policyValues[channel * BOARD_SIZE * BOARD_SIZE +
                row * BOARD_SIZE + col] *
                Number(policyConv2Weights[0][0][0][channel]);
            }
            policyConv2Values.push(sum);
          }
        }
        const policyBoardRows = BOARD_SIZE - 2;
        const policyBoardCols = BOARD_SIZE;
        const policyBoardOriginY =
          policyInput.centerY - (policyBoardRows - 1) * filterSpacing / 2;
        const policyBoardOriginZ =
          policyInput.centerZ - (policyBoardCols - 1) * filterSpacing / 2;
        const policyConv2Stats = featureOutputRenderer.addValueCubes(
          policyConv2Values, policyBoardRows, policyBoardCols, 1,
          [
            policyConv2OutputOriginX,
            policyBoardOriginY,
            policyBoardOriginZ,
          ],
          COLORS.output,
          {
            spacing: filterSpacing,
            cellSize: filterCubeSize / 0.9,
            // Conv2 has no BatchNorm or absolute activation: these are
            // signed policy logits. Use their actual symmetric range so
            // negative and positive values are represented proportionally.
            colorMax: Math.max(1e-8, ...policyConv2Values.map((value) =>
              Math.abs(Number(value)))),
          },
        );

        // Stable softmax over all 528 playable intersections. The cropped
        // 22×24 board remains two-dimensional; no flattening layer is shown.
        const maxLogit = Math.max(...policyConv2Values);
        const exponentials = policyConv2Values.map((value) =>
          Math.exp(Number(value) - maxLogit));
        const exponentialSum = exponentials.reduce((sum, value) => sum + value, 0);
        const policyProbabilities = exponentials.map((value) =>
          value / exponentialSum);
        // Match the Softmax spacing to the full output-tensor-to-output-tensor
        // gap from Policy Conv1 to Policy Conv2 (including the Conv2 filter bank).
        const policyConv1OutputRight = outputOriginX + outputChannelStep +
          filterCubeSize / 2;
        const policyConv2OutputLeft = policyConv2OutputOriginX - filterCubeSize / 2;
        const policySoftmaxOutputGap = policyConv2OutputLeft -
          policyConv1OutputRight;
        const policySoftmaxOriginX = policyConv2OutputOriginX +
          policySoftmaxOutputGap + filterCubeSize;
        const policySoftmaxStats = featureOutputRenderer.addValueCubes(
          policyProbabilities, policyBoardRows, policyBoardCols, 1,
          [
            policySoftmaxOriginX,
            policyBoardOriginY,
            policyBoardOriginZ,
          ],
          COLORS.output,
          {
            spacing: filterSpacing,
            cellSize: filterCubeSize / 0.9,
            // Probability colour runs from green (zero) to red (highest).
            positiveOnly: true,
            colorMax: Math.max(1e-8, ...policyProbabilities),
          },
        );
        if (!policyConv2AnnotationsBuilt) {
          addPolicyLabel(
            'Policy Conv₂ · 1×1×2×1',
            [
              policyConv2StartX + filterSpacing / 2,
              policyConv2StartY + 1.0,
              policyConv2StartZ,
            ],
            24, '#ffffff', 90,
          );
          addPolicyLabel(
            'Policy Conv₂ output · 22×24×1',
            [
              policyConv2OutputOriginX,
              5,
              policyInput.centerZ,
            ],
            24, '#ffffff', 90,
          );
          const addCenteredPolicyArrow = (left, right, name) => {
            const gap = right - left;
            const length = gap * 0.58;
            const arrow = new THREE.ArrowHelper(
              new THREE.Vector3(1, 0, 0),
              new THREE.Vector3(
                (left + right - length) / 2,
                policyInput.centerY,
                policyInput.centerZ,
              ),
              length,
              0x71879b,
              Math.min(0.14, length * 0.38),
              Math.min(0.10, length * 0.26),
            );
            arrow.name = name;
            policyBranchFilterGroup.add(arrow);
          };
          addCenteredPolicyArrow(
            policyOutputRight + 0.08,
            policyConv2StartX - filterCubeSize / 2,
            'policy-output-to-conv2',
          );
          addCenteredPolicyArrow(
            policyConv2BankRight + 0.08,
            policyConv2OutputOriginX - filterCubeSize / 2,
            'policy-conv2-to-policy-logits',
          );
          addPolicyFormulaLabel(
            'y = Conv(x)',
            [
              (policyConv2BankRight + 0.08 +
                policyConv2OutputOriginX - filterCubeSize / 2) / 2,
              policyInput.centerY + 0.62,
              policyInput.centerZ,
            ],
            23, '#f2f4f8', 92,
          );
          policyConv2AnnotationsBuilt = true;
        }
        if (!policySoftmaxAnnotationsBuilt) {
          addPolicyLabel(
            'Softmax output · 22×24×1',
            [
              policySoftmaxOriginX,
              5,
              policyInput.centerZ,
            ],
            24, '#ffffff', 90,
          );
          const arrowLeft = policyConv2OutputOriginX +
            filterCubeSize / 2 + 0.08;
          const arrowRight = policySoftmaxOriginX - filterCubeSize / 2;
          const gap = arrowRight - arrowLeft;
          const length = gap * 0.58;
          const arrow = new THREE.ArrowHelper(
            new THREE.Vector3(1, 0, 0),
            new THREE.Vector3(
              (arrowLeft + arrowRight - length) / 2,
              policyInput.centerY,
              policyInput.centerZ,
            ),
            length,
            0x71879b,
            Math.min(0.14, length * 0.38),
            Math.min(0.10, length * 0.26),
          );
          arrow.name = 'policy-logits-to-softmax';
          policyBranchFilterGroup.add(arrow);
          addPolicyFormulaLabel(
            'pᵢ = exp(zᵢ) / Σ exp(z)',
            [
              (arrowLeft + arrowRight) / 2,
              policyInput.centerY + 0.62,
              policyInput.centerZ,
            ],
            23, '#f2f4f8', 92,
          );
          policySoftmaxAnnotationsBuilt = true;
        }
        policyOutputStats.policyConv2 = policyConv2Stats;
        policyOutputStats.policySoftmax = policySoftmaxStats;
        policyOutputStats.policyConv2FilterCount = policyConv2Info.count;
      }
    }

    const blockCount = Math.max(1, Number(getData().residualBlockCount) || 20);
    const selectedResidualBlockIndex = residualRenderer.draw(rebuild, blockCount);
    cachedFilterWeights = weights;

    return {
      title: 'Trained convolution filters',
      description: 'Trained input convolutions and an interactive residual stack feed the policy branch. Policy Conv₂ produces logits for the cropped 22×24 Black-to-move board; Softmax converts those 528 logits into move probabilities while preserving the two-dimensional board layout. Residual outputs include batch normalization, absolute-value activations, and the skip connection.',
      shape: 'Locations Conv 1×1×2×48; Pegs Conv 5×5×2×48; Links Conv 4×4×8×48; Policy Conv₁ 1×1×48×2; Policy Conv₂ 1×1×2×1',
      range: 'Trained signed weights; Policy Conv₁ is 1×1×48×2 (two output filters, 48 input features each) and has no bias.' +
        (policyOutputStats ? ' Policy Conv1 output ' + policyOutputStats.min.toFixed(4) +
          ' … ' + policyOutputStats.max.toFixed(4) + ' (colour scale 0 … ' +
          policyOutputStats.colorMax.toFixed(4) + ').' : ''),
      count: locationInfo.count + ' location filters + ' + pegInfo.count + ' peg filters + ' + linkInfo.count + ' link filters + ' + policyInfo.count + ' policy filters' +
        (policyOutputStats ? ' · ' + policyOutputStats.count.toLocaleString() + ' policy output cells' : ''),
      residualCount: 2 * FILTER_COUNT,
      policyOutputStats,
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


  function refreshTensorColors() {
    // Recolour persistent location input instances and rebuild colour-coded
    // convolution/filter meshes using the current tanh scale.
    inputRenderer.refreshLocationColors();
    cachedFilterWeights = null;
  }

  return {
    refreshTensorColors,
    addValueCubes: featureOutputRenderer.addValueCubes,
    drawFilterKernel: filterBankRenderer.drawFilterKernel,
    drawInputs,
    drawFilters,
    drawOutput: featureOutputRenderer.drawOutput,
    setResidualFiltersVisible: residualRenderer.setResidualFiltersVisible,
    setShowIntermediateResidualBlocks: residualRenderer.setShowIntermediateResidualBlocks,
    getBlock20CameraOffset: residualRenderer.getBlock20CameraOffset,
    getCompactBlock20CameraOffset: residualRenderer.getCompactBlock20CameraOffset,
    setSelectedResidualBlock: residualRenderer.setSelectedResidualBlock,
    pickResidualBlock: residualRenderer.pickResidualBlock,
  };
}
