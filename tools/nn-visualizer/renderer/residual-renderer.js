/** Rendering and interaction for the residual-block stack. */
export function createResidualRenderer({
  THREE, BOARD_SIZE, FILTER_COUNT, cubeGeometry, filterSpacing,
  filterCubeSize, filterStaticGroup, flattenTensor, addLabel, addFormulaLabel, getData,
  heatMapColor,
}) {
  let cachedResidualOutput = null;
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
  const convGapX = 2.0 * 3 * 3 * 0.5;
  const outputGapX = 1.25 * 1.15 * 3;
  const baseBlockPitchX =
    2 * bankSizeX + convGapX + outputGapX +
    outputCubeSize / 2 + outputSizeX + 1.25 + filterCubeSize / 2;
  // Increase the complete pitch between successive residual blocks by 20%.
  const blockPitchX = baseBlockPitchX * 1.2;
  const blockStartX =
    outputMaxX + firstResidualGapX + filterCubeSize / 2 +
    blockIndex * blockPitchX;
  // Place Conv2 to the right of Conv1, then leave room for the block output.
  const currentBlockRightX =
    blockStartX - filterCubeSize / 2 + 2 * bankSizeX + convGapX;
  const nextBlockLeftX =
    blockStartX + blockPitchX - filterCubeSize / 2;
  const outputCenterX = (currentBlockRightX + nextBlockLeftX) / 2;
  const outputStartX =
    outputCenterX - (channels - 1) * outputChannelStep / 2;
  return {
    channels, kernelHeight, kernelWidth, columns, rows,
    filterWidth, filterHeight, gapY, gapZ,
    bankSizeX, bankSizeY, bankSizeZ, bankStartY,
    bankStartZ,
    bankStartXs: [blockStartX, blockStartX + bankSizeX + convGapX],
    bankStartZs: [bankStartZ, bankStartZ],
    blockStartX, outputStartX, outputCenterX, outputChannelStep,
    outputSizeX, outputSizeY, outputSizeZ,
    outputCenterY, outputCenterZ, outputCubeSize,
  };
}

function addResidualOutputValues(group, layout, blockIndex, tensor) {
  const count = FILTER_COUNT * BOARD_SIZE * BOARD_SIZE;
  const mesh = new THREE.InstancedMesh(
    cubeGeometry,
    new THREE.MeshStandardMaterial({
      color: 0xffffff, roughness: 0.55, transparent: true, opacity: 0.96,
    }),
    count,
  );
  mesh.name = 'residual-block-' + blockIndex + '-actual-output-values';
  const values = [];
  for (let channel = 0; channel < FILTER_COUNT; channel += 1) {
    for (let row = 0; row < BOARD_SIZE; row += 1) {
      for (let col = 0; col < BOARD_SIZE; col += 1) {
        values.push(Number(tensor[row][col][channel]));
      }
    }
  }
  const maxAbs = Math.max(1e-8, ...values.map(Math.abs));
  const transform = new THREE.Object3D();
  let instance = 0;
  for (let channel = 0; channel < FILTER_COUNT; channel += 1) {
    for (let row = 0; row < BOARD_SIZE; row += 1) {
      for (let col = 0; col < BOARD_SIZE; col += 1) {
        const value = Number(tensor[row][col][channel]);
        transform.position.set(
          layout.outputStartX + channel * layout.outputChannelStep,
          layout.outputCenterY + (BOARD_SIZE - 1 - row) * filterSpacing -
            (BOARD_SIZE - 1) * filterSpacing / 2,
          layout.outputCenterZ + col * filterSpacing -
            (BOARD_SIZE - 1) * filterSpacing / 2,
        );
        transform.scale.setScalar(layout.outputCubeSize * 0.72);
        transform.updateMatrix();
        mesh.setMatrixAt(instance, transform.matrix);
        // Residual outputs are nonnegative by construction (abs activations
        // in the stem and both residual convolutions, followed by h + x).
        // Use a positive-only green-to-red scale so blue is not mistaken for
        // a negative value.
        mesh.setColorAt(instance, new THREE.Color().setHSL(
          (1 / 3) * (1 - Math.max(0, Math.min(1, value / maxAbs))),
          1,
          0.5,
        ));
        instance += 1;
      }
    }
  }
  mesh.count = instance;
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  group.add(mesh);
  mesh.userData.valueRange = { min: Math.min(...values), max: maxAbs };
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
          layout.bankStartZs[bankIndex] + column * (layout.filterWidth + layout.gapZ);
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
    const startZ = layout.bankStartZs[index];
    addLabel(
      'Residual block ' + blockIndex + ' · ' + name + ' · 48 × 5×5×48 filters',
      [
        startX + (layout.bankSizeX - filterCubeSize) / 2,
        layout.bankStartY + layout.filterHeight + 0.8,
        startZ + (layout.bankSizeZ - filterCubeSize) / 2,
      ],
      22,
    );
  }
}

  function addResidualArrow(
    group, startX, endX, y, z, label, name,
    labelYOffset = 0.58, startClearanceFraction = 0.22,
  ) {
    const gap = endX - startX;
    if (gap <= 0.15) return;
    // Most arrows leave a deliberate gap after their source object. The
    // post-merge arrow is the exception: it must begin exactly at the
    // curved convolution/identity join point.
    const startClearance = gap * startClearanceFraction;
    const length = gap * (0.89 - startClearanceFraction);
    const arrow = new THREE.ArrowHelper(
      new THREE.Vector3(1, 0, 0),
      new THREE.Vector3(startX + startClearance, y, z),
      length,
      0xaeb7c2,
      0.42,
      0.22,
    );
    arrow.name = name;
    group.add(arrow);
    if (label) {
      addFormulaLabel(label, [(startX + endX) / 2, y + labelYOffset, z], 25, '#f2f4f8', 92);
    }
  }

  function addResidualCurvedArrow(group, startX, endX, startY, startZ, endY, endZ, name) {
    const dx = endX - startX;
    if (dx <= 0.2) return;
    const curve = new THREE.CubicBezierCurve3(
      new THREE.Vector3(startX, startY, startZ),
      new THREE.Vector3(startX + dx * 0.42, startY, startZ),
      new THREE.Vector3(endX - dx * 0.42, endY, endZ),
      new THREE.Vector3(endX, endY, endZ),
    );
    const shaft = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(curve.getPoints(64)),
      new THREE.LineBasicMaterial({ color: 0xaeb7c2 }),
    );
    shaft.name = name + '-shaft';
    group.add(shaft);
  }

  function addResidualBypass(group, startX, endX, centerZ, offsetZ, name) {
    const dx = endX - startX;
    if (dx <= 1) return;
    const turn = Math.min(dx * 0.22, 3.0);
    const outsideZ = centerZ + offsetZ;
    const path = new THREE.CurvePath();
    path.add(new THREE.CubicBezierCurve3(
      new THREE.Vector3(startX, 0, centerZ),
      new THREE.Vector3(startX + turn * 0.45, 0, centerZ),
      new THREE.Vector3(startX + turn * 0.55, 0, outsideZ),
      new THREE.Vector3(startX + turn, 0, outsideZ),
    ));
    path.add(new THREE.LineCurve3(
      new THREE.Vector3(startX + turn, 0, outsideZ),
      new THREE.Vector3(endX - turn, 0, outsideZ),
    ));
    path.add(new THREE.CubicBezierCurve3(
      new THREE.Vector3(endX - turn, 0, outsideZ),
      new THREE.Vector3(endX - turn * 0.55, 0, outsideZ),
      new THREE.Vector3(endX - turn * 0.45, 0, centerZ),
      new THREE.Vector3(endX, 0, centerZ),
    ));
    const shaft = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(path.getPoints(96)),
      new THREE.LineBasicMaterial({ color: 0xaeb7c2 }),
    );
    shaft.name = name + '-shaft';
    group.add(shaft);
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
      {
        for (let bankIndex = 0; bankIndex < 2; bankIndex += 1) {
          const position = new THREE.Vector3(
            layout.bankStartXs[bankIndex] + (layout.bankSizeX - filterCubeSize) / 2,
            layout.bankStartY +
              (-(layout.rows - 1) * (layout.filterHeight + layout.gapY) +
                (layout.kernelHeight - 1) * filterSpacing) / 2,
            layout.bankStartZs[bankIndex] + (layout.bankSizeZ - filterCubeSize) / 2,
          );
          const glassGeometry = new THREE.BoxGeometry(
            layout.bankSizeX * 1.08, layout.bankSizeY * 1.08, layout.bankSizeZ * 1.08,
          );
          // Keep the enclosure as an open frame rather than solid glass
          // panes, so the camera can move inside without the walls obscuring
          // the filters.
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

      // Connect each neighbouring stage along X. The first arrow carries
      // the input into Conv1; the two convolution arrows show their transforms.
      const bankLeft1 = layout.bankStartXs[0] - filterCubeSize / 2;
      const bankRight1 = layout.bankStartXs[0] +
        (layout.channels - 1) * filterSpacing + filterCubeSize / 2;
      const bankLeft2 = layout.bankStartXs[1] - filterCubeSize / 2;
      const bankRight2 = layout.bankStartXs[1] +
        (layout.channels - 1) * filterSpacing + filterCubeSize / 2;
      const bankCenterY = layout.outputCenterY;
      const bankCenterZ = layout.outputCenterZ;
      const outputLeftX = layout.outputCenterX - layout.outputSizeX / 2;
      const outputRightX = layout.outputCenterX + layout.outputSizeX / 2;

      if (index === 0) {
        const inputRightX = 20 +
          (FILTER_COUNT - 1) * layout.outputChannelStep +
          layout.outputCubeSize / 2;
        addResidualArrow(
          residualPlaceholderGroup, inputRightX, bankLeft1,
          bankCenterY, bankCenterZ, null,
          'feature-output-to-residual-0-conv1',
        );
      } else {
        const previous = residualLayout(index - 1);
        const previousRightX = previous.outputCenterX + previous.outputSizeX / 2;
        addResidualArrow(
          residualPlaceholderGroup, previousRightX, bankLeft1,
          bankCenterY, bankCenterZ, null,
          'residual-output-' + (index - 1) + '-to-conv1-' + index,
        );
      }

      addResidualArrow(
        residualPlaceholderGroup, bankRight1, bankLeft2,
        bankCenterY, bankCenterZ,
        null,
        'residual-' + index + '-conv1-to-conv2',
      );
      // Main path bends into a shared merge point just before the output.
      const mergeX = outputLeftX - 1.35;
      addResidualCurvedArrow(
        residualPlaceholderGroup, bankRight2, mergeX,
        0, bankCenterZ, 0, bankCenterZ,
        'residual-' + index + '-conv2-to-merge',
      );

      // The identity path bypasses both convolutions with a double-S curve,
      // then joins the main path at the same merge point.
      const skipStartX = index === 0
        ? 20 + (FILTER_COUNT - 1) * layout.outputChannelStep + layout.outputCubeSize / 2
        : residualLayout(index - 1).outputCenterX + residualLayout(index - 1).outputSizeX / 2;
      // Keep the bypass in the y=0 plane. It leaves the bank smoothly,
      // runs straight outside the full Z extent of both convolution banks,
      // then turns back to the shared merge point.
      const bypassOffsetZ = layout.bankSizeZ * 1.08 / 2 + 1.0;
      addResidualBypass(
        residualPlaceholderGroup, skipStartX, mergeX,
        bankCenterZ, bypassOffsetZ,
        'residual-' + index + '-identity-skip',
      );
      addResidualArrow(
        residualPlaceholderGroup, mergeX, outputLeftX,
        bankCenterY, bankCenterZ, null,
        'residual-' + index + '-merged-output',
        0.58, 0,
      );



      // Output tensors remain unenclosed; show actual values for block 0.
      if (index === 0 && getData().residualBlock0Output) {
        addResidualOutputValues(residualPlaceholderGroup, layout, index, getData().residualBlock0Output);
      }

      addLabel(
        'residual ' + index,
        [
          layout.bankStartXs[0] + layout.bankSizeX / 2,
          layout.outputCenterY + layout.bankSizeY / 2 + 1.0,
          layout.bankStartZ + layout.bankPairSizeZ / 2,
        ],
        16, '#dce7f5', 110,
      );
      addLabel(
        (index === 0 && getData().residualBlock0Output ? 'Block 0 actual output · 24×24×48' : 'Block ' + index + ' output · 24×24×48'),
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

  function drawResidualPlaceholderLabels(blockCount) {
    // Arrow geometry is persistent, but formula sprites belong to the
    // per-render label pool and therefore need to be attached every frame.
    for (let index = 0; index < blockCount; index += 1) {
      const layout = residualLayout(index);
      const bankRight1 = layout.bankStartXs[0] +
        (layout.channels - 1) * filterSpacing + filterCubeSize / 2;
      const bankLeft2 = layout.bankStartXs[1] - filterCubeSize / 2;
      const bankRight2 = layout.bankStartXs[1] +
        (layout.channels - 1) * filterSpacing + filterCubeSize / 2;
      const outputLeftX = layout.outputCenterX - layout.outputSizeX / 2;
      const centerY = layout.outputCenterY;
      const centerZ = layout.outputCenterZ;

      addFormulaLabel(
        'y = |BN(Conv(x))|',
        [(bankRight1 + bankLeft2) / 2, centerY + 0.58, centerZ],
        25, '#f2f4f8', 92,
      );
      const mergeX = outputLeftX - 1.35;
      addFormulaLabel(
        'y = |BN(Conv(x))|',
        [(bankRight2 + mergeX) / 2, centerY + 1.15, centerZ],
        25, '#f2f4f8', 92,
      );

      const skipStartX = index === 0
        ? 20 + (FILTER_COUNT - 1) * layout.outputChannelStep + layout.outputCubeSize / 2
        : residualLayout(index - 1).outputCenterX + residualLayout(index - 1).outputSizeX / 2;
      // Place the identity formula beside the straight section of the
      // bypass itself (y=0, outside the convolution banks in Z).
      const bypassOffsetZ = layout.bankSizeZ * 1.08 / 2 + 1.0;
      addFormulaLabel(
        'y = x',
        [(skipStartX + mergeX) / 2, 0.62, centerZ + bypassOffsetZ],
        25, '#f2f4f8', 92,
      );
      // The two convolution paths and identity path are summed at this arrow.
      addFormulaLabel(
        'Σ',
        [(mergeX + outputLeftX) / 2, centerY + 0.58, centerZ],
        25, '#f2f4f8', 92,
      );
    }
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


  function reset() {
    residualFilterGroup = null;
    residualPlaceholderGroup = null;
    residualPlaceholderDirty = false;
    residualHitTargets = [];
  }

  function draw(rebuild, blockCount) {
    if (getData().residualBlock0Output !== cachedResidualOutput) {
      residualPlaceholderDirty = true;
    }
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
      cachedResidualOutput = getData().residualBlock0Output || null;
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
    return selectedResidualBlockIndex;
  }

  return {
    draw,
    reset,
    setResidualFiltersVisible,
    setSelectedResidualBlock,
    pickResidualBlock,
  };
}
