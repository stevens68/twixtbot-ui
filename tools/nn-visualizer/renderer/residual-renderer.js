/** Rendering and interaction for the residual-block stack. */
export function createResidualRenderer({
  THREE, BOARD_SIZE, FILTER_COUNT, cubeGeometry, filterSpacing,
  filterCubeSize, filterStaticGroup, flattenTensor, addLabel, getData,
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
  const bankStartZ = outputCenterZ - bankSizeZ / 2 + filterCubeSize / 2;
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

      // Connect the initial post-BN feature tensor to residual block 0.
      if (index === 0) {
        const initialOutputRightX =
          20 + (FILTER_COUNT - 1) * layout.outputChannelStep +
          filterCubeSize / 2;
        const residualBankLeftX = layout.blockStartX - filterCubeSize / 2;
        const fullArrowLength = Math.max(0.1, residualBankLeftX - initialOutputRightX);
        // Match the inter-block arrows: use 80% of the available gap and
        // center the shortened arrow exactly between source and target.
        const shortenedArrowLength = fullArrowLength * 0.8;
        const arrowY = layout.outputCenterY + layout.outputSizeY / 2 + 1.2;
        const arrow = new THREE.ArrowHelper(
          new THREE.Vector3(1, 0, 0),
          new THREE.Vector3(
            initialOutputRightX + (fullArrowLength - shortenedArrowLength) / 2,
            arrowY,
            layout.outputCenterZ,
          ),
          shortenedArrowLength,
          0xaeb7c2,
          0.48,
          0.26,
        );
        arrow.name = 'initial-feature-output-to-residual-0';
        residualPlaceholderGroup.add(arrow);
      }

      // Connect consecutive residual output tensors with a straight grey
      // arrow. Raise it above the filter banks so it remains visible.
      if (index < blockCount - 1) {
        const nextLayout = residualLayout(index + 1);
        const arrowY = layout.outputCenterY + layout.outputSizeY / 2 + 1.2;
        const arrowStartX = layout.outputCenterX + layout.outputSizeX / 2 + 0.12;
        const arrowEndX = nextLayout.outputCenterX - nextLayout.outputSizeX / 2 - 0.12;
        const fullArrowLength = Math.max(0.1, arrowEndX - arrowStartX);
        const shortenedArrowLength = fullArrowLength * 0.8;
        const arrow = new THREE.ArrowHelper(
          new THREE.Vector3(1, 0, 0),
          new THREE.Vector3(arrowStartX + (fullArrowLength - shortenedArrowLength) / 2, arrowY, layout.outputCenterZ),
          shortenedArrowLength,
          0xaeb7c2,
          0.48,
          0.26,
        );
        arrow.name = 'residual-output-' + index + '-to-output-' + (index + 1);
        residualPlaceholderGroup.add(arrow);
      }

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
