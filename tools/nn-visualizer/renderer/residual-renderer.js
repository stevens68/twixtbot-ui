/** Rendering and interaction for the residual-block stack. */
export function createResidualRenderer({
  THREE, BOARD_SIZE, FILTER_COUNT, cubeGeometry, filterSpacing,
  filterCubeSize, filterStaticGroup, flattenTensor, addLabel, addFormulaLabel, getData,
  heatMapColor,
}) {
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
  // The output tensor is no longer part of the residual-stack view.
  // Keep only a compact separation between the two convolution banks.
  const convGapX = 4.5;
  const blockGapX = 2.5;
  const blockSpanX = 2 * bankSizeX + convGapX;
  const blockPitchX = blockSpanX + blockGapX;
  const outputCenterY = -2.5 + (BOARD_SIZE - 1) * filterSpacing / 2;
  const outputCenterZ =
    -((channels - 1) * 0.105) / 2 +
    (BOARD_SIZE - 1) * filterSpacing / 2;
  const bankStartY = outputCenterY -
    (bankMinYOffset + bankMaxYOffset) / 2;
  const bankStartZ = outputCenterZ - bankSizeZ / 2 + filterCubeSize / 2;
  const firstResidualGapX = 5.0;
  const firstBlockX = 20 + (channels - 1) *
    (filterCubeSize / 0.9 + filterSpacing - filterCubeSize) +
    filterCubeSize / 0.9 / 2 + firstResidualGapX + filterCubeSize / 2;
  const blockStartX = firstBlockX + blockIndex * blockPitchX;
  return {
    channels, kernelHeight, kernelWidth, columns, rows,
    filterWidth, filterHeight, gapY, gapZ,
    bankSizeX, bankSizeY, bankSizeZ, bankStartY, bankStartZ,
    bankStartXs: [blockStartX, blockStartX + bankSizeX + convGapX],
    bankStartZs: [bankStartZ, bankStartZ],
    blockStartX, blockPitchX, blockSpanX,
    outputCenterY, outputCenterZ,
  };
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
      'Residual block ' + (blockIndex + 1) + ' · ' + name + ' · 48 × 5×5×48 filters',
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

    for (let index = 0; index < blockCount; index += 1) {
      const layout = residualLayout(index);
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
        const outline = new THREE.LineSegments(
          new THREE.EdgesGeometry(glassGeometry),
          new THREE.LineBasicMaterial({
            color: 0x6f879a, transparent: true, opacity: 0.95,
          }),
        );
        glassGeometry.dispose();
        outline.position.copy(position);
        outline.name =
          'residual-block-' + (index + 1) + '-conv' + (bankIndex + 1) + '-outline';
        residualPlaceholderGroup.add(outline);

        const hitTarget = new THREE.Mesh(
          new THREE.BoxGeometry(layout.bankSizeX, layout.bankSizeY, layout.bankSizeZ),
          new THREE.MeshBasicMaterial({ visible: false }),
        );
        hitTarget.position.copy(position);
        hitTarget.name =
          'residual-block-' + (index + 1) + '-conv' + (bankIndex + 1) + '-hit-target';
        hitTarget.userData.residualBlockIndex = index;
        residualPlaceholderGroup.add(hitTarget);
        residualHitTargets.push(hitTarget);
      }
    }
    filterStaticGroup.add(residualPlaceholderGroup);
  }

  function drawResidualPlaceholderLabels(blockCount) {
    for (let index = 0; index < blockCount; index += 1) {
      const layout = residualLayout(index);
      const blockCenterX = layout.blockStartX + layout.blockSpanX / 2;
      const labelY = layout.bankStartY + layout.bankSizeY / 2 + 1.0;
      const labelZ = layout.bankStartZ + layout.bankSizeZ / 2;

      addLabel(
        'Residual block ' + (index + 1),
        [blockCenterX, labelY + 1.0, labelZ],
        20, '#dce7f5', 110,
      );
      addLabel(
        'Conv1 · 48 × 5×5×48 weights',
        [layout.bankStartXs[0] + layout.bankSizeX / 2, labelY, labelZ],
        15, '#dce7f5', 110,
      );
      addLabel(
        'Conv2 · 48 × 5×5×48 weights',
        [layout.bankStartXs[1] + layout.bankSizeX / 2, labelY, labelZ],
        15, '#dce7f5', 110,
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
