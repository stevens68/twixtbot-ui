/** Shared rendering primitives for trained convolution filter banks. */
export function createFilterBankRenderer({
  THREE, cubeGeometry, filterStaticGroup, flattenTensor, addLabel,
  heatMapColor, filterSpacing, filterCubeSize, inputSpacing, inputCellSize,
}) {
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


  function addBankGlass(name, x, y, z, sizeX, sizeY, sizeZ) {
    const geometry = new THREE.BoxGeometry(sizeX * 1.08, sizeY * 1.08, sizeZ * 1.08);
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(geometry),
      new THREE.LineBasicMaterial({ color: 0x9eabb9, transparent: true, opacity: 0.48 }),
    );
    geometry.dispose();
    edges.position.set(x + sizeX / 2, y + sizeY / 2, z + sizeZ / 2);
    edges.name = name + '-convolution-glass-outline';
    filterStaticGroup.add(edges);
  }

  function drawBank({
    name, weights, startX, startY, startZ, columns, gapY, gapZ,
    showBias = false, biasValues = null, buildGeometry = true,
    slotWidth, slotHeight, bankWidth, bankHeight,
  }) {
    const count = weights.length;
    const kernelHeight = weights[0].length;
    const kernelWidth = weights[0][0].length;
    const channels = weights[0][0][0].length;
    const rows = Math.ceil(count / columns);
    if (buildGeometry) {
      addBankGlass(
        name, startX, startY - (rows - 1) * (slotHeight + gapY),
        startZ, channels * filterSpacing, bankHeight,
        columns * slotWidth + (columns - 1) * gapZ,
      );
    }
    for (let filter = 0; filter < count; filter += 1) {
      const column = filter % columns;
      const row = Math.floor(filter / columns);
      drawFilterKernel(
        filter,
        startX,
        startY - row * (slotHeight + gapY),
        startZ + column * (slotWidth + gapZ),
        weights, kernelHeight, kernelWidth, channels,
        String(filter + 1), showBias, buildGeometry, biasValues,
      );
    }
    return { count, kernelHeight, kernelWidth, channels, rows };
  }

  return { drawFilterKernel, drawBank };
}
