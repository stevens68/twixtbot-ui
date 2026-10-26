/** Rendering helpers for input tensors, convolution kernels, and feature outputs. */
export function createTensorRenderer({
  THREE, BOARD_SIZE, FILTER_COUNT, COLORS, cubeGeometry, flattenTensor,
  createTensor3, getSceneGroup, getData, createSampleGame,
  drawSampleBoard, drawEffectiveBoard, logLinkEncodingDiagnostic,
  logFullInputTensorValidation, calculateConvolutionOutput, addLabel,
}) {
  function addValueCubes(values, rows, cols, channels, origin, color, options = {}) {
    const maxAbs = Math.max(1e-8, ...values.map(Math.abs));
    const spacing = options.spacing || 0.34;
    const cellSize = options.cellSize || 0.26;
    const channelGap = options.channelGap ?? 0.2;
    const depthStep = cellSize + channelGap;
    const maxThickness = options.maxThickness || 0.4;
    // Zero values must have exactly zero thickness and therefore be invisible.
  
    const mesh = new THREE.InstancedMesh(
      cubeGeometry,
      new THREE.MeshStandardMaterial({
        color,
        roughness: 0.6,
        transparent: true,
        opacity: 0.82,
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
          // Keep empty tensor cells visible as tiny dot-like cubes instead of
          // full-size blades. Nonzero values retain their signed blade shape.
          const dotSize = Math.min(cellSize, 0.055);
          const thickness = isZero
            ? dotSize
            : Math.max(0.025, (Math.abs(value) / maxAbs) * maxThickness);
  
          // Signed values extend away from the zero plane along the X axis.
          // Each channel is a separate blade. Stack blades along X, perpendicular
          // to the 24×24 board plane, rather than along the column axis (Z).
          // Zero values sit on the channel plane as small cube dots.
          transform.position.set(
            origin[0] + channel * depthStep +
              (isZero ? 0 : value < 0 ? -thickness / 2 : thickness / 2),
            origin[1] + (rows - 1 - row) * spacing,
            origin[2] + col * spacing,
          );
          transform.scale.set(
            thickness,
            isZero ? dotSize : cellSize,
            isZero ? dotSize : cellSize,
          );
          transform.updateMatrix();
  
          mesh.setMatrixAt(index, transform.matrix);
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
  
  
  function drawFilterKernel(
    filterIndex,
    originX,
    originY,
    originZ,
    weights,
    kernelHeight,
    kernelWidth,
    channels,
    label,
    showBias = false,
  ) {
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
  
    const transform = new THREE.Object3D();
    let index = 0;
  
    for (let channel = 0; channel < channels; channel += 1) {
      for (let row = 0; row < kernelHeight; row += 1) {
        for (let col = 0; col < kernelWidth; col += 1) {
          const value = values[index];
          const thickness = Math.max(
            0.025,
            (Math.abs(value) / maxAbs) * 0.17,
          );
  
          transform.position.set(
            originX + (value < 0 ? -thickness / 2 : thickness / 2),
            originY + (kernelHeight - 1 - row) * spacing,
            originZ + col * spacing + channel * channelDepth,
          );
          transform.scale.set(thickness, cellSize, cellSize);
          transform.updateMatrix();
  
          mesh.setMatrixAt(index, transform.matrix);
          index += 1;
        }
      }
    }
  
    mesh.instanceMatrix.needsUpdate = true;
    getSceneGroup().add(mesh);
  
    if (showBias) {
      const biasCube = new THREE.Mesh(
        cubeGeometry,
        new THREE.MeshStandardMaterial({ color: COLORS.bias }),
      );
  
      biasCube.position.set(
        originX + 0.42,
        originY + ((kernelHeight - 1) * spacing) / 2,
        originZ + kernelWidth * spacing + channels * channelDepth * 0.5,
      );
      biasCube.scale.setScalar(0.22);
      getSceneGroup().add(biasCube);
    }
  
    addLabel(
      label,
      [
        originX,
        originY + kernelHeight * spacing + 0.2,
        originZ + channels * channelDepth * 0.3,
      ],
      17,
    );
  }
  
  
  function drawInputs() {
    const game = createSampleGame();
    drawSampleBoard();
    drawEffectiveBoard();
    if (!getBackendSnapshot()) {
      logLinkEncodingDiagnostic(game, getData().links);
      logFullInputTensorValidation(game, getData());
    }
  
    const tensors = [
      {
        key: 'pegs',
        label: 'Pegs · 24×24×2',
        x: -1.8,
        z: -13,
        channels: 2,
        bladeLabels: ['Red', 'Black'],
        color: COLORS.input,
      },
      {
        key: 'links',
        label: 'Links · 24×24×8',
        x: -1.8,
        z: -3.5,
        channels: 8,
        // Channel bits follow the Python Game/NetInputs convention:
        // bit 0 is colour (even=Black, odd=White), bit 2 is long-y,
        // and bit 1 is different-sign slope. These describe the tensor channels.
        bladeLabels: [
          'Ch 0: red SSE/NNW',
          'Ch 1: black SSE/NNW',
          'Ch 2: red ENE/WSW',
          'Ch 3: black ENE/WSW',
          'Ch 4: red ESE',
          'Ch 5: black ESE',
          'Ch 6: red ENE/WSW',
          'Ch 7: black ENE/WSW',
        ],
        color: COLORS.input,
      },
      {
        key: 'locations',
        label: 'Locations · 24×24×2',
        x: -1.8,
        z: 6,
        channels: 2,
        bladeLabels: ['X', 'Y'],
        color: COLORS.fixed,
      },
    ];
  
    tensors.forEach((tensor) => {
      addValueCubes(
        flattenTensor(getData()[tensor.key]),
        BOARD_SIZE,
        BOARD_SIZE,
        tensor.channels,
        [
          tensor.x - ((tensor.channels - 1) * (0.26 + 0.28)) / 2,
          -4,
          tensor.z,
        ],
        tensor.color,
        {
          spacing: 0.34,
          cellSize: 0.26,
          maxThickness: 0.34,
          channelGap: 0.28,
        },
      );
  
      // Keep the tensor-level label and the per-blade labels.
      addLabel(
        tensor.label,
        [tensor.x, 7.0, tensor.z + ((BOARD_SIZE - 1) * 0.34) / 2],
        22,
        '#ffffff',
        70,
      );
  
      const firstBladeX =
        tensor.x - ((tensor.channels - 1) * (0.26 + 0.28)) / 2;
      const boardTopY = -4 + (BOARD_SIZE - 1) * 0.34;
  
      tensor.bladeLabels.forEach((bladeLabel, channel) => {
        addLabel(
          bladeLabel,
          [firstBladeX + channel * (0.26 + 0.28), 5.65, tensor.z - 0.08],
          20,
          '#e7edf6',
          115,
          'left',
        );
      });
  
      // Tensor coordinates are array indices: both axes are numbered 0–23.
      // Show the coordinate grid only for the links tensor, as requested.
      if (tensor.key === 'links') {
        for (let col = 0; col < BOARD_SIZE; col += 1) {
          addLabel(
            String(col),
            [firstBladeX, 4.6, tensor.z + col * 0.34],
            17,
            '#dce7f5',
            145,
            'center',
          );
        }
  
        for (let row = 0; row < BOARD_SIZE; row += 1) {
          addLabel(
            String(row),
            [firstBladeX - 0.12, boardTopY - row * 0.34, tensor.z - 0.48],
            17,
            '#dce7f5',
            145,
            'center',
          );
        }
      }
    });
  
    return {
      title: 'Input tensors',
      description:
        'The 13-move sample game is encoded into the peg and link tensors. Locations remain fixed coordinate inputs; convolution weights are reproducible demo values.',
      shape: '24 × 24 × (2 + 8 + 2)',
      range: 'Binary demo inputs',
      count: '3 tensors · 6,912 cells',
    };
  }
  
  function drawFilters() {
    for (let filter = 0; filter < FILTER_COUNT; filter += 1) {
      const column = filter % 8;
      const row = Math.floor(filter / 8);
      const x = -18.5 + column * 4.8;
      const y = -4 + row * 5.3;
  
      drawFilterKernel(
        filter,
        x,
        y,
        -1.5,
        getData().weights.links,
        5,
        5,
        8,
        'F' + filter + ' · link',
        true,
      );
  
      drawFilterKernel(
        filter,
        x + 1.65,
        y,
        2.7,
        getData().weights.pegs,
        1,
        1,
        2,
        'peg',
      );
  
      drawFilterKernel(
        filter,
        x + 2.75,
        y,
        2.7,
        getData().weights.locations,
        1,
        1,
        2,
        'loc',
      );
    }
  
    return {
      title: '48 convolution filters',
      description:
        'Eight columns × six rows. Each filter contains link, peg, and location kernels plus a bias cube.',
      shape: '48 filters · 8 × 6 grid',
      range: 'Signed weights and bias',
      count: '48 filter groups',
    };
  }
  
  function drawOutput() {
    const values = flattenTensor(calculateConvolutionOutput());
  
    const stats = addValueCubes(
      values,
      BOARD_SIZE,
      BOARD_SIZE,
      FILTER_COUNT,
      [-4.5, -4, -((FILTER_COUNT - 1) * 0.105) / 2],
      COLORS.output,
      {
        spacing: 0.34,
        cellSize: 0.24,
        maxThickness: 0.42,
        minThickness: 0.025,
        channelGap: 0.105,
      },
    );
  
    addLabel('Summed convolution output · 24×24×48', [0, 5, 2], 28);
  
    return {
      title: 'Convolution output',
      description:
        'Three branch convolutions are summed into one feature tensor. Thickness is scaled for visibility.',
      shape: '24 × 24 × 48',
      range: stats.min.toFixed(4) + ' … ' + stats.max.toFixed(4),
      count: stats.count.toLocaleString() + ' cells',
    };
  }
  

  return { addValueCubes, drawFilterKernel, drawInputs, drawFilters, drawOutput };
}
