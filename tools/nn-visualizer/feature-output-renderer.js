/** Generic tensor-value meshes and the first convolution feature output. */
export function createFeatureOutputRenderer({
  THREE, BOARD_SIZE, FILTER_COUNT, COLORS, cubeGeometry, flattenTensor,
  getSceneGroup, calculateConvolutionOutput, addLabel, filterSpacing,
  filterCubeSize, heatMapColor,
}) {
  function addValueCubes(values, rows, cols, channels, origin, color, options = {}) {
    const maxAbs = Math.max(1e-8, ...values.map(Math.abs));
    const spacing = options.spacing || 0.34;
    const cellSize = options.cellSize || 0.26;
    const channelGap = options.channelGap ?? 0.2;
    const depthStep = cellSize + channelGap;
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

  return { addValueCubes, drawOutput };
}
