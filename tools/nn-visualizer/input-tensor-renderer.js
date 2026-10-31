/** Incremental rendering of the persistent input tensor layers. */
export function createInputTensorRenderer({
  THREE, BOARD_SIZE, COLORS, cubeGeometry, getData,
  getInputStaticGroup, getInputDynamicGroup, inputTensorSpecs,
  inputCellSize, inputSpacing, minimumCubeScale, locationHeatMapColor,
}) {
  const inputStaticGroup = getInputStaticGroup();
  const inputDynamicGroup = getInputDynamicGroup();
  const inputChannelGap = 0.28;
  const inputDepthStep = inputCellSize + inputChannelGap;
  const inputOriginY = -4;
  let inputGeometryBuilt = false;
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
    transform.scale.set(thickness, cellSize, cellSize);
    transform.updateMatrix();
  }

  function makeInputMaterial(color) {
    return new THREE.MeshStandardMaterial({
      color, roughness: 0.6, transparent: true, opacity: 0.82,
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
      cubeGeometry,
      new THREE.MeshStandardMaterial({
        color: 0xffffff, roughness: 0.6, transparent: true, opacity: 0.9,
      }),
      capacity,
    );
    mesh.name = 'locations-persistent-values';
    const origin = inputOrigin(spec);
    const transform = new THREE.Object3D();
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

  return { ensureInputGeometry, updateOccupiedTensor };
}
