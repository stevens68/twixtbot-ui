/**
 * Board click hit-testing and drag suppression for the interactive Three.js canvas.
 * The caller owns move validation and any higher-priority scene picking.
 */
export function createBoardInteraction({
  THREE,
  boardSize,
  camera,
  controls,
  canvas,
  isEnabled,
  onBoardMove,
  onSceneClick,
}) {
  const boardSpacing = 0.34;
  const boardWorldSize = (boardSize - 1) * boardSpacing;
  const boardX = -26.675 + 0.026;
  const boardStart = -boardWorldSize / 2;
  const dragThresholdPx = 5;
  let pointerDown = null;
  let pointerDragged = false;

  function projectIntersection(row, col, rect) {
    const point = new THREE.Vector3(
      boardX,
      boardStart + (boardSize - 1 - row) * boardSpacing,
      boardStart + col * boardSpacing,
    );
    point.project(camera);
    if (point.z < -1 || point.z > 1) return null;
    return {
      x: rect.left + (point.x + 1) * rect.width / 2,
      y: rect.top + (1 - point.y) * rect.height / 2,
    };
  }

  function handleBoardClick(event) {
    if (!isEnabled()) return;
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    controls.update();
    camera.updateMatrixWorld(true);

    let closest = null;
    for (let row = 0; row < boardSize; row += 1) {
      for (let col = 0; col < boardSize; col += 1) {
        const screen = projectIntersection(row, col, rect);
        if (!screen) continue;
        const dx = screen.x - event.clientX;
        const dy = screen.y - event.clientY;
        const distanceSquared = dx * dx + dy * dy;
        if (!closest || distanceSquared < closest.distanceSquared) {
          closest = { row, col, screen, distanceSquared };
        }
      }
    }
    if (!closest) return;

    // Scale hit tolerance with the projected grid spacing at the current camera angle.
    let nearestGridSpacing = Infinity;
    for (const [dr, dc] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
      const row = closest.row + dr;
      const col = closest.col + dc;
      if (row < 0 || row >= boardSize || col < 0 || col >= boardSize) continue;
      const neighbor = projectIntersection(row, col, rect);
      if (!neighbor) continue;
      nearestGridSpacing = Math.min(
        nearestGridSpacing,
        Math.hypot(neighbor.x - closest.screen.x, neighbor.y - closest.screen.y),
      );
    }
    const hitRadius = Number.isFinite(nearestGridSpacing)
      ? Math.max(3, Math.min(18, nearestGridSpacing * 0.45))
      : 8;
    if (closest.distanceSquared > hitRadius * hitRadius) return;

    onBoardMove(String.fromCharCode(65 + closest.col) + (closest.row + 1));
  }

  canvas.addEventListener('pointerdown', (event) => {
    if (event.button !== 0) return;
    pointerDown = { x: event.clientX, y: event.clientY };
    pointerDragged = false;
  });

  canvas.addEventListener('pointermove', (event) => {
    if (!pointerDown || pointerDragged) return;
    const dx = event.clientX - pointerDown.x;
    const dy = event.clientY - pointerDown.y;
    if (dx * dx + dy * dy > dragThresholdPx * dragThresholdPx) pointerDragged = true;
  });

  canvas.addEventListener('pointerup', () => {
    pointerDown = null;
  });

  canvas.addEventListener('pointercancel', () => {
    pointerDown = null;
    pointerDragged = false;
  });

  canvas.addEventListener('click', (event) => {
    if (pointerDragged) {
      pointerDragged = false;
      return;
    }
    if (onSceneClick(event)) return;
    handleBoardClick(event);
  });
}
