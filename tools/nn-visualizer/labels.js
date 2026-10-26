// Canvas-based labels used by the Three.js scene.
export function createLabelHelpers(THREE, getSceneGroup) {

function addLabel(
  text,
  position,
  fontSize = 30,
  color = '#e7edf6',
  scaleDivisor = 45,
  alignment = 'center',
) {
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');

  context.font = '600 ' + fontSize + 'px Arial';
  canvas.width = Math.max(128, context.measureText(text).width + 28);
  canvas.height = fontSize + 26;

  context.font = '600 ' + fontSize + 'px Arial';
  context.fillStyle = color;
  context.textAlign = alignment;
  context.textBaseline = 'middle';
  context.fillText(
    text,
    alignment === 'left' ? 12 : canvas.width / 2,
    canvas.height / 2,
  );

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;

  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      depthTest: false,
    }),
  );

  const spriteWidth = canvas.width / scaleDivisor;
  // Anchor left-aligned sprites at their left edge. Sprite.center controls
  // the texture anchor; shifting the position alone can still look centered.
  if (alignment === 'left') sprite.center.set(0, 0.5);
  sprite.position.set(position[0], position[1], position[2]);
  sprite.scale.set(spriteWidth, canvas.height / scaleDivisor, 1);
  getSceneGroup().add(sprite);
}

function addRichLabel(parts, position, fontSize = 30, scaleDivisor = 45, alignment = 'left') {
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  context.font = '600 ' + fontSize + 'px Arial';
  const widths = parts.map((part) => context.measureText(part.text).width);
  canvas.width = Math.max(128, widths.reduce((sum, width) => sum + width, 0) + 28);
  canvas.height = fontSize + 26;
  context.font = '600 ' + fontSize + 'px Arial';
  context.textAlign = 'left';
  context.textBaseline = 'middle';
  let x = alignment === 'center' ? (canvas.width - widths.reduce((sum, width) => sum + width, 0)) / 2 : 12;
  parts.forEach((part, index) => {
    context.fillStyle = part.color;
    context.fillText(part.text, x, canvas.height / 2);
    x += widths[index];
  });
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map: texture, transparent: true, depthTest: false,
  }));
  sprite.center.set(alignment === 'center' ? 0.5 : 0, 0.5);
  sprite.position.set(position[0], position[1], position[2]);
  sprite.scale.set(canvas.width / scaleDivisor, canvas.height / scaleDivisor, 1);
  getSceneGroup().add(sprite);
}


  return { addLabel, addRichLabel };
}
