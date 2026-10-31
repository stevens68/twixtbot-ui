// Canvas-based labels used by the Three.js scene.
//
// Sprites and their canvas textures are pooled across renders. A game move
// changes many positions, but most labels are identical; reusing them avoids
// allocating hundreds of canvases, textures and sprite objects on every ply.
export function createLabelHelpers(THREE, getSceneGroup) {
  const spritePool = [];
  const textureCache = new Map();
  let spriteCursor = 0;
  const TEXTURE_CACHE_LIMIT = 4096;

  function beginFrame() {
    spriteCursor = 0;
  }

  function getTexture(key, draw) {
    if (textureCache.has(key)) {
      const texture = textureCache.get(key);
      // LRU refresh.
      textureCache.delete(key);
      textureCache.set(key, texture);
      return texture;
    }

    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    const dimensions = draw(context, canvas);
    canvas.width = dimensions.width;
    canvas.height = dimensions.height;
    const resizedContext = canvas.getContext('2d');
    draw(resizedContext, canvas);
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    textureCache.set(key, texture);

    while (textureCache.size > TEXTURE_CACHE_LIMIT) {
      const oldestKey = textureCache.keys().next().value;
      textureCache.delete(oldestKey);
      // Do not dispose the texture here: a pooled sprite may still reference it.
    }
    return texture;
  }

  function acquireSprite(texture, position, scale, alignment) {
    let sprite = spritePool[spriteCursor];
    if (!sprite) {
      sprite = new THREE.Sprite(new THREE.SpriteMaterial({
        map: texture,
        transparent: true,
        depthTest: false,
      }));
      sprite.userData.pooledLabel = true;
      spritePool.push(sprite);
    } else if (sprite.material.map !== texture) {
      sprite.material.map = texture;
      sprite.material.needsUpdate = true;
    }

    spriteCursor += 1;
    sprite.center.set(alignment === 'left' ? 0 : 0.5, 0.5);
    sprite.position.set(position[0], position[1], position[2]);
    sprite.scale.set(scale[0], scale[1], 1);
    getSceneGroup().add(sprite);
  }

  function addLabel(
    text,
    position,
    fontSize = 30,
    color = '#e7edf6',
    scaleDivisor = 45,
    alignment = 'center',
  ) {
    const key = JSON.stringify(['label', text, fontSize, color, scaleDivisor, alignment]);
    const texture = getTexture(key, (context) => {
      context.font = '600 ' + fontSize + 'px Arial';
      const width = Math.max(128, context.measureText(text).width + 28);
      const height = fontSize + 26;
      context.font = '600 ' + fontSize + 'px Arial';
      context.fillStyle = color;
      context.textAlign = alignment;
      context.textBaseline = 'middle';
      context.fillText(text, alignment === 'left' ? 12 : width / 2, height / 2);
      return { width, height };
    });

    acquireSprite(
      texture,
      position,
      [texture.image.width / scaleDivisor, texture.image.height / scaleDivisor],
      alignment,
    );
  }

  function addFormulaLabel(
    text,
    position,
    fontSize = 26,
    color = '#f2f4f8',
    scaleDivisor = 90,
  ) {
    const font = 'italic 500 ' + fontSize + 'px "Cambria Math", "STIX Two Math", "Times New Roman", serif';
    const key = JSON.stringify(['formula', text, fontSize, color, scaleDivisor]);
    const texture = getTexture(key, (context) => {
      context.font = font;
      const width = Math.max(160, context.measureText(text).width + 32);
      const height = fontSize + 30;
      context.font = font;
      context.fillStyle = color;
      context.textAlign = 'center';
      context.textBaseline = 'middle';
      context.fillText(text, width / 2, height / 2);
      return { width, height };
    });
    acquireSprite(
      texture,
      position,
      [texture.image.width / scaleDivisor, texture.image.height / scaleDivisor],
      'center',
    );
  }

  // Render a whole move-list column as one canvas texture and one sprite.
  // This preserves the 20-row column layout while avoiding one draw call per move.
  function addMoveColumnLabel(moves, position, fontSize = 14, scaleDivisor = (fontSize + 8) / 0.38) {
    const lineHeight = fontSize + 8;
    const key = JSON.stringify(['move-column', moves, fontSize, scaleDivisor]);
    const texture = getTexture(key, (context) => {
      context.font = '600 ' + fontSize + 'px Arial';
      const textWidth = Math.max(128, ...moves.map((move) => context.measureText(move).width + 24));
      const width = Math.ceil(textWidth);
      const height = Math.max(lineHeight + 12, moves.length * lineHeight + 12);
      context.font = '600 ' + fontSize + 'px Arial';
      context.textAlign = 'left';
      context.textBaseline = 'middle';
      moves.forEach((move, index) => {
        // Columns start at plies 0, 20, 40... so each column starts on Red's move.
        context.fillStyle = index % 2 === 0 ? '#ff4545' : '#e7edf6';
        context.fillText(move, 12, 6 + (index + 0.5) * lineHeight);
      });
      return { width, height };
    });

    acquireSprite(
      texture,
      position,
      [texture.image.width / scaleDivisor, texture.image.height / scaleDivisor],
      'left',
    );
  }

  function addRichLabel(parts, position, fontSize = 30, scaleDivisor = 45, alignment = 'left') {
    const key = JSON.stringify(['rich', parts, fontSize, scaleDivisor, alignment]);
    const texture = getTexture(key, (context) => {
      context.font = '600 ' + fontSize + 'px Arial';
      const widths = parts.map((part) => context.measureText(part.text).width);
      const totalWidth = widths.reduce((sum, width) => sum + width, 0);
      const width = Math.max(128, totalWidth + 28);
      const height = fontSize + 26;
      context.font = '600 ' + fontSize + 'px Arial';
      context.textAlign = 'left';
      context.textBaseline = 'middle';
      let x = alignment === 'center' ? (width - totalWidth) / 2 : 12;
      parts.forEach((part, index) => {
        context.fillStyle = part.color;
        context.fillText(part.text, x, height / 2);
        x += widths[index];
      });
      return { width, height };
    });

    acquireSprite(
      texture,
      position,
      [texture.image.width / scaleDivisor, texture.image.height / scaleDivisor],
      alignment,
    );
  }

  return { addLabel, addRichLabel, addFormulaLabel, addMoveColumnLabel, beginFrame };
}
