/** Rendering and interaction for the residual-block stack. */
export function createResidualRenderer({
  THREE, BOARD_SIZE, FILTER_COUNT, cubeGeometry, filterSpacing,
  filterCubeSize, filterStaticGroup, flattenTensor, addLabel, addFormulaLabel, getData,
  heatMapColor,
}) {
  const residualFilterGroups = new Map();
  const residualWeightsByIndex = new Map();
  const residualOutputGroups = new Map();
  const residualOutputMeshes = new Map();
  let residualLabelsBuilt = false;
  // These labels are attached to the persistent annotation group, so retain
  // their sprites and explicitly hide them when blocks 2–19 are collapsed.
  let intermediateResidualLabelSprites = [];
  let residualPlaceholderGroup = null;
  let residualPlaceholderDirty = false;
  let residualHitTargets = [];
  let selectedResidualBlockIndex = -1;
  let selectedResidualWeights = null;
  let residualGeometryDirty = false;
  let residualFiltersVisible = true;
  let showIntermediateResidualBlocks = false;
  let lastRenderedOutputTensor = null;
  let lastRenderedLastOutputTensor = null;
  function disposeResidualDetail(blockIndex) {
    const group = residualFilterGroups.get(blockIndex);
    if (!group) return;
    filterStaticGroup.remove(group);
    const geometries = new Set();
    const materials = new Set();
    group.traverse((object) => {
      if (object.geometry && object.geometry !== cubeGeometry) geometries.add(object.geometry);
      if (object.material) (Array.isArray(object.material) ? object.material : [object.material])
        .forEach((material) => material && materials.add(material));
    });
    geometries.forEach((geometry) => geometry.dispose());
    materials.forEach((material) => material.dispose());
    residualFilterGroups.delete(blockIndex);
  }


function residualLayout(blockIndex, forceExpanded = false) {
 const actualLastIndex=Number(getData().residualBlockCount || 20)-1;
 // Camera framing sometimes needs the full-stack coordinates even while the
 // intermediate blocks are hidden and the last block is rendered compactly.
 const layoutIndex=(!showIntermediateResidualBlocks && !forceExpanded && blockIndex===actualLastIndex)?1:blockIndex;
 const channels=FILTER_COUNT,kernelHeight=5,kernelWidth=5,columns=8,rows=Math.ceil(FILTER_COUNT/columns);
 const filterWidth=kernelWidth*filterSpacing,filterHeight=kernelHeight*filterSpacing,gapY=.42,gapZ=.42;
 const bankSizeX=(channels-1)*filterSpacing+filterCubeSize;
 const bankMinYOffset=-(rows-1)*(filterHeight+gapY)-filterCubeSize/2,bankMaxYOffset=(kernelHeight-1)*filterSpacing+filterCubeSize/2;
 const bankSizeY=bankMaxYOffset-bankMinYOffset,bankSizeZ=(columns-1)*(filterWidth+gapZ)+(kernelWidth-1)*filterSpacing+filterCubeSize;
 const outputChannelStep=filterCubeSize/.9+filterSpacing-filterCubeSize,outputCubeSize=filterCubeSize/.9;
 const outputSizeX=(channels-1)*outputChannelStep+outputCubeSize,outputSizeY=(BOARD_SIZE-1)*filterSpacing+outputCubeSize;
 const outputCenterY=-2.5+(BOARD_SIZE-1)*filterSpacing/2,outputCenterZ=-((channels-1)*.105)/2+(BOARD_SIZE-1)*filterSpacing/2;
 const bankStartY=outputCenterY-(bankMinYOffset+bankMaxYOffset)/2,bankStartZ=outputCenterZ-bankSizeZ/2+filterCubeSize/2;
 const firstX=20+(channels-1)*outputChannelStep+outputCubeSize/2+5+filterCubeSize/2,firstConvGap=9;
 const firstPitch=(2*bankSizeX+firstConvGap+1.25*1.15*3+outputCubeSize/2+outputSizeX+1.25+filterCubeSize/2)*1.2;
 const nextLeft=firstX+firstPitch-filterCubeSize/2,right=firstX-filterCubeSize/2+2*bankSizeX+firstConvGap;
 const outputCenterX=(right+nextLeft)/2,outputStartX=outputCenterX-(channels-1)*outputChannelStep/2;
 const compactConvGap=2.5;
 // Match the clear space after block 1 (which contains its output tensor),
 // while retaining the compact conv1-to-conv2 gap inside every later block.
 const firstInterBlockGap=firstPitch-(2*bankSizeX+firstConvGap);
 const compactPitch=2*bankSizeX+compactConvGap+firstInterBlockGap;
 const blockStartX=layoutIndex===0?firstX:nextLeft+filterCubeSize/2+(layoutIndex-1)*compactPitch;
 const isEndpoint=blockIndex===0 || blockIndex===actualLastIndex;
 const convGap=isEndpoint?firstConvGap:compactConvGap;
 const shift=blockStartX-firstX;
 return {channels,kernelHeight,kernelWidth,columns,rows,filterWidth,filterHeight,gapY,gapZ,bankSizeX,bankSizeY,bankSizeZ,bankStartY,bankStartZ,
 bankStartXs:[blockStartX,blockStartX+bankSizeX+convGap],bankStartZs:[bankStartZ,bankStartZ],blockStartX,blockSpanX:2*bankSizeX+convGap,
 outputStartX:outputStartX+shift,outputCenterX:outputCenterX+shift,outputChannelStep,outputSizeX,outputSizeY,outputSizeZ:outputSizeY,outputCenterY,outputCenterZ,outputCubeSize};
}

function drawSelectedResidualBlock(residualWeights, buildGeometry, blockIndex) {
  if (!residualWeights?.conv1 || !residualWeights?.conv2) {
    throw new Error(
      'Residual block ' + blockIndex + ' convolution weights are unavailable.'
    );
  }

  const layout = residualLayout(blockIndex);
  if (buildGeometry) {
    const residualFilterGroup = new THREE.Group();
    residualFilterGroup.name = 'residual-block-' + blockIndex + '-filter-banks';
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
    residualFilterGroups.set(blockIndex, residualFilterGroup);
  }

}

function addResidualOutputValues(group,layout,index,tensor,mesh=null){
 const count=FILTER_COUNT*BOARD_SIZE*BOARD_SIZE;
 if(!mesh){mesh=new THREE.InstancedMesh(cubeGeometry,new THREE.MeshStandardMaterial({color:0xffffff,roughness:.55,transparent:true,opacity:.96}),count);mesh.name='residual-block-'+(index+1)+'-actual-output-values';group.add(mesh);}
 let max=1e-8;
 for(let c=0;c<FILTER_COUNT;c++)for(let r=0;r<BOARD_SIZE;r++)for(let col=0;col<BOARD_SIZE;col++)max=Math.max(max,Math.abs(Number(tensor[r][col][c])));
 const o=new THREE.Object3D();let i=0;
 for(let c=0;c<FILTER_COUNT;c++)for(let r=0;r<BOARD_SIZE;r++)for(let col=0;col<BOARD_SIZE;col++){
 const v=Number(tensor[r][col][c]);o.position.set(layout.outputStartX+c*layout.outputChannelStep,layout.outputCenterY+(BOARD_SIZE-1-r)*filterSpacing-(BOARD_SIZE-1)*filterSpacing/2,layout.outputCenterZ+col*filterSpacing-(BOARD_SIZE-1)*filterSpacing/2);
 o.scale.setScalar(layout.outputCubeSize*.72);o.updateMatrix();mesh.setMatrixAt(i,o.matrix);mesh.setColorAt(i,heatMapColor(v,0,max));i++;}
 mesh.count=i;mesh.instanceMatrix.needsUpdate=true;if(mesh.instanceColor)mesh.instanceColor.needsUpdate=true;return mesh;
}
function addResidualArrow(group,x1,x2,y,z,name,label=null){const gap=x2-x1;if(gap<=.15)return;const arrow=new THREE.ArrowHelper(new THREE.Vector3(1,0,0),new THREE.Vector3(x1+gap*.22,y,z),gap*.67,0x71879b,.42,.22);arrow.name=name;group.add(arrow);if(label)addFormulaLabel(label,[(x1+x2)/2,y+1.15,z],25,'#f2f4f8',92);}
function addResidualCurvedArrow(group,x1,x2,y1,z1,y2,z2,name,label=null){const dx=x2-x1;if(dx<=.2)return;const c=new THREE.CubicBezierCurve3(new THREE.Vector3(x1,y1,z1),new THREE.Vector3(x1+dx*.42,y1,z1),new THREE.Vector3(x2-dx*.42,y2,z2),new THREE.Vector3(x2,y2,z2));const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints(c.getPoints(64)),new THREE.LineBasicMaterial({color:0x71879b}));line.name=name+'-shaft';group.add(line);if(label){const p=c.getPoint(.5);addFormulaLabel(label,[p.x,p.y+1.15,p.z],25,'#f2f4f8',92);}}
function addResidualBypass(group,x1,x2,z,offset,name,label=null){const dx=x2-x1;if(dx<=1)return null;const t=Math.min(dx*.22,3),oz=z+offset,p=new THREE.CurvePath();p.add(new THREE.CubicBezierCurve3(new THREE.Vector3(x1,0,z),new THREE.Vector3(x1+t*.45,0,z),new THREE.Vector3(x1+t*.55,0,oz),new THREE.Vector3(x1+t,0,oz)));p.add(new THREE.LineCurve3(new THREE.Vector3(x1+t,0,oz),new THREE.Vector3(x2-t,0,oz)));p.add(new THREE.CubicBezierCurve3(new THREE.Vector3(x2-t,0,oz),new THREE.Vector3(x2-t*.55,0,oz),new THREE.Vector3(x2-t*.45,0,z),new THREE.Vector3(x2,0,z)));const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints(p.getPoints(96)),new THREE.LineBasicMaterial({color:0x71879b}));line.name=name+'-shaft';group.add(line);const head=new THREE.ArrowHelper(new THREE.Vector3(1,0,0),new THREE.Vector3(x2-.48,0,z),.48,0x71879b,.34,.18);head.name=name+'-arrowhead';group.add(head);if(label)return addFormulaLabel(label,[(x1+x2)/2,1.15,z+offset],25,'#f2f4f8',92);return null;}
  function createResidualBlockPlaceholders(blockCount) {
    residualPlaceholderGroup = new THREE.Group();
    residualPlaceholderGroup.name = 'residual-block-placeholders';
    residualHitTargets = [];

    for (let index = 0; index < blockCount; index += 1) {
      if (!showIntermediateResidualBlocks && index > 0 && index < blockCount - 1) continue;
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
      const br1=layout.bankStartXs[0]+(layout.channels-1)*filterSpacing+filterCubeSize/2,bl2=layout.bankStartXs[1]-filterCubeSize/2;
      const br2=layout.bankStartXs[1]+(layout.channels-1)*filterSpacing+filterCubeSize/2,cy=layout.outputCenterY,cz=layout.outputCenterZ;
      if(index===0 || index===blockCount-1){
       const prefix='block-'+(index+1);
       let inputRight=20+(FILTER_COUNT-1)*layout.outputChannelStep+layout.outputCubeSize/2;
       if(index===blockCount-1 && index>0){
        const previous=residualLayout((!showIntermediateResidualBlocks && index===blockCount-1)?0:index-1);
        const previousRight=previous.bankStartXs[1]+(previous.channels-1)*filterSpacing+filterCubeSize/2;
        const currentLeft=layout.bankStartXs[0]-filterCubeSize/2;
        const previousGapCenter=(previousRight+currentLeft)/2;
        inputRight=previousGapCenter+layout.outputSizeX/2;
       }
       addResidualArrow(residualPlaceholderGroup,inputRight,layout.bankStartXs[0]-filterCubeSize/2,cy,cz,prefix+'-input-to-block');
       addResidualArrow(residualPlaceholderGroup,br1,bl2,cy,cz,prefix+'-conv1-to-conv2','y₁ = |BN(Conv₁(x))|');
       const outLeft=layout.outputCenterX-layout.outputSizeX/2,merge=outLeft-1.35;
       addResidualCurvedArrow(residualPlaceholderGroup,br2,merge,0,cz,0,cz,prefix+'-conv2-to-merge','y₂ = |BN(Conv₂(y₁))|');
       addResidualBypass(residualPlaceholderGroup,inputRight,merge,cz,layout.bankSizeZ*1.08/2+1,prefix+'-identity-skip','y₃ = x');
       addResidualArrow(residualPlaceholderGroup,merge,outLeft,0,cz,prefix+'-merged-output','Σ');
       if(index===0){
        const outRight=layout.outputCenterX+layout.outputSizeX/2,nextIndex=showIntermediateResidualBlocks?1:blockCount-1,nextLeft=residualLayout(nextIndex).bankStartXs[0]-filterCubeSize/2;
        addResidualArrow(residualPlaceholderGroup,outRight,nextLeft,cy,cz,'block-1-output-to-block-2');
       }
      }else if(index<blockCount-1){
       const mergeX=br2+1.1;
       const intermediateSkipLabel = addResidualBypass(
         residualPlaceholderGroup,
         layout.bankStartXs[0] - filterCubeSize / 2,
         mergeX,
         cz,
         layout.bankSizeZ * 1.08 / 2 + 1,
         'block-' + (index + 1) + '-identity-skip',
         'y₃ = x',
       );
       if (intermediateSkipLabel) intermediateResidualLabelSprites.push(intermediateSkipLabel);
       const next=residualLayout(index+1),nextLeft=next.bankStartXs[0]-filterCubeSize/2;
       const gapCenterX=(br2+nextLeft)/2;
       const outputWidth=layout.outputSizeX;
       const glassLeft=gapCenterX-outputWidth/2;
       const glassRight=gapCenterX+outputWidth/2;

       // A borderless green glass volume represents the 24x24x48 tensor
       // produced by this residual block.
       const outputGlass=new THREE.Mesh(
         new THREE.BoxGeometry(outputWidth,layout.outputSizeY,layout.outputSizeZ),
         new THREE.MeshPhysicalMaterial({
           color:0x00ff00,
           transparent:true,
           opacity:0.16,
           transmission:0.72,
           roughness:0.08,
           metalness:0,
           depthWrite:false,
           side:THREE.DoubleSide,
         }),
       );
       outputGlass.position.set(gapCenterX,cy,cz);
       outputGlass.name='residual-block-'+(index+1)+'-output-tensor-glass';
       residualPlaceholderGroup.add(outputGlass);

       addResidualArrow(
         residualPlaceholderGroup,br2,glassLeft,cy,cz,
         'block-'+(index+1)+'-to-output-tensor',
       );
       addResidualArrow(
         residualPlaceholderGroup,glassRight,nextLeft,cy,cz,
         'output-tensor-to-block-'+(index+2),
       );
      }
    }
    filterStaticGroup.add(residualPlaceholderGroup);
  }

  function drawResidualPlaceholderLabels(blockCount) {
    for (let index = 0; index < blockCount; index += 1) {
      if (!showIntermediateResidualBlocks && index > 0 && index < blockCount - 1) continue;
      const layout = residualLayout(index);
      const cx = layout.blockStartX + layout.blockSpanX / 2;
      const y = layout.bankStartY + layout.bankSizeY / 2 + 2;
      const z = layout.bankStartZ + layout.bankSizeZ / 2;
      const sprite = addLabel('Residual block ' + (index + 1), [cx, y, z], 24, '#ffffff', 70);
      if (index > 0 && index < blockCount - 1 && sprite) {
        intermediateResidualLabelSprites.push(sprite);
      }
    }
  }

  function setShowIntermediateResidualBlocks(visible) {
    const next=Boolean(visible);
    if(showIntermediateResidualBlocks===next)return;
    showIntermediateResidualBlocks=next;
    // Annotation sprites are in a persistent scene group. Hide the previous
    // intermediate labels before rebuilding the set for the new layout.
    for (const sprite of intermediateResidualLabelSprites) sprite.visible = false;
    intermediateResidualLabelSprites = [];
    residualLabelsBuilt = false;
    residualPlaceholderDirty=true;
    residualGeometryDirty=true;
    // Output tensors move with their blocks; dispose and recreate them at the new positions.
    for(const index of [...residualOutputGroups.keys()]) updateResidualOutput(index,null);
    lastRenderedOutputTensor=null;
    lastRenderedLastOutputTensor=null;
  }

  function getBlock20CameraOffset() {
    const lastIndex=Number(getData().residualBlockCount || 20)-1;
    // The preset coordinates describe the expanded stack. When blocks 2–19
    // are hidden, shift the camera by the distance the final block moved left.
    if (showIntermediateResidualBlocks) return 0;
    return residualLayout(lastIndex, true).blockStartX-residualLayout(1, true).blockStartX;
  }

  function setSelectedResidualBlock(blockIndex, weights) {
    selectedResidualBlockIndex = blockIndex;
    selectedResidualWeights = weights;
    residualWeightsByIndex.set(blockIndex, weights);
    residualGeometryDirty = true;
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
    for (const group of residualFilterGroups.values()) group.visible = residualFiltersVisible;
  }

  function updateResidualOutput(index, tensor) {
    let group = residualOutputGroups.get(index);
    if (!tensor) {
      if (group) {
        filterStaticGroup.remove(group);
        group.traverse((object) => {
          if (object.geometry && object.geometry !== cubeGeometry) object.geometry.dispose();
          if (object.material) (Array.isArray(object.material) ? object.material : [object.material])
            .forEach((material) => material?.dispose());
        });
        residualOutputGroups.delete(index);
        residualOutputMeshes.delete(index);
      }
      return;
    }
    if (!group) {
      group = new THREE.Group();
      group.name = 'residual-block-' + (index + 1) + '-output-values';
      residualOutputGroups.set(index, group);
      filterStaticGroup.add(group);
    }
    residualOutputMeshes.set(index, addResidualOutputValues(
      group, residualLayout(index), index, tensor, residualOutputMeshes.get(index),
    ));
  }

  function reset() {
    for (const index of [...residualFilterGroups.keys()]) disposeResidualDetail(index);
    for (const index of [...residualOutputGroups.keys()]) updateResidualOutput(index, null);
    residualPlaceholderGroup = null;
    residualPlaceholderDirty = false;
    for (const sprite of intermediateResidualLabelSprites) sprite.visible = false;
    intermediateResidualLabelSprites = [];
    residualLabelsBuilt = false;
    lastRenderedOutputTensor = null;
    lastRenderedLastOutputTensor = null;
    residualHitTargets = [];
    // clearFilterGeometry() calls reset() before rebuilding the filter scene.
    // Keep the trained payloads: draw() needs them to recreate both endpoint
    // banks (and the currently selected block) after that geometry rebuild.
    residualGeometryDirty = true;
  }

  function draw(rebuild, blockCount) {
    const data = getData();
    const outputTensor = data.residualBlock0Output || null;
    const lastOutputTensor = data.residualBlockLastOutput || null;
    if (outputTensor !== lastRenderedOutputTensor) {
      lastRenderedOutputTensor = outputTensor;
      updateResidualOutput(0, outputTensor);
    }
    if (lastOutputTensor !== lastRenderedLastOutputTensor) {
      lastRenderedLastOutputTensor = lastOutputTensor;
      updateResidualOutput(blockCount - 1, lastOutputTensor);
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
      residualPlaceholderDirty = false;
    }
    // Annotation labels live in a persistent scene group. Build them once,
    // then explicitly replace/hide the intermediate set when the toggle changes.
    if (!residualLabelsBuilt) {
      drawResidualPlaceholderLabels(blockCount);
      residualLabelsBuilt = true;
    }
    const visibleBlockIndices = showIntermediateResidualBlocks
      ? [0, blockCount - 1, selectedResidualBlockIndex]
      : [0, blockCount - 1];
    for (const index of new Set(visibleBlockIndices)) {
      const weights = residualWeightsByIndex.get(index);
      if (index < 0 || !weights) continue;
      if (rebuild || residualGeometryDirty || !residualFilterGroups.has(index)) {
        disposeResidualDetail(index);
        drawSelectedResidualBlock(weights, true, index);
      }
    }
    residualGeometryDirty = false;
    return selectedResidualBlockIndex;
  }

  return {
    draw,
    reset,
    setResidualFiltersVisible,
    setShowIntermediateResidualBlocks,
    getBlock20CameraOffset,
    setSelectedResidualBlock,
    pickResidualBlock,
  };
}
