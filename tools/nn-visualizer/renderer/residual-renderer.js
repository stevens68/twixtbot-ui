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
  let lastRenderedOutputTensor = null;
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
 const compactConvGap=2.5,interBlockGap=8,compactPitch=2*bankSizeX+compactConvGap+interBlockGap;
 const blockStartX=blockIndex===0?firstX:nextLeft+filterCubeSize/2+(blockIndex-1)*compactPitch,convGap=blockIndex===0?firstConvGap:compactConvGap;
 return {channels,kernelHeight,kernelWidth,columns,rows,filterWidth,filterHeight,gapY,gapZ,bankSizeX,bankSizeY,bankSizeZ,bankStartY,bankStartZ,
 bankStartXs:[blockStartX,blockStartX+bankSizeX+convGap],bankStartZs:[bankStartZ,bankStartZ],blockStartX,blockSpanX:2*bankSizeX+convGap,
 outputStartX,outputCenterX,outputChannelStep,outputSizeX,outputSizeY,outputSizeZ:outputSizeY,outputCenterY,outputCenterZ,outputCubeSize};
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


}

function addResidualOutputValues(group,layout,index,tensor){
 const count=FILTER_COUNT*BOARD_SIZE*BOARD_SIZE,mesh=new THREE.InstancedMesh(cubeGeometry,new THREE.MeshStandardMaterial({color:0xffffff,roughness:.55,transparent:true,opacity:.96}),count);
 mesh.name='residual-block-1-actual-output-values';const vals=[];
 for(let c=0;c<FILTER_COUNT;c++)for(let r=0;r<BOARD_SIZE;r++)for(let col=0;col<BOARD_SIZE;col++)vals.push(Number(tensor[r][col][c]));
 const max=Math.max(1e-8,...vals.map(Math.abs)),o=new THREE.Object3D();let i=0;
 for(let c=0;c<FILTER_COUNT;c++)for(let r=0;r<BOARD_SIZE;r++)for(let col=0;col<BOARD_SIZE;col++){
 const v=Number(tensor[r][col][c]);o.position.set(layout.outputStartX+c*layout.outputChannelStep,layout.outputCenterY+(BOARD_SIZE-1-r)*filterSpacing-(BOARD_SIZE-1)*filterSpacing/2,layout.outputCenterZ+col*filterSpacing-(BOARD_SIZE-1)*filterSpacing/2);
 o.scale.setScalar(layout.outputCubeSize*.72);o.updateMatrix();mesh.setMatrixAt(i,o.matrix);mesh.setColorAt(i,new THREE.Color().setHSL((1/3)*(1-Math.max(0,Math.min(1,v/max))),1,.5));i++;}
 mesh.count=i;mesh.instanceMatrix.needsUpdate=true;if(mesh.instanceColor)mesh.instanceColor.needsUpdate=true;group.add(mesh);
}
function addResidualArrow(group,x1,x2,y,z,name){const gap=x2-x1;if(gap<=.15)return;const arrow=new THREE.ArrowHelper(new THREE.Vector3(1,0,0),new THREE.Vector3(x1+gap*.22,y,z),gap*.67,0x71879b,.42,.22);arrow.name=name;group.add(arrow);}
function addResidualCurvedArrow(group,x1,x2,y1,z1,y2,z2,name){const dx=x2-x1;if(dx<=.2)return;const c=new THREE.CubicBezierCurve3(new THREE.Vector3(x1,y1,z1),new THREE.Vector3(x1+dx*.42,y1,z1),new THREE.Vector3(x2-dx*.42,y2,z2),new THREE.Vector3(x2,y2,z2));const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints(c.getPoints(64)),new THREE.LineBasicMaterial({color:0x71879b}));line.name=name+'-shaft';group.add(line);}
function addResidualBypass(group,x1,x2,z,offset,name){const dx=x2-x1;if(dx<=1)return;const t=Math.min(dx*.22,3),oz=z+offset,p=new THREE.CurvePath();p.add(new THREE.CubicBezierCurve3(new THREE.Vector3(x1,0,z),new THREE.Vector3(x1+t*.45,0,z),new THREE.Vector3(x1+t*.55,0,oz),new THREE.Vector3(x1+t,0,oz)));p.add(new THREE.LineCurve3(new THREE.Vector3(x1+t,0,oz),new THREE.Vector3(x2-t,0,oz)));p.add(new THREE.CubicBezierCurve3(new THREE.Vector3(x2-t,0,oz),new THREE.Vector3(x2-t*.55,0,oz),new THREE.Vector3(x2-t*.45,0,z),new THREE.Vector3(x2,0,z)));const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints(p.getPoints(96)),new THREE.LineBasicMaterial({color:0x71879b}));line.name=name+'-shaft';group.add(line);}
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
      const br1=layout.bankStartXs[0]+(layout.channels-1)*filterSpacing+filterCubeSize/2,bl2=layout.bankStartXs[1]-filterCubeSize/2;
      const br2=layout.bankStartXs[1]+(layout.channels-1)*filterSpacing+filterCubeSize/2,cy=layout.outputCenterY,cz=layout.outputCenterZ;
      if(index===0){
       const inputRight=20+(FILTER_COUNT-1)*layout.outputChannelStep+layout.outputCubeSize/2;
       addResidualArrow(residualPlaceholderGroup,inputRight,layout.bankStartXs[0]-filterCubeSize/2,cy,cz,'input-to-block-1');
       addResidualArrow(residualPlaceholderGroup,br1,bl2,cy,cz,'block-1-conv1-to-conv2');
       const outLeft=layout.outputCenterX-layout.outputSizeX/2,merge=outLeft-1.35;
       addResidualCurvedArrow(residualPlaceholderGroup,br2,merge,0,cz,0,cz,'block-1-conv2-to-merge');
       addResidualBypass(residualPlaceholderGroup,inputRight,merge,cz,layout.bankSizeZ*1.08/2+1,'block-1-identity-skip');
       addResidualArrow(residualPlaceholderGroup,merge,outLeft,0,cz,'block-1-merged-output');
       if(getData().residualBlock0Output)addResidualOutputValues(residualPlaceholderGroup,layout,index,getData().residualBlock0Output);
       const outRight=layout.outputCenterX+layout.outputSizeX/2,nextLeft=residualLayout(1).bankStartXs[0]-filterCubeSize/2;
       addResidualArrow(residualPlaceholderGroup,outRight,nextLeft,cy,cz,'block-1-output-to-block-2');
      }else if(index<blockCount-1){
       const next=residualLayout(index+1),nextLeft=next.bankStartXs[0]-filterCubeSize/2;
       addResidualArrow(residualPlaceholderGroup,br2,nextLeft,cy,cz,'block-'+(index+1)+'-to-block-'+(index+2));
      }
    }
    filterStaticGroup.add(residualPlaceholderGroup);
  }

  function drawResidualPlaceholderLabels(blockCount){
   for(let index=0;index<blockCount;index++){const layout=residualLayout(index),cx=layout.blockStartX+layout.blockSpanX/2,y=layout.bankStartY+layout.bankSizeY/2+2,z=layout.bankStartZ+layout.bankSizeZ/2;
    addLabel('Residual block '+(index+1),[cx,y,z],24,'#ffffff',70);}
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
    const outputTensor = getData().residualBlock0Output || null;
    if (outputTensor !== lastRenderedOutputTensor) {
      lastRenderedOutputTensor = outputTensor;
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
