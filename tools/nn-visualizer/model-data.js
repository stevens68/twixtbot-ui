/** Model data factories and inference helpers shared by the visualizer. */
import { createRandomGenerator, createTensor3, createTensor4 } from './tensor-utils.js';

export function createDemoData({ BOARD_SIZE, FILTER_COUNT, encodeGameInputs }) {
  const random=createRandomGenerator(2026);
    const {pegs,links}=encodeGameInputs({pegs:[],links:[],nextPlayer:1});
    const locations=createTensor3(BOARD_SIZE,BOARD_SIZE,2);
    for(let row=0;row<BOARD_SIZE;row++) for(let col=0;col<BOARD_SIZE;col++) {
      locations[row][col][0]=col/BOARD_SIZE; locations[row][col][1]=row/BOARD_SIZE;
    }
    const weights={
      pegs:createTensor4(FILTER_COUNT,5,5,2,()=> (random()-0.5)*0.4),
      links:createTensor4(FILTER_COUNT,4,4,8,()=> (random()-0.5)*0.16),
      locations:createTensor4(FILTER_COUNT,1,1,2,()=> (random()-0.5)*0.3),
      residualBlock0: {
        conv1: createTensor4(FILTER_COUNT,5,5,FILTER_COUNT,()=> (random()-0.5)*0.08),
        conv2: createTensor4(FILTER_COUNT,5,5,FILTER_COUNT,()=> (random()-0.5)*0.08),
      },
    };
    const bias=Array.from({length:FILTER_COUNT},()=> (random()-0.5)*0.15);
    // Identity BN keeps the generated fallback numerically usable, but it is
    // explicitly a demo and is never confused with checkpoint inference.
    const primaryBn = {
      weight: Array(FILTER_COUNT).fill(1),
      bias: Array(FILTER_COUNT).fill(0),
      running_mean: Array(FILTER_COUNT).fill(0),
      running_var: Array(FILTER_COUNT).fill(1),
      eps: 1e-3,
    };
    return {pegs,links,locations,weights,bias,primaryBn,residualBlockCount:20,isDemo:true};
}

export function calculateConvolutionOutput({ data, BOARD_SIZE, FILTER_COUNT, createTensor3 }) {
  const output = createTensor3(BOARD_SIZE, BOARD_SIZE, FILTER_COUNT);
    const branches = [
      { name: 'pegs', input: data.pegs, weights: data.weights.pegs, metadata: data.pegConvMetadata },
      { name: 'links', input: data.links, weights: data.weights.links, metadata: data.linkConvMetadata },
      { name: 'locations', input: data.locations, weights: data.weights.locations, metadata: data.locationConvMetadata },
    ];
  
    // The checkpoint uses TensorFlow-style SAME padding. For even-sized kernels,
    // the extra padding cell belongs on the bottom/right (before = floor(total/2)).
    for (const branch of branches) {
      const kernelHeight = branch.weights[0].length;
      const kernelWidth = branch.weights[0][0].length;
      const channels = branch.weights[0][0][0].length;
      const padTop = Math.floor((kernelHeight - 1) / 2);
      const padLeft = Math.floor((kernelWidth - 1) / 2);
  
      for (let row = 0; row < BOARD_SIZE; row += 1) {
        for (let col = 0; col < BOARD_SIZE; col += 1) {
          for (let filter = 0; filter < FILTER_COUNT; filter += 1) {
            let sum = branch.metadata?.has_bias
              ? Number(branch.metadata.bias[filter] || 0)
              : 0;
  
            for (let kernelRow = 0; kernelRow < kernelHeight; kernelRow += 1) {
              const inputRow = row + kernelRow - padTop;
              if (inputRow < 0 || inputRow >= BOARD_SIZE) continue;
  
              for (let kernelCol = 0; kernelCol < kernelWidth; kernelCol += 1) {
                const inputCol = col + kernelCol - padLeft;
                if (inputCol < 0 || inputCol >= BOARD_SIZE) continue;
  
                for (let channel = 0; channel < channels; channel += 1) {
                  sum += Number(branch.input[inputRow][inputCol][channel]) *
                    Number(branch.weights[filter][kernelRow][kernelCol][channel]);
                }
              }
            }
  
            output[row][col][filter] += sum;
          }
        }
      }
    }
  
    // Match TwixtNet.forward exactly: sum the three convolution branches,
    // apply primary BatchNorm2d in inference mode, then torch.abs().
    const bn = data.primaryBn;
    if (!bn) {
      throw new Error('Primary batch-normalization parameters are unavailable; refusing to label raw convolution sums as model features.');
    }
    const epsilon = Number(bn.eps ?? 1e-3);
    for (let row = 0; row < BOARD_SIZE; row += 1) {
      for (let col = 0; col < BOARD_SIZE; col += 1) {
        for (let channel = 0; channel < FILTER_COUNT; channel += 1) {
          const variance = Number(bn.running_var[channel]);
          const normalized =
            (output[row][col][channel] - Number(bn.running_mean[channel])) /
            Math.sqrt(variance + epsilon);
          const scaled = normalized * Number(bn.weight[channel]) +
            Number(bn.bias[channel]);
          output[row][col][channel] = Math.abs(scaled);
        }
      }
    }
  
    return output;
}

export function validateResidualBlockPayload(payload, expectedIndex, FILTER_COUNT) {
  if (payload?.index !== expectedIndex || payload.config?.channels !== FILTER_COUNT) {
      throw new Error('The server returned unexpected metadata for residual block ' + expectedIndex + '.');
    }
    for (const name of ['conv1', 'conv2']) {
      const layer = payload.layers?.[name];
      const filters = payload.weights?.[name];
      if (!layer || layer.filter_count !== FILTER_COUNT || layer.kernel_height !== 5 ||
          layer.kernel_width !== 5 || layer.channels !== FILTER_COUNT ||
          !Array.isArray(filters) || filters.length !== FILTER_COUNT ||
          filters.some(filter => filter.length !== 5 || filter.some(row => row.length !== 5 ||
            row.some(cell => cell.length !== FILTER_COUNT ||
              cell.some(value => !Number.isFinite(Number(value))))))) {
        throw new Error('The server returned an invalid ' + name + ' filter bank for residual block ' + expectedIndex + '.');
      }
    }
    return payload.weights;
}
