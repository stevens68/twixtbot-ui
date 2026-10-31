/**
 * Fetch and validate the trained first-layer and initial residual-block weights.
 * Mutates the supplied demo model so the renderers can keep one shared data object.
 */
export async function loadTrainedModelWeights({
  demoModel,
  filterCount,
  residualBlockWeightCache,
  onInitialResidualBlock,
}) {
  try {
    const response = await fetch('/api/weights', { cache: 'no-store' });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || response.statusText);
    if (!result.layers || !result.weights ||
        !result.layers.links || !result.layers.pegs || !result.layers.locations ||
        !Array.isArray(result.weights.links) ||
        !Array.isArray(result.weights.pegs) ||
        !Array.isArray(result.weights.locations) ||
        result.weights.links.length !== result.layers.links.filter_count ||
        result.weights.pegs.length !== result.layers.pegs.filter_count ||
        result.weights.locations.length !== result.layers.locations.filter_count) {
      throw new Error('The model-weight endpoint returned an unexpected peg/link/location weight shape.');
    }

    for (const [name, layer] of Object.entries(result.layers)) {
      if (layer.has_bias &&
          (!Array.isArray(layer.bias) ||
           layer.bias.length !== layer.filter_count ||
           layer.bias.some((value) => !Number.isFinite(Number(value))))) {
        throw new Error('The model-weight endpoint returned invalid bias values for ' + name + '.');
      }
    }

    demoModel.weights.links = result.weights.links;
    demoModel.weights.pegs = result.weights.pegs;
    demoModel.weights.locations = result.weights.locations;
    const residualBlock0 = result.residual_block_0;
    if (!residualBlock0 || residualBlock0.config?.channels !== filterCount) {
      throw new Error('The model-weight endpoint did not return a ' + filterCount + '-channel first residual block.');
    }
    for (const name of ['conv1', 'conv2']) {
      const layer = residualBlock0.layers?.[name];
      const filters = residualBlock0.weights?.[name];
      if (!layer || layer.filter_count !== filterCount || layer.kernel_height !== 5 ||
          layer.kernel_width !== 5 || layer.channels !== filterCount ||
          !Array.isArray(filters) || filters.length !== filterCount ||
          filters.some(filter => filter.length !== 5 || filter.some(row => row.length !== 5 ||
            row.some(cell => cell.length !== filterCount || cell.some(value => !Number.isFinite(Number(value))))))) {
        throw new Error('The model-weight endpoint returned an invalid first residual ' + name + ' filter bank.');
      }
    }
    demoModel.weights.residualBlock0 = residualBlock0.weights;
    demoModel.residualBlock0Metadata = residualBlock0.layers;
    demoModel.residualBlockCount = Number(result.residual_block_count) || 1;
    residualBlockWeightCache.set(0, residualBlock0.weights);
    onInitialResidualBlock(residualBlock0.weights);
    demoModel.linkConvMetadata = result.layers.links;
    demoModel.pegConvMetadata = result.layers.pegs;
    demoModel.locationConvMetadata = result.layers.locations;

    const primaryBn = result.primary_bn;
    for (const key of ['weight', 'bias', 'running_mean', 'running_var']) {
      if (!Array.isArray(primaryBn?.[key]) ||
          primaryBn[key].length !== filterCount ||
          primaryBn[key].some((value) => !Number.isFinite(Number(value)))) {
        throw new Error('The model-weight endpoint returned invalid primary batch-normalization parameter: ' + key);
      }
    }
    demoModel.primaryBn = primaryBn;

    const status = 'Trained peg/link/location weights and primary batch-normalization parameters loaded from ' + result.layers.links.source +
      ' (pegs: ' + result.layers.pegs.filter_count + ' filters, bias=' +
      (result.layers.pegs.has_bias ? 'loaded' : 'none') + '; links: ' +
      result.layers.links.filter_count + ' filters, bias=' +
      (result.layers.links.has_bias ? 'loaded' : 'none') + '; locations: ' +
      result.layers.locations.filter_count + ' filters, bias=' +
      (result.layers.locations.has_bias ? 'loaded' : 'none') + ').';
    console.info('[NN visualizer] ' + status, {
      pegs: result.layers.pegs,
      links: result.layers.links,
      locations: result.layers.locations,
    });
    return status;
  } catch (error) {
    const status = 'Using generated demo weights; trained weights and biases could not be loaded: ' + error.message;
    console.warn('[NN visualizer] ' + status);
    return status;
  }
}
