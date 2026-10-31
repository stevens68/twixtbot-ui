/**
 * Fetch and validate the trained first-layer and initial residual-block weights.
 * Mutates the supplied demo model so the renderers can keep one shared data object.
 */
export async function loadTrainedModelWeights({
  demoModel,
  filterCount,
  residualBlockWeightCache,
  onInitialResidualBlock,
  onInitialLastResidualBlock,
}) {
  try {
    const response = await fetch('/api/weights', { cache: 'no-store' });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || response.statusText);
    if (!result.layers || !result.weights ||
        !result.layers.links || !result.layers.pegs || !result.layers.locations ||
        !result.layers.policy_conv1 || !Array.isArray(result.weights.policy_conv1) ||
        result.layers.policy_conv1.filter_count !== 2 ||
        result.layers.policy_conv1.kernel_height !== 1 ||
        result.layers.policy_conv1.kernel_width !== 1 ||
        result.layers.policy_conv1.channels !== filterCount ||
        result.weights.policy_conv1.length !== 2 ||
        result.weights.policy_conv1.some(filter => filter.length !== 1 ||
          filter[0].length !== 1 || filter[0][0].length !== filterCount ||
          filter[0][0].some(value => !Number.isFinite(Number(value)))) ||
        !result.layers.policy_conv2 || !Array.isArray(result.weights.policy_conv2) ||
        result.layers.policy_conv2.filter_count !== 1 ||
        result.layers.policy_conv2.kernel_height !== 1 ||
        result.layers.policy_conv2.kernel_width !== 1 ||
        result.layers.policy_conv2.channels !== 2 ||
        result.weights.policy_conv2.length !== 1 ||
        result.weights.policy_conv2[0].length !== 1 ||
        result.weights.policy_conv2[0][0].length !== 1 ||
        result.weights.policy_conv2[0][0][0].length !== 2 ||
        result.weights.policy_conv2[0][0][0].some(value => !Number.isFinite(Number(value))) ||
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
    demoModel.weights = {
      ...demoModel.weights,
      links: result.weights.links,
      pegs: result.weights.pegs,
      locations: result.weights.locations,
      policyConv1: result.weights.policy_conv1,
      policyConv2: result.weights.policy_conv2,
    };
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
    const lastIndex = demoModel.residualBlockCount - 1;
    if (lastIndex > 0 && typeof onInitialLastResidualBlock === 'function') {
      const lastResponse = await fetch('/api/residual-block?index=' + encodeURIComponent(lastIndex), { cache: 'no-store' });
      const lastPayload = await lastResponse.json();
      if (!lastResponse.ok) throw new Error(lastPayload.error || lastResponse.statusText);
      if (lastPayload?.index !== lastIndex || lastPayload.config?.channels !== filterCount) {
        throw new Error('Unexpected metadata for final residual block.');
      }
      residualBlockWeightCache.set(lastIndex, lastPayload.weights);
      onInitialLastResidualBlock(lastPayload.weights);
    }
    demoModel.linkConvMetadata = result.layers.links;
    demoModel.pegConvMetadata = result.layers.pegs;
    demoModel.locationConvMetadata = result.layers.locations;
    demoModel.policyConv1Metadata = result.layers.policy_conv1;
    demoModel.policyConv2Metadata = result.layers.policy_conv2;

    const primaryBn = result.primary_bn;
    for (const key of ['weight', 'bias', 'running_mean', 'running_var']) {
      if (!Array.isArray(primaryBn?.[key]) ||
          primaryBn[key].length !== filterCount ||
          primaryBn[key].some((value) => !Number.isFinite(Number(value)))) {
        throw new Error('The model-weight endpoint returned invalid primary batch-normalization parameter: ' + key);
      }
    }
    demoModel.primaryBn = primaryBn;

    const policyBn = result.policy_bn;
    for (const key of ['weight', 'bias', 'running_mean', 'running_var']) {
      if (!Array.isArray(policyBn?.[key]) ||
          policyBn[key].length !== 2 ||
          policyBn[key].some((value) => !Number.isFinite(Number(value)))) {
        throw new Error('The model-weight endpoint returned invalid policy batch-normalization parameter: ' + key);
      }
    }
    demoModel.policyBn = policyBn;

    const status = 'Trained peg/link/location and policy-convolution weights plus primary batch-normalization parameters loaded from ' + result.layers.links.source +
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
