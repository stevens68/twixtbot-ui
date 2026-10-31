/** Cached, instrumented client for authoritative game-position snapshots. */
export function createSnapshotClient({ recordPerf, cacheLimit = 128 }) {
  const snapshotCache = new Map();

  function getCachedSnapshot(key) {
    if (!snapshotCache.has(key)) return null;
    const value = snapshotCache.get(key);
    snapshotCache.delete(key);
    snapshotCache.set(key, value);
    return value;
  }

  function cacheSnapshot(key, value) {
    snapshotCache.delete(key);
    snapshotCache.set(key, value);
    while (snapshotCache.size > cacheLimit) {
      snapshotCache.delete(snapshotCache.keys().next().value);
    }
  }

  async function requestSnapshot(moves) {
    const cacheKey = JSON.stringify({ moves, allow_scl: false, allow_swap: true });
    const cached = getCachedSnapshot(cacheKey);
    if (cached) {
      recordPerf('snapshotCacheHit', {
        moves: moves.length,
        responseBytes: cached.responseBytes,
        wireResponseBytes: cached.wireResponseBytes,
        cacheEntries: snapshotCache.size,
      });
      return cached.result;
    }

    const started = performance.now();
    const response = await fetch('/api/trace', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ moves, allow_scl: false, allow_swap: true }),
    });
    const headersMs = performance.now() - started;
    const bodyStarted = performance.now();
    // Measure first-byte availability separately from draining the full body.
    const reader = response.body.getReader();
    const chunks = [];
    let responseLength = 0;
    let firstChunkMs = null;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (firstChunkMs === null) firstChunkMs = performance.now() - bodyStarted;
      chunks.push(value);
      responseLength += value.length;
    }
    const responseBuffer = new Uint8Array(responseLength);
    let responseOffset = 0;
    for (const chunk of chunks) {
      responseBuffer.set(chunk, responseOffset);
      responseOffset += chunk.length;
    }
    const responseText = new TextDecoder().decode(responseBuffer);
    const bodyMs = performance.now() - bodyStarted;
    const parseStarted = performance.now();
    let result;
    try {
      result = JSON.parse(responseText);
    } catch (error) {
      throw new Error('The visualizer API returned invalid JSON: ' + error.message);
    }
    const jsonParseMs = performance.now() - parseStarted;
    if (!response.ok) throw new Error(result.error || response.statusText);
    // Fetch exposes decompressed bytes; retain the content-length separately.
    const responseBytes = new TextEncoder().encode(responseText).length;
    const wireResponseBytes = Number(response.headers.get('content-length')) || responseBytes;
    recordPerf('requestSnapshot', {
      moves: moves.length,
      headersMs: Number(headersMs.toFixed(2)),
      firstChunkMs: firstChunkMs === null ? null : Number(firstChunkMs.toFixed(2)),
      bodyMs: Number(bodyMs.toFixed(2)),
      jsonParseMs: Number(jsonParseMs.toFixed(2)),
      totalMs: Number((performance.now() - started).toFixed(2)),
      responseBytes,
      wireResponseBytes,
      compression: response.headers.get('content-encoding') || 'none',
    });
    cacheSnapshot(cacheKey, { result, responseBytes, wireResponseBytes });
    return result;
  }

  return { requestSnapshot };
}
