// Shared helpers for deterministic demo data and tensor arrays.
function createRandomGenerator(seed) {
  return function random() {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };
}

function createTensor3(dim1, dim2, dim3) {
  return Array.from({ length: dim1 }, () =>
    Array.from({ length: dim2 }, () => Array(dim3).fill(0)),
  );
}

function createTensor4(dim1, dim2, dim3, dim4, valueFactory) {
  return Array.from({ length: dim1 }, () =>
    Array.from({ length: dim2 }, () =>
      Array.from({ length: dim3 }, () =>
        Array.from({ length: dim4 }, valueFactory),
      ),
    ),
  );
}

function flattenTensor(tensor) {
  const values = [];

  for (let channel = 0; channel < tensor[0][0].length; channel += 1) {
    for (let row = 0; row < tensor.length; row += 1) {
      for (let col = 0; col < tensor[row].length; col += 1) {
        values.push(Number(tensor[row][col][channel]));
      }
    }
  }

  return values;
}


export { createRandomGenerator, createTensor3, createTensor4, flattenTensor };
