/**
 * Link input encoding diagnostics, kept separate from the Three.js renderer.
 * These checks independently reproduce the upstream naf.py transforms and
 * compare the production encoder output against that reference.
 */
export function createLinkEncodingDiagnostics({ BOARD_SIZE, createTensor3 }) {
  function logLinkEncodingDiagnostic(game, linkTensor) {
    const transpose = game.nextPlayer === 0;
    // The upstream Game.links arrays are indexed [x][y]. On Black's
    // turn naf.py transposes that array, producing [row=y][col=x], which
    // matches the visualizer's { row, col } board coordinates. Do not swap
    // row/col again here.
    const canonicalPoint = (peg) => ({
      row: peg.row,
      col: peg.col,
    });
    const canonicalPlayer = (player) => transpose ? 1 - player : player;
    const channelNames = [
      'index 0', 'index 1', 'index 2', 'index 3',
      'index 4', 'index 5', 'index 6', 'index 7',
    ];
  
    // Reproduce the raw-link index calculation and the upstream naf.py
    // transpose/color swap and to_input_arrays shift independently of the
    // rendered tensor. This lets the log compare expected cells to actual data.
    const trace = game.links.map((link, index) => {
      const x1 = link.from.col;
      const y1 = link.from.row;
      const x2 = link.to.col;
      const y2 = link.to.row;
      const rawChannel = link.player
        + (((x1 + x2) % 2 !== 0) ? 4 : 0)
        + (((y2 - y1) * (x2 - x1) < 0) ? 2 : 0);
      const rawRow = Math.floor((x1 + x2) / 2);
      const rawCol = Math.floor((y1 + y2) / 2);
  
      // In the original encoder, game.links are indexed [x][y], then
      // transposed into [row][col]. Canonical row/col therefore start as y/x.
      const swappedColor = 1 - link.player;
      // Match NetInputs.init_from_game_black() in upstream naf.py:
      // raw 0-1 -> canonical 4-5; raw 2-3 -> 6-7;
      // raw 4-5 -> 0-1; raw 6-7 -> 2-3.
      // The colour within each pair is swapped.
      const midChannel =
        rawChannel < 2 ? 4 + swappedColor :
        rawChannel < 4 ? 6 + swappedColor :
        rawChannel < 6 ? swappedColor : 2 + swappedColor;
  
      // Canonical board coordinates use row=x, col=y when transposed.
      const canonicalA = canonicalPoint(link.from);
      const canonicalB = canonicalPoint(link.to);
      const canonicalRow = Math.floor((canonicalA.row + canonicalB.row) / 2);
      const canonicalCol = Math.floor((canonicalA.col + canonicalB.col) / 2);
  
      // to_input_arrays shifts the midpoint cell according to the channel bits.
      const vertical = (midChannel & 4) !== 0;
      const diff = (midChannel & 2) !== 0;
      const expectedRow = canonicalRow + ((vertical || diff) ? 1 : 0);
      const expectedCol = canonicalCol + ((!vertical || diff) ? 1 : 0);
      const inBounds = expectedRow >= 0 && expectedRow < BOARD_SIZE &&
        expectedCol >= 0 && expectedCol < BOARD_SIZE;
      const actualValue = inBounds ? linkTensor[expectedRow][expectedCol][midChannel] : null;
  
      return {
        link: index,
        originalPlayer: link.player === 1 ? 'Black' : 'White',
        originalFrom: String.fromCharCode(65 + link.from.col) + (link.from.row + 1),
        originalTo: String.fromCharCode(65 + link.to.col) + (link.to.row + 1),
        transpose,
        canonicalPlayer: canonicalPlayer(link.player) === 1 ? 'Black' : 'White',
        canonicalFrom: String.fromCharCode(65 + canonicalA.col) + (canonicalA.row + 1),
        canonicalTo: String.fromCharCode(65 + canonicalB.col) + (canonicalB.row + 1),
        rawChannel,
        rawMidpoint: '(' + rawRow + ',' + rawCol + ')',
        expectedChannel: midChannel,
        expectedChannelName: channelNames[midChannel],
        expectedCell: inBounds
          ? String.fromCharCode(65 + expectedCol) + (expectedRow + 1)
          : 'out of bounds',
        expectedCellValue: actualValue,
        foundAtExpectedCell: actualValue === 1,
      };
    });
  
    const actualCells = [];
    for (let channel = 0; channel < 8; channel += 1) {
      for (let row = 0; row < BOARD_SIZE; row += 1) {
        for (let col = 0; col < BOARD_SIZE; col += 1) {
          if (linkTensor[row][col][channel] !== 0) {
            actualCells.push({
              channel,
              cell: String.fromCharCode(65 + col) + (row + 1),
              row,
              col,
              value: linkTensor[row][col][channel],
            });
          }
        }
      }
    }
  
    console.groupCollapsed(
      '[Link encoding trace] transpose=' + transpose +
      ', nextPlayer=' + game.nextPlayer +
      ', board links=' + trace.length +
      ', nonzero tensor cells=' + actualCells.length,
    );
    console.log('Per-link trace: original -> canonical -> expected shifted tensor cell');
    console.table(trace);
    console.log('Actual nonzero tensor cells (channel labels intentionally omitted until verified):');
    console.table(actualCells);
    console.log('Loaded diagnostic build: link-trace-v4 (upstream channel + coordinate mapping). If this exact group title is absent, hard-reload before interpreting the output.');
    console.groupEnd();
  }
  
  /**
   * Independent full-tensor check against the upstream naf.py / NetInputs
   * transformations. It builds the raw [x][y][channel] link planes first,
   * applies the upstream transpose/color swap, then applies to_input_arrays'
   * per-channel shifts. This is a parity check of the JS encoder's complete
   * output, not a comparison against a saved TensorFlow runtime dump.
   */
  function buildUpstreamReferenceInputs(game) {
    const rawPegs = Array.from({ length: 2 }, () => createTensor3(BOARD_SIZE, BOARD_SIZE, 1));
    const rawLinks = Array.from({ length: 8 }, () => createTensor3(BOARD_SIZE, BOARD_SIZE, 1));
  
    for (const peg of game.pegs) {
      rawPegs[peg.player][peg.col][peg.row][0] = 1;
    }
    for (const link of game.links) {
      const ax = link.from.col, ay = link.from.row;
      const bx = link.to.col, by = link.to.row;
      const channel = link.player
        + (((ax + bx) % 2 !== 0) ? 4 : 0)
        + (((by - ay) * (bx - ax) < 0) ? 2 : 0);
      const x = Math.floor((ax + bx) / 2);
      const y = Math.floor((ay + by) / 2);
      rawLinks[channel][x][y][0] = 1;
    }
  
    const blackToMove = game.nextPlayer === 0;
    const pegs = createTensor3(BOARD_SIZE, BOARD_SIZE, 2);
    const midpointLinks = createTensor3(BOARD_SIZE, BOARD_SIZE, 8);
  
    for (let x = 0; x < BOARD_SIZE; x += 1) {
      for (let y = 0; y < BOARD_SIZE; y += 1) {
        const row = blackToMove ? x : y;
        const col = blackToMove ? y : x;
        // naf.py changes perspective on Black's turn and swaps player planes.
        pegs[row][col][0] = rawPegs[blackToMove ? 1 : 0][blackToMove ? row : x][blackToMove ? col : y][0];
        pegs[row][col][1] = rawPegs[blackToMove ? 0 : 1][blackToMove ? row : x][blackToMove ? col : y][0];
  
        for (let rawChannel = 0; rawChannel < 8; rawChannel += 1) {
          const sourceChannel = blackToMove ? 1 - (rawChannel & 1) : rawChannel & 1;
          const group = rawChannel & 6;
          const sourceGroup = blackToMove
            ? (group === 0 ? 4 : group === 2 ? 6 : group === 4 ? 0 : 2)
            : group;
          const sourceIndex = sourceGroup + sourceChannel;
          midpointLinks[row][col][rawChannel] =
            rawLinks[sourceIndex][blackToMove ? row : x][blackToMove ? col : y][0];
        }
      }
    }
  
    const links = createTensor3(BOARD_SIZE, BOARD_SIZE, 8);
    for (let row = 0; row < BOARD_SIZE; row += 1) {
      for (let col = 0; col < BOARD_SIZE; col += 1) {
        for (let channel = 0; channel < 8; channel += 1) {
          const vertical = (channel & 4) !== 0;
          const diff = (channel & 2) !== 0;
          const sourceRow = row - ((vertical || diff) ? 1 : 0);
          const sourceCol = col - ((!vertical || diff) ? 1 : 0);
          if (sourceRow >= 0 && sourceCol >= 0) {
            links[row][col][channel] = midpointLinks[sourceRow][sourceCol][channel];
          }
        }
      }
    }
    return { pegs, links };
  }
  
  function compareTensors(label, actual, expected) {
    let mismatchCount = 0;
    let actualNonzero = 0;
    let expectedNonzero = 0;
    const examples = [];
  
    for (let row = 0; row < BOARD_SIZE; row += 1) {
      for (let col = 0; col < BOARD_SIZE; col += 1) {
        for (let channel = 0; channel < actual[row][col].length; channel += 1) {
          const a = actual[row][col][channel];
          const e = expected[row][col][channel];
          if (a !== 0) actualNonzero += 1;
          if (e !== 0) expectedNonzero += 1;
          if (a !== e) {
            mismatchCount += 1;
            if (examples.length < 12) {
              examples.push({
                cell: String.fromCharCode(65 + col) + (row + 1),
                row, col, channel, actual: a, expected: e,
              });
            }
          }
        }
      }
    }
  
    console.groupCollapsed(
      '[Full tensor validation] ' + label + ': ' +
      (mismatchCount === 0 ? 'PASS' : 'FAIL') +
      ', mismatches=' + mismatchCount,
    );
    console.table([{
      tensor: label, result: mismatchCount === 0 ? 'PASS' : 'FAIL',
      mismatches: mismatchCount, actualNonzero, expectedNonzero,
    }]);
    if (examples.length) console.table(examples);
    console.groupEnd();
    return { label, mismatchCount, actualNonzero, expectedNonzero };
  }
  
  function logFullInputTensorValidation(game, inputs) {
    const reference = buildUpstreamReferenceInputs(game);
    const results = [
      compareTensors('pegs 24×24×2', inputs.pegs, reference.pegs),
      compareTensors('links 24×24×8', inputs.links, reference.links),
    ];
    const totalMismatches = results.reduce((sum, result) => sum + result.mismatchCount, 0);
    console.log(
      '[Full tensor validation] ' +
      (totalMismatches === 0 ? 'PASS' : 'FAIL') +
      ': ' + totalMismatches + ' total cell mismatches. ' +
      'Reference reimplements upstream naf.py and NetInputs shifts; it is not yet a saved TF runtime comparison.',
    );
  }

  return { logLinkEncodingDiagnostic, logFullInputTensorValidation };
}
