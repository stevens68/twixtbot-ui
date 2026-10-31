import { createTensor3 } from './tensor-utils.js';

/**
 * Encode a Twixt position from the current player's canonical network viewpoint.
 * The returned tensors use [row][column][channel] order and match NetInputs.
 */
export function encodeGameInputs(game, BOARD_SIZE = 24) {
  
    // Canonicalize every position to the network's Black-to-move viewpoint:
    // transpose x/y on White's turn, and swap the peg/link colour planes.
    // Game player IDs follow twixt.Game: BLACK=0, WHITE=1.
    const rawPegs = Array.from({ length: 2 }, () =>
      createTensor3(BOARD_SIZE, BOARD_SIZE, 1),
    );
    const rawLinks = Array.from({ length: 8 }, () =>
      createTensor3(BOARD_SIZE, BOARD_SIZE, 1),
    );
  
    // Backend game arrays are indexed [x][y] = [column][row].
    for (const peg of game.pegs) {
      rawPegs[peg.player][peg.col][peg.row][0] = 1;
    }
  
    // Link channel encoding: bit 0 = colour, bit 2 = LONGY, bit 1 = DIFFSIGN.
    for (const link of game.links) {
      const x1 = link.from.col;
      const y1 = link.from.row;
      const x2 = link.to.col;
      const y2 = link.to.row;
      const channel =
        link.player +
        (((x1 + x2) % 2 !== 0) ? 4 : 0) +
        (((y2 - y1) * (x2 - x1) < 0) ? 2 : 0);
      rawLinks[channel][Math.floor((x1 + x2) / 2)][Math.floor((y1 + y2) / 2)][0] = 1;
    }
  
    const pegs = createTensor3(BOARD_SIZE, BOARD_SIZE, 2);
    const midLinks = createTensor3(BOARD_SIZE, BOARD_SIZE, 8);
    const whiteToMove = game.nextPlayer === 1;
  
    for (let row = 0; row < BOARD_SIZE; row += 1) {
      for (let col = 0; col < BOARD_SIZE; col += 1) {
        // The canonical tensor is [row][column]. On White's turn, transpose
        // the raw [x][y] axes and exchange the player colour planes.
        const sourceX = whiteToMove ? row : col;
        const sourceY = whiteToMove ? col : row;
        const firstColor = whiteToMove ? 1 : 0;
        const secondColor = whiteToMove ? 0 : 1;
  
        pegs[row][col][0] = rawPegs[firstColor][sourceX][sourceY][0];
        pegs[row][col][1] = rawPegs[secondColor][sourceX][sourceY][0];
  
        for (let channel = 0; channel < 8; channel += 1) {
          const sourceChannel = whiteToMove
            ? (channel ^ 1)
            : channel;
          midLinks[row][col][channel] =
            rawLinks[sourceChannel][sourceX][sourceY][0];
        }
      }
    }
  
    // Match NetInputs.to_input_arrays: shift link channels after perspective
    // mapping. Peg tensors are not shifted.
    const links = createTensor3(BOARD_SIZE, BOARD_SIZE, 8);
    for (let row = 0; row < BOARD_SIZE; row += 1) {
      for (let col = 0; col < BOARD_SIZE; col += 1) {
        for (let channel = 0; channel < 8; channel += 1) {
          const vertical = (channel & 4) !== 0;
          const diffSign = (channel & 2) !== 0;
          const sourceRow = row - ((vertical || diffSign) ? 1 : 0);
          const sourceCol = col - ((!vertical || diffSign) ? 1 : 0);
          if (sourceRow >= 0 && sourceCol >= 0) {
            links[row][col][channel] = midLinks[sourceRow][sourceCol][channel];
          }
        }
      }
    }
  
    return { pegs, links };
}
