/**
 * Jest test setup
 */

// Set test timeout globally
jest.setTimeout(120000); // 2 minutes for proof generation tests

// Add custom matchers if needed
expect.extend({
  toBeValidProof(received) {
    const pass = received.hasOwnProperty('approved') &&
                 received.hasOwnProperty('verification') &&
                 received.hasOwnProperty('zkmlProof') &&
                 received.zkmlProof.hasOwnProperty('commitment');

    if (pass) {
      return {
        message: () => `expected ${received} not to be a valid proof`,
        pass: true,
      };
    } else {
      return {
        message: () => `expected ${received} to be a valid proof with all required fields`,
        pass: false,
      };
    }
  },
});
