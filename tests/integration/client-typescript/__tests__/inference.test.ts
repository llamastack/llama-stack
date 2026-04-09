// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the terms described in the LICENSE file in
// the root directory of this source tree.

/**
 * Integration tests for Inference API (Chat Completions).
 *
 * NOTE: These tests are temporarily disabled because the Python test file
 * (test_openai_completion.py) that generated the recordings these tests
 * depend on was removed as part of the /v1/completions API removal.
 * The chat completion tests will be re-added when the Python chat
 * completion integration tests are restored in a separate file.
 */

describe('Inference API - Chat Completions', () => {
  test.skip('chat completion tests temporarily disabled', () => {
    // See note above
  });
});
