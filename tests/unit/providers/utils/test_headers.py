# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.providers.utils.headers import BLOCKED_HEADERS, filter_extra_headers


class TestFilterExtraHeaders:
    """Tests for the shared header-filtering utility."""

    def test_passes_safe_headers(self):
        headers = {"X-MAAS-SUBSCRIPTION": "free-tier", "X-Request-ID": "abc"}
        assert filter_extra_headers(headers) == headers

    def test_filters_authorization(self):
        result = filter_extra_headers({"Authorization": "Bearer evil", "X-Safe": "ok"})
        assert result == {"X-Safe": "ok"}

    def test_filters_hop_by_hop(self):
        result = filter_extra_headers({"Transfer-Encoding": "chunked", "Host": "evil", "X-Safe": "ok"})
        assert result == {"X-Safe": "ok"}

    def test_filters_proxy_headers(self):
        result = filter_extra_headers(
            {
                "X-Forwarded-For": "1.2.3.4",
                "X-Real-IP": "5.6.7.8",
                "X-Custom": "value",
            }
        )
        assert result == {"X-Custom": "value"}

    def test_case_insensitive_blocking(self):
        result = filter_extra_headers({"AUTHORIZATION": "bad", "host": "bad", "X-Ok": "yes"})
        assert result == {"X-Ok": "yes"}

    def test_all_blocked_returns_none(self):
        result = filter_extra_headers({"Authorization": "x", "Host": "y", "Cookie": "z"})
        assert result is None

    def test_empty_dict_returns_none(self):
        assert filter_extra_headers({}) is None

    def test_none_returns_none(self):
        assert filter_extra_headers(None) is None

    def test_non_dict_returns_none(self):
        assert filter_extra_headers("not-a-dict") is None

    def test_filters_additional_proxy_headers(self):
        result = filter_extra_headers(
            {
                "X-Forwarded-Prefix": "/api",
                "CF-Connecting-IP": "1.2.3.4",
                "True-Client-IP": "5.6.7.8",
                "X-Safe": "ok",
            }
        )
        assert result == {"X-Safe": "ok"}

    def test_blocked_headers_is_frozen(self):
        assert isinstance(BLOCKED_HEADERS, frozenset)
