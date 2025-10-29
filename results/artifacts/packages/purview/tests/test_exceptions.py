# Copyright (c) Microsoft. All rights reserved.

"""Purview 例外のテスト。"""

from agent_framework_purview import (
    PurviewAuthenticationError,
    PurviewRateLimitError,
    PurviewRequestError,
    PurviewServiceError,
)


class TestPurviewExceptions:
    """カスタム Purview 例外クラスのテスト。"""

    def test_purview_service_error(self) -> None:
        """PurviewServiceError 基底例外のテスト。"""
        error = PurviewServiceError("Service error occurred")
        assert str(error) == "Service error occurred"
        assert isinstance(error, Exception)

    def test_purview_authentication_error(self) -> None:
        """PurviewAuthenticationError 例外のテスト。"""
        error = PurviewAuthenticationError("Authentication failed")
        assert str(error) == "Authentication failed"
        assert isinstance(error, PurviewServiceError)

    def test_purview_rate_limit_error(self) -> None:
        """PurviewRateLimitError 例外のテスト。"""
        error = PurviewRateLimitError("Rate limit exceeded")
        assert str(error) == "Rate limit exceeded"
        assert isinstance(error, PurviewServiceError)

    def test_purview_request_error(self) -> None:
        """PurviewRequestError 例外のテスト。"""
        error = PurviewRequestError("Request failed")
        assert str(error) == "Request failed"
        assert isinstance(error, PurviewServiceError)
