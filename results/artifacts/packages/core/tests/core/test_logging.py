# Copyright (c) Microsoft. All rights reserved.


import pytest

from agent_framework import get_logger
from agent_framework.exceptions import AgentFrameworkException


def test_get_logger():
    """正しい名前でloggerが作成されることをテストします。"""
    logger = get_logger()
    assert logger.name == "agent_framework"


def test_get_logger_custom_name():
    """カスタム名でloggerが作成できることをテストします。"""
    custom_name = "agent_framework.custom"
    logger = get_logger(custom_name)
    assert logger.name == custom_name


def test_get_logger_invalid_name():
    """無効なlogger名に対して例外が発生することをテストします。"""
    with pytest.raises(AgentFrameworkException):
        get_logger("invalid_name")


def test_log(caplog):
    """loggerがメッセージをログでき、期待されるフォーマットに従っていることをテストします。"""
    logger = get_logger()
    with caplog.at_level("DEBUG"):
        logger.debug("This is a debug message")
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.levelname == "DEBUG"
        assert record.message == "This is a debug message"
        assert record.name == "agent_framework"
        assert record.pathname.endswith("test_logging.py")
