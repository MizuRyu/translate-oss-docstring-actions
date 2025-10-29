# Copyright (c) Microsoft. All rights reserved.

"""Agent Framework DevUIモデル - OpenAI互換の型とカスタム拡張。"""

# discoveryモデルをインポート openaiパッケージからすべてのOpenAI型を直接インポートします
from openai.types.conversations import Conversation, ConversationDeletedResource
from openai.types.conversations.conversation_item import ConversationItem
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseErrorEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallOutputItem,
    ResponseInputParam,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningTextDeltaEvent,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
    ResponseUsage,
    ToolParam,
)
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails
from openai.types.shared import Metadata, ResponsesModel

from ._discovery_models import DiscoveryResponse, EntityInfo
from ._openai_custom import (
    AgentFrameworkRequest,
    CustomResponseOutputItemAddedEvent,
    CustomResponseOutputItemDoneEvent,
    ExecutorActionItem,
    OpenAIError,
    ResponseFunctionResultComplete,
    ResponseTraceEvent,
    ResponseTraceEventComplete,
    ResponseWorkflowEventComplete,
)

# 互換性のための型エイリアス
OpenAIResponse = Response

# 簡単にインポートできるようにすべての型をエクスポート
__all__ = [
    "AgentFrameworkRequest",
    "Conversation",
    "ConversationDeletedResource",
    "ConversationItem",
    "CustomResponseOutputItemAddedEvent",
    "CustomResponseOutputItemDoneEvent",
    "DiscoveryResponse",
    "EntityInfo",
    "ExecutorActionItem",
    "InputTokensDetails",
    "Metadata",
    "OpenAIError",
    "OpenAIResponse",
    "OutputTokensDetails",
    "Response",
    "ResponseCompletedEvent",
    "ResponseErrorEvent",
    "ResponseFunctionCallArgumentsDeltaEvent",
    "ResponseFunctionResultComplete",
    "ResponseFunctionToolCall",
    "ResponseFunctionToolCallOutputItem",
    "ResponseInputParam",
    "ResponseOutputItemAddedEvent",
    "ResponseOutputItemDoneEvent",
    "ResponseOutputMessage",
    "ResponseOutputText",
    "ResponseReasoningTextDeltaEvent",
    "ResponseStreamEvent",
    "ResponseTextDeltaEvent",
    "ResponseTraceEvent",
    "ResponseTraceEventComplete",
    "ResponseUsage",
    "ResponseWorkflowEventComplete",
    "ResponsesModel",
    "ToolParam",
]
