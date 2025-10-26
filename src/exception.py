class ExtractionError(Exception):
    """抽出処理に関する基底例外。"""


class TranslationError(Exception):
    """翻訳処理に関する基底例外。"""


class TranslationRequestError(TranslationError):
    """LLMリクエストに失敗した場合の例外。"""


class TranslationParseError(TranslationError):
    """翻訳結果の解析に失敗した場合の例外。"""


class TokenLimitExceededError(TranslationError):
    """トークン制限によりリクエストを構成できなかった場合の例外。"""
