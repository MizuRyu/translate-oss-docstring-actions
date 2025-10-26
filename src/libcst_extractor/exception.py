"""翻訳および抽出処理で利用する例外定義。"""

class ExtractorError(Exception):
    """抽出処理における基底例外。"""


class TranslationError(Exception):
    """翻訳処理に関する基底例外。"""


class TranslationRequestError(TranslationError):
    """LLM問い合わせに失敗した場合の例外。"""


class TranslationParseError(TranslationError):
    """翻訳結果のパースに失敗した場合の例外。"""


class TokenLimitExceededError(TranslationError):
    """トークン制限によりリクエストを構成できなかった場合の例外。"""
