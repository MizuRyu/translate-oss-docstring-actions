# Copyright (c) Microsoft. All rights reserved.


from typing import Annotated, Any, ClassVar, TypeVar

from pydantic import Field, UrlConstraints
from pydantic.networks import AnyUrl
from pydantic_settings import BaseSettings, SettingsConfigDict

HTTPsUrl = Annotated[AnyUrl, UrlConstraints(max_length=2083, allowed_schemes=["https"])]

__all__ = ["AFBaseSettings", "HTTPsUrl"]


TSettings = TypeVar("TSettings", bound="AFBaseSettings")


class AFBaseSettings(BaseSettings):
    """Agent Frameworkのすべての設定クラスの基底クラス。

    サブクラスはフィールドを作成し、環境変数のプレフィックスとしてenv_prefixクラス変数をオーバーライドします。

    同じSettingsフィールドに複数の方法で値が指定された場合、選択される値は以下の優先順位（降順）で決定されます:
    - Settingsクラスのイニシャライザに渡された引数。
    - 環境変数（例: my_prefix_special_function）。
    - dotenv (.env)ファイルから読み込まれた変数。
    - secretsディレクトリから読み込まれた変数。
    - Settingsモデルのデフォルトフィールド値。

    """

    env_prefix: ClassVar[str] = ""
    env_file_path: str | None = Field(default=None, exclude=True)
    env_file_encoding: str | None = Field(default="utf-8", exclude=True)

    model_config = SettingsConfigDict(
        extra="ignore",
        case_sensitive=False,
    )

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """settingsクラスを初期化します。"""
        # kwargsからNoneの値を削除し、デフォルトが使用されるようにします。
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        super().__init__(**kwargs)

    def __new__(cls: type["TSettings"], *args: Any, **kwargs: Any) -> "TSettings":
        """env_prefixを設定するために__new__メソッドをオーバーライドします。"""
        # 両方とも、指定されていてNoneの場合はデフォルトに設定します。
        if "env_file_encoding" in kwargs and kwargs["env_file_encoding"] is not None:
            env_file_encoding = kwargs["env_file_encoding"]
        else:
            env_file_encoding = "utf-8"
        if "env_file_path" in kwargs and kwargs["env_file_path"] is not None:
            env_file_path = kwargs["env_file_path"]
        else:
            env_file_path = ".env"
        cls.model_config.update(  # type: ignore
            env_prefix=cls.env_prefix,
            env_file=env_file_path,
            env_file_encoding=env_file_encoding,
        )
        cls.model_rebuild()
        return super().__new__(cls)  # type: ignore[return-value]
