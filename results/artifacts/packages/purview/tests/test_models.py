# Copyright (c) Microsoft. All rights reserved.

"""Purview モデルとシリアライゼーションのテスト。"""

from agent_framework_purview._models import (
    Activity,
    ActivityMetadata,
    ContentToProcess,
    DeviceMetadata,
    IntegratedAppMetadata,
    OperatingSystemSpecifications,
    PolicyLocation,
    ProcessContentRequest,
    ProcessContentResponse,
    ProcessConversationMetadata,
    ProtectedAppMetadata,
    ProtectionScopeActivities,
    ProtectionScopesRequest,
    ProtectionScopesResponse,
    PurviewTextContent,
    deserialize_flag,
    serialize_flag,
)


class TestFlagOperations:
    """フラグのシリアライズとデシリアライズ操作をテスト。"""

    def test_protection_scope_activities_flag_combination(self) -> None:
        """フラグの結合をテスト。"""
        combined = ProtectionScopeActivities.UPLOAD_TEXT | ProtectionScopeActivities.UPLOAD_FILE
        assert combined.value == 3
        assert ProtectionScopeActivities.UPLOAD_TEXT in combined
        assert ProtectionScopeActivities.UPLOAD_FILE in combined

    def test_deserialize_flag_with_string(self) -> None:
        """カンマ区切り文字列からのフラグのデシリアライズをテスト。"""
        mapping = {
            "uploadText": ProtectionScopeActivities.UPLOAD_TEXT,
            "uploadFile": ProtectionScopeActivities.UPLOAD_FILE,
        }

        result = deserialize_flag("uploadText,uploadFile", mapping, ProtectionScopeActivities)
        assert result is not None
        assert ProtectionScopeActivities.UPLOAD_TEXT in result
        assert ProtectionScopeActivities.UPLOAD_FILE in result

    def test_deserialize_flag_with_none(self) -> None:
        """None のデシリアライズは None を返すことをテスト。"""
        mapping = {"uploadText": ProtectionScopeActivities.UPLOAD_TEXT}
        result = deserialize_flag(None, mapping, ProtectionScopeActivities)
        assert result is None

    def test_serialize_flag_with_none(self) -> None:
        """None のシリアライズは None を返すことをテスト。"""
        result = serialize_flag(None, [])
        assert result is None

    def test_serialize_flag_with_values(self) -> None:
        """値を持つフラグのシリアライズをテスト。"""
        flag = ProtectionScopeActivities.UPLOAD_TEXT | ProtectionScopeActivities.UPLOAD_FILE
        ordered = [
            ("uploadText", ProtectionScopeActivities.UPLOAD_TEXT),
            ("uploadFile", ProtectionScopeActivities.UPLOAD_FILE),
        ]
        result = serialize_flag(flag, ordered)
        assert result == "uploadText,uploadFile"


class TestComplexModels:
    """ネストされた構造を持つ複雑なモデルをテスト。"""

    def test_content_to_process_with_nested_structures(self) -> None:
        """すべてのネストされた構造を持つ ContentToProcess をテスト。"""
        text_content = PurviewTextContent(data="Test")
        metadata = ProcessConversationMetadata(
            identifier="msg-1",
            content=text_content,
            name="Test",
            is_truncated=False,
        )

        activity_meta = ActivityMetadata(activity=Activity.UPLOAD_TEXT)
        device_meta = DeviceMetadata(
            operating_system_specifications=OperatingSystemSpecifications(
                operating_system_platform="Windows", operating_system_version="10"
            )
        )
        integrated_app = IntegratedAppMetadata(name="App", version="1.0")
        location = PolicyLocation(data_type="microsoft.graph.policyLocationApplication", value="app-id")
        protected_app = ProtectedAppMetadata(name="Protected", version="1.0", application_location=location)

        content = ContentToProcess(
            content_entries=[metadata],
            activity_metadata=activity_meta,
            device_metadata=device_meta,
            integrated_app_metadata=integrated_app,
            protected_app_metadata=protected_app,
        )

        assert len(content.content_entries) == 1
        assert content.activity_metadata.activity == Activity.UPLOAD_TEXT
        assert content.device_metadata.operating_system_specifications.operating_system_platform == "Windows"
        assert content.integrated_app_metadata.name == "App"
        assert content.protected_app_metadata.name == "Protected"


class TestRequestResponseSerialization:
    """エイリアスを使ったリクエスト/レスポンスのシリアライズをテスト。"""

    def test_protection_scopes_request_serialization(self) -> None:
        """ProtectionScopesRequest がアクティビティを正しくシリアライズすることをテスト。"""
        location = PolicyLocation(data_type="microsoft.graph.policyLocationApplication", value="app-id")

        request = ProtectionScopesRequest(
            user_id="user-123",
            tenant_id="tenant-456",
            activities=ProtectionScopeActivities.UPLOAD_TEXT | ProtectionScopeActivities.UPLOAD_FILE,
            locations=[location],
        )

        dumped = request.model_dump(by_alias=True, exclude_none=True, mode="json")

        assert "activities" in dumped
        assert isinstance(dumped["activities"], str)
        assert "uploadText" in dumped["activities"]


class TestModelDeserialization:
    """API レスポンスからのモデルのデシリアライズをテスト。"""

    def test_protection_scopes_response_deserialization(self) -> None:
        """ProtectionScopesResponse が 'value' を 'scopes' にデシリアライズすることをテスト。"""
        api_data = {
            "scopeIdentifier": "scope-123",
            "value": [
                {
                    "activities": "uploadText,downloadText",
                    "locations": [{"@odata.type": "location.type", "value": "/path"}],
                    "policyActions": [{"action": "warn", "restrictionAction": "blockAccess"}],
                    "executionMode": "evaluateInline",
                }
            ],
        }

        response = ProtectionScopesResponse.model_validate(api_data)

        assert response.scope_identifier == "scope-123"
        assert response.scopes is not None
        assert len(response.scopes) == 1
        assert response.scopes[0].execution_mode == "evaluateInline"

    def test_process_content_response_deserialization(self) -> None:
        """ProcessContentResponse がエイリアス付きフィールドを正しくデシリアライズすることをテスト。"""
        api_data = {
            "id": "response-123",
            "protectionScopeState": "blocked",
            "policyActions": [{"action": "block", "restrictionAction": "blockAccess"}],
        }

        response = ProcessContentResponse.model_validate(api_data)

        assert response.id == "response-123"
        assert response.protection_scope_state == "blocked"
        assert len(response.policy_actions) == 1

    def test_content_serialization_uses_aliases(self) -> None:
        """ContentToProcess が camelCase エイリアスでシリアライズされることをテスト。"""
        text_content = PurviewTextContent(data="Test")
        metadata = ProcessConversationMetadata(
            identifier="msg-1",
            content=text_content,
            name="Test",
            is_truncated=False,
        )

        activity_meta = ActivityMetadata(activity=Activity.UPLOAD_TEXT)
        device_meta = DeviceMetadata(
            operating_system_specifications=OperatingSystemSpecifications(
                operating_system_platform="Windows", operating_system_version="10"
            )
        )
        integrated_app = IntegratedAppMetadata(name="App", version="1.0")
        location = PolicyLocation(data_type="microsoft.graph.policyLocationApplication", value="app-id")
        protected_app = ProtectedAppMetadata(name="Protected", version="1.0", application_location=location)

        content = ContentToProcess(
            content_entries=[metadata],
            activity_metadata=activity_meta,
            device_metadata=device_meta,
            integrated_app_metadata=integrated_app,
            protected_app_metadata=protected_app,
        )

        dumped = content.model_dump(by_alias=True, exclude_none=True, mode="json")

        assert "contentEntries" in dumped
        assert "activityMetadata" in dumped
        assert "deviceMetadata" in dumped
        assert "integratedAppMetadata" in dumped
        assert "protectedAppMetadata" in dumped

    def test_process_content_request_excludes_private_fields(self) -> None:
        """ProcessContentRequest がシリアライズ時にプライベートフィールドを除外することをテスト。"""
        text_content = PurviewTextContent(data="Test")
        metadata = ProcessConversationMetadata(
            identifier="msg-1",
            content=text_content,
            name="Test",
            is_truncated=False,
        )

        activity_meta = ActivityMetadata(activity=Activity.UPLOAD_TEXT)
        device_meta = DeviceMetadata(
            operating_system_specifications=OperatingSystemSpecifications(
                operating_system_platform="Windows", operating_system_version="10"
            )
        )
        integrated_app = IntegratedAppMetadata(name="App", version="1.0")
        location = PolicyLocation(data_type="microsoft.graph.policyLocationApplication", value="app-id")
        protected_app = ProtectedAppMetadata(name="Protected", version="1.0", application_location=location)

        content = ContentToProcess(
            content_entries=[metadata],
            activity_metadata=activity_meta,
            device_metadata=device_meta,
            integrated_app_metadata=integrated_app,
            protected_app_metadata=protected_app,
        )

        request = ProcessContentRequest(
            content_to_process=content,
            user_id="user-123",
            tenant_id="tenant-456",
            correlation_id="corr-789",
        )

        dumped = request.model_dump(by_alias=True, exclude_none=True, mode="json")

        # 除外されたフィールドが存在しないことをチェック。
        assert "user_id" not in dumped
        assert "tenant_id" not in dumped
        assert "correlation_id" not in dumped

        # content が存在することをチェック。
        assert "contentToProcess" in dumped
