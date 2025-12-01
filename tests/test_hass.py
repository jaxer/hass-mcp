import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock, ANY
import json
import httpx
from typing import Dict, List, Any

from app.hass import (
    get_entity_state, call_service, get_entities, get_automations, handle_api_errors,
    list_labels, create_label, update_label, delete_label, update_entity_labels,
    reload_home_assistant, check_home_assistant_config, list_automation_traces,
    get_automation_trace
)

class TestHassAPI:
    """Test the Home Assistant API functions."""

    @pytest.mark.asyncio
    async def test_get_entities(self, mock_config):
        """Test getting all entities."""
        # Mock response data
        mock_states = [
            {"entity_id": "light.living_room", "state": "on", "attributes": {"brightness": 255}},
            {"entity_id": "switch.kitchen", "state": "off", "attributes": {}}
        ]
        
        # Create mock response
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = mock_states
        
        # Create properly awaitable mock
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        
        # Setup client mocking
        with patch('app.hass.get_client', return_value=mock_client):
            with patch('app.hass.HA_URL', mock_config["hass_url"]):
                with patch('app.hass.HA_TOKEN', mock_config["hass_token"]):
                            # Test function
                            states = await get_entities()
                            
                            # Assertions
                            assert isinstance(states, list)
                            assert len(states) == 2
                            
                            # Verify API was called correctly
                            mock_client.get.assert_called_once()
                            called_url = mock_client.get.call_args[0][0]
                            assert called_url == f"{mock_config['hass_url']}/api/states"

    @pytest.mark.asyncio
    async def test_get_entity_state(self, mock_config):
        """Test getting a specific entity state."""
        # Mock response data
        mock_state = {"entity_id": "light.living_room", "state": "on"}
        
        # Create mock response
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = mock_state
        
        # Create properly awaitable mock
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        
        # Patch the client
        with patch('app.hass.get_client', return_value=mock_client):
            with patch('app.hass.HA_URL', mock_config["hass_url"]):
                with patch('app.hass.HA_TOKEN', mock_config["hass_token"]):
                    # Test function - use_cache parameter has been removed
                    state = await get_entity_state("light.living_room")
                    
                    # Assertions
                    assert isinstance(state, dict)
                    assert state["entity_id"] == "light.living_room"
                    assert state["state"] == "on"
                    
                    # Verify API was called correctly
                    mock_client.get.assert_called_once()
                    called_url = mock_client.get.call_args[0][0]
                    assert called_url == f"{mock_config['hass_url']}/api/states/light.living_room"

    @pytest.mark.asyncio
    async def test_call_service(self, mock_config):
        """Test calling a service."""
        domain = "light"
        service = "turn_on"
        data = {"entity_id": "light.living_room", "brightness": 255}
        
        # Create mock response
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"result": "ok"}
        
        # Create properly awaitable mock
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        
        # Patch the client
        with patch('app.hass.get_client', return_value=mock_client):
            with patch('app.hass.HA_URL', mock_config["hass_url"]):
                with patch('app.hass.HA_TOKEN', mock_config["hass_token"]):
                        # Test function
                        result = await call_service(domain, service, data)
                        
                        # Assertions
                        assert isinstance(result, dict)
                        assert result["result"] == "ok"
                        
                        # Verify API was called correctly
                        mock_client.post.assert_called_once()
                        called_url = mock_client.post.call_args[0][0]
                        called_data = mock_client.post.call_args[1].get('json')
                        assert called_url == f"{mock_config['hass_url']}/api/services/{domain}/{service}"
                        assert called_data == data

    @pytest.mark.asyncio
    async def test_call_service_retries_plural_domains(self, mock_config, mock_get_client):
        """Service calls should retry with a singularized domain when HA returns 404."""
        domain = "automations"
        service = "reload"
        data = {}
        
        request = httpx.Request("POST", f"{mock_config['hass_url']}/api/services/{domain}/{service}")
        response = httpx.Response(status_code=404, request=request)
        http_error = httpx.HTTPStatusError("not found", request=request, response=response)
        
        first_response = MagicMock()
        first_response.raise_for_status.side_effect = http_error
        first_response.json.return_value = {}
        
        second_response = MagicMock()
        second_response.raise_for_status = MagicMock()
        second_response.json.return_value = {"status": "ok"}
        
        mock_get_client.post = AsyncMock(side_effect=[first_response, second_response])
        
        with patch('app.hass.HA_URL', mock_config["hass_url"]), patch('app.hass.HA_TOKEN', mock_config["hass_token"]):
            result = await call_service(domain, service, data)
        
        assert result == {"status": "ok"}
        assert mock_get_client.post.call_count == 2
        first_url = mock_get_client.post.call_args_list[0][0][0]
        second_url = mock_get_client.post.call_args_list[1][0][0]
        assert first_url.endswith("/api/services/automations/reload")
        assert second_url.endswith("/api/services/automation/reload")

    @pytest.mark.asyncio
    async def test_reload_home_assistant(self, mock_config):
        """Reload helper should call the right Home Assistant service."""
        mock_result = {"status": "ok"}

        with patch('app.hass.HA_TOKEN', mock_config["hass_token"]):
            with patch('app.hass.call_service', AsyncMock(return_value=mock_result)) as mock_call:
                result = await reload_home_assistant()

                assert result == mock_result
                mock_call.assert_awaited_once_with("homeassistant", "reload_core_config", {})

    @pytest.mark.asyncio
    async def test_check_home_assistant_config(self, mock_config):
        """Config check helper should hit the correct endpoint."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"result": "valid"}

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch('app.hass.get_client', return_value=mock_client):
            with patch('app.hass.HA_URL', mock_config["hass_url"]):
                with patch('app.hass.HA_TOKEN', mock_config["hass_token"]):
                    with patch('app.hass.get_ha_headers', return_value={"Authorization": "Bearer test"}) as mock_headers:
                        result = await check_home_assistant_config()

        assert result == {"result": "valid"}
        mock_client.post.assert_awaited_once()
        called_url = mock_client.post.call_args[0][0]
        assert called_url == f"{mock_config['hass_url']}/api/config/core/check_config"
        called_headers = mock_client.post.call_args[1]["headers"]
        assert called_headers == {"Authorization": "Bearer test"}

    @pytest.mark.asyncio
    async def test_get_automations(self, mock_config):
        """Test getting automations from the states API."""
        # Mock states response with automation entities
        mock_automation_states = [
            {
                "entity_id": "automation.morning_lights", 
                "state": "on", 
                "attributes": {
                    "friendly_name": "Turn on lights in the morning",
                    "last_triggered": "2025-03-15T07:00:00Z"
                }
            },
            {
                "entity_id": "automation.night_lights",
                "state": "off",
                "attributes": {
                    "friendly_name": "Turn off lights at night"
                }
            }
        ]
        
        # Patch the token to avoid the "No token" error
        with patch('app.hass.HA_TOKEN', mock_config["hass_token"]):
            with patch('app.hass.HA_URL', mock_config["hass_url"]):
                # For get_automations we need to mock the get_entities function
                with patch('app.hass.get_entities', AsyncMock(return_value=mock_automation_states)):
                    # Test function
                    automations = await get_automations()
                    
                    # Assertions
                    assert isinstance(automations, list)
                    assert len(automations) == 2
                    
                    # Verify contents of first automation
                    assert automations[0]["entity_id"] == "automation.morning_lights"
                    assert automations[0]["state"] == "on"
                    assert automations[0]["alias"] == "Turn on lights in the morning"
                    assert automations[0]["last_triggered"] == "2025-03-15T07:00:00Z"
                    
                # Test error response
                with patch('app.hass.get_entities', AsyncMock(return_value={"error": "HTTP error: 404 - Not Found"})):
                    # Test function with error
                    automations = await get_automations()
                    
                    # In our new implementation, it should pass through the error
                    assert isinstance(automations, dict)
                    assert "error" in automations
                    assert "404" in automations["error"]

    @pytest.mark.asyncio
    async def test_get_entity_history(self, mock_config):
        """Test getting entity history."""
        entity_id = "sensor.temperature"
        hours = 24

        # Mock response data for history
        mock_history_data = [
            [
                {
                    "state": "25.0",
                    "last_changed": "2025-06-30T10:00:00.000Z",
                    "attributes": {"unit_of_measurement": "°C"}
                },
                {
                    "state": "26.0",
                    "last_changed": "2025-06-30T11:00:00.000Z",
                    "attributes": {"unit_of_measurement": "°C"}
                }
            ]
        ]

        # Create mock response
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = mock_history_data

        # Create properly awaitable mock
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        # Patch the client and HA_URL/HA_TOKEN
        with patch('app.hass.get_client', return_value=mock_client):
            with patch('app.hass.HA_URL', mock_config["hass_url"]):
                with patch('app.hass.HA_TOKEN', mock_config["hass_token"]):
                    from app.hass import get_entity_history
                    history = await get_entity_history(entity_id, hours)

                    # Assertions
                    assert isinstance(history, list)
                    assert len(history) == 1  # History API returns list of lists
                    assert len(history[0]) == 2
                    assert history[0][0]["state"] == "25.0"
                    assert history[0][1]["state"] == "26.0"

                    # Verify API was called correctly
                    mock_client.get.assert_called_once()
                    called_url = mock_client.get.call_args[0][0]
                    assert f"{mock_config['hass_url']}/api/history/period/" in called_url
                    assert mock_client.get.call_args[1]["params"]["filter_entity_id"] == entity_id

    def test_handle_api_errors_decorator(self):
        """Test the handle_api_errors decorator."""
        from app.hass import handle_api_errors
        import inspect
        
        # Create a simple test function with a Dict return annotation
        @handle_api_errors
        async def test_dict_function() -> Dict:
            """Test function that returns a dict."""
            return {}
        
        # Create a simple test function with a str return annotation
        @handle_api_errors
        async def test_str_function() -> str:
            """Test function that returns a string."""
            return ""
        
        # Verify that both functions have their return type annotations preserved
        assert "Dict" in str(inspect.signature(test_dict_function).return_annotation)
        assert "str" in str(inspect.signature(test_str_function).return_annotation)
        
        # Verify that both functions have a docstring
        assert test_dict_function.__doc__ == "Test function that returns a dict."
        assert test_str_function.__doc__ == "Test function that returns a string."

    @pytest.mark.asyncio
    async def test_list_labels(self, mock_config):
        """Ensure list_labels triggers the websocket registry command."""
        mock_labels = [{"label_id": "abc", "name": "Lighting"}]
        ws_mock = AsyncMock(return_value=mock_labels)

        with patch('app.hass._call_ws_command', ws_mock):
            with patch('app.hass.HA_URL', mock_config["hass_url"]):
                with patch('app.hass.HA_TOKEN', mock_config["hass_token"]):
                    labels = await list_labels()
                    assert labels == mock_labels
                    ws_mock.assert_awaited_once_with("config/label_registry/list")

    @pytest.mark.asyncio
    async def test_create_label(self, mock_config):
        """Ensure create_label sends the websocket payload with optional fields."""
        mock_result = {"label_id": "abc", "name": "Lighting"}
        ws_mock = AsyncMock(return_value=mock_result)

        with patch('app.hass._call_ws_command', ws_mock):
            with patch('app.hass.HA_URL', mock_config["hass_url"]):
                with patch('app.hass.HA_TOKEN', mock_config["hass_token"]):
                    result = await create_label("Lighting", icon="mdi:lightbulb", color="#FFEE00")
                    assert result == mock_result
                    ws_mock.assert_awaited_once_with(
                        "config/label_registry/create",
                        {"name": "Lighting", "icon": "mdi:lightbulb", "color": "#FFEE00"}
                    )

    @pytest.mark.asyncio
    async def test_update_label(self, mock_config):
        """Ensure update_label sends only provided fields via websocket and enforces validation."""
        mock_result = {"label_id": "abc", "name": "Updated"}
        ws_mock = AsyncMock(return_value=mock_result)

        with patch('app.hass._call_ws_command', ws_mock):
            with patch('app.hass.HA_URL', mock_config["hass_url"]):
                with patch('app.hass.HA_TOKEN', mock_config["hass_token"]):
                    result = await update_label("abc", name="Updated")
                    assert result == mock_result
                    ws_mock.assert_awaited_once_with(
                        "config/label_registry/update",
                        {"name": "Updated", "label_id": "abc"}
                    )

        # Ensure we short-circuit when no new fields are provided
        with patch('app.hass.HA_TOKEN', mock_config["hass_token"]):
            no_update = await update_label("abc")
            assert "error" in no_update

    @pytest.mark.asyncio
    async def test_delete_label(self, mock_config):
        """Ensure delete_label sends the expected websocket payload."""
        mock_result = {"label_id": "abc", "deleted": True}
        ws_mock = AsyncMock(return_value=mock_result)

        with patch('app.hass._call_ws_command', ws_mock):
            with patch('app.hass.HA_URL', mock_config["hass_url"]):
                with patch('app.hass.HA_TOKEN', mock_config["hass_token"]):
                    result = await delete_label("abc")
                    assert result == mock_result
                    ws_mock.assert_awaited_once_with(
                        "config/label_registry/delete",
                        {"label_id": "abc"}
                    )

    @pytest.mark.asyncio
    async def test_update_entity_labels(self, mock_config):
        """Ensure entity label assignment uses the entity registry websocket command."""
        mock_result = {"entity_id": "automation.test", "labels": ["abc"]}
        ws_mock = AsyncMock(return_value=mock_result)

        with patch('app.hass._call_ws_command', ws_mock):
            with patch('app.hass.HA_URL', mock_config["hass_url"]):
                with patch('app.hass.HA_TOKEN', mock_config["hass_token"]):
                    result = await update_entity_labels("automation.test", ["abc"])
                    assert result == mock_result
                    ws_mock.assert_awaited_once_with(
                        "config/entity_registry/update",
                        {"entity_id": "automation.test", "labels": ["abc"]}
                    )

    @pytest.mark.asyncio
    async def test_list_automation_traces(self, mock_config):
        """Ensure automation trace listing uses the websocket trace API and normalizes IDs."""
        mock_traces = [{"run_id": "123", "timestamp": "2025-03-01T00:00:00Z"}]
        ws_mock = AsyncMock(return_value=mock_traces)
        state_mock = AsyncMock(return_value={"attributes": {"id": "1724195401005"}})

        with patch('app.hass.call_websocket_api', ws_mock), \
             patch('app.hass.get_entity_state', state_mock):
            with patch('app.hass.HA_URL', mock_config["hass_url"]):
                with patch('app.hass.HA_TOKEN', mock_config["hass_token"]):
                    traces = await list_automation_traces("automation.morning_lights")
                    assert traces == mock_traces
                    state_mock.assert_awaited_once_with("automation.morning_lights", lean=False)
                    ws_mock.assert_awaited_once_with(
                        "trace/list",
                        domain="automation",
                        item_id="1724195401005"
                    )

        # When called without a filter we should omit item_id and skip entity lookups
        ws_mock.reset_mock()
        state_mock.reset_mock()
        ws_mock.return_value = mock_traces

        with patch('app.hass.call_websocket_api', ws_mock), \
             patch('app.hass.get_entity_state', state_mock):
            with patch('app.hass.HA_URL', mock_config["hass_url"]):
                with patch('app.hass.HA_TOKEN', mock_config["hass_token"]):
                    await list_automation_traces()
                    state_mock.assert_not_called()
                    ws_mock.assert_awaited_once_with(
                        "trace/list",
                        domain="automation"
                    )

    @pytest.mark.asyncio
    async def test_get_automation_trace(self, mock_config):
        """Ensure the detailed trace fetch validates inputs and calls the API."""
        mock_trace = {"domain": "automation", "item_id": "morning_lights", "trace": []}
        ws_mock = AsyncMock(return_value=mock_trace)
        state_mock = AsyncMock(return_value={"attributes": {"id": "1724195401005"}})

        with patch('app.hass.call_websocket_api', ws_mock), \
             patch('app.hass.get_entity_state', state_mock):
            with patch('app.hass.HA_URL', mock_config["hass_url"]):
                with patch('app.hass.HA_TOKEN', mock_config["hass_token"]):
                    trace = await get_automation_trace("automation.morning_lights", "1")
                    assert trace == mock_trace
                    state_mock.assert_awaited_once_with("automation.morning_lights", lean=False)
                    ws_mock.assert_awaited_once_with(
                        "trace/get",
                        domain="automation",
                        item_id="1724195401005",
                        run_id="1"
                    )

        invalid = await get_automation_trace(" ", "1")
        assert "error" in invalid

        missing_run = await get_automation_trace("automation.morning", "")
        assert "error" in missing_run

        # Numeric ids should bypass entity lookups
        ws_mock.reset_mock()
        state_mock.reset_mock()
        ws_mock.return_value = mock_trace

        with patch('app.hass.call_websocket_api', ws_mock), \
             patch('app.hass.get_entity_state', state_mock):
            with patch('app.hass.HA_URL', mock_config["hass_url"]):
                with patch('app.hass.HA_TOKEN', mock_config["hass_token"]):
                    await get_automation_trace("1724195401005", "1")
                    state_mock.assert_not_called()
                    ws_mock.assert_awaited_once_with(
                        "trace/get",
                        domain="automation",
                        item_id="1724195401005",
                        run_id="1"
                    )
