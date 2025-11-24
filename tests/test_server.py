import pytest
import json
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import os
import sys
import uuid

# Add the app directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
app_dir = os.path.join(parent_dir, "app")
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

class TestMCPServer:
    """Test the MCP server functionality."""
    
    def test_server_version(self):
        """Test that the server has a version attribute."""
        # Import the server module directly without mocking
        # This ensures we're testing the actual code
        from app.server import mcp
        
        # All MCP servers should have a name, and it should be "Hass-MCP"
        assert hasattr(mcp, "name")
        assert mcp.name == "Hass-MCP"

    def test_async_handler_decorator(self):
        """Test the async_handler decorator."""
        # Import the decorator
        from app.server import async_handler
        
        # Create a test async function
        async def test_func(arg1, arg2=None):
            return f"{arg1}_{arg2}"
        
        # Apply the decorator
        decorated_func = async_handler("test_command")(test_func)
        
        # Run the decorated function
        result = asyncio.run(decorated_func("val1", arg2="val2"))
        
        # Verify the result
        assert result == "val1_val2"
    
    def test_tool_functions_exist(self):
        """Test that tool functions exist in the server module."""
        # Import the server module directly
        import app.server
        
        # List of expected tool functions
        expected_tools = [
            "get_version",
            "get_entity",
            "list_entities",
            "entity_action",
            "domain_summary_tool",  # Domain summaries tool
            "call_service_tool",
            "reload_ha",
            "restart_ha",
            "list_automations",
            "get_history",
            "get_history_range",
            "get_statistics",
            "get_statistics_range",
            "get_error_log",
            "list_labels_tool",
            "create_label_tool",
            "update_label_tool",
            "delete_label_tool",
            "set_entity_labels",
        ]
        
        # Check that each expected tool function exists
        for tool_name in expected_tools:
            assert hasattr(app.server, tool_name)
            assert callable(getattr(app.server, tool_name))
    
    def test_resource_functions_exist(self):
        """Test that resource functions exist in the server module."""
        # Import the server module directly
        import app.server
        
        # List of expected resource functions - Use only the ones actually in server.py
        expected_resources = [
            "get_entity_resource", 
            "get_entity_resource_detailed",
            "get_all_entities_resource", 
            "list_states_by_domain_resource",     # Domain-specific resource
            "search_entities_resource_with_limit"  # Search resource with limit parameter
        ]
        
        # Check that each expected resource function exists
        for resource_name in expected_resources:
            assert hasattr(app.server, resource_name)
            assert callable(getattr(app.server, resource_name))
            
    @pytest.mark.asyncio
    async def test_list_automations_error_handling(self):
        """Test that list_automations handles errors properly."""
        from app.server import list_automations
        
        # Mock the get_automations function with different scenarios
        with patch("app.server.get_automations") as mock_get_automations:
            # Case 1: Test with 404 error response format (list with single dict with error key)
            mock_get_automations.return_value = [{"error": "HTTP error: 404 - Not Found"}]
            
            result = await list_automations()
            assert isinstance(result, dict)
            assert result["count"] == 0
            assert result["automations"] == []
            assert "HTTP error" in result.get("error", "")
            
            # Case 2: Test with dict error response
            mock_get_automations.return_value = {"error": "HTTP error: 404 - Not Found"}
            
            result = await list_automations()
            assert isinstance(result, dict)
            assert result["count"] == 0
            assert result["automations"] == []
            assert "HTTP error" in result.get("error", "")
            
            # Case 3: Test with unexpected error
            mock_get_automations.side_effect = Exception("Unexpected error")
            
            result = await list_automations()
            assert isinstance(result, dict)
            assert result["count"] == 0
            assert result["automations"] == []
            assert "Unexpected error" in result.get("error", "")
            
            # Case 4: Test with successful response
            mock_automations = [
                {
                    "id": "morning_lights",
                    "entity_id": "automation.morning_lights",
                    "state": "on",
                    "alias": "Turn on lights in the morning"
                }
            ]
            mock_get_automations.side_effect = None
            mock_get_automations.return_value = mock_automations
            
            result = await list_automations()
            assert isinstance(result, dict)
            assert result["count"] == 1
            assert result["automations"][0]["id"] == "morning_lights"
            assert "error" not in result

    @pytest.mark.asyncio
    async def test_call_service_tool_wraps_list_response(self):
        """call_service_tool should wrap non-dict responses for MCP clients."""
        from app.server import call_service_tool
        
        with patch("app.server.call_service", AsyncMock(return_value=[])) as mock_call:
            result = await call_service_tool("automation", "reload")
            mock_call.assert_awaited_once_with("automation", "reload", {})
            assert result == {"response": []}

    @pytest.mark.asyncio
    async def test_reload_ha_runs_check_before_reload(self):
        """reload_ha should run the config check and include both results."""
        from app.server import reload_ha

        check_payload = {"result": "valid"}
        reload_payload = {"status": "reloaded"}

        with patch("app.server.check_home_assistant_config", AsyncMock(return_value=check_payload)) as mock_check, \
             patch("app.server.reload_home_assistant", AsyncMock(return_value=reload_payload)) as mock_reload:
            result = await reload_ha()

        assert result == {"check_result": check_payload, "reload_result": reload_payload}
        mock_check.assert_awaited_once()
        mock_reload.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_reload_ha_aborts_on_invalid_config(self):
        """reload_ha should not reload when the config check fails."""
        from app.server import reload_ha

        invalid_payload = {"result": "invalid", "errors": "Bad config"}

        with patch("app.server.check_home_assistant_config", AsyncMock(return_value=invalid_payload)) as mock_check, \
             patch("app.server.reload_home_assistant", AsyncMock()) as mock_reload:
            result = await reload_ha()

        assert result["error"] == "Bad config"
        assert result["check_result"] == invalid_payload
        mock_check.assert_awaited_once()
        mock_reload.assert_not_called()
            
    def test_tools_have_proper_docstrings(self):
        """Test that tool functions have proper docstrings"""
        # Import the server module directly
        import app.server
        
        # List of expected tool functions
        tool_functions = [
            "get_version",
            "get_entity",
            "list_entities",
            "entity_action",
            "domain_summary_tool",
            "call_service_tool",
            "reload_ha",
            "restart_ha",
            "list_automations",
            "search_entities_tool",
            "system_overview",
            "get_history",
            "get_history_range",
            "get_statistics",
            "get_statistics_range",
            "get_error_log",
            "list_labels_tool",
            "create_label_tool",
            "update_label_tool",
            "delete_label_tool",
            "set_entity_labels",
        ]
        
        # Check that each tool function has a proper docstring and exists
        for tool_name in tool_functions:
            assert hasattr(app.server, tool_name), f"{tool_name} function missing"
            tool_function = getattr(app.server, tool_name)
            assert tool_function.__doc__ is not None, f"{tool_name} missing docstring"
            assert len(tool_function.__doc__.strip()) > 10, f"{tool_name} has insufficient docstring"
    
    def test_prompt_functions_exist(self):
        """Test that prompt functions exist in the server module."""
        # Import the server module directly
        import app.server
        
        # List of expected prompt functions
        expected_prompts = [
            "create_automation",
            "debug_automation",
            "troubleshoot_entity"
        ]
        
        # Check that each expected prompt function exists
        for prompt_name in expected_prompts:
            assert hasattr(app.server, prompt_name)
            assert callable(getattr(app.server, prompt_name))
            
    @pytest.mark.asyncio
    async def test_search_entities_resource(self):
        """Test the search_entities_tool function"""
        from app.server import search_entities_tool
        
        # Mock the get_entities function with test data
        mock_entities = [
            {"entity_id": "light.living_room", "state": "on", "attributes": {"friendly_name": "Living Room Light", "brightness": 255}},
            {"entity_id": "light.kitchen", "state": "off", "attributes": {"friendly_name": "Kitchen Light"}}
        ]
        
        with patch("app.server.get_entities", return_value=mock_entities) as mock_get:
            # Test search with a valid query
            result = await search_entities_tool(query="living")
            
            # Verify the function was called with the right parameters including lean format
            mock_get.assert_called_once_with(search_query="living", limit=20, lean=True)
            
            # Check that the result contains the expected entity data
            assert result["count"] == 2
            assert any(e["entity_id"] == "light.living_room" for e in result["results"])
            assert result["query"] == "living"
            
            # Check that domain counts are included
            assert "domains" in result
            assert "light" in result["domains"]
            
            # Test with empty query (returns all entities instead of error)
            result = await search_entities_tool(query="")
            assert "error" not in result
            assert result["count"] > 0
            assert "all entities (no filtering)" in result["query"]
            
            # Test that simplified representation includes domain-specific attributes
            result = await search_entities_tool(query="living")
            assert any("brightness" in e for e in result["results"])
            
            # Test with custom limit as an integer
            mock_get.reset_mock()
            result = await search_entities_tool(query="light", limit=5)
            mock_get.assert_called_once_with(search_query="light", limit=5, lean=True)
            
            # Test with a different limit to ensure it's respected
            mock_get.reset_mock()
            result = await search_entities_tool(query="light", limit=10)
            mock_get.assert_called_once_with(search_query="light", limit=10, lean=True)

    @pytest.mark.asyncio
    async def test_label_tools_delegate(self):
        """Ensure label MCP tools call the underlying helpers."""
        from app.server import (
            list_labels_tool, create_label_tool, update_label_tool,
            delete_label_tool, set_entity_labels
        )
        
        with patch("app.server.list_labels", AsyncMock(return_value=[{"label_id": "abc"}])) as mock_list, \
             patch("app.server.create_label", AsyncMock(return_value={"label_id": "abc"})) as mock_create, \
             patch("app.server.update_label", AsyncMock(return_value={"label_id": "abc", "name": "New"})) as mock_update, \
             patch("app.server.delete_label", AsyncMock(return_value={"status": "deleted"})) as mock_delete, \
             patch("app.server.update_entity_labels", AsyncMock(return_value={"entity_id": "automation.test"})) as mock_set:
            
            assert await list_labels_tool() == {"labels": [{"label_id": "abc"}], "count": 1}
            mock_list.assert_awaited_once()
            
            assert await create_label_tool("Lighting") == {"label_id": "abc"}
            mock_create.assert_awaited_once_with(name="Lighting", icon=None, color=None)
            
            assert await update_label_tool("abc", name="New") == {"label_id": "abc", "name": "New"}
            mock_update.assert_awaited_once_with(label_id="abc", name="New", icon=None, color=None)
            
            assert await delete_label_tool("abc") == {"status": "deleted"}
            mock_delete.assert_awaited_once_with(label_id="abc")
            
            assert await set_entity_labels("automation.test", ["abc"]) == {"entity_id": "automation.test"}
            mock_set.assert_awaited_once_with(entity_id="automation.test", labels=["abc"])
            
    @pytest.mark.asyncio
    async def test_domain_summary_tool(self):
        """Test the domain_summary_tool function"""
        from app.server import domain_summary_tool
        
        # Mock the summarize_domain function
        mock_summary = {
            "domain": "light",
            "total_count": 2,
            "state_distribution": {"on": 1, "off": 1},
            "examples": {
                "on": [{"entity_id": "light.living_room", "friendly_name": "Living Room Light"}],
                "off": [{"entity_id": "light.kitchen", "friendly_name": "Kitchen Light"}]
            },
            "common_attributes": [("friendly_name", 2), ("brightness", 1)]
        }
        
        with patch("app.server.summarize_domain", return_value=mock_summary) as mock_summarize:
            # Test the function
            result = await domain_summary_tool(domain="light", example_limit=3)
            
            # Verify the function was called with the right parameters
            mock_summarize.assert_called_once_with("light", 3)
            
            # Check that the result matches the mock data
            assert result == mock_summary
            
    @pytest.mark.asyncio        
    async def test_get_entity_with_field_filtering(self):
        """Test the get_entity function with field filtering"""
        from app.server import get_entity
        
        # Mock entity data
        mock_entity = {
            "entity_id": "light.living_room",
            "state": "on",
            "attributes": {
                "friendly_name": "Living Room Light",
                "brightness": 255,
                "color_temp": 370
            }
        }
        
        # Mock filtered entity data
        mock_filtered = {
            "entity_id": "light.living_room",
            "state": "on"
        }
        
        # Set up mock for get_entity_state to handle different calls
        with patch("app.server.get_entity_state") as mock_get_state:
            # Configure mock to return different responses based on parameters
            mock_get_state.return_value = mock_filtered
            
            # Test with field filtering
            result = await get_entity(entity_id="light.living_room", fields=["state"])
            
            # Verify the function call with fields parameter
            mock_get_state.assert_called_with("light.living_room", fields=["state"])
            assert result == mock_filtered
            
            # Test with detailed=True
            mock_get_state.reset_mock()
            mock_get_state.return_value = mock_entity
            result = await get_entity(entity_id="light.living_room", detailed=True)
            
            # Verify the function call with detailed parameter
            mock_get_state.assert_called_with("light.living_room", lean=False)
            assert result == mock_entity
            
            # Test default lean mode
            mock_get_state.reset_mock()
            mock_get_state.return_value = mock_filtered
            result = await get_entity(entity_id="light.living_room")
            
        # Verify the function call with lean=True parameter
        mock_get_state.assert_called_with("light.living_room", lean=True)
        assert result == mock_filtered

    @pytest.mark.asyncio
    async def test_get_history_range_tool_success(self):
        """get_history_range should flatten history data and include metadata."""
        from app.server import get_history_range

        history_payload = [
            [
                {"last_changed": "2025-01-01T00:00:00+00:00", "state": "on"},
                {"last_changed": "2025-01-01T01:00:00+00:00", "state": "off"},
            ]
        ]

        with patch(
            "app.server.get_entity_history_range",
            AsyncMock(return_value=history_payload),
        ) as mock_history:
            result = await get_history_range(
                "light.living_room",
                "2025-01-01T00:00:00Z",
                "2025-01-01T02:00:00Z",
            )

            mock_history.assert_awaited_once_with(
                "light.living_room",
                "2025-01-01T00:00:00Z",
                "2025-01-01T02:00:00Z",
                True,
            )
            assert result["entity_id"] == "light.living_room"
            assert result["count"] == 2
            assert result["states"][0]["state"] == "on"
            assert result["first_changed"] == "2025-01-01T00:00:00+00:00"
            assert result["last_changed"] == "2025-01-01T01:00:00+00:00"

    @pytest.mark.asyncio
    async def test_get_history_range_tool_handles_errors(self):
        """get_history_range should bubble up API errors in a structured payload."""
        from app.server import get_history_range

        with patch(
            "app.server.get_entity_history_range",
            AsyncMock(return_value={"error": "boom"}),
        ) as mock_history:
            result = await get_history_range("sensor.temp", "yesterday", "today")

            mock_history.assert_awaited_once_with("sensor.temp", "yesterday", "today", True)
            assert result["entity_id"] == "sensor.temp"
            assert result["error"] == "boom"
            assert result["states"] == []
            assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_get_statistics_tool_success(self):
        """get_statistics should delegate to hass helper and summarize count."""
        from app.server import get_statistics

        stats_payload = {
            "entity_id": "sensor.power",
            "statistics": [
                {"start": 1, "end": 2, "mean": 10.5},
                {"start": 2, "end": 3, "mean": 9.5},
            ],
            "period": "hour",
        }

        with patch(
            "app.server.get_entity_statistics",
            AsyncMock(return_value=stats_payload),
        ) as mock_stats:
            result = await get_statistics("sensor.power", hours=6, period="hour")

            mock_stats.assert_awaited_once_with("sensor.power", 6, "hour")
            assert result["entity_id"] == "sensor.power"
            assert result["count"] == 2
            assert result["statistics"][0]["mean"] == 10.5

    @pytest.mark.asyncio
    async def test_get_statistics_range_tool_handles_value_errors(self):
        """get_statistics_range should capture parsing errors cleanly."""
        from app.server import get_statistics_range

        with patch(
            "app.server.get_entity_statistics_range",
            AsyncMock(side_effect=ValueError("bad date")),
        ) as mock_stats:
            result = await get_statistics_range("sensor.humidity", "invalid")

            mock_stats.assert_awaited_once_with("sensor.humidity", "invalid", None, "hour")
            assert result["entity_id"] == "sensor.humidity"
            assert "bad date" in result["error"]
            assert result["statistics"] == []
            assert result["count"] == 0
