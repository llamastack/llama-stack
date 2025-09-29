import asyncio
import re
from typing import List
from mcp import ListToolsResult
import pytest
from unittest.mock import AsyncMock, patch
import json

from llama_stack.apis.tools.tools import ToolParameter
from llama_stack.providers.utils.tools.mcp import list_mcp_tools


def find_param(params:List[ToolParameter], param_name: str)->ToolParameter| None:
    return next((p for p in params if p.name == param_name), None)

@pytest.mark.asyncio
async def test_list_mcp_tools_with_ref_defs():
    mcp_tools_resp_json = """
    {"tools":[{"name":"book_reservation",
    "inputSchema":{
        "$defs":{
            "FlightInfo":{"properties":{"flight_number":{"description":"Flight number, such as \'HAT001\'.","title":"Flight Number","type":"string"},"date":{"description":"The date for the flight in the format \'YYYY-MM-DD\', such as \'2024-05-01\'.","title":"Date","type":"string"}},"required":["flight_number","date"],"title":"FlightInfo","type":"object"},
            "Passenger":{"properties":{"first_name":{"description":"Passenger\'s first name","title":"First Name","type":"string"},"last_name":{"description":"Passenger\'s last name","title":"Last Name","type":"string"},"dob":{"description":"Date of birth in YYYY-MM-DD format","title":"Dob","type":"string"}},"required":["first_name","last_name","dob"],"title":"Passenger","type":"object"},
            "Payment":{"properties":{"payment_id":{"description":"Unique reference for the payment method in the user\'s payment methods.","title":"Payment Id","type":"string"},"amount":{"description":"Payment amount in dollars","title":"Amount","type":"integer"}},"required":["payment_id","amount"],"title":"Payment","type":"object"}
        },
        "properties":{
            "user_id":{"title":"User Id","type":"string"},
            "origin":{"title":"Origin","type":"string"},
            "destination":{"title":"Destination","type":"string"},
            "flight_type":{"enum":["round_trip","one_way"],"title":"Flight Type","type":"string"},
            "cabin":{"enum":["business","economy","basic_economy"],"title":"Cabin","type":"string"},
            "flights":{"items":{"$ref":"#/$defs/FlightInfo"},"title":"Flights","type":"array"},
            "passengers":{"items":{"anyOf":[{"$ref":"#/$defs/Passenger"},{"additionalProperties":true,"type":"object"}]},"title":"Passengers","type":"array"},
            "payment":{"$ref":"#/$defs/Payment","title":"Payment Methods"},
            "total_baggages":{"title":"Total Baggages","type":"integer"},
            "nonfree_baggages":{"title":"Nonfree Baggages","type":"integer"},
            "insurance":{"enum":["yes","no"],"title":"Insurance","type":"string"}
        },
        "required":["user_id","origin","destination","flight_type","cabin","flights","passengers","payment_methods","total_baggages","nonfree_baggages","insurance"],
        "type":"object"
    },
        
    "outputSchema":{
        "$defs":{
            "Passenger":{"properties":{"first_name":{"description":"Passenger\'s first name","title":"First Name","type":"string"},"last_name":{"description":"Passenger\'s last name","title":"Last Name","type":"string"},"dob":{"description":"Date of birth in YYYY-MM-DD format","title":"Dob","type":"string"}},"required":["first_name","last_name","dob"],"title":"Passenger","type":"object"},
            "Payment":{"properties":{"payment_id":{"description":"Unique reference for the payment method in the user\'s payment methods.","title":"Payment Id","type":"string"},"amount":{"description":"Payment amount in dollars","title":"Amount","type":"integer"}},"required":["payment_id","amount"],"title":"Payment","type":"object"},
            "ReservationFlight":{"properties":{"flight_number":{"description":"Unique flight identifier","title":"Flight Number","type":"string"},"origin":{"description":"IATA code for origin airport","title":"Origin","type":"string"},"destination":{"description":"IATA code for destination airport","title":"Destination","type":"string"},"date":{"description":"Flight date in YYYY-MM-DD format","title":"Date","type":"string"},"price":{"description":"Flight price in dollars.","title":"Price","type":"integer"}},"required":["flight_number","origin","destination","date","price"],"title":"ReservationFlight","type":"object"}
        },
        "properties":{
            "reservation_id":{"description":"Unique identifier for the reservation","title":"Reservation Id","type":"string"},
            "user_id":{"description":"ID of the user who made the reservation","title":"User Id","type":"string"},
            "origin":{"description":"IATA code for trip origin","title":"Origin","type":"string"},"destination":{"description":"IATA code for trip destination","title":"Destination","type":"string"},
            "flight_type":{"description":"Type of trip","enum":["round_trip","one_way"],"title":"Flight Type","type":"string"},
            "cabin":{"description":"Selected cabin class","enum":["business","economy","basic_economy"],"title":"Cabin","type":"string"},
            "flights":{"description":"List of flights in the reservation","items":{"$ref":"#/$defs/ReservationFlight"},"title":"Flights","type":"array"},"passengers":{"description":"List of passengers on the reservation","items":{"$ref":"#/$defs/Passenger"},"title":"Passengers","type":"array"},
            "payment_history":{"description":"History of payments for this reservation","items":{"$ref":"#/$defs/Payment"},"title":"Payment History","type":"array"},
            "created_at":{"description":"Timestamp when reservation was created in the format YYYY-MM-DDTHH:MM:SS","title":"Created At","type":"string"},
            "total_baggages":{"description":"Total number of bags in reservation","title":"Total Baggages","type":"integer"},
            "nonfree_baggages":{"description":"Number of paid bags in reservation","title":"Nonfree Baggages","type":"integer"},
            "insurance":{"description":"Whether travel insurance was purchased","enum":["yes","no"],"title":"Insurance","type":"string"},
            "status":{"anyOf":[{"const":"cancelled","type":"string"},{"type":"null"}],"default":null,"description":"Status of the reservation","title":"Status"}
        },
        "required":["reservation_id","user_id","origin","destination","flight_type","cabin","flights","passengers","payment_history","created_at","total_baggages","nonfree_baggages","insurance"],"title":"Reservation","type":"object"},
        "annotations":null,"meta":{"_fastmcp":{"tags":[]}}
    }]}"""
    mcp_tools_resp = ListToolsResult.model_validate_json(mcp_tools_resp_json)

    # Mock the client_wrapper context manager
    mock_session = AsyncMock()
    mock_session.list_tools.return_value = mcp_tools_resp

    mock_cm = AsyncMock()
    mock_cm.__aenter__.return_value = mock_session
    mock_cm.__aexit__.return_value = None

    # Patch the client_wrapper to return our mock context manager
    with patch("llama_stack.providers.utils.tools.mcp.client_wrapper", return_value=mock_cm):
        tools_data = await list_mcp_tools("fake_endpoint", {"Authorization": "Bearer X"})
    tools = tools_data.data
    assert len(tools) == 1
    book_resv = tools[0]
    assert book_resv.name == "book_reservation"
    assert find_param(book_resv.parameters, "payment").properties.payment_id.type == "string"
    assert find_param(book_resv.parameters, "flights").items.properties.flight_number.type == "string"
    assert find_param(book_resv.parameters, "passengers").items.properties.first_name.title == "First Name"
    mock_session.list_tools.assert_awaited_once()

asyncio.run(
    test_list_mcp_tools_with_ref_defs()
)
