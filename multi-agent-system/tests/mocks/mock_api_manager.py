import unittest.mock

class MockAPIManager:
    def __init__(self):
        self.make_request = unittest.mock.MagicMock()
        # Add any other methods from APIManager that agents might call, if any.
        # For now, make_request is the primary one.

    def set_make_request_response(self, response_data, is_error=False, error_content=None):
        if is_error:
            # Default error structure from APIManager.make_request
            error_response = {
                "status": "error", # This key is not standard in APIManager error returns, 'error' is usually the key for the type
                "error": response_data, # e.g. "HTTPError", "Timeout"
                "message": error_content if error_content else "Simulated error",
                "status_code": 500 # Default status code for simulated error
            }
            if isinstance(response_data, dict) and "status_code" in response_data : # if a full error dict is passed
                 self.make_request.return_value = response_data
            else: # construct from parts
                 self.make_request.return_value = error_response

            # If APIManager is expected to raise exceptions on certain errors:
            # self.make_request.side_effect = SomeExpectedException(response_data)
        else:
            self.make_request.return_value = response_data

    def reset_mocks(self):
        self.make_request.reset_mock()

# Example of how it might be used (optional to include in the file itself)
if __name__ == '__main__':
    mock_api = MockAPIManager()

    # Test setting a success response
    mock_api.set_make_request_response({"data": "success_payload", "status_code": 200})
    success_result = mock_api.make_request("test_service", "test_endpoint", method="GET")
    print(f"Success call result: {success_result}")
    mock_api.make_request.assert_called_once_with("test_service", "test_endpoint", method="GET")
    mock_api.reset_mocks()

    # Test setting an error response (simple message)
    mock_api.set_make_request_response("SimulatedErrorType", is_error=True, error_content="Something went wrong")
    error_result = mock_api.make_request("test_service", "test_error_endpoint", method="POST", data={})
    print(f"Error call result: {error_result}")
    mock_api.make_request.assert_called_once_with("test_service", "test_error_endpoint", method="POST", data={})
    mock_api.reset_mocks()

    # Test setting a full error response dict
    full_error_response = {
        "error": "SpecificErrorType",
        "message": "A detailed error message",
        "status_code": 404,
        "content": {"details": "Resource not found"}
    }
    mock_api.set_make_request_response(full_error_response, is_error=True) # is_error=True helps if logic changes
    error_result_full = mock_api.make_request("test_service", "test_another_error", method="GET")
    print(f"Full error dict result: {error_result_full}")
    assert error_result_full["status_code"] == 404
    assert error_result_full["error"] == "SpecificErrorType"

    print("MockAPIManager basic tests completed.")
