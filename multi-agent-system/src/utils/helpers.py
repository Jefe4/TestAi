# src/utils/helpers.py
"""
Utility functions for the multi-agent system.
"""
import re
from typing import Dict, Any, Optional, Union

def get_nested_value(data_dict: Dict[str, Any], path: str, default: Optional[Any] = None) -> Any:
    """
    Retrieves a value from a nested dictionary and list structure using a dot-separated path.

    Args:
        data_dict: The dictionary to traverse.
        path: A dot-separated string representing the path to the desired value.
              List indices can be specified using [index] notation, e.g., "key1.key2[0].key3".
        default: The value to return if the path is not found or invalid.

    Returns:
        The value found at the specified path, or the default value.
    """
    if not path:
        return default

    # Regex to split path by dots or by list indices like [0]
    # It will capture keys and indices separately.
    # e.g., "a.b[0].c" -> ["a", "b", "[0]", "c"]
    # Then we process "[0]" into just "0" for list index.
    path_parts = re.split(r'\.(?![^\[]*\])|(\[[0-9]+\])', path)
    # Filter out None and empty strings that can result from split
    processed_parts: List[str] = []
    for part in path_parts:
        if part is None or part == '':
            continue
        if part.startswith('[') and part.endswith(']'):
            processed_parts.append(part[1:-1]) # Store index as string e.g. "0"
        else:
            processed_parts.append(part)
            
    current_value: Any = data_dict
    for part_str in processed_parts:
        if isinstance(current_value, dict):
            if part_str in current_value:
                current_value = current_value[part_str]
            else:
                return default
        elif isinstance(current_value, list):
            try:
                index = int(part_str)
                if 0 <= index < len(current_value):
                    current_value = current_value[index]
                else:
                    return default # Index out of bounds
            except ValueError:
                return default # Part is not a valid integer index for a list
            except IndexError: # Should be caught by length check, but as a safeguard
                return default
        else:
            # Path tries to go deeper, but current_value is not a dict or list
            return default
            
    return current_value

if __name__ == '__main__':
    test_dict = {
        "key1": "value1",
        "key2": {
            "nested_key1": "nested_value1",
            "nested_list": [
                {"list_item_key1": "item0_value1"},
                {"list_item_key1": "item1_value1", "deep_item": {"final_val": 123}},
                "simple_list_item"
            ]
        },
        "key3": [10, 20, {"nested_in_list": "found_me"}]
    }

    print(f"Path 'key1': {get_nested_value(test_dict, 'key1')}")  # Expected: value1
    print(f"Path 'key2.nested_key1': {get_nested_value(test_dict, 'key2.nested_key1')}") # Expected: nested_value1
    print(f"Path 'key2.nested_list[0].list_item_key1': {get_nested_value(test_dict, 'key2.nested_list[0].list_item_key1')}") # Expected: item0_value1
    print(f"Path 'key2.nested_list[1].deep_item.final_val': {get_nested_value(test_dict, 'key2.nested_list[1].deep_item.final_val')}") # Expected: 123
    print(f"Path 'key2.nested_list[2]': {get_nested_value(test_dict, 'key2.nested_list[2]')}") # Expected: simple_list_item
    print(f"Path 'key3[1]': {get_nested_value(test_dict, 'key3[1]')}") # Expected: 20
    print(f"Path 'key3[2].nested_in_list': {get_nested_value(test_dict, 'key3[2].nested_in_list')}") # Expected: found_me

    # Test missing paths
    print(f"Path 'key_nonexistent': {get_nested_value(test_dict, 'key_nonexistent', default='MISSING')}") # Expected: MISSING
    print(f"Path 'key1.nonexistent': {get_nested_value(test_dict, 'key1.nonexistent', default='MISSING')}") # Expected: MISSING
    print(f"Path 'key2.nested_list[5]': {get_nested_value(test_dict, 'key2.nested_list[5]', default='MISSING INDEX')}") # Expected: MISSING INDEX
    print(f"Path 'key2.nested_list[0].nonexistent': {get_nested_value(test_dict, 'key2.nested_list[0].nonexistent', default='MISSING')}") # Expected: MISSING
    print(f"Path 'key2.nested_list[badindex].key': {get_nested_value(test_dict, 'key2.nested_list[badindex].key', default='BAD INDEX FORMAT')}") # Expected: BAD INDEX FORMAT
    print(f"Path 'key3[0].nonexistent_key': {get_nested_value(test_dict, 'key3[0].nonexistent_key', default='MISSING')}") # Expected: MISSING
    print(f"Path 'key2.nested_list[1][0]': {get_nested_value(test_dict, 'key2.nested_list[1][0]', default='INVALID_ACCESS')}") # Expected: INVALID_ACCESS (item is dict)

    # Test path that tries to access key on a non-dict/list type
    print(f"Path 'key1.subpath': {get_nested_value(test_dict, 'key1.subpath', default='NOT_A_DICT')}") # Expected: NOT_A_DICT
    
    # Test with empty path
    print(f"Path '': {get_nested_value(test_dict, '', default='EMPTY_PATH')}") # Expected: EMPTY_PATH

    # Test with dict that is not at top level
    data_list_of_dicts = [{"id":1, "val": {"a":10}}, {"id":2, "val": {"a":20}}]
    # This utility is designed for a top-level dict. To access items in a list like this,
    # the path would need to start with an index, e.g. "[0].val.a"
    # print(f"Path '[0].val.a': {get_nested_value(data_list_of_dicts, '[0].val.a', default='ERROR')}") # This line will error because data_dict must be Dict.

    # Test for paths like "a.b[0]" where the last part is an index
    print(f"Path 'key2.nested_list[0]': {get_nested_value(test_dict, 'key2.nested_list[0]')}") # Expected: {'list_item_key1': 'item0_value1'}

    # Test for direct index access at root (not supported by current design, path must start with key)
    # print(f"Path '[0]': {get_nested_value(test_dict, '[0]', default='ROOT_INDEX_FAIL')}") # Expected: ROOT_INDEX_FAIL
    # Current implementation: `re.split` for `[0]` would give `['', '[0]', '']`. `processed_parts` would be `['0']`.
    # Then it would try `test_dict['0']` which is not there. So it would return default.
    print(f"Path '[0]': {get_nested_value(test_dict, '[0]', default='ROOT_INDEX_FAIL')}") # Expected: ROOT_INDEX_FAIL (correct)
    
    # Test path like a[0] without dot
    test_dict_no_dot = {"a[0]": "no_dot_value"}
    print(f"Path 'a[0]' (no dot): {get_nested_value(test_dict_no_dot, 'a[0]')}") # Expected: no_dot_value (correct, as 'a[0]' is a single key)

    # Test path like a.b[0].c where b is not a list
    test_dict_b_not_list = {"a": {"b": "not_a_list"}}
    print(f"Path 'a.b[0].c' (b not list): {get_nested_value(test_dict_b_not_list, 'a.b[0].c', default='B_NOT_LIST')}") # Expected: B_NOT_LIST

    # Test path like a.b.c where b is a list but accessed with key
    test_dict_b_is_list = {"a": {"b": [{"c": 100}]}}
    print(f"Path 'a.b.c' (b is list): {get_nested_value(test_dict_b_is_list, 'a.b.c', default='B_IS_LIST_NEEDS_INDEX')}") # Expected: B_IS_LIST_NEEDS_INDEX**File `multi-agent-system/src/utils/helpers.py` created successfully.**

Now, I'll create the test file `multi-agent-system/tests/utils/test_helpers.py` and populate it with unit tests for `get_nested_value`.
