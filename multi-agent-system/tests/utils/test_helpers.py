# tests/utils/test_helpers.py
import unittest

try:
    from src.utils.helpers import get_nested_value
except ImportError:
    # Fallback for different execution contexts
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.utils.helpers import get_nested_value


class TestGetNestedValue(unittest.TestCase):

    def setUp(self):
        self.test_dict = {
            "key1": "value1",
            "key2": {
                "nested_key1": "nested_value1",
                "nested_list": [
                    {"list_item_key1": "item0_value1"},
                    {"list_item_key1": "item1_value1", "deep_item": {"final_val": 123}},
                    "simple_list_item",
                    [1000, 2000] # nested list
                ]
            },
            "key3": [10, 20, {"nested_in_list": "found_me"}],
            "a[0]": "key_with_brackets_in_name" # Test for keys that look like list access
        }

    def test_get_simple_key(self):
        self.assertEqual(get_nested_value(self.test_dict, "key1"), "value1")

    def test_get_nested_key(self):
        self.assertEqual(get_nested_value(self.test_dict, "key2.nested_key1"), "nested_value1")

    def test_get_from_list_in_dict(self):
        self.assertEqual(get_nested_value(self.test_dict, "key2.nested_list[0].list_item_key1"), "item0_value1")

    def test_get_deeply_nested_value(self):
        self.assertEqual(get_nested_value(self.test_dict, "key2.nested_list[1].deep_item.final_val"), 123)

    def test_get_simple_list_item_by_index(self):
        self.assertEqual(get_nested_value(self.test_dict, "key2.nested_list[2]"), "simple_list_item")
        
    def test_get_from_nested_list_item_by_index(self):
        self.assertEqual(get_nested_value(self.test_dict, "key2.nested_list[3][1]"), 2000)


    def test_get_from_list_at_root_level_by_index(self):
        self.assertEqual(get_nested_value(self.test_dict, "key3[1]"), 20)

    def test_get_from_dict_in_list_at_root_level(self):
        self.assertEqual(get_nested_value(self.test_dict, "key3[2].nested_in_list"), "found_me")

    def test_key_with_brackets_in_name(self):
        self.assertEqual(get_nested_value(self.test_dict, "a[0]"), "key_with_brackets_in_name")


    # --- Test Default Value and Missing Paths ---
    def test_nonexistent_top_level_key(self):
        self.assertIsNone(get_nested_value(self.test_dict, "nonexistent_key"))
        self.assertEqual(get_nested_value(self.test_dict, "nonexistent_key", default="not_found"), "not_found")

    def test_nonexistent_nested_key(self):
        self.assertIsNone(get_nested_value(self.test_dict, "key2.nonexistent_nested"))
        self.assertEqual(get_nested_value(self.test_dict, "key2.nonexistent_nested", default="missing"), "missing")

    def test_index_out_of_bounds(self):
        self.assertIsNone(get_nested_value(self.test_dict, "key2.nested_list[10]"))
        self.assertEqual(get_nested_value(self.test_dict, "key2.nested_list[10]", default="idx_error"), "idx_error")

    def test_nonexistent_key_in_list_item_dict(self):
        self.assertIsNone(get_nested_value(self.test_dict, "key2.nested_list[0].nonexistent"))
        self.assertEqual(get_nested_value(self.test_dict, "key2.nested_list[0].nonexistent", default="missing_in_list_item"), "missing_in_list_item")

    def test_invalid_index_format(self):
        self.assertIsNone(get_nested_value(self.test_dict, "key2.nested_list[abc].key"))
        self.assertEqual(get_nested_value(self.test_dict, "key2.nested_list[abc].key", default="bad_idx"), "bad_idx")

    def test_accessing_key_on_a_list_directly(self):
        # Trying to access "key2.nested_list.some_key" where nested_list is a list, not a dict
        self.assertIsNone(get_nested_value(self.test_dict, "key2.nested_list.some_key"))
        self.assertEqual(get_nested_value(self.test_dict, "key2.nested_list.some_key", default="not_dict"), "not_dict")

    def test_accessing_index_on_a_dict_directly(self):
        # Trying to access "key2[0]" where key2 is a dict
        self.assertIsNone(get_nested_value(self.test_dict, "key2[0]"))
        self.assertEqual(get_nested_value(self.test_dict, "key2[0]", default="not_list"), "not_list")
        
    def test_accessing_sub_item_on_non_container(self):
        # key1 is "value1" (a string)
        self.assertIsNone(get_nested_value(self.test_dict, "key1.subpath"))
        self.assertEqual(get_nested_value(self.test_dict, "key1.subpath", default="not_container"), "not_container")
        self.assertIsNone(get_nested_value(self.test_dict, "key1[0]")) # String is sequence, but helper might not support string indexing explicitly
        self.assertEqual(get_nested_value(self.test_dict, "key1[0]", default="not_list_or_dict"), "not_list_or_dict")


    def test_empty_path(self):
        self.assertIsNone(get_nested_value(self.test_dict, ""))
        self.assertEqual(get_nested_value(self.test_dict, "", default="empty_path_val"), "empty_path_val")

    def test_path_ends_with_index(self):
        self.assertEqual(get_nested_value(self.test_dict, "key2.nested_list[0]"), {"list_item_key1": "item0_value1"})
    
    def test_root_index_access(self):
        # Current implementation of get_nested_value supports this if the path starts with an index
        # and the data_dict itself is treated as something indexable (which it isn't, it's a Dict)
        # The regex `re.split` for `[0]` on `test_dict` would result in `processed_parts = ['0']`.
        # Then `test_dict['0']` would be attempted, returning `default`.
        self.assertIsNone(get_nested_value(self.test_dict, "[0]"))
        self.assertEqual(get_nested_value(self.test_dict, "[0]", default="root_index_default"), "root_index_default")

    def test_path_with_only_index_for_non_list_at_root(self):
        # This test is to confirm behavior when the dict is not a list but path is just an index
        self.assertEqual(get_nested_value(self.test_dict, "[0]", default="not_a_list_at_root"), "not_a_list_at_root")


if __name__ == '__main__':
    unittest.main()
