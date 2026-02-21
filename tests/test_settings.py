import unittest
from unittest.mock import patch, MagicMock
from src.settings import Settings

class DummyConstants:
    LOGGER = 'dummy_logger'
    SETTINGS_FILE = 'dummy_settings.json'
    SETTING_KEYS = [
        ('desc', 'key1', 'key2', 'default1', 'default2'),
        ('desc2', 'key3', 'key4', 'default3', 'default4')
    ]
    K_BOARD_SIZE = ('desc', 'key1')
    BOARD_SIZE_LIST = [500, 600]
    K_MODEL_FOLDER = ('desc', 'key1', 'key2')
    K_ALLOW_SWAP = ('Allow Swap', 'key1')
    K_ALLOW_SCL = ('Allow SCL', 'key3')
    K_SMART_ACCEPT = ('Smart Accept', 'key4')
    K_ROTATION = ('Rotation', 'key1', 'key2')
    K_LEVEL = ('Level', 'key1', 'key2')
    K_SMART_ROOT = ('Smart Root', 'key1', 'key2')
    K_TEMPERATURE = ('Temperature', 'key1', 'key2')
    K_ADD_NOISE = ('Add Noise', 'key1', 'key2')
    K_CPUCT = ('CPUCT', 'key1', 'key2')
    K_TRIALS = ('desc', 'key1', 'key3')
    MSG_NO_CONFIG_FILE = 'No config file'
    MSG_ERROR_UPDATING_KEY = 'Error updating key'

import src.settings
src.settings.ct = DummyConstants

class TestSettings(unittest.TestCase):
    @patch('src.settings.logging.getLogger')
    @patch('src.settings.sg.popup')
    @patch('src.settings.jsonload', return_value={'key1': 500, 'key2': 600, 'key3': 'val3', 'key4': 'val4'})
    @patch('src.settings.pathlib.Path.absolute', return_value='dummy_path')
    def test_init_and_defaults(self, mock_abs, mock_jsonload, mock_popup, mock_logger):
        s = Settings()
        self.assertIsInstance(s.settings, dict)
        self.assertIn('key1', s.settings)
        self.assertIn('key2', s.settings)
        self.assertIn('key3', s.settings)
        self.assertIn('key4', s.settings)

    @patch('src.settings.logging.getLogger')
    @patch('src.settings.sg.popup')
    @patch('src.settings.jsonload', return_value={'key1': 500, 'key2': 600, 'key3': 'val3', 'key4': 'val4'})
    @patch('src.settings.pathlib.Path.absolute', return_value='dummy_path')
    def test_get_set(self, mock_abs, mock_jsonload, mock_popup, mock_logger):
        s = Settings()
        s.set('key1', 42)
        self.assertEqual(s.get('key1'), 42)

    @patch('src.settings.logging.getLogger')
    @patch('src.settings.sg.popup')
    @patch('src.settings.jsonload', return_value={'key1': 500, 'key2': 600, 'key3': 'val3', 'key4': 'val4'})
    @patch('src.settings.pathlib.Path.absolute', return_value='dummy_path')
    def test_update(self, mock_abs, mock_jsonload, mock_popup, mock_logger):
        s = Settings()
        s.update('key1', {'key1': '123'})
        self.assertEqual(s.get('key1'), 123)
        s.update('key3', {'key3': '123'})
        self.assertEqual(s.get('key3'), 123)

    @patch('src.settings.logging.getLogger')
    @patch('src.settings.sg.popup')
    @patch('src.settings.jsonload', return_value={'key1': 500, 'key2': 'dummy_path', 'key3': '123', 'key4': 'val4'})
    @patch('src.settings.pathlib.Path')
    def test_same_models(self, mock_path, mock_jsonload, mock_popup, mock_logger):
        s = Settings()
        # Create two mock Path objects
        mock_path.side_effect = lambda val: MagicMock(as_posix=lambda: val, absolute=lambda: val)
        s.set('key1', 'dummy_path')
        s.set('key2', 'dummy_path')
        self.assertTrue(s.same_models())
        s.set('key2', 'other_path')
        self.assertFalse(s.same_models())

    @patch('src.settings.jsondump')
    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data='{"key1": 500, "key2": 600, "key3": "val3", "key4": "val4"}')
    @patch('src.settings.logging.getLogger')
    @patch('src.settings.sg.popup')
    @patch('src.settings.jsonload', return_value={'key1': 500, 'key2': 600, 'key3': 'val3', 'key4': 'val4'})
    @patch('src.settings.pathlib.Path.absolute', return_value='dummy_path')
    def test_save(self, mock_abs, mock_jsonload, mock_popup, mock_logger, mock_open, mock_jsondump):
        s = Settings()
        s.settings['key1'] = 'val1'
        s.settings['key2'] = 'val2'
        s.save()
        mock_open.assert_called_with('dummy_settings.json', 'w')
        mock_jsondump.assert_called()

    @patch('src.settings.logging.getLogger')
    @patch('src.settings.sg.popup')
    @patch('src.settings.jsonload', return_value={'key1': 500, 'key2': 600, 'key3': 'val3', 'key4': 'val4'})
    @patch('src.settings.pathlib.Path.absolute', return_value='dummy_path')
    def test_get_tooltip(self, mock_abs, mock_jsonload, mock_popup, mock_logger):
        s = Settings()
        s.set('key1', 'val1')
        s.set('key2', 'val2')
        tooltip = s.get_tooltip(2)
        self.assertIsInstance(tooltip, str)

if __name__ == '__main__':
    unittest.main()

