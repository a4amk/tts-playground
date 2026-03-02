import os
import importlib.util
import logging
from typing import Dict, List, Optional, Type
from .interface import TTSPlugin

import threading
from .interface import TTSPlugin

logger = logging.getLogger(__name__)

class PluginManager:
    """
    Handles dynamic discovery and lifecycle management of TTS engines.
    """
    def __init__(self, engines_dir: str = "app/engines"):
        self.engines_dir = engines_dir
        self._plugins: Dict[str, TTSPlugin] = {}
        self._plugin_classes: Dict[str, Type[TTSPlugin]] = {}
        self._lock = threading.Lock()
        self.discover_plugins()

    def discover_plugins(self):
        """
        Scans the engines directory for subdirectories containing a valid plugin.
        Expected structure: app/engines/<name>/engine.py
        """
        with self._lock:
            if not os.path.exists(self.engines_dir):
                logger.error(f"Engines directory not found: {self.engines_dir}")
                return

            for entry in os.scandir(self.engines_dir):
                if entry.is_dir() and not entry.name.startswith("__"):
                    engine_file = os.path.join(entry.path, "engine.py")
                    if os.path.exists(engine_file):
                        self._load_plugin_from_file(entry.name, engine_file)

    def _load_plugin_from_file(self, plugin_name: str, file_path: str):
        """
        Dynamically imports the engine.py file and finds the TTSPlugin subclass.
        """
        try:
            module_name = f"app.engines.{plugin_name}.engine"
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find the class that inherits from TTSPlugin but isn't TTSPlugin itself
            for attr in dir(module):
                cls = getattr(module, attr)
                if (isinstance(cls, type) and 
                    issubclass(cls, TTSPlugin) and 
                    cls is not TTSPlugin):
                    
                    # Instantiate (but don't call load() yet)
                    plugin_instance = cls()
                    self._plugins[plugin_instance.id] = plugin_instance
                    self._plugin_classes[plugin_instance.id] = cls
                    logger.info(f"Discovered plugin: {plugin_instance.display_name} ({plugin_instance.id})")
                    return

        except Exception as e:
            logger.error(f"Failed to load plugin from {file_path}: {e}")

    def get_plugin(self, plugin_id: str) -> Optional[TTSPlugin]:
        """Returns the plugin instance, or None if not found."""
        return self._plugins.get(plugin_id)

    def get_all_plugins(self) -> List[TTSPlugin]:
        """Returns all discovered plugin instances."""
        return list(self._plugins.values())

    def get_all_ids(self) -> List[str]:
        """Returns IDs of all discovered plugins."""
        return list(self._plugins.keys())

# Global Manager Instance
plugin_manager = PluginManager()
