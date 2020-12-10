from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.utils.mjcf_utils import string_to_array


class StairsArena(Arena):
    """Empty workspace."""

    def __init__(self):
        super().__init__(xml_path_completion("arenas/stairs_arena.xml"))
        self.floor = self.worldbody.find("./geom[@name='floor']")
