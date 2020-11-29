from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.utils.mjcf_utils import string_to_array


class WalkingArena(Arena):
    """Empty workspace."""

    def __init__(self):
        super().__init__(xml_path_completion("arenas/empty_arena.xml"))
        self.floor = self.worldbody.find("./geom[@name='floor']")

    #OLEG: Leave out for now, maybe reintroduce later only if you want to reconfigure position of
    #      everything compared to the floor (if it is not (0,0,0) then need to start changing the floors
    #      etc
    '''
    def configure_location(self):
        """Configures correct locations for this arena"""
        self.bottom_pos = np.array([0, 0, 0])
        self.floor.set("pos", array_to_string(self.bottom_pos))
        
    def floor_pos(self):
        """
        Grabs the position of the floor

        Returns:
            np.array: (x,y,z) table position
        """
        return string_to_array(self.floor.get("pos"))
    '''