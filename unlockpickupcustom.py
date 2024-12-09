from minigrid.envs import UnlockPickupEnv

class UnlockPickupCustomEnv(UnlockPickupEnv):
    def __init__(self, max_steps: int | None = None, verbose=0, **kwargs):
        super().__init__(max_steps, **kwargs)
        self.verbose = verbose

        self.gave_key_reward = False
        self.gave_open_reward = False
        self.gave_drop_reward = False

    def _gen_grid(self, width, height):
        super(UnlockPickupEnv, self)._gen_grid(width, height)

        # Add a box to the room on the right
        obj, _ = self.add_object(1, 0, kind="box")
        # Make sure the two rooms are directly connected by a locked door
        door, _ = self.add_door(0, 0, 0, locked=True)
        # Add a key to unlock the door
        key = self.add_object(0, 0, "key", door.color)

        self.place_agent(0, 0)

        self.obj = obj
        self.mission = f"pick up the {obj.color} {obj.type}"

        self.door = door
        self.key = key

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        """
        # We don't really need this reward if we are jump start using problem 1
        # picked up an item; if it is key, alter reward
        if action == self.actions.pickup and self.carrying == self.key and (not self.gave_key_reward):
            reward = self._reward() / 5
            self.gave_key_reward = True
        """
        # We still need this if we jump start using problem 1 in order not to lose progress
        # toggled an item; if door is now unlocked and opened, alter reward
        if action == self.actions.toggle \
           and (not self.door.is_locked) and self.door.is_open \
           and (not self.gave_open_reward):
            reward = self._reward() / 3
            self.gave_open_reward = True

            if self.verbose > 0:
                print("gave agent door open reward")

        # Apparently some people are getting messed up by this, so add an extra reward
        # dropped an item; if door is unlocked by the time we drop, alter reward
        if action == self.actions.drop and (not self.door.is_locked) \
           and (not self.gave_drop_reward):
            reward = self._reward() / 5
            self.gave_drop_reward = True

            if self.verbose > 0:
                print("gave agent key drop reward")

        if terminated and self.verbose > 0:
            print("agent completed task")

        return obs, reward, terminated, truncated, info
    
    def reset(self, *, seed = None, options = None):
        if self.verbose > 0:
            print()
        self.gave_open_reward = False
        self.gave_drop_reward = False
        return super().reset(seed=seed, options=options)

