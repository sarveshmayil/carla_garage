import carla


class Metrics:
    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle

        self.collision_count = 0
        self.collision_ids = []

        self.lane_invasion_count = 0

    def setup(self):
        blueprint_library = self.world.get_blueprint_library()

        self.collision_sensor = self.world.spawn_actor(
            blueprint_library.find("sensor.other.collision"),
            carla.Transform(),
            attach_to=self.vehicle,
        )
        self.collision_sensor.listen((lambda event: self._collision_callback(event)))

        self.lane_sensor = self.world.spawn_actor(
            blueprint_library.find("sensor.other.lane_invasion"),
            carla.Transform(),
            attach_to=self.vehicle,
        )
        self.lane_sensor.listen((lambda event: self._lane_invasion_callback(event)))

    def _collision_callback(self, event):
        if event.other_actor.id not in self.collision_ids:
            self.collision_count += 1
            self.collision_ids.append(event.other_actor.id)

    def _lane_invasion_callback(self, event):
        self.lane_invasion_count += 1
