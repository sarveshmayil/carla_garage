class Config:
    def __init__(self) -> None:
        self.cameras = [
            {
                "type": 'sensor.camera.rgb',
                "position": [-1.5, 0.0, 2.0],  # [x y z] position wrt car
                "rotation": [0.0, 0.0, 0.0],   # [r p y]
                "size": [640, 480],            # [w h] of image
                "fov": 110,
                "id": 'rgb_front'
            }
        ]

        self.lidar = {
            "type": 'sensor.lidar.ray_cast',
            "position": [0.0, 0.0, 2.5],  # [x y z] position wrt car
            "rotation": [0.0, 0.0, -90.0],   # [r p y]
            "rot_freq": 10,  # Hz of lidar sensor
            "id": 'lidar'
        }

