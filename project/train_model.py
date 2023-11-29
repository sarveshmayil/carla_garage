from model import tf_model_minimal
from config import GlobalConfig
import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm
from vehicle import Vehicle
from autonomous_agent import Agent
from agents.navigation.global_route_planner import GlobalRoutePlanner
import carla
import sys
import os
import threading
import traceback

class Trainer():
    def __init__(self) -> None:
        self.EPOCH = 1
        self.LR = 1e-5
        self.device = "cuda"
        
        self.model = None
        self.vehicle = None



    def train(self):
        

        with ClientManager() as client_manager:
            self.vehicle = client_manager.get_vehicle()
            self.model = tf_model_minimal.LidarCenterNet(self.vehicle.model_config).to(self.device)
            state_dict = torch.load(os.path.join(self.vehicle.vehicle_config.model["dir"], self.vehicle.vehicle_config.model["weights"]), map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.backbone.image_encoder.s4.parameters():
                param.requires_grad = True
            


            optim = torch.optim.AdamW(self.model.parameters(), lr=self.LR)

            data_listener = client_manager.data_listener

            pbar = tqdm(range(self.EPOCH))
            self.model.train()
            for epoch in pbar:
                if epoch != 0: client_manager.setup()
                self.vehicle = client_manager.get_vehicle()
                thread_vehicle = threading.Thread(target=client_manager.vehicle.follow_route, args=(5.0, 7.0, True, True)) # (tp_threshold, wp_threshold, visualize, debug)
                thread_vehicle.start()

                avg_loss = 0
                ticks = 0
                while thread_vehicle.is_alive():
                    print("main thread waiting", end='\r')
                    if len(data_listener.data) != 0:
                        optim.zero_grad()
                        data = data_listener.data.pop()
                        preds = self.model(data["rgb"],
                                            data["lidar"],
                                            target_point=data["target_point"], 
                                            ego_vel=data["ego_vel"])
                        truth = data["preds"]

                        loss_wp = torch.mean(torch.abs(preds[2][0] - truth[2][0]))
                        loss_speed = torch.mean(torch.abs(preds[1][0] - truth[1][0]))
                        loss = loss_speed + loss_wp

                        avg_loss += loss
                        ticks += 1

                        loss.backward()
                        optim.step()
                        pbar.set_description(f"loss = {loss}")

                        data_listener.is_listening = True
                tqdm.write(f"epoch = {epoch} | avg_loss = {avg_loss/ticks}")
                
        
    
        return

    def setup_config(self):
        import json, pickle
        self.model_config = GlobalConfig()
        with open(os.path.join(self.vehicle_config.model["dir"], self.vehicle_config.model["args"])) as file:
            args = json.load(file)
        with open(os.path.join(self.vehicle_config.model["dir"], self.vehicle_config.model["config"]),'rb') as file:
            loaded_config = pickle.load(file)
        self.model_config.__dict__.update(loaded_config.__dict__)

        for k, v in args.items():
            if not isinstance(v, str):
                exec(f'self.model_config.{k} = {v}')
        
        self.model_config.use_our_own_config()
    


class ClientManager():
    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.vehicle = None
        self.data_listener = DataListener()
        self.world = None
        self.setup()
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
        self.vehicle.__del__()
        print("plz stop crashing")

    def setup(self):
        self.world:carla.World = self.client.load_world('Town01')

        settings = self.world.get_settings()
        settings.no_rendering_mode = False
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        wmap:carla.Map = self.world.get_map()
        spawn_points = wmap.get_spawn_points()
        a = carla.Location(spawn_points[3].location)
        b = carla.Location(spawn_points[100].location)

        # vehicle = Vehicle(world=world)
        self.vehicle = Agent(world=self.world, data_listener=self.data_listener)
        self.vehicle.spawn(location=carla.Location(spawn_points[3].location))
        self.vehicle.set_controller_pid()
        self.vehicle.planner = GlobalRoutePlanner(wmap, sampling_resolution=10)
        self.vehicle.set_route(start=a, target=b)
    
    def get_vehicle(self):
        return self.vehicle

class DataListener():
    def __init__(self) -> None:
        self.is_listening = True
        self.data = []
        
    def publish(self, data):
        self.data.append(data)
        self.is_listening = False
    

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
