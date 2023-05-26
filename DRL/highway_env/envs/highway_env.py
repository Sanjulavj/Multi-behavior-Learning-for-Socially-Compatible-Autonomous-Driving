import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
import random

class HighwayEnv(AbstractEnv):
    collision = 0;  
    velocity_impact = 0
    braking_impact = 0
    vehicle_gap_front = 0
    vehicle_gap_back = 0
    overtaking_frontend_vehicle_gap = 0
    overtaking_rearend_vehicle_gap = 0
    iterations = 0
    episode_count = 0

    velocity_impact_count = 0
    braking_impact_count = 0
    vehicle_gap_frontend_count = 0
    vehicle_gap_rearend_count = 0
    overtaking_frontend_vehicle_gap_count = 0
    overtaking_rearend_vehicle_gap_count = 0
    overtaking_count = 0

    AVMeanList = []
    HVMeanList = []
    HVDistanceList = []
    total_distance = 0
 
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1.2,  # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.08,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "reward_speed_range": [20, 30],
            "offroad_terminal": False
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            controlled_vehicle = self.action_type.vehicle_class.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
     
        R = speed * 0.4 + rightmostlane * 0.04 + collison* (-1.2) + (-)vehicle_gap + AV_utility_function  * (cos(o)) + human_utility_function  * (sin(o))
        """
        vehicle_gap_reward = 0
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                if  0.0 <= vehicle.position[0] - self.vehicle.position[0] <= 20.0 and vehicle.target_lane_index[2] == self.vehicle.target_lane_index[2]:
                    if 14.0 <= vehicle.position[0] - self.vehicle.position[0] <= 18.0:
                        vehicle_gap_reward -= 0.8
                    elif 0.0 <= vehicle.position[0] - self.vehicle.position[0] < 14.0:
                        vehicle_gap_reward -= 1

        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]

        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.config["right_lane_reward"] * lane / max(len(neighbours) - 1, 1) \
            + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1) \
            + vehicle_gap_reward

        reward = utils.lmap(reward,
                           [self.config["collision_reward"] + vehicle_gap_reward,
                            self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                           [0, 1])

        reward = 0 if not self.vehicle.on_road else reward
       
        AV_utility_function = self.AV_utility_function()


        human_utility_function = self.human_utility_function()

        if self.vehicle._random_number == -0.45:
            cosAV_utility_function = AV_utility_function * 0.70710678118
            sinhuman_utility_function = human_utility_function * 0.70710678118
            Final_REWRD = reward + cosAV_utility_function - sinhuman_utility_function
            scaled_Final_REWRD = utils.lmap(Final_REWRD, [0,1], [0, 1])

            return scaled_Final_REWRD
        
        elif self.vehicle._random_number == 0:
            cosAV_utility_function = AV_utility_function 
            sinhuman_utility_function = human_utility_function 
            Final_REWRD = reward + cosAV_utility_function 
            scaled_Final_REWRD = utils.lmap(Final_REWRD, [0,2], [0, 1])

            return scaled_Final_REWRD
        
        elif self.vehicle._random_number == 0.45:
            cosAV_utility_function = AV_utility_function * 0.70710678118
            sinhuman_utility_function = human_utility_function * 0.70710678118
            Final_REWRD = reward + cosAV_utility_function + sinhuman_utility_function
            scaled_Final_REWRD = utils.lmap(Final_REWRD, [0,3], [0, 1])

            return scaled_Final_REWRD
        
        elif self.vehicle._random_number == 1:
            cosAV_utility_function = AV_utility_function 
            sinhuman_utility_function = human_utility_function 
            Final_REWRD = reward + sinhuman_utility_function 
            scaled_Final_REWRD = utils.lmap(Final_REWRD, [0,2], [0, 1])

            return scaled_Final_REWRD


    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        if self.vehicle.crashed or \
            self.steps >= self.config["duration"] or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road):
            self.episode_count +=1 
            utils.episode_changed  = True
            
        evaluation_success_rate = self._evaluation_success_rate()
        evaluation_change_of_velocity = self._evaluation_change_of_velocity()
        evaluation_change_of_braking = self._evaluation_change_of_braking()
        evaluation_vehicle_gap_rearend = self._evaluation_vehicle_gap_rearend()
        evaluation_vehicle_gap_frontend = self._evaluation_vehicle_gap_frontend()
        evaluation_overtaking_vehicle_gap_frontend = self._evaluation_overtaking_vehicle_gap_frontend()
        evaluation_overtaking_vehicle_gap_rearend = self._evaluation_overtaking_vehicle_gap_rearend()
        evaluation_overtakings_by_AV = self._evaluation_overtakings_by_AV()
        evaluation_human_utility_mean = self._evaluation_human_utility_mean()
        evaluation_AV_utility_mean = self._evaluation_AV_utility_mean()
        evaluation_iterations = self._evaluation_iterations()

        print("iterations")
        print(evaluation_iterations)

        print("evaluation_success_rate")
        print(evaluation_success_rate)
        
        print("evaluation_change_of_velocity")
        print(evaluation_change_of_velocity)
       
        print("evaluation_change_of_braking")
        print(evaluation_change_of_braking)
          
        print("evaluation_vehicle_gap_rearend")
        print(evaluation_vehicle_gap_rearend)

        print("evaluation_vehicle_gap_frontend")
        print(evaluation_vehicle_gap_frontend)

        print("evaluation_overtaking_vehicle_gap_frontend")
        print(evaluation_overtaking_vehicle_gap_frontend)

        print("evaluation_overtaking_vehicle_gap_rearend")
        print(evaluation_overtaking_vehicle_gap_rearend)

        print("evaluation_overtakings_by_AV") 
        print(evaluation_overtakings_by_AV)

        print("evaluation_human_utility_mean")
        print(evaluation_human_utility_mean)

        print("evaluation_AV_utility_mean")
        print(evaluation_AV_utility_mean)


        if evaluation_iterations == 1500:
            with open('multitasking_readme.txt', 'w') as f:
                f.write(str("evaluation_iterations"))
                f.write('\n')
                f.write(str(evaluation_iterations))
                f.write('\n')
                f.write(str("velocity_impact_count"))
                f.write('\n')
                f.write(str(self.velocity_impact_count))
                f.write('\n')
                f.write(str("braking_impact_count"))
                f.write('\n')
                f.write(str(self.braking_impact_count))
                f.write('\n')
                f.write(str("vehicle_gap_rearend_count"))
                f.write('\n')
                f.write(str(self.vehicle_gap_rearend_count))
                f.write('\n')
                f.write(str("vehicle_gap_frontend_count"))
                f.write('\n')
                f.write(str(self.vehicle_gap_frontend_count))
                f.write('\n')
                f.write(str("overtaking_frontend_vehicle_gap_count"))
                f.write('\n')                
                f.write(str(self.overtaking_frontend_vehicle_gap_count))
                f.write('\n')
                f.write(str("overtaking_rearend_vehicle_gap_count"))
                f.write('\n')  
                f.write(str(self.overtaking_rearend_vehicle_gap_count))
                f.write('\n')
                f.write(str("evaluation_success_rate"))
                f.write('\n')
                f.write(str(evaluation_success_rate))
                f.write('\n')
                f.write(str("evaluation_change_of_velocity"))
                f.write('\n')
                f.write(str(evaluation_change_of_velocity))
                f.write('\n')
                f.write(str("evaluation_change_of_braking"))
                f.write('\n')
                f.write(str(evaluation_change_of_braking))
                f.write('\n')
                f.write(str("evaluation_vehicle_gap_rearend"))
                f.write('\n')
                f.write(str(evaluation_vehicle_gap_rearend))
                f.write('\n')
                f.write(str("evaluation_vehicle_gap_frontend"))
                f.write('\n')
                f.write(str(evaluation_vehicle_gap_frontend))
                f.write('\n')
                f.write(str("evaluation_overtaking_vehicle_gap_frontend"))
                f.write('\n')
                f.write(str(evaluation_overtaking_vehicle_gap_frontend))
                f.write('\n')
                f.write(str("evaluation_overtaking_vehicle_gap_rearend"))
                f.write('\n')
                f.write(str(evaluation_overtaking_vehicle_gap_rearend))
                f.write('\n')
                f.write(str("evaluation_overtakings_by_AV"))
                f.write('\n')
                f.write(str(evaluation_overtakings_by_AV))
                f.write('\n')
                f.write(str("evaluation_human_utility_mean"))
                f.write('\n')
                f.write(str(evaluation_human_utility_mean))
                f.write('\n')
                f.write(str("evaluation_AV_utility_mean"))
                f.write('\n')
                f.write(str(evaluation_AV_utility_mean))

        return self.vehicle.crashed or \
            self.steps >= self.config["duration"] or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _AV_vehcile_features(self):
        """
        Hand-crafted features
        :return: the array of the defined features
        """
        # ego motion
        ego_longitudial_positions = self.vehicle.traj.reshape(-1, 2)[self.time-3:, 0]
        ego_longitudial_speeds = (ego_longitudial_positions[1:] - ego_longitudial_positions[:-1]) / 0.1 if self.time >= 3 else [0]
        ego_longitudial_accs = (ego_longitudial_speeds[1:] - ego_longitudial_speeds[:-1]) / 0.1 if self.time >= 3 else [0]
        ego_longitudial_jerks = (ego_longitudial_accs[1:] - ego_longitudial_accs[:-1]) / 0.1 if self.time >= 3 else [0]

        ego_lateral_positions = self.vehicle.traj.reshape(-1, 2)[self.time-3:, 1]
        ego_lateral_speeds = (ego_lateral_positions[1:] - ego_lateral_positions[:-1]) / 0.1 if self.time >= 3 else [0]
        ego_lateral_accs = (ego_lateral_speeds[1:] - ego_lateral_speeds[:-1]) / 0.1 if self.time >= 3 else [0]
  
        ego_speed = abs(ego_longitudial_speeds[-1]) 
        
        # comfort
        ego_longitudial_acc = ego_longitudial_accs[-1]
        ego_lateral_acc = ego_lateral_accs[-1]
        ego_longitudial_jerk = ego_longitudial_jerks[-1]

        # time headway front (THWF) and time headway behind (THWB)
        THWFs = [100]; THWBs = [100]
        for v in self.road.vehicles:
            if v.position[0] > self.vehicle.position[0] and abs(v.position[1]-self.vehicle.position[1]) < self.vehicle.WIDTH and self.vehicle.velocity[0] >= 1:
                THWF = (v.position[0] - self.vehicle.position[0]) / self.vehicle.velocity[0]
                THWFs.append(THWF)

            elif v.position[0] < self.vehicle.position[0] and abs(v.position[1]-self.vehicle.position[1]) < self.vehicle.WIDTH and v.velocity[0] >= 1:
                THWB = (self.vehicle.position[0] - v.position[0]) / v.velocity[0]
                THWBs.append(THWB)

        THWF = np.exp(-min(THWFs))
        THWB = np.exp(-min(THWBs)) 
        # avoid collision
        collision = 1 if self.vehicle.crashed or not self.vehicle.on_road else 0

        # interaction (social) impact
        social_impact = 0
        for v in self.road.vehicles:
                if v not in self.controlled_vehicles and v.overtaken and v.velocity[0] != 0:
                    social_impact += (v.velocity[0] - v.velocity_history[-1])/0.1 if v.velocity[0] - v.velocity_history[-1] < 0 else 0

        # feature array
        fetures = np.array([ego_speed, abs(ego_longitudial_acc), abs(ego_lateral_acc), abs(ego_longitudial_jerk),
                            THWF, THWB, collision, social_impact])
  
        return fetures

    def _human_vehcile_features(self):
       """
        Hand-crafted features
        :return: the array of the defined features
        """

       self.total_distance = 0

       HVUList = []
       self.HVDistanceList = []
       for v in self.road.vehicles:
           if v not in self.controlled_vehicles:
               if  -65.0 <= v.position[0] - self.vehicle.position[0] <= 65.0 and (v.target_lane_index[2] == self.vehicle.target_lane_index[2] or v.target_lane_index[2] == self.vehicle.target_lane_index[2] + 1 or v.target_lane_index[2] == self.vehicle.target_lane_index[2] - 1):
                    if v.position[0] - self.vehicle.position[0] >= 0:
                        distance = v.position[0] - self.vehicle.position[0]
                    else:
                        distance = self.vehicle.position[0] - v.position[0]
                    
                    self.total_distance += distance

                    ego_longitudial_positions = v.humanTraj.reshape(-1, 2)[self.time-3:, 0]
                    ego_longitudial_speeds = (ego_longitudial_positions[1:] - ego_longitudial_positions[:-1]) / 0.1 if self.time >= 3 else [0]
                    ego_longitudial_accs = (ego_longitudial_speeds[1:] - ego_longitudial_speeds[:-1]) / 0.1 if self.time >= 3 else [0]
                    ego_longitudial_jerks = (ego_longitudial_accs[1:] - ego_longitudial_accs[:-1]) / 0.1 if self.time >= 3 else [0]               

                    ego_lateral_positions = v.humanTraj.reshape(-1, 2)[self.time-3:, 1]
                    ego_lateral_speeds = (ego_lateral_positions[1:] - ego_lateral_positions[:-1]) / 0.1 if self.time >= 3 else [0]
                    ego_lateral_accs = (ego_lateral_speeds[1:] - ego_lateral_speeds[:-1]) / 0.1 if self.time >= 3 else [0]
        
                    
                    ego_speed = abs(ego_longitudial_speeds[-1]) 

                    # comfort
                    ego_longitudial_acc = ego_longitudial_accs[-1]
                    ego_lateral_acc = ego_lateral_accs[-1]
                    ego_longitudial_jerk = ego_longitudial_jerks[-1]

                    collision = 1 if v.crashed or not v.on_road else 0

                    THWFs = []; THWBs = []
                    THWFs.append(100)
                    THWBs.append(100)
                    # time headway front (THWF) and time headway behind (THWB)
                    if v.position[0] < self.vehicle.position[0] and abs(self.vehicle.position[1] - v.position[1]) < v.WIDTH and v.velocity[0] >= 1:
                        THWF = (self.vehicle.position[0] - v.position[0]) / v.velocity[0]
                        THWFs.append(THWF)

                    elif v.position[0] > self.vehicle.position[0] and abs(self.vehicle.position[1] - v.position[1]) < v.WIDTH and self.vehicle.velocity[0] >= 1:
                        THWB = (v.position[0] - self.vehicle.position[0]) / self.vehicle.velocity[0]
                        THWBs.append(THWB)

                    THWF = np.exp(-min(THWFs)) 
                    THWB = np.exp(-min(THWBs)) 

                    # interaction (social) impact
                    social_impact = 0
                    if v.overtaken and v.velocity[0] != 0:
                        social_impact = (v.velocity[0] - v.velocity_history[-1])/0.1 if v.velocity[0] - v.velocity_history[-1] < 0 else 0
                    
                    
                    features = np.array([ego_speed, abs(ego_longitudial_acc), abs(ego_lateral_acc), abs(ego_longitudial_jerk),
                             THWF, THWB, collision, social_impact])
     
                    weights_arr = np.array([1.753971587 , -0.5295225956 , -0.5097452474 , -3.058776862,-1.799733349 , -1.1420489 , -9.950304272 , -3.498063628 ])
                    weighted_features = features * weights_arr
                    total_weighted_features = np.sum(weighted_features)
                    self.HVDistanceList.append(distance)
                    HVUList.append(total_weighted_features)

       return HVUList    
 
    def AV_utility_function(self) -> float:
        weights_arr = np.array([1.753971587 , -0.5295225956 , -0.5097452474 , -3.058776862,-1.799733349 , -1.1420489 , -9.950304272 , -3.498063628 ])
        weighted_features = self._AV_vehcile_features() * weights_arr    
        total_weighted_features = np.sum(weighted_features)
        self.AVMeanList.append(total_weighted_features)
        scaled_total_weighted_features = utils.lmap(total_weighted_features, [0,30], [0, 1])
        return scaled_total_weighted_features 
    
    def human_utility_function(self) -> float:
        weightedDistanceElements = []
        sumDistanceElements = []
        human_features = self._human_vehcile_features()
        
        for element in self.HVDistanceList:
            weighted_distance_element = self.total_distance / element
            weightedDistanceElements.append(weighted_distance_element)

        weighted_distance_element_total = np.sum(weightedDistanceElements)

        for e in weightedDistanceElements:
            value_weighted_element = e / weighted_distance_element_total
            sumDistanceElements.append(value_weighted_element)

        numpy_human_features = np.array(human_features)
        numpysumDistanceElements = np.array(sumDistanceElements)
        total_weighted_features = numpy_human_features * numpysumDistanceElements
        total_weighted_features_sum = np.sum(total_weighted_features)
        self.HVMeanList.append(total_weighted_features_sum)
        scaled_total_weighted_features = utils.lmap(total_weighted_features_sum, [0,22], [0, 1])
        return scaled_total_weighted_features
    
    ##_________Evaluation metrics_________

    def _evaluation_iterations(self) ->float:
        self.iterations += 1  
        return self.iterations

    def _evaluation_success_rate(self) ->float:
        self.iterations += 1  
        for v in self.road.vehicles:
            if v in self.controlled_vehicles:
                if self.vehicle.crashed:
                    self.collision += 1
        success_rate = ((self.iterations - self.collision) / self.iterations) * 100
        return success_rate

    def _evaluation_change_of_velocity(self) ->float:
        for v in self.road.vehicles:
                if v not in self.controlled_vehicles:
                    if self.vehicle.overtaken and self.vehicle.velocity[0] != 0 and 0.0 <= self.vehicle.position[0] - v.position[0] <= 30.0 and v.target_lane_index[2] == self.vehicle.target_lane_index[2]:
                        self.velocity_impact += (v.velocity[0] - v.velocity_history[-1])
                        self.velocity_impact_count += 1
        if self.velocity_impact_count != 0 and self.iterations !=0:
            final_velocity_impact = self.velocity_impact/(self.velocity_impact_count * self.iterations)
        else:
            final_velocity_impact = 0
        print("Velocity_impact_count", self.velocity_impact_count)
        return final_velocity_impact
    
    def _evaluation_change_of_braking(self) ->float: 
        for v in self.road.vehicles:
                if v not in self.controlled_vehicles:
                    if  -20.0 <= self.vehicle.position[0] - v.position[0] <= 30.0 and v.velocity[0] - v.velocity_history[-1] < 0:
                        if v.target_lane_index[2] == self.vehicle.target_lane_index[2] or v.target_lane_index[2] == self.vehicle.target_lane_index[2] + 1 or v.target_lane_index[2] == self.vehicle.target_lane_index[2] - 1:
                            self.braking_impact += v.velocity_history[-1] - v.velocity[0]
                            self.braking_impact_count +=1
        if self.braking_impact_count != 0 and self.iterations !=0 :
            final_braking_impact = self.braking_impact/(self.braking_impact_count * self.iterations)
        else:
            final_braking_impact = 0        
                                                   
        print("Braking_impact_count", self.braking_impact_count)
        return final_braking_impact
    
    def _evaluation_vehicle_gap_rearend(self) ->float: 
        for v in self.road.vehicles:
                if v not in self.controlled_vehicles:
                    if  0.0 <= v.position[0] - self.vehicle.position[0] <= 20.0 and v.target_lane_index[2] == self.vehicle.target_lane_index[2]:
                        self.vehicle_gap_back += v.position[0] - self.vehicle.position[0]
                        self.vehicle_gap_rearend_count +=1

        if self.vehicle_gap_rearend_count != 0 and self.iterations !=0 :
            final_vehicle_gap_back = self.vehicle_gap_back/(self.vehicle_gap_rearend_count * self.iterations)
        else:
            final_vehicle_gap_back = 0
        print("Vehicle_gap_rearend_count", self.vehicle_gap_rearend_count)
        return final_vehicle_gap_back
    
    def _evaluation_vehicle_gap_frontend(self) ->float: 
        for v in self.road.vehicles:
                if v not in self.controlled_vehicles:
                    if  0.0 <= self.vehicle.position[0] - v.position[0] <= 20.0 and v.target_lane_index[2] == self.vehicle.target_lane_index[2]:
                        self.vehicle_gap_front += self.vehicle.position[0] -  v.position[0]
                        self.vehicle_gap_frontend_count +=1

        if self.vehicle_gap_frontend_count != 0 and self.iterations !=0 :
            final_vehicle_gap_front = self.vehicle_gap_front/(self.vehicle_gap_frontend_count * self.iterations )
        else:
            final_vehicle_gap_front = 0
        print("Vehicle_gap_frontend_count", self.vehicle_gap_frontend_count) 
        return final_vehicle_gap_front
    
    def _evaluation_overtaking_vehicle_gap_frontend(self) ->float: 
        for v in self.road.vehicles:
                if v not in self.controlled_vehicles:
                    if self.vehicle.overtaken and self.vehicle.velocity[0] != 0 and 0.0 <= self.vehicle.position[0] - v.position[0] <= 20.0 and v.target_lane_index[2] == self.vehicle.target_lane_index[2]:
                        self.overtaking_frontend_vehicle_gap += self.vehicle.position[0] - v.position[0]
                        self.overtaking_frontend_vehicle_gap_count +=1
        

        if self.overtaking_frontend_vehicle_gap_count != 0 and self.iterations !=0:
            final_overtaking_front_vehicle_gap = self.overtaking_frontend_vehicle_gap/(self.overtaking_frontend_vehicle_gap_count * self.iterations)
        else:
            final_overtaking_front_vehicle_gap = 0
        print("Overtaking_vehicle_gap_frontend_count", self.overtaking_frontend_vehicle_gap_count) 
        return final_overtaking_front_vehicle_gap

    def _evaluation_overtaking_vehicle_gap_rearend(self) ->float:       
        for v in self.road.vehicles:
                if v not in self.controlled_vehicles and len(v.lane_history) > 16 and len(v.position_history) > 16:
                    if v.lane_history[-16] == self.vehicle.lane_history[-16] and v.target_lane_index[2] != self.vehicle.target_lane_index[2] and 0.0 <= v.position_history[-16] - self.vehicle.position_history[-16] <= 15.0 :
                        self.overtaking_rearend_vehicle_gap += v.position_history[-16] - self.vehicle.position_history[-16]
                        self.overtaking_rearend_vehicle_gap_count += 1
        

        if self.overtaking_rearend_vehicle_gap_count != 0 and self.iterations !=0:
            final_overtaking_back_vehicle_gap = self.overtaking_rearend_vehicle_gap/(self.overtaking_rearend_vehicle_gap_count * self.iterations)
        else:
            final_overtaking_back_vehicle_gap = 0
        print("Overtaking_vehicle_gap_rearend_count", self.overtaking_rearend_vehicle_gap_count) 
        return final_overtaking_back_vehicle_gap 
    
    def _evaluation_overtakings_by_AV(self) ->float: 
        if self.vehicle.overtaken:
            self.overtaking_count +=1
        return self.overtaking_count
    
    def _evaluation_human_utility_mean(self) ->float: 
        mean = np.mean(self.HVMeanList)
        return mean
    
    def _evaluation_AV_utility_mean(self) ->float: 
        mean = np.mean(self.AVMeanList)
        return mean
    

    ##_________Evaluation metrics_________

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)


class HighwayEnvFast(HighwayEnv):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "simulation_frequency": 5,
            "lanes_count": 3,
            "vehicles_count": 20,
            "duration": 20,  # [s]
            "ego_spacing": 1.5,
        })
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        # for vehicle in self.road.vehicles:
        #     if vehicle not in self.controlled_vehicles:
        #         vehicle.check_collisions = False


register(
    id='highway-v0',
    entry_point='highway_env.envs:HighwayEnv',
)

register(
    id='highway-fast-v0',
    entry_point='highway_env.envs:HighwayEnvFast',
)
