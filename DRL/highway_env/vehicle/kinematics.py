from typing import Union, Optional, Tuple, List
import numpy as np
import copy
from collections import deque

from highway_env import utils
from highway_env.road.road import Road, LaneIndex
from highway_env.vehicle.objects import RoadObject, Obstacle, Landmark
from highway_env.types import Vector
import random


class Vehicle(RoadObject):

    """
    A moving vehicle on a road, and its kinematics.

    The vehicle is represented by a dynamical system: a modified bicycle model.
    It's state is propagated depending on its steering and acceleration actions.
    """

    LENGTH = 5.0
    """ Vehicle length [m] """
    WIDTH = 2.0
    """ Vehicle width [m] """
    DEFAULT_SPEEDS = [23, 25]
    """ Range for random initial speeds [m/s] """
    MAX_SPEED = 40.
    """ Maximum reachable speed [m/s] """
    HISTORY_SIZE = 30
    """ Length of the vehicle state history, for trajectory display"""

    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 predition_type: str = 'constant_steering'):
        super().__init__(road, position, heading, speed)
        self.prediction_type = predition_type
        self.action = {'steering': 0, 'acceleration': 0}
        self.crashed = False
        self.impact = None
        self.log = []
        self.history = deque(maxlen=self.HISTORY_SIZE)
        self.velocity_history = []
        self.position_history = []
        self.lane_history = []
        self.humanTraj = np.array(self.position)
        self.overtaken = False
        self._random_number = 1
        

    @classmethod
    def make_on_lane(cls, road: Road, lane_index: LaneIndex, longitudinal: float, speed: float = 0) -> "Vehicle":
        """
        Create a vehicle on a given lane at a longitudinal position.

        :param road: the road where the vehicle is driving
        :param lane_index: index of the lane where the vehicle is located
        :param longitudinal: longitudinal position along the lane
        :param speed: initial speed in [m/s]
        :return: A vehicle with at the specified position
        """
        lane = road.network.get_lane(lane_index)
        if speed is None:
            speed = lane.speed_limit
        return cls(road, lane.position(longitudinal, 0), lane.heading_at(longitudinal), speed)

    @classmethod
    def create_random(cls, road: Road,
                      speed: float = None,
                      lane_from: Optional[str] = None,
                      lane_to: Optional[str] = None,
                      lane_id: Optional[int] = None,
                      spacing: float = 1) \
            -> "Vehicle":
        """
        Create a random vehicle on the road.

        The lane and /or speed are chosen randomly, while longitudinal position is chosen behind the last
        vehicle in the road with density based on the number of lanes.

        :param road: the road where the vehicle is driving
        :param speed: initial speed in [m/s]. If None, will be chosen randomly
        :param lane_from: start node of the lane to spawn in
        :param lane_to: end node of the lane to spawn in
        :param lane_id: id of the lane to spawn in
        :param spacing: ratio of spacing to the front vehicle, 1 being the default
        :return: A vehicle with random position and/or speed
        """
        _from = lane_from or road.np_random.choice(list(road.network.graph.keys()))
        _to = lane_to or road.np_random.choice(list(road.network.graph[_from].keys()))
        _id = lane_id if lane_id is not None else road.np_random.choice(len(road.network.graph[_from][_to]))
        lane = road.network.get_lane((_from, _to, _id))
        if speed is None:
            if lane.speed_limit is not None:
                speed = road.np_random.uniform(0.7*lane.speed_limit, 0.8*lane.speed_limit)
            else:
                speed = road.np_random.uniform(Vehicle.DEFAULT_SPEEDS[0], Vehicle.DEFAULT_SPEEDS[1])
        default_spacing = 12+1.0*speed
        offset = spacing * default_spacing * np.exp(-5 / 40 * len(road.network.graph[_from][_to]))
        x0 = np.max([lane.local_coordinates(v.position)[0] for v in road.vehicles]) \
            if len(road.vehicles) else 3*offset
        x0 += offset * road.np_random.uniform(0.9, 1.1)
        v = cls(road, lane.position(x0, 0), lane.heading_at(x0), speed)
        return v

    @classmethod
    def create_from(cls, vehicle: "Vehicle") -> "Vehicle":
        """
        Create a new vehicle from an existing one.

        Only the vehicle dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.position, vehicle.heading, vehicle.speed)
        return v

    def act(self, action: Union[dict, str] = None) -> None:
        """
        Store an action to be repeated.

        :param action: the input action
        """
        if action:
            self.action = action

    def step(self, dt: float) -> None:
        """
        Propagate the vehicle state given its actions.

        Integrate a modified bicycle model with a 1st-order response on the steering wheel dynamics.
        If the vehicle is crashed, the actions are overridden with erratic steering and braking until complete stop.
        The vehicle's current lane is updated.

        :param dt: timestep of integration of the model [s]
        """
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
        if front_vehicle is not None and front_vehicle.overtaken:
            gap = self.lane_distance_to(front_vehicle)
            desired_gap = self.desired_gap(self, front_vehicle)
        elif front_vehicle is not None:
            gap = self.lane_distance_to(front_vehicle)
            desired_gap = self.desired_gap(self, front_vehicle)
        else:
            gap = 100
            desired_gap = 50
      
        if gap >= desired_gap and not self.overtaken:
            self.overtaken = False
        elif self.overtaken:     
            self.overtaken = False
        else:
            self.overtaken = True



        self.velocity_history.append(self.velocity[0])
        self.position_history.append(self.position[0])
        self.lane_history.append(self.target_lane_index[2])

        #self.sim_steps += 1
        #super(MDPVehicle, self).step(dt)
        self.humanTraj = np.append(self.humanTraj, self.position, axis=0)


        self.clip_actions()
        delta_f = self.action['steering']
        beta = np.arctan(1 / 2 * np.tan(delta_f))
        v = self.speed * np.array([np.cos(self.heading + beta),
                                   np.sin(self.heading + beta)])
        self.position += v * dt
        if self.impact is not None:
            self.position += self.impact
            self.crashed = True
            self.impact = None
        self.heading += self.speed * np.sin(beta) / (self.LENGTH / 2) * dt
        self.speed += self.action['acceleration'] * dt
        self.on_state_update()

    def clip_actions(self) -> None:
        if self.crashed:
            self.action['steering'] = 0
            self.action['acceleration'] = -1.0*self.speed
        self.action['steering'] = float(self.action['steering'])
        self.action['acceleration'] = float(self.action['acceleration'])
        if self.speed > self.MAX_SPEED:
            self.action['acceleration'] = min(self.action['acceleration'], 1.0 * (self.MAX_SPEED - self.speed))
        elif self.speed < -self.MAX_SPEED:
            self.action['acceleration'] = max(self.action['acceleration'], 1.0 * (self.MAX_SPEED - self.speed))

    def on_state_update(self) -> None:
        if self.road:
            self.lane_index = self.road.network.get_closest_lane_index(self.position, self.heading)
            self.lane = self.road.network.get_lane(self.lane_index)
            if self.road.record_history:
                self.history.appendleft(self.create_from(self))

    def predict_trajectory_constant_speed(self, times: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        if self.prediction_type == 'zero_steering':
            action = {'acceleration': 0.0, 'steering': 0.0}
        elif self.prediction_type == 'constant_steering':
            action = {'acceleration': 0.0, 'steering': self.action['steering']}
        else:
            raise ValueError("Unknown predition type")

        dt = np.diff(np.concatenate(([0.0], times)))

        positions = []
        headings = []
        v = copy.deepcopy(self)
        v.act(action)
        for t in dt:
            v.step(t)
            positions.append(v.position.copy())
            headings.append(v.heading)
        return (positions, headings)

    @property
    def velocity(self) -> np.ndarray:
        return self.speed * self.direction  # TODO: slip angle beta should be used here

    @property
    def destination(self) -> np.ndarray:
        if getattr(self, "route", None):
            last_lane_index = self.route[-1]
            last_lane_index = last_lane_index if last_lane_index[-1] is not None else (*last_lane_index[:-1], 0)
            last_lane = self.road.network.get_lane(last_lane_index)
            return last_lane.position(last_lane.length, 0)
        else:
            return self.position

    @property
    def destination_direction(self) -> np.ndarray:
        if (self.destination != self.position).any():
            return (self.destination - self.position) / np.linalg.norm(self.destination - self.position)
        else:
            return np.zeros((2,))
        
    def _random_num(self) -> float:
        my_array = [-0.45,0, 0.45,1]
        #my_array = [1]

        # generate a random index to access a random element from the array
        random_index = random.randint(0, len(my_array) - 1)

        # access the element at the random index
        random_number = my_array[random_index]
        if utils.episode_changed == True:
            self._random_number = random_number
            utils.episode_changed  = False
        return self._random_number 

    def to_dict(self, origin_vehicle: "Vehicle" = None, observe_intentions: bool = True, vehicle_type_HV: int = 0) -> dict:
        if vehicle_type_HV == 2:
            d = {
                'presence': 1,
                'x': self.position[0],
                'y': self.position[1],
                'vx': self.velocity[0],
                'vy': self.velocity[1],
                'SVO': self._random_num(),
                'heading': self.heading,
                'cos_h': self.direction[0],
                'sin_h': self.direction[1],
                'cos_d': self.destination_direction[0],
                'sin_d': self.destination_direction[1] }

        elif vehicle_type_HV == 1:
            d = {
                'presence': 1,
                'x': self.position[0],
                'y': self.position[1],
                'vx': self.velocity[0],
                'vy': self.velocity[1],
                'SVO':0,
                'heading': self.heading,
                'cos_h': self.direction[0],
                'sin_h': self.direction[1],
                'cos_d': self.destination_direction[0],
                'sin_d': self.destination_direction[1]}
        else:
            d = {
                'presence': 1,
                'x': self.position[0],
                'y': self.position[1],
                'vx': self.velocity[0],
                'vy': self.velocity[1],
                'SVO':0,
                'heading': self.heading,
                'cos_h': self.direction[0],
                'sin_h': self.direction[1],
                'cos_d': self.destination_direction[0],
                'sin_d': self.destination_direction[1]}

        if not observe_intentions:
            d["cos_d"] = d["sin_d"] = 0
        if origin_vehicle:
            origin_dict = origin_vehicle.to_dict()
            for key in ['x', 'y', 'vx', 'vy']:
                d[key] -= origin_dict[key]
        return d

    def __str__(self):
        return "{} #{}: {}".format(self.__class__.__name__, id(self) % 1000, self.position)

    def __repr__(self):
        return self.__str__()
