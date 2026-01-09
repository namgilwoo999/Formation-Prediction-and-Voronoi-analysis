import sys
import os
# sys.path.append('../')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.bbox_utils import get_bbox_center, measure_distance

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70
    
    def assign_ball_to_player(self, players, ball_bbox):
        ball_position = get_bbox_center(ball_bbox)
        if ball_position is None:
            return -1

        min_distance = float("inf")
        assigned_player = -1

        for player_id, player in players.items():
            bbox = player.get('bbox')
            if bbox is None or len(bbox) != 4:
                continue

            foot_left = (bbox[0], bbox[3])
            foot_right = (bbox[2], bbox[3])
            dist_left = measure_distance(foot_left, ball_position)
            dist_right = measure_distance(foot_right, ball_position)
            distance = min(dist_left, dist_right)

            if distance < self.max_player_ball_distance and distance < min_distance:
                min_distance = distance
                assigned_player = player_id

        return assigned_player
