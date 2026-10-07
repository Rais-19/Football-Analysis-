import sys
sys.path.append('../')
from utils import get_center_of_bbox,measure_distance
class PlayerBallAssigner:
    def  __init__(self):
        self.max_player_ball_distance=70 # anything above 70 ball won't be assigned to any player   
    def assign_ball_to_player(self,players,ball_bbox):
        ball_position=get_center_of_bbox(ball_bbox)  
        
        minimum_distance= 99999 #s "assume the worst possible distance at the start, so the first real player always wins the first comparison."
        assigned_player=-1 # no player assigned yet 
        
        for player_id,player in players.items():
            player_bbox=player['bbox']

            distance_left=measure_distance((player_bbox[0],player_bbox[-1]),ball_position)
            distance_right=measure_distance((player_bbox[2],player_bbox[-1]),ball_position)
            distance=min(distance_left,distance_right)


            if distance < self.max_player_ball_distance:   # ← guard: ball must be reasonably close
                if distance < minimum_distance:             # ← is this the closest player so far?
                    minimum_distance = distance             # update best
                    assigned_player = player_id            # this player is currently winning

        return assigned_player
