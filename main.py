from utils.video_utils import read_video, save_video
from trackers.tracker import Tracker
import cv2
from team_assigner.team_assigner import TeamAssigner
from player_ball_assigner.player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator
from view_transformer.view_transformer import ViewTransformer
from speed_and_distance_estimator.speed_and_distance_estimator import SpeedAndDistanceEstimator
import numpy as np
def main():
    #read in the video
    video_frames=read_video(r'D:\Computer Vision\Football Analysis\input_videos\video.mp4')

    #initialize the tracker
    tracker=Tracker(model_path=r'D:\Computer Vision\Football Analysis\model\best.pt')
    tracks = tracker.get_object_tracks(
    video_frames,
    read_from_stub=True,  # Set to True to read from the stub file
    stub_path='stubs/track_stubs.pkl'
)
    """ one time needeed code to export a single img
    #save cropped image of a player:
    for track_id , player in tracks["players"][0].items():
          bbox=player['bbox']
          frame=video_frames[0]
          #crop the player from the frame using the bbox
          cropped_image=frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] #y1:y2, x1:x2
          #save the cropped image
          cv2.imwrite(f"output_videos/cropped_image.jpg", cropped_image)
          break # we just need one img  , without break we"ll get all players cropped images in the first frame but we just got one player because we used break so iteration won't finish 
    """
    
    #get object positions
    tracker.add_position_to_tracks(tracks)
    #camera movement estimator:
    camera_movement_estimator=CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame=camera_movement_estimator.get_camera_movement(video_frames,
                                                                            read_from_stub=True,
                                                                            stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)

    
     
    #view transformer 
    view_transformer=ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)
    
    #interpolate ball positions:
    tracks["ball"]=tracker.interpolate_ball_positions(tracks["ball"])
   
    #speed and distance estimator
    speed_and_distance_estimator=SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
    
    #assign players to teams based on their jersey colors
    team_assigner=TeamAssigner()
    team_assigner.assign_team_color(video_frames[0],tracks["players"][0]) #assign team colors based on the first frame
    #loop through each player in all frames and assign team colors to players
    for frame_num,player_track in enumerate(tracks["players"]):
        for player_id ,track in player_track.items():
             team=team_assigner.get_player_team(video_frames[frame_num],
                                                track["bbox"],
                                                player_id)
             tracks["players"][frame_num][player_id]["team"]=team
             tracks["players"][frame_num][player_id]["team_color"]=team_assigner.team_colors[team]
    #assign ball acquisition:
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            if team_ball_control:  # only carry over if list is not empty
                team_ball_control.append(team_ball_control[-1])
            # if empty (frame 0 with no possession), skip — don't append anything small fix so it won't crush if no player has the ball at frame 0

    team_ball_control = np.array(team_ball_control)
    
    #draw outputs
    #draw object tracks on the video frames
    output_video_frames=tracker.draw_annotaions(video_frames,tracks,team_ball_control)
    #draw camera movement:
    output_video_frames=camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)
    
    #draw speed and distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)
    #save the video
    save_video(output_video_frames, r'D:\Computer Vision\Football Analysis\output_videos\output_video.avi')


if __name__=="__main__":
        main() 