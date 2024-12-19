import cv2
# from cv2.typing import MatLike
from tracker import PlayerTracker, CourtLineDetector
import os
from utils import get_minimum_distance, find_middle

def draw_bboxes(frame, player_detections: dict):
    for track_id, box_result in player_detections.items():
        x1, y1, x2, y2 = box_result
        cv2.putText(frame,text=f"Player ID: {track_id}",
                    org=(int(x1),int(y1)-10),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    color = (0,0,255),
                    fontScale=1,
                    thickness=2)
        cv2.rectangle(img = frame, 
                                pt1= (int(x1),int(y1)), pt2 = (int(x2),int(y2)),
                                color=(0,0,255),
                                thickness=2)

def find_player(court_keypoints, player_detection):
    player_dict = player_detection
    players = []
    for id, bbox in player_dict.items():
        player_point = find_middle(bbox)
        distance = get_minimum_distance(player_point,court_keypoints)
        players.append((id,distance))
    players.sort(key=lambda x: x[1])
    return (players[0][0],players[1][0])

def filter_players_only(court_keypoints,player_detections):
    selected_track_ids  = find_player(court_keypoints, player_detections)
    player_detect = {id: bbox for id,bbox in player_detections.items() if id in selected_track_ids}
    return player_detect


def draw_points(frame,keypoints):
    for i in range(0,len(keypoints),2):
        x = int(keypoints[i])
        y = int(keypoints[i+1])
        cv2.putText(frame,
                    text = str(i//2),
                    org = (x,y-10),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1,
                    color=(255,0,0),
                    thickness=2
                    )
        cv2.circle(frame,
                    center = (x,y),
                    radius= 5,
                    color = (0,255,0),
                    thickness=1)
    return frame

def main():
    video_path = './data/tennis-video.mp4'

    cap = cv2.VideoCapture(video_path)

    cwd = os.getcwd()
    player_model_path = os.path.join(cwd,'saved_models/PLAYER_TRACK/yolov8x.pt')
    player_tracker = PlayerTracker(player_model_path)

    court_model_path = os.path.join(cwd,'saved_models/KEYPOINTSMODEL/keypoints2.pth')
    court_detector = CourtLineDetector(court_model_path)

    while(True):
        ret, frame = cap.read()
        # Get the frame rate of the video
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = int(1000 / fps)  # Delay in milliseconds

        player_detections = player_tracker.detect_frame(frame)
        # print(player_detections)

        court_keypoints = court_detector.predict(frame)

        # player_detections = filter_players_only(court_keypoints, player_detections)
        #p rint(player_detections)

        draw_bboxes(frame, player_detections)
        draw_points(frame, court_keypoints)

        cv2.imshow('frame', frame)
        if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
            print('frame is running')
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

"""
if __name__ == '__main__':
    player_model_path = './saved_models/PLAYER_TRACK/yolov8x.pt'
    tracker = PlayerTracker(player_model_path)
    print("PlayerTracker instantiated successfully")
"""

