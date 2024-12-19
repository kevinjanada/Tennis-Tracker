import cv2

def read_video(video_path, max_frames=0):
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"width = {width}")
    print(f"height = {height}")
    print(f"fps = {fps}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if (max_frames > 0) and len(frames) == max_frames:
            break
        frames.append(frame)
    cap.release()
    return frames


def save_video(frames, video_path):
    height, width,layers = frames[0].shape
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
    print(f'Video saved to {video_path}')
