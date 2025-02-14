import cv2
import math
import numpy as np
# import torch
import os
from ultralytics import YOLO
from make_videos import make_video

'''
    reference: https://docs.ultralytics.com/modes/track/
    conda env: bytetrack
'''

def store_results(results, cur_frame):
    frame_info = []
    frame_info_np = np.array(frame_info)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            if box.is_track:
                id = int(box.id[0])
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w,h = x2-x1, y2-y1

                # our db format
                # game_id = -998
                # frame_info = [game_id, cur_frame, x1, y1, x2, y2, id, id]
                # SN eval format
                frame_info = [cur_frame, id, x1, y1, w, h, 1, -1, -1, -1]

                frame_info_np = np.append(frame_info_np, frame_info)

    if len(frame_info) > 0:
        # this is for our db
        # frame_info_np = frame_info_np.reshape(-1, 8)
        # this is for SN Eval
        frame_info_np = frame_info_np.reshape(-1, 10)
                
    return frame_info_np



def bytetrack(weight_path, video_path):
    model = YOLO(weight_path)

    track_info = np.array([])

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened()

    video_name = video_path
    slash_index = video_name.rfind('/')
    underscore_index = video_name.rfind('_')
    video_name = video_name[slash_index + 1 : underscore_index]
    print(video_name)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    format_string = "{:03d}"
    print(total_frames)

    while True:
        success, frame = cap.read()
        cur_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if not success:
            if cur_frame < total_frames:
                print("Failed at frame " + str(cur_frame))
                break
            else:
                print("Video ended")
                break

        print(str(cur_frame) + "/" + str(total_frames))

        results = model.track(frame, persist=True, tracker="bytetrack.yaml")

        # for visualization
        annotated_frame = results[0].plot()

        frame_info = store_results(results, cur_frame)
        track_info = np.append(track_info, frame_info)
        # our db
        # track_info = track_info.reshape(-1, 8)
        # SN Eval
        # track_info = track_info.reshape(-1, 10)

        # print(track_info.shape)
        # if track_info.shape[0] > 0:
        #     print(track_info[-1])

        # save visualization
        format_string = "{:08d}"
        save_path = os.getcwd()
        save_path = save_path + '\\detections'
        print(cv2.imwrite(save_path + '\\frame_'+ format_string.format(cur_frame) +'.png', annotated_frame))

        # cv2.imshow("tracking", annotated_frame)

        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    # np.savetxt('mot_results_bytetrack/' + video_name + '.txt', track_info, delimiter=',', fmt='%.0f')
    output_path = os.getcwd() + '\\detections'
    make_video(output_path, 'output.avi')


if __name__ == '__main__':
    weight_path = 'last.pt'
    video_path = 'vid.avi'

    sn_vid_root = 'C:/Users/kyle1/Desktop/PADDLE_DETECTION_ROOT/dataset/mot/SN/images/test'

    # run detector on all videos in SN test folder
    # for fld in os.listdir(sn_vid_root):
    #   print(fld)
    #   dir = sn_vid_root + '/' + fld
    #   for file in os.listdir(dir):
    #     if os.path.splitext(file)[1] == '.avi':
    #       vid_path = dir + '/' + file
    #   print(vid_path)
    #   bytetrack(weight_path, vid_path)

    bytetrack(weight_path, video_path)

