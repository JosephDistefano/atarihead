import matplotlib.pyplot as plt
import os
import h5py
import numpy as np
import pandas as pd
import cv2


def Vis_Validate(game, config):
    interim_data_directory = config["interim_data_dir"]
    processed_data_directory = config["proccessed_data_dir"]
    interim_folder = os.path.join(interim_data_directory, game)
    for file in os.listdir(processed_data_directory):
        if game in file:
            print(file)
            with h5py.File(os.path.join(processed_data_directory, file), "a") as f:
                print(list(f.keys()))
                game_session_id = list(f.keys())[0]
                H5_gaze = np.array(f[game_session_id]["gazes"])
                H5_frame = np.array(f[game_session_id]["images"])
                # H5_action = np.array(f[game_session_id]["actions"])
            for session in os.listdir(interim_folder):
                if session == game_session_id:
                    game_dir = os.path.join(interim_folder, session)
                    frame_labels = [x for x in os.listdir(game_dir) if "png" in x]
                    frames = np.zeros((len(frame_labels), 210, 160, 3), dtype="uint8")
                    for i in range(len(frame_labels)):
                        label = frame_labels[i]
                        index = int(label.strip(".png").split("_")[-1]) - 1
                        frames[index] = cv2.imread(os.path.join(game_dir, label))
                    gaze_file = os.path.join(game_dir, session + "_gaze_data.csv")
                    gaze_data = pd.read_csv(gaze_file)["gaze_positions"]
                    break
            display_visual([H5_gaze, H5_frame], [gaze_data, frames])


def display_visual(H5s, Raws):
    wait_time = 0
    data_length = len(H5s[0])
    raw_gazes = []
    Fy = []
    Fx = []
    for crds in Raws[0]:
        raw_gazes.append([[], []])
        coords = crds.strip("[[").strip("]]").split("], [")
        for crd in coords:
            x, y = crd.split(", ")
            # print(x)
            # print(np.floor(np.float(x)))
            x = np.floor(float(x))
            y = np.floor(float(y))
            if x > 159:
                Fx.append(x)
                continue
            if y > 209:
                Fy.append(y)
                continue
            raw_gazes[-1][0].append(x)
            raw_gazes[-1][1].append(y)
    print("Gaze Out of Bounds: ", len(Fx))
    for i in range(data_length):
        H5_frame_stack = (H5s[1][i] * 255).astype("uint8")
        H5_gaze_map = np.zeros((84, 84, 3), dtype="uint8")
        temp_gaze = np.array((H5s[0][i] / np.max(H5s[0][i])) * 255)  # , dtype="uint8")
        H5_gaze_map[:, :, 1] = H5_gaze_map[:, :, 2] = temp_gaze

        H5_disp = cv2.cvtColor(H5_frame_stack[0], cv2.COLOR_GRAY2RGB) + H5_gaze_map
        for frame in H5_frame_stack[1:]:
            temp_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) + H5_gaze_map
            H5_disp = np.hstack((H5_disp, temp_frame))  # , dtype="uint8")

        raw_frames = Raws[1][i].astype("uint8")
        raw_gaze = np.array(raw_gazes[i], dtype=int).transpose()
        for x, y in raw_gaze:
            raw_frames[y - 1 : y + 1, x - 1 : x + 1] = [0, 255, 255]
        for j in range(1, 4):
            raw_frame = Raws[1][i + j].astype("uint8")
            raw_gaze = np.array(raw_gazes[i + j], dtype=int).transpose()
            for x, y in raw_gaze:
                raw_frame[y - 1 : y + 1, x - 1 : x + 1] = [0, 255, 255]
            raw_frames = np.hstack((raw_frames, raw_frame))  # , dtype="uint8")
        raw_width = len(raw_frames[0, :, 0])
        H5_width = len(H5_disp[0, :, 0])
        H5_disp = cv2.resize(
            H5_disp, (0, 0), fx=(raw_width / H5_width), fy=(raw_width / H5_width)
        )
        img = np.vstack((H5_disp, raw_frames))  # , dtype="uint8")
        img = cv2.resize(img, (0, 0), fx=2, fy=2)
        cv2.imshow("frame1", img)
        # cv2.imshow("frame", H5_disp)
        if cv2.waitKey(wait_time) == ord("q"):
            exit()
        elif cv2.waitKey(wait_time) == ord("f"):
            wait_time += 1
        elif cv2.waitKey(wait_time) == ord("s"):
            wait_time -= 1
        elif cv2.waitKey(wait_time) == ord("d"):
            wait_time = 0
        # cv2.destroyAllWindows()
