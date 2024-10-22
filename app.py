#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
import threading
import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from utils import DrawOnCamera
from utils import Calcs
from utils import Talks
from utils import ModeManager
from model import KeyPointClassifier
from model import PointHistoryClassifier
from repository.signsDescription import SignsDescriptionClient


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help="cap width", type=int, default=960)
    parser.add_argument("--height", help="cap height", type=int, default=540)

    parser.add_argument("--use_static_image_mode", action="store_true")
    parser.add_argument(
        "--min_detection_confidence",
        help="min_detection_confidence",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--min_tracking_confidence",
        help="min_tracking_confidence",
        type=int,
        default=0.5,
    )

    args = parser.parse_args()

    return args


def main():
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=use_static_image_mode,
        model_complexity=0,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    repo = SignsDescriptionClient()

    mode_manager = ModeManager()

    # Read labels ###########################################################
    with open(
        "model/keypoint_classifier/keypoint_classifier_label.csv", encoding="utf-8-sig"
    ) as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]
    with open(
        "model/point_history_classifier/point_history_classifier_label.csv",
        encoding="utf-8-sig",
    ) as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    point_history = {
        "L": deque(maxlen=history_length),
        "R": deque(maxlen=history_length),
    }
    pre_processed_point_history_list = {"L": [], "R": []}

    # Finger gesture history ################################################
    finger_gesture_history = {
        "L": deque(maxlen=history_length // 4),
        "R": deque(maxlen=history_length),
    }

    #  ########################################################################

    cm_timer = 0
    blank_timer = 0
    CM = ""
    draw_word = ""
    has_a_new_word = False

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break

        mode_manager.alter_mode_by_key(key)

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        hand_results = hands.process(image)
        pose_results = pose.process(image)
        image.flags.writeable = True
        
        hand_side_history = []

        wrist_hand_points = {"L":None, "R":None}
        if pose_results.pose_landmarks is not None:
            pose_landmarks = pose_results.pose_landmarks

            pose_landmark_list = calc_pose_landmark_list(debug_image, pose_landmarks)

            pose_landmark_list = calc_new_pose_landmarks(pose_landmark_list)

        hand_side_history = []
        
        draw = DrawOnCamera(cv)
        calcs = Calcs(cv)

        wrist_hand_points = {"L": None, "R": None}
        if pose_results.pose_landmarks is not None:
            pose_landmarks = pose_results.pose_landmarks

            pose_landmark_list = calcs.calc_pose_landmark_list(debug_image, pose_landmarks)

            pose_landmark_list = calcs.calc_new_pose_landmarks(pose_landmark_list)

        #  ####################################################################
        if hand_results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(
                hand_results.multi_hand_landmarks, hand_results.multi_handedness
            ):
                # Bounding box calculation
                brect = calcs.calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calcs.calc_landmark_list(debug_image, hand_landmarks)

                hand_side = handedness.classification[0].label[0]
                hand_side_history.append(hand_side)

                hand_side = handedness.classification[0].label[0]
                hand_side_history.append(hand_side)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list[hand_side] = pre_process_point_history(
                    debug_image, point_history[hand_side]
                )
                # Write to the dataset file ####################################################################
                logging_csv(
                    mode_manager,
                    pre_processed_landmark_list,
                    pre_processed_point_history_list[hand_side],
                )

                # Hand sign classification
                sign_percantage = keypoint_classifier(pre_processed_landmark_list)

                y_da_mao_size = [
                    calcs.calc_euclidian_distance(landmark_list[0], landmark_list[17])
                ]
                x_da_mao_size = [
                    calcs.calc_euclidian_distance(landmark_list[5], landmark_list[17])
                ]

                point_history[hand_side].append(
                    landmark_list[0] + y_da_mao_size + x_da_mao_size
                )

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list[hand_side])
                if point_history_len == (history_length * 4):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list[hand_side]
                    )

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history[hand_side].append(finger_gesture_id)
                most_common_fg_id = 0
                if hand_side == "L":
                    most_common_fg_id = Counter(
                        finger_gesture_history[hand_side]
                    ).most_common()
                    
                else:
                    most_common_fg_id = [[finger_gesture_history[hand_side][-1]]]

                # Getting the top 3 more probable signs
                probability_rank = ranking_sign_probability(
                    keypoint_classifier_labels, list(sign_percantage)
                )

                wrist_hand_points[hand_side] = landmark_list[0]
                location = identify_hand_area(
                    landmark_list[5], hand_side, pose_landmark_list
                )

                if mode_manager.is_body_able():
                    debug_image = draw.draw_pose_landmarks(
                        debug_image,
                        pose_landmark_list,
                        wrist_hand_points,
                        location,
                        hand_side,
                    )

                if mode_manager.is_hand_able():
                    # Drawing part
                    debug_image = draw.draw_bounding_rect(use_brect, debug_image, brect)
                    debug_image = draw.draw_landmarks(debug_image, landmark_list)
                    debug_image = draw.draw_info_text(
                        debug_image,
                        brect,
                        hand_side,
                        point_history_classifier_labels[most_common_fg_id[0][0]],
                        probability_rank,
                    )

                if CM != probability_rank[0][0]:
                    cm_timer = 0
                    has_a_new_word = False

                CM = probability_rank[0][0]
                if 15 < cm_timer  and not has_a_new_word:
                    result = repo.getSignByCMAndLocal(CM,location)

                    if len(result) == 1:
                        word = result.getFirstMottoEn() if mode_manager.is_english_on() else result.getFirstMotto()

                        if draw_word == "" or draw_word != word:
                            draw_word = word
                            has_a_new_word = True
                            threading.Thread(target=play_word_in_background, args=(word,)).start()
                    
                    elif len(result) > 1:
                        for trajectory_index in most_common_fg_id:
                            trajectory = point_history_classifier_labels[trajectory_index[0]]
                            result_filtered = result.filterSignBySense(trajectory)
                            if len(result_filtered) == 1:
                                word = result_filtered.getFirstMottoEn() if mode_manager.is_english_on() else result_filtered.getFirstMotto()
                                if draw_word == "" or draw_word != word:
                                    draw_word = word
                                    has_a_new_word = True
                                    threading.Thread(target=play_word_in_background, args=(word,)).start()
                                    break
            
                if cm_timer > 150:
                    CM = ""
                    cm_timer = 0
                    blank_timer = 0
                    draw_word = ""

            cm_timer += 1
            [
                point_history[side].append([0, 0, 0, 0])
                for side in ("L", "R")
                if side not in hand_side_history
            ]
        else:
            point_history["L"].append([0, 0, 0, 0])
            point_history["R"].append([0, 0, 0, 0])

            blank_timer += 1
            if blank_timer == 150:
                CM = ""
                cm_timer = 0
                blank_timer = 0

        if mode_manager.is_hand_able():
            debug_image = draw.draw_point_history(debug_image, point_history)
        debug_image = draw.draw_info(
            debug_image, fps, mode_manager, cm_timer, draw_word
        )

        # Screen reflection #############################################################
        cv.imshow("Hand Gesture Recognition", debug_image)

    cap.release()
    cv.destroyAllWindows()

def play_word_in_background(word):
    Talks.play(word)



def identify_hand_area(point, hand_side, pose_landmark):
    location = ""
    if (
        (pose_landmark[1][0] < point[0] < pose_landmark[0][0]
        and pose_landmark[0][1] < point[1] < pose_landmark[10][1]) 
    ):
        location = "TESTA"  

    elif (
        pose_landmark[1][0] < point[0] < pose_landmark[0][0]
        and pose_landmark[10][1] < point[1] < pose_landmark[2][1]
    ):
        location = "BOCA"

    elif (
        pose_landmark[3][0] < point[0] < pose_landmark[2][0]
        and pose_landmark[2][1] < point[1] < pose_landmark[4][1]
    ):
        neck_side = "L" if point[0] < pose_landmark[12][0] else "R"
        if neck_side == hand_side:
            location = "PESCOCO IPSILATERAL"
        else:
            location = "PESCOCO CONTRALATERAL"

    elif (
        pose_landmark[5][0] < point[0] < pose_landmark[4][0]
        and pose_landmark[4][1] < point[1] < pose_landmark[8][1]
    ):
        
        side = "L" if point[0] < pose_landmark[13][0] else "R"
        location = "IPSILATERAL" if side == hand_side else "CONTRALATERAL"

        if pose_landmark[16][1] < point[1] < pose_landmark[8][1]:
            location = "BARRIGA " + location
        else:
            location = "PEITORAL " + location
        
    else:
        location = "NEUTRA"

    return location

def ranking_sign_probability(hand_sign_list, percentage_list):
    atribuition_list = dict(zip(hand_sign_list, percentage_list))
    sorted_items = sorted(
        atribuition_list.items(), key=lambda item: item[1], reverse=True
    )

    probability_rank = [
        [sign, f"{percentage*100:.1f}"] for sign, percentage in sorted_items[:3]
    ]
    return probability_rank


def calc_euclidian_distance(x1, x2):
    return int(((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2) ** (1/2))


def calc_pose_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_points = []

    landmark_list = [landmarks.landmark[i] for i in range(25) 
                     if i not in (0,1,2,4,5,7,8,15,16,17,18,19,20,21,22)] 
    for _, landmark in enumerate(landmark_list):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_points.append([landmark_x, landmark_y])

    return landmark_points

def calc_new_pose_landmarks(landmark_point):
    head_wrist_left = [(landmark_point[2][i] * 3) // 5 + (landmark_point[4][i] * 2) // 5 for i in range(2)]
    head_wrist_right = [(landmark_point[3][i] * 3) // 5 + (landmark_point[5][i] * 2) // 5 for i in range(2)]

    eye_mean = [(landmark_point[0][i] + landmark_point[1][i]) // 2 for i in range(2)]
    head_wrist_mean = [(head_wrist_left[i] + head_wrist_right[i]) // 2 for i in range(2)]
    shoulder_mean = [(landmark_point[4][i] + landmark_point[5][i]) // 2 for i in range(2)]
    hip_mean = [(landmark_point[8][i] + landmark_point[9][i]) // 2 for i in range(2)]

    head_top = [head_wrist_mean[0],eye_mean[1] * 2 - head_wrist_mean[1]]

    head_top_left = [head_wrist_left[0],head_top[1]]
    head_top_right = [head_wrist_right[0],head_top[1]]

    head_mean_left = [head_wrist_left[0],eye_mean[1]]
    head_mean_right = [head_wrist_right[0],eye_mean[1]]

    new_external_landmarks=[head_top_left, head_top_right, head_wrist_left, head_wrist_right]
    new_internal_landmarks=[head_mean_left, head_mean_right, head_wrist_mean, shoulder_mean, hip_mean]

    return new_external_landmarks + landmark_point[4:] + new_internal_landmarks

def identify_hand_area(point, hand_side, pose_landmark):
    location = ""
    if pose_landmark[1][0] < point[0] < pose_landmark[0][0] and pose_landmark[0][1] < point[1] < pose_landmark[10][1]:
        location = "TESTA"

    elif pose_landmark[1][0] < point[0] < pose_landmark[0][0] and pose_landmark[10][1] < point[1] < pose_landmark[2][1]:
        location = "BOCA"

    elif pose_landmark[3][0] < point[0] < pose_landmark[2][0] and pose_landmark[2][1] < point[1] < pose_landmark[4][1]:
        neck_side = "L" if point[0] < pose_landmark[12][0] else "R"
        if neck_side == hand_side:
            location = "PESCOCO IPSILATERAL"
        else:
            location = "PESCOCO CONTRALATERAL"
        
    elif pose_landmark[5][0] < point[0] < pose_landmark[4][0] and pose_landmark[4][1] < point[1] < pose_landmark[8][1]:
        torso_side = "L" if point[0] < pose_landmark[13][0] else "R"
        if torso_side == hand_side:
            location = "TORSO IPSILATERAL"
        else:
            location = "TORSO CONTRALATERAL"

    else:
        location = "NEUTRA"
    
    return location



def ranking_sign_probability(hand_sign_list, percentage_list):
    atribuition_list = dict(zip(hand_sign_list,percentage_list))
    sorted_items = sorted(atribuition_list.items(), key=lambda item: item[1], reverse=True)

    probability_rank = [[sign,f"{percentage*100:.1f}"] for sign, percentage in sorted_items[:3]]
    return probability_rank


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y, base_zy, base_zx = 0, 0, 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y, base_zy, base_zx = point[0], point[1], point[2], point[3]

        temp_point_history[index][0] = (
            temp_point_history[index][0] - base_x
        ) / image_width
        temp_point_history[index][1] = (
            temp_point_history[index][1] - base_y
        ) / image_height
        temp_point_history[index][2] = (
            temp_point_history[index][2] - base_zy
        ) / image_height
        temp_point_history[index][3] = (
            temp_point_history[index][3] - base_zx
        ) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(mode_manager, landmark_list, point_history_list):
    mode, train_index = mode_manager.get_current_train_mode() #(Nothing, CM, Movement, Rotation)

    if mode == 1:
        csv_path = "model/keypoint_classifier/keypoint.csv"
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([train_index, *landmark_list])

    if mode == 2:
        csv_path = "model/point_history_classifier/point_history.csv"
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([train_index, *point_history_list])
    return


if __name__ == "__main__":
    main()
