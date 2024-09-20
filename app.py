#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from utils import DrawOnCamera
from utils import Calcs
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
    mode = 0
    number = ""
    record_on = False

    cm_timer = 0
    blank_timer = 0
    CM = ""
    phrase = []
    language = "pt-br"

    has_a_new_word = False

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, new_mode, record_on = select_mode(key, mode, number, record_on)

        if new_mode != mode and (new_mode == 4 or new_mode == 5):
            phrase = []
            language = "pt-br" if new_mode == 4 else "en"

        mode = new_mode

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

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list[hand_side] = pre_process_point_history(
                    debug_image, point_history[hand_side]
                )
                # Write to the dataset file ####################################################################
                logging_csv(
                    number,
                    mode,
                    pre_processed_landmark_list,
                    pre_processed_point_history_list[hand_side],
                    record_on,
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
                if mode == 3:
                    debug_image = draw.draw_pose_landmarks(
                        debug_image,
                        pose_landmark_list,
                        wrist_hand_points,
                        location,
                        hand_side,
                    )

                if mode != 6:
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
                if 15 < cm_timer and not has_a_new_word:
                    result = repo.getSignByCMAndLocal(CM, location)
                    print("RESULT:" + str(result.get()))
                                
                    if len(result) == 1:
                        word = (
                            result.getFirstMotto()
                            if language == "pt-br"
                            else result.getFirstMottoEn()
                        )

                        if len(phrase) == 0 or phrase[-1] != word:
                            phrase.append(word)
                            has_a_new_word = True

                
                    elif len(result) > 1:
                        for trajectory_index in most_common_fg_id:
                            trajectory = point_history_classifier_labels[
                                trajectory_index[0]
                            ]
                            
                            print(trajectory)
                            result_filtered = result.filterSignBySense(trajectory)
                            if len(result_filtered) == 1:
                                word = (
                                    result_filtered.getFirstMotto()
                                    if language == "pt-br"
                                    else result_filtered.getFirstMottoEn()
                                )
                                if len(phrase) == 0 or phrase[-1] != word:
                                    phrase.append(word)
                                    has_a_new_word = True
                                    break
                    
                        
                        
                        
                    
                if cm_timer > 150:
                    CM = ""
                    cm_timer = 0
                    blank_timer = 0
                    phrase = []

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

        if mode != 6:
            debug_image = draw.draw_point_history(debug_image, point_history)
        debug_image = draw.draw_info(
            debug_image, fps, mode, number, cm_timer, phrase, record_on
        )

        # Screen reflection #############################################################
        cv.imshow("Hand Gesture Recognition", debug_image)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode, number, record_on):
    if mode != 1 and mode != 2:
        number = ""
    if ord("0") <= key <= ord("9"):
        if len(number) == 2:
            number = str(key - 48)
        else:
            number = number + str(key - 48)
    if key == ord("n"):
        mode = 0
    if key == ord("k"):  # CM configuration mode
        mode = 1
    if key == ord("h"):
        mode = 2
    if key == ord("r"):
        record_on = not record_on
    if key == ord("b"):  # to view the body
        mode = 3
    if key == ord("p"):
        mode = 4
    if key == ord("e"):
        mode = 5
    if key == ord("x"):
        mode = 6
    return number, mode, record_on



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


def logging_csv(number, mode, landmark_list, point_history_list, record_on):
    number = int(number) if number != "" else 0
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9) and record_on:
        csv_path = "model/keypoint_classifier/keypoint.csv"
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9) and record_on:
        csv_path = "model/point_history_classifier/point_history.csv"
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


if __name__ == "__main__":
    main()
