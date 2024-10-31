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
import mediapipe as mp

from utils import CvFpsCalc
from utils import DrawOnCamera
from utils import Calcs
from utils import Talks
from utils import ModeManager
from utils import TimerManager
from model import KeyPointClassifier
from model import PointHistoryClassifier
from repository.signsDescription import SignsDescriptionClient
from repository.letterDescription import LetterDescriptionClient


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
    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    repo_sign = SignsDescriptionClient()
    repo_letter = LetterDescriptionClient()

    mode_manager = ModeManager()
    timer_manager = TimerManager()
    draw = DrawOnCamera(cv)
    calcs = Calcs(cv)

    cvFpsCalc = CvFpsCalc(buffer_len=10)

    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

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

    history_length = 16
    point_history = {
        "L": deque(maxlen=history_length),
        "R": deque(maxlen=history_length),
    }
    pre_processed_point_history_list = {"L": [], "R": []}

    finger_gesture_history = {
        "L": deque(maxlen=history_length // 4),
        "R": deque(maxlen=history_length),
    }

    word = ""

    while True:
        fps = cvFpsCalc.get()

        key = cv.waitKey(10)
        if key == 27:  # ESC
            break

        received_command = mode_manager.alter_mode_by_key(key)
        if received_command == "s":
            spelled_word = word
            word = ""

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

        hand_side_history = (
            []
        )  # ######################################################### que isso?

        wrist_hand_points = {"L": None, "R": None}
        if pose_results.pose_landmarks is not None:
            pose_landmarks = pose_results.pose_landmarks
            pose_landmark_list = calcs.calc_pose_landmark_list(
                debug_image, pose_landmarks
            )
            pose_landmark_list = calcs.calc_new_pose_landmarks(pose_landmark_list)

        #  ####################################################################
        if hand_results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(
                hand_results.multi_hand_landmarks, hand_results.multi_handedness
            ):
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
                    received_command,
                )

                # Hand sign classification
                sign_percantage = keypoint_classifier(pre_processed_landmark_list)

                y_hand_axis_size = calcs.calc_euclidian_distance(
                    landmark_list[0], landmark_list[17]
                )
                x_hand_axis_size = calcs.calc_euclidian_distance(
                    landmark_list[5], landmark_list[17]
                )

                point_history[hand_side].append(
                    landmark_list[0] + [y_hand_axis_size] + [x_hand_axis_size]
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
                location = calcs.identify_hand_area(
                    landmark_list[5], hand_side, pose_landmark_list
                )

                if mode_manager.is_hand_able():
                    brect = calcs.calc_bounding_rect(debug_image, hand_landmarks)

                    debug_image = draw.draw_bounding_rect(debug_image, brect)
                    debug_image = draw.draw_landmarks(debug_image, landmark_list)
                    debug_image = draw.draw_info_text(
                        debug_image,
                        brect,
                        hand_side,
                        point_history_classifier_labels[most_common_fg_id[0][0]],
                        probability_rank,
                    )

                if mode_manager.is_body_able():
                    debug_image = draw.draw_pose_landmarks(
                        debug_image,
                        pose_landmark_list,
                        wrist_hand_points,
                        location,
                        hand_side,
                    )

                if timer_manager.check_if_movement_updated(
                    most_common_fg_id[0][0]
                ) or timer_manager.check_if_CM_updated(probability_rank[0][0]):
                    timer_manager.reset_timer()

                if mode_manager.is_spelling_on():
                    if 13 < timer_manager.get_timer() and timer_manager.is_able():
                        result = repo_letter.getLetterByCM(probability_rank[0][0])
                        if len(result) == 1:
                            word += result.getFirstLetter()
                            timer_manager.enable()

                else:
                    # SEMITIR AINDA QUE A MÃO NÃO ESTEJA EXPOSTA
                    if received_command == "s":  # reconheceu quando acabou a soletragem
                        threading.Thread(
                            target=play_word_in_background,
                            args=(spelled_word.lower(), True),
                        ).start()
                    if 1 < timer_manager.get_timer() and timer_manager.is_able():
                        result = repo_sign.getSignByCMAndLocal(
                            probability_rank[0][0], location
                        )
                        print(probability_rank[0][0], location, len(result))

                        if len(result) == 1:
                            word = (
                                result.getFirstMottoEn()
                                if mode_manager.is_english_on()
                                else result.getFirstMotto()
                            )

                            # if draw_word == "" or draw_word != word:
                            # draw_word = word
                            timer_manager.enable()
                            threading.Thread(
                                target=play_word_in_background, args=(word,)
                            ).start()

                        elif len(result) > 1:
                            for trajectory_index in most_common_fg_id:
                                trajectory = point_history_classifier_labels[
                                    trajectory_index[0]
                                ]
                                result_filtered = result.filterSignBySense(trajectory)
                                if len(result_filtered) == 1:
                                    word = (
                                        result_filtered.getFirstMottoEn()
                                        if mode_manager.is_english_on()
                                        else result_filtered.getFirstMotto()
                                    )
                                    # if draw_word == "" or draw_word != word:
                                    # draw_word = word
                                    timer_manager.enable()

                                    threading.Thread(
                                        target=play_word_in_background, args=(word,)
                                    ).start()
                                    break

                if timer_manager.get_timer() > 150:
                    timer_manager.reset_timer()

            timer_manager.increase_timer()
            [
                point_history[side].append([0, 0, 0, 0])
                for side in ("L", "R")
                if side not in hand_side_history
            ]
        else:
            point_history["L"].append([0, 0, 0, 0])
            point_history["R"].append([0, 0, 0, 0])

            if timer_manager.get_blank_timer() == 150:
                # draw_word = ""
                timer_manager.reset()

            timer_manager.increase_blank_timer()

        if mode_manager.is_hand_able():
            debug_image = draw.draw_point_history(debug_image, point_history)
        debug_image = draw.draw_info(
            debug_image, fps, mode_manager, timer_manager.get_timer(), word
        )

        # Screen reflection #############################################################
        cv.imshow("Hand Gesture Recognition", debug_image)

    cap.release()
    cv.destroyAllWindows()


def play_word_in_background(word, isSpelling=False):
    Talks.play(word, isSpelling)


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


cm_list = []


def logging_csv(mode_manager, landmark_list, point_history_list, received_command):
    global cm_list
    mode, train_index = mode_manager.get_current_train_mode(
        received_command
    )  # (Nothing, CM, Movement, Rotation)

    if mode == 1:
        cm_list.append([train_index, *landmark_list])
        if received_command == "r":
            csv_path = "model/keypoint_classifier/keypoint.csv"
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                for cm in cm_list:
                    writer.writerow(cm)
            cm_list = []

    if mode == 2:
        csv_path = "model/point_history_classifier/point_history.csv"
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([train_index, *point_history_list])
    return


if __name__ == "__main__":
    main()
