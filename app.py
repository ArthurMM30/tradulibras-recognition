#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
from unidecode import unidecode

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier
from model import RotationHistoryClassifier
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

    rotation_history_classifier = RotationHistoryClassifier()

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
    with open(
        "model/rotation_history_classifier/rotation_history_classifier_label.csv",
        encoding="utf-8-sig",
    ) as f:
        rotation_history_classifier_labels = csv.reader(f)
        rotation_history_classifier_labels = [
            row[0] for row in rotation_history_classifier_labels
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

    rotation_history = {
        "L": deque(maxlen=history_length),
        "R": deque(maxlen=history_length),
    }
    pre_processed_rotation_history_list = {"L": [], "R": []}

    rotation_gesture_history = {
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

        wrist_hand_points = {"L": None, "R": None}
        if pose_results.pose_landmarks is not None:
            pose_landmarks = pose_results.pose_landmarks

            pose_landmark_list = calc_pose_landmark_list(debug_image, pose_landmarks)

            pose_landmark_list = calc_new_pose_landmarks(pose_landmark_list)

        #  ####################################################################
        if hand_results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(
                hand_results.multi_hand_landmarks, hand_results.multi_handedness
            ):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                hand_side = handedness.classification[0].label[0]
                hand_side_history.append(hand_side)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list[hand_side] = pre_process_point_history(
                    debug_image, point_history[hand_side]
                )

                pre_processed_rotation_history_list[hand_side] = (
                    pre_process_rotation_history(
                        debug_image, rotation_history[hand_side]
                    )
                )
                # Write to the dataset file ####################################################################
                logging_csv(
                    number,
                    mode,
                    pre_processed_landmark_list,
                    pre_processed_point_history_list[hand_side],
                    pre_processed_rotation_history_list[hand_side],
                    record_on,
                )

                # Hand sign classification
                sign_percantage = keypoint_classifier(pre_processed_landmark_list)

                y_da_mao_size = [
                    calc_euclidian_distance(landmark_list[0], landmark_list[17])
                ]
                x_da_mao_size = [
                    calc_euclidian_distance(landmark_list[5], landmark_list[17])
                ]

                # Para a rotação
                k_da_mao_size = [
                    calc_euclidian_distance(landmark_list[0], landmark_list[9])
                ]

                point_history[hand_side].append(
                    landmark_list[0] + y_da_mao_size + x_da_mao_size
                )

                rotation_history[hand_side].append(x_da_mao_size + k_da_mao_size)

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list[hand_side])
                if point_history_len == (history_length * 4):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list[hand_side]
                    )

                # Rotation gesture classification
                rotation_gesture_id = 0
                rotation_history_len = len(
                    pre_processed_rotation_history_list[hand_side]
                )
                if rotation_history_len == (history_length * 2):
                    rotation_gesture_id = rotation_history_classifier(
                        pre_processed_rotation_history_list[hand_side]
                    )

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history[hand_side].append(finger_gesture_id)
                most_common_fg_id = 0
                if hand_side == "L":
                    most_common_fg_id = Counter(
                        finger_gesture_history[hand_side]
                    ).most_common()
                    print(most_common_fg_id)
                else:
                    most_common_fg_id = [[finger_gesture_history[hand_side][-1]]]

                rotation_gesture_history[hand_side].append(rotation_gesture_id)

                print("testeeee", rotation_gesture_id)
                most_common_rotation_id = 0
                most_common_rotation_id = Counter(
                    rotation_gesture_history[hand_side]
                ).most_common()
                print(most_common_rotation_id)

                # Getting the top 3 more probable signs
                probability_rank = ranking_sign_probability(
                    keypoint_classifier_labels, list(sign_percantage)
                )

                wrist_hand_points[hand_side] = landmark_list[0]
                location = identify_hand_area(
                    landmark_list[5], hand_side, pose_landmark_list
                )
                if mode == 3:
                    debug_image = draw_pose_landmarks(
                        debug_image,
                        pose_landmark_list,
                        wrist_hand_points,
                        location,
                        hand_side,
                    )

                if mode != 6:
                    # Drawing part
                    debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    debug_image = draw_info_text(
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
                            result_filtered = result.filterSignBySense(trajectory)
                            print(
                                "RESULT FILTERED:"
                                + str(result_filtered.get())
                                + "\nTRAJECTORY:"
                                + trajectory
                            )
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
            debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(
            debug_image, fps, mode, number, cm_timer, phrase, record_on
        )

        # Screen reflection #############################################################
        cv.imshow("Hand Gesture Recognition", debug_image)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode, number, record_on):
    if mode != 1 and mode != 2 and mode != 7:
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
    if key == ord("f"):
        mode = 7
    return number, mode, record_on


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def calc_euclidian_distance(x1, x2):
    return int(((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2) ** (1 / 2))


def calc_pose_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_points = []

    landmark_list = [
        landmarks.landmark[i]
        for i in range(25)
        if i not in (0, 1, 2, 4, 5, 7, 8, 15, 16, 17, 18, 19, 20, 21, 22)
    ]
    for _, landmark in enumerate(landmark_list):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_points.append([landmark_x, landmark_y])

    return landmark_points


def calc_new_pose_landmarks(landmark_point):
    head_wrist_left = [
        (landmark_point[2][i] * 3) // 5 + (landmark_point[4][i] * 2) // 5
        for i in range(2)
    ]
    head_wrist_right = [
        (landmark_point[3][i] * 3) // 5 + (landmark_point[5][i] * 2) // 5
        for i in range(2)
    ]

    eye_mean = [(landmark_point[0][i] + landmark_point[1][i]) // 2 for i in range(2)]
    head_wrist_mean = [
        (head_wrist_left[i] + head_wrist_right[i]) // 2 for i in range(2)
    ]
    shoulder_mean = [
        (landmark_point[4][i] + landmark_point[5][i]) // 2 for i in range(2)
    ]
    hip_mean = [(landmark_point[8][i] + landmark_point[9][i]) // 2 for i in range(2)]

    head_top = [head_wrist_mean[0], eye_mean[1] * 2 - head_wrist_mean[1]]

    head_top_left = [head_wrist_left[0], head_top[1]]
    head_top_right = [head_wrist_right[0], head_top[1]]

    head_mean_left = [head_wrist_left[0], eye_mean[1]]
    head_mean_right = [head_wrist_right[0], eye_mean[1]]

    new_external_landmarks = [
        head_top_left,
        head_top_right,
        head_wrist_left,
        head_wrist_right,
    ]
    new_internal_landmarks = [
        head_mean_left,
        head_mean_right,
        head_wrist_mean,
        shoulder_mean,
        hip_mean,
    ]

    return new_external_landmarks + landmark_point[4:] + new_internal_landmarks


def identify_hand_area(point, hand_side, pose_landmark):
    location = ""
    if (
        pose_landmark[1][0] < point[0] < pose_landmark[0][0]
        and pose_landmark[0][1] < point[1] < pose_landmark[10][1]
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
        torso_side = "L" if point[0] < pose_landmark[13][0] else "R"
        if torso_side == hand_side:
            location = "TORSO IPSILATERAL"
        else:
            location = "TORSO CONTRALATERAL"

    else:
        location = "NEUTRA"

    return location


def identify_hand_rotation(point, hand_side, pose_landmark):
    location = ""
    if (
        pose_landmark[1][0] < point[0] < pose_landmark[0][0]
        and pose_landmark[0][1] < point[1] < pose_landmark[10][1]
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
        torso_side = "L" if point[0] < pose_landmark[13][0] else "R"
        if torso_side == hand_side:
            location = "TORSO IPSILATERAL"
        else:
            location = "TORSO CONTRALATERAL"

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


def pre_process_rotation_history(image, rotation_history):
    image_height = image.shape[0]

    temp_rotation_history = copy.deepcopy(rotation_history)

    # Convert to relative coordinates
    base_zy, base_zx = 0, 0
    for index, point in enumerate(temp_rotation_history):
        if index == 0:
            base_zy, base_zx = point[0], point[1]

        temp_rotation_history[index][0] = (
            temp_rotation_history[index][0] - base_zy
        ) / image_height
        temp_rotation_history[index][1] = (
            temp_rotation_history[index][1] - base_zx
        ) / image_height

    # Convert to a one-dimensional list
    temp_rotation_history = list(itertools.chain.from_iterable(temp_rotation_history))

    return temp_rotation_history


def logging_csv(
    number, mode, landmark_list, point_history_list, rotation_history_list, record_on
):

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

    if mode == 7 and (0 <= number <= 9) and record_on:
        csv_path = "model/rotation_history_classifier/rotation_history.csv"
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *rotation_history_list])
    return


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[2]),
            tuple(landmark_point[3]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[3]),
            tuple(landmark_point[4]),
            (255, 255, 255),
            2,
        )

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[5]),
            tuple(landmark_point[6]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[6]),
            tuple(landmark_point[7]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[7]),
            tuple(landmark_point[8]),
            (255, 255, 255),
            2,
        )

        # Middle finger
        cv.line(
            image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[9]),
            tuple(landmark_point[10]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[10]), tuple(landmark_point[11]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[10]),
            tuple(landmark_point[11]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[11]), tuple(landmark_point[12]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[11]),
            tuple(landmark_point[12]),
            (255, 255, 255),
            2,
        )

        # Ring finger
        cv.line(
            image, tuple(landmark_point[13]), tuple(landmark_point[14]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[13]),
            tuple(landmark_point[14]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[14]), tuple(landmark_point[15]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[14]),
            tuple(landmark_point[15]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[15]), tuple(landmark_point[16]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[15]),
            tuple(landmark_point[16]),
            (255, 255, 255),
            2,
        )

        # Little finger
        cv.line(
            image, tuple(landmark_point[17]), tuple(landmark_point[18]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[17]),
            tuple(landmark_point[18]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[18]), tuple(landmark_point[19]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[18]),
            tuple(landmark_point[19]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[19]), tuple(landmark_point[20]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[19]),
            tuple(landmark_point[20]),
            (255, 255, 255),
            2,
        )

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[0]),
            tuple(landmark_point[1]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[1]),
            tuple(landmark_point[2]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[2]),
            tuple(landmark_point[5]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[5]),
            tuple(landmark_point[9]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[9]),
            tuple(landmark_point[13]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[13]), tuple(landmark_point[17]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[13]),
            tuple(landmark_point[17]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[17]),
            tuple(landmark_point[0]),
            (20, 180, 240),
            2,
        )
        cv.line(
            image, tuple(landmark_point[17]), tuple(landmark_point[5]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[17]),
            tuple(landmark_point[5]),
            (20, 180, 240),
            2,
        )

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # Wrist 1
            cv.circle(image, (landmark[0], landmark[1]), 7, (45, 0, 210), -1)
            cv.circle(image, (landmark[0], landmark[1]), 7, (0, 0, 0), 1)
        if index == 1:  # Wrist 2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # Thumb: base
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # Thumb: first joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # Thumb: fingertip
            cv.circle(image, (landmark[0], landmark[1]), 7, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 7, (0, 0, 0), 1)
        if index == 5:  # Index finger: base
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # Index finger: second joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # Index finger: first joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # Index finger: fingertip
            cv.circle(image, (landmark[0], landmark[1]), 7, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 7, (0, 0, 0), 1)
        if index == 9:  # Middle finger: base
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # Middle finger: second joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # Middle finger: first joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # Middle finger: fingertip
            cv.circle(image, (landmark[0], landmark[1]), 7, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 7, (0, 0, 0), 1)
        if index == 13:  # Ring finger: base
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # Ring finger: second joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # Ring finger: first joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # Ring finger: fingertip
            cv.circle(image, (landmark[0], landmark[1]), 7, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 7, (0, 0, 0), 1)
        if index == 17:  # Little finger: base
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # Little finger: second joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # Little finger: first joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # Little finger: fingertip
            cv.circle(image, (landmark[0], landmark[1]), 7, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 7, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, hand_side, finger_gesture_text, probability_rank):
    image_width = image.shape[1]

    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)

    info_text = hand_side
    if probability_rank[0][0] != None:
        info_text = info_text + ":" + probability_rank[0][0]
    cv.putText(
        image,
        info_text,
        (brect[0] + 5, brect[1] - 4),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )
    cv.putText(
        image,
        probability_rank[0][1],
        (brect[2] - 44, brect[1] - 4),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )

    if finger_gesture_text != "":
        if hand_side == "L":
            cv.putText(
                image,
                "T: " + finger_gesture_text,
                (10, 60),
                cv.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 0),
                4,
                cv.LINE_AA,
            )
            cv.putText(
                image,
                "T: " + finger_gesture_text,
                (10, 60),
                cv.FONT_HERSHEY_SIMPLEX,
                1.0,
                (152, 251, 152),
                2,
                cv.LINE_AA,
            )
        else:
            cv.putText(
                image,
                "T: " + finger_gesture_text,
                (image_width - 350, 60),
                cv.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 0),
                4,
                cv.LINE_AA,
            )
            cv.putText(
                image,
                "T: " + finger_gesture_text,
                (image_width - 350, 60),
                cv.FONT_HERSHEY_SIMPLEX,
                1.0,
                (152, 152, 251),
                2,
                cv.LINE_AA,
            )

    cv.rectangle(
        image, (brect[0], brect[3]), (brect[2], brect[3] + 44), (63, 63, 63), -1
    )

    cv.putText(
        image,
        probability_rank[1][0],
        (brect[0] + 5, brect[3] + 16),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )
    cv.putText(
        image,
        probability_rank[1][1],
        (brect[2] - 44, brect[3] + 16),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )

    cv.putText(
        image,
        probability_rank[2][0],
        (brect[0] + 5, brect[3] + 40),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )
    cv.putText(
        image,
        probability_rank[2][1],
        (brect[2] - 44, brect[3] + 40),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history["L"]):
        if point[0] != 0 and point[1] != 0:
            cv.circle(
                image,
                (point[0], point[1]),
                (point[2] ** 2 * 6) // 700 + index,
                (152, 251, 152),
                2,
            )

    for index, point in enumerate(point_history["R"]):
        if point[0] != 0 and point[1] != 0:
            cv.circle(
                image,
                (point[0], point[1]),
                (point[2] ** 2 * 6) // 700 + index,
                (152, 152, 251),
                2,
            )

    return image


def draw_info(image, fps, mode, number, timer, phrase, record_on):
    image_width, image_height = image.shape[1], image.shape[0]
    cv.putText(
        image,
        str(fps),
        (image_width // 2 - 30, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        4,
        cv.LINE_AA,
    )
    cv.putText(
        image,
        str(fps),
        (image_width // 2 - 30, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )

    mode_string = [
        "Logging Key Point",
        "Logging Point History",
        "Logging Rotation History",
    ]
    if number == "":
        number = "0"

    if 1 <= mode <= 2:
        active = " ON" if record_on else " OFF"
        cv.putText(
            image,
            "MODE:" + mode_string[mode - 1] + active,
            (image_width // 2 - 100, 70),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv.LINE_AA,
        )
        cv.putText(
            image,
            "MODE:" + mode_string[mode - 1] + active,
            (image_width // 2 - 100, 70),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )
        if 0 <= int(number) <= 99:

            cv.putText(
                image,
                "K:" + number,
                (image.shape[1] - 250, image_height - 25),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                4,
                cv.LINE_AA,
            )
            cv.putText(
                image,
                "K:" + number,
                (image.shape[1] - 250, image_height - 25),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv.LINE_AA,
            )

    if mode != 6:
        cv.putText(
            image,
            "TIMER:" + str(timer),
            (10, image_height - 25),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            4,
            cv.LINE_AA,
        )
        cv.putText(
            image,
            "TIMER:" + str(timer),
            (10, image_height - 25),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv.LINE_AA,
        )

    if mode == 7:
        active = " ON" if record_on else " OFF"
        cv.putText(
            image,
            "MODE:" + mode_string[2] + active,
            (image_width // 2 - 100, 70),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv.LINE_AA,
        )
        cv.putText(
            image,
            "MODE:" + mode_string[2] + active,
            (image_width // 2 - 100, 70),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )
        if 0 <= int(number) <= 99:
            cv.putText(
                image,
                "K:" + number,
                (image.shape[1] - 250, image_height - 25),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                4,
                cv.LINE_AA,
            )
            cv.putText(
                image,
                "K:" + number,
                (image.shape[1] - 250, image_height - 25),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv.LINE_AA,
            )

    phrase = [unidecode(word) for word in phrase]
    cv.putText(
        image,
        " ".join(phrase),
        (image_width // 2 - 80, image_height - 25),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        3,
        cv.LINE_AA,
    )
    cv.putText(
        image,
        " ".join(phrase),
        (image_width // 2 - 80, image_height - 25),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )

    return image


def draw_pose_landmarks(image, landmark_point, landmark_wrist, location, hand_side):
    cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 6)
    cv.line(
        image, tuple(landmark_point[0]), tuple(landmark_point[1]), (255, 255, 255), 2
    )

    cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[2]), (0, 0, 0), 6)
    cv.line(
        image, tuple(landmark_point[0]), tuple(landmark_point[2]), (255, 255, 255), 2
    )

    cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[3]), (0, 0, 0), 6)
    cv.line(
        image, tuple(landmark_point[1]), tuple(landmark_point[3]), (255, 255, 255), 2
    )

    cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6)
    cv.line(
        image, tuple(landmark_point[2]), tuple(landmark_point[3]), (255, 255, 255), 2
    )

    cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[4]), (0, 0, 0), 6)
    cv.line(
        image, tuple(landmark_point[2]), tuple(landmark_point[4]), (255, 255, 255), 2
    )

    cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[5]), (0, 0, 0), 6)
    cv.line(
        image, tuple(landmark_point[3]), tuple(landmark_point[5]), (255, 255, 255), 2
    )

    cv.line(image, tuple(landmark_point[4]), tuple(landmark_point[5]), (0, 0, 0), 6)
    cv.line(
        image, tuple(landmark_point[4]), tuple(landmark_point[5]), (255, 255, 255), 2
    )

    cv.line(image, tuple(landmark_point[4]), tuple(landmark_point[6]), (0, 0, 0), 6)
    cv.line(
        image, tuple(landmark_point[4]), tuple(landmark_point[6]), (255, 255, 255), 2
    )

    cv.line(image, tuple(landmark_point[4]), tuple(landmark_point[8]), (0, 0, 0), 6)
    cv.line(
        image, tuple(landmark_point[4]), tuple(landmark_point[8]), (255, 255, 255), 2
    )

    cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[7]), (0, 0, 0), 6)
    cv.line(
        image, tuple(landmark_point[5]), tuple(landmark_point[7]), (255, 255, 255), 2
    )

    cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 6)
    cv.line(
        image, tuple(landmark_point[5]), tuple(landmark_point[9]), (255, 255, 255), 2
    )

    if landmark_wrist["R"] != None:
        cv.line(
            image, tuple(landmark_point[6]), tuple(landmark_wrist["R"]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[6]),
            tuple(landmark_wrist["R"]),
            (255, 255, 255),
            2,
        )

    if landmark_wrist["L"] != None:
        cv.line(
            image, tuple(landmark_point[7]), tuple(landmark_wrist["L"]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[7]),
            tuple(landmark_wrist["L"]),
            (255, 255, 255),
            2,
        )

    cv.line(image, tuple(landmark_point[8]), tuple(landmark_point[9]), (0, 0, 0), 6)
    cv.line(
        image, tuple(landmark_point[8]), tuple(landmark_point[9]), (255, 255, 255), 2
    )

    cv.line(
        image, tuple(landmark_point[10]), tuple(landmark_point[11]), (45, 0, 210), 2
    )

    cv.line(
        image, tuple(landmark_point[12]), tuple(landmark_point[13]), (45, 0, 210), 2
    )

    cv.line(
        image, tuple(landmark_point[13]), tuple(landmark_point[14]), (45, 0, 210), 2
    )

    for landmark in landmark_point[:10]:
        cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
        cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)

    location = location.replace("LATERAL", ".")
    if hand_side == "L":
        cv.putText(
            image,
            "L: " + location,
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            4,
            cv.LINE_AA,
        )
        cv.putText(
            image,
            "L: " + location,
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (152, 251, 152),
            2,
            cv.LINE_AA,
        )
    else:
        cv.putText(
            image,
            "L: " + location,
            (image.shape[1] - 350, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            4,
            cv.LINE_AA,
        )
        cv.putText(
            image,
            "L: " + location,
            (image.shape[1] - 350, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (152, 152, 251),
            2,
            cv.LINE_AA,
        )

    return image


if __name__ == "__main__":
    main()
