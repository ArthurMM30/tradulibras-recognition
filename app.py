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
from utils import TimerManager
from model import SignKeyPointClassifier
from model import SpellingKeyPointClassifier
from model import PointHistoryClassifier
from model import RotationHistoryClassifier
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
        default=0.8,
    )
    parser.add_argument(
        "--min_tracking_confidence",
        help="min_tracking_confidence",
        type=int,
        default=0.5,
    )
    parser.add_argument(
        "--min_detection_confidence_pose",
        help="min_detection_confidence_pose",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--min_tracking_confidence_pose",
        help="min_tracking_confidence_pose",
        type=int,
        default=0.5,
    )

    args = parser.parse_args()

    return args


def main():
    sign_keypoint_classifier = SignKeyPointClassifier()
    spelling_keypoint_classifier = SpellingKeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()
    rotation_history_classifier = RotationHistoryClassifier()

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
    min_detection_confidence_pose = args.min_detection_confidence_pose
    min_tracking_confidence_pose = args.min_tracking_confidence_pose

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
        min_detection_confidence=min_detection_confidence_pose,
        min_tracking_confidence=min_tracking_confidence_pose,
    )

    # Read labels ###########################################################
    with open(
        "model/sign_keypoint_classifier/keypoint_classifier_label.csv",
        encoding="utf-8-sig",
    ) as f:
        sign_keypoint_classifier_labels = csv.reader(f)
        sign_keypoint_classifier_labels = [
            row[0] for row in sign_keypoint_classifier_labels
        ]

    with open(
        "model/spelling_keypoint_classifier/keypoint_classifier_label.csv",
        encoding="utf-8-sig",
    ) as f:
        spelling_keypoint_classifier_labels = csv.reader(f)
        spelling_keypoint_classifier_labels = [
            row[0] for row in spelling_keypoint_classifier_labels
        ]

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

    history_length = 16
    point_history = {
        "L": deque(maxlen=history_length),
        "R": deque(maxlen=history_length),
    }
    pre_processed_point_history_list = {"L": [], "R": []}

    finger_gesture_history = {
        "L": deque(maxlen=history_length // 2),
        "R": deque(maxlen=history_length // 2),
    }

    rotation_history = {
        "L": deque(maxlen=history_length),
        "R": deque(maxlen=history_length),
    }
    pre_processed_rotation_history_list = {"L": [], "R": []}

    rotation_gesture_history = {
        "L": deque(maxlen=history_length // 4),
        "R": deque(maxlen=history_length // 4),
    }
    
    validate_if_a_sign_can_finish = False
    
    sign_history = []
    params_history = []

    word = ""
    hand_fidelity = 20.0

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break

        received_command = mode_manager.alter_mode_by_key(key)
        if received_command == "s":
            if not mode_manager.is_spelling_on() and word != "":
                threading.Thread(
                    target=play_word_in_background, args=(word.lower(), True)
                ).start()
            timer_manager.reset_timer()
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

        hand_side_history = []

        draw = DrawOnCamera(cv)
        calcs = Calcs(cv)

        wrist_hand_points = {"L": None, "R": None}
        if pose_results.pose_landmarks is not None:
            pose_landmarks = pose_results.pose_landmarks
            pose_landmark_list = calcs.calc_pose_landmark_list(
                debug_image, pose_landmarks
            )
            pose_landmark_list = calcs.calc_new_pose_landmarks(pose_landmark_list)

        # print(pose_results.pose_landmarks)
        # print("a")
        # print(hand_results.multi_hand_landmarks)
        #  ####################################################################
        if (
            hand_results.multi_hand_landmarks is not None
            and pose_results.pose_landmarks is not None
        ):
            for hand_landmarks, handedness in zip(
                hand_results.multi_hand_landmarks, hand_results.multi_handedness
            ):
                # Landmark calculation
                landmark_list = calcs.calc_landmark_list(debug_image, hand_landmarks)

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
                    mode_manager,
                    pre_processed_landmark_list,
                    pre_processed_point_history_list[hand_side],
                    pre_processed_rotation_history_list[hand_side],
                    received_command,
                )

                if mode_manager.is_spelling_on():
                    # Hand sign classification
                    sign_percantage = spelling_keypoint_classifier(
                        pre_processed_landmark_list
                    )

                    # Getting the top 3 more probable signs
                    probability_rank = ranking_sign_probability(
                        spelling_keypoint_classifier_labels, list(sign_percantage)
                    )
                else:
                    sign_percantage = sign_keypoint_classifier(
                        pre_processed_landmark_list
                    )

                    probability_rank = ranking_sign_probability(
                        sign_keypoint_classifier_labels, list(sign_percantage)
                    )

                y_hand_axis_size = [
                    calcs.calc_euclidian_distance(landmark_list[0], landmark_list[17])
                ]

                x_hand_axis_size = [
                    calcs.calc_euclidian_distance(landmark_list[5], landmark_list[17])
                ]

                k_hand_axis_size = [
                    calcs.calc_euclidian_distance(landmark_list[0], landmark_list[9])
                ]

                point_history[hand_side].append(
                    landmark_list[0] + y_hand_axis_size + x_hand_axis_size
                )

                rotation_history[hand_side].append(x_hand_axis_size + k_hand_axis_size)
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
                else:
                    most_common_fg_id = [[finger_gesture_history[hand_side][-1]]]

                rotation_gesture_history[hand_side].append(rotation_gesture_id)

                most_common_rotation_id = 0
                most_common_rotation_id = Counter(
                    rotation_gesture_history[hand_side]
                ).most_common()

                wrist_hand_points[hand_side] = landmark_list[0]
                location = identify_hand_area(
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
                        rotation_history_classifier_labels[
                            most_common_rotation_id[0][0]
                        ],
                        probability_rank,
                        mode_manager,
                    )

                if mode_manager.is_body_able():
                    debug_image = draw.draw_pose_landmarks(
                        debug_image,
                        pose_landmark_list,
                        wrist_hand_points,
                        location,
                        hand_side,
                    )

                if timer_manager.check_if_CM_updated(probability_rank[0][0], hand_side):
                    #Tira o eu do mock
                    timer_manager.reset_timer()

                    if validate_if_a_sign_can_finish and sign_history != []:
                        sign_history.pop(0)
                        validate_if_a_sign_can_finish = False

                if not mode_manager.is_train_mode():
                    if mode_manager.is_spelling_on():
                        cm = (
                            probability_rank[0][0]
                            if float(probability_rank[0][1]) > hand_fidelity
                            else "null"
                        )
                        if 12 < timer_manager.get_timer() and timer_manager.is_able():
                            result = repo_letter.getLetterByCM(cm)
                            if len(result) == 1:
                                if result.validateSense("REPOUSO", 0):
                                    word += result.getFirstLetter()
                                    timer_manager.enable()
                                else:
                                    for trajectory_index in most_common_fg_id:
                                        trajectory = point_history_classifier_labels[
                                            trajectory_index[0]
                                        ]
                                        result_validation = result.validateSense(
                                            trajectory,
                                            timer_manager.get_spelling_index(),
                                        )
                                        if result_validation:

                                            if (
                                                len(result.data[0]["sense"])
                                                == timer_manager.get_spelling_index()
                                                + 1
                                            ):
                                                word += result.getFirstLetter()
                                                timer_manager.enable()
                                                timer_manager.set_spelling_index(0)

                                            else:
                                                timer_manager.set_spelling_index(
                                                    timer_manager.get_spelling_index()
                                                    + 1
                                                )

                                        elif (
                                            not result_validation
                                            and timer_manager.get_spelling_index() > 0
                                        ):
                                            if result.validateSense("REPOUSO", 0):
                                                word += result.getFirstLetter()
                                                timer_manager.enable()
                                                timer_manager.set_spelling_index(0)
                            elif len(result) == 2:
                                if result.validate_if_have_rotation():
                                    result = result.filter_by_sense(
                                        rotation_history_classifier_labels[
                                            most_common_rotation_id[0][0]
                                        ]
                                    )
                                    if len(result) > 0:
                                        word += result.getFirstLetter()
                                        timer_manager.enable()
                                        timer_manager.set_spelling_index(0)
                    else:
                        #SEMITIR AINDA QUE A MÃO NÃO ESTEJA EXPOSTA                    
                        if 7 < timer_manager.get_timer() and timer_manager.is_able():
                            CM = probability_rank[0][0]
                    
                            for trajectory_index in most_common_fg_id:
                                    rotation  = rotation_history_classifier_labels[most_common_rotation_id[0][0]]
                                    trajectory = point_history_classifier_labels[trajectory_index[0]]
                                    isDominant = hand_side == "R"
                                    if(timer_manager.get_save_result_hand() != {} and timer_manager.get_save_result_hand()["side"] != hand_side): # Condicional para sinais com 2 mãos
                                        hand_temp = timer_manager.get_save_result_hand() 
                                        if(hand_temp["CM"] == CM and hand_temp["sense"] == trajectory and hand_temp["final_local"] == location and hand_temp["rotation"] == rotation) :
                                            word = hand_temp["motto"]
                                            timer_manager.set_save_result_hand({})
                                            timer_manager.enable()
                                            timer_manager.set_index(0)
                                            sign_history = []
                                            validate_if_a_sign_can_finish = False
                                            threading.Thread(target=play_word_in_background, args=(word,)).start()
                                    else:    
                                        if sign_history == []:
                                            result = repo_sign.getSignByCMAndLocalAndTrajectory(CM,location,trajectory, rotation, timer_manager.get_index(), isDominant)    
                                            if len(result) > 0:
                                                old_word = result.getFirstMotto() # Descobre se houve desistencia do sinal
                                                validate_if_a_sign_can_finish = result.validate_if_a_sign_can_finish(timer_manager.get_index() + 1)
                                                if validate_if_a_sign_can_finish:
                                                    if len(result) == 1:
                                                        if(isDominant): # Verifica se é mão direita, usado nos casos onde tem 2 mãos
                                                            auxiliar_hand = result.data[0]["phonology"][timer_manager.get_index()]["auxiliar_hand"]
                                                            if(auxiliar_hand != None): # Valida se é sinal de 2 mãos
                                                                auxiliar_hand["side"] = hand_side
                                                                auxiliar_hand["motto"] = result.data[0]["motto"]
                                                                timer_manager.set_save_result_hand(auxiliar_hand)
                                                            else:
                                                                timer_manager.set_index(0)
                                                                word = result.getFirstMotto() if not mode_manager.is_english_on() else result.getFirstMottoEn()
                                                                timer_manager.enable()
                                                                timer_manager.set_save_result_hand({})
                                                                sign_history = []
                                                                validate_if_a_sign_can_finish = False
                                                                threading.Thread(target=play_word_in_background, args=(word,)).start()
                                                                
                                                        else: # O sinal está em sua última, na mão não dominante
                                                            dominant_hand = result.data[0]["phonology"][timer_manager.get_index()]["dominant_hand"]
                                                            if(dominant_hand != None): # Valida se é sinal de 2 mãos
                                                                dominant_hand["side"] = hand_side
                                                                dominant_hand["motto"] = result.data[0]["motto"]
                                                                timer_manager.set_save_result_hand(dominant_hand)
                                                            else:
                                                                timer_manager.set_index(0)
                                                                word = result.getFirstMotto() if not mode_manager.is_english_on() else result.getFirstMottoEn()
                                                                timer_manager.enable()
                                                                timer_manager.set_save_result_hand({})
                                                                sign_history = []
                                                                validate_if_a_sign_can_finish = False
                                                                threading.Thread(target=play_word_in_background, args=(word,)).start()
                                                    else:
                                                        sign_history = result.data.copy()    
                                                        params_history.append({"CM" : CM, "trajectory" : "RETO", "sense" : trajectory, "rotation" : rotation, "final_local" : location}) 
                                                else:
                                                    timer_manager.set_index(timer_manager.get_index() + 1)   
                                                    params_history.append({"CM" : CM, "trajectory" : "RETO", "sense" : trajectory, "rotation" : rotation, "final_local" : location}) 
                                            elif len(result) == 0 and timer_manager.get_index() > 0: # Valida se houve desistência do sinal
                                                result = repo_sign.getSignByCMAndLocalAndTrajectory(CM,location,trajectory, 0)
                                                if len(result) == 1:
                                                    word_retry = result.getFirstMotto()
                                                    if old_word != word_retry: # Valida se a desistência na verda era algum oiutro sinal
                                                        timer_manager.set_index(0)
                                                        if len(result.data[0]["phonology"]) == 1:
                                                            word = result.getFirstMotto() if not mode_manager.is_english_on() else result.getFirstMottoEn()
                                                            threading.Thread(target=play_word_in_background, args=(word,)).start()     
                                        else:
                                            if 25 < timer_manager.get_timer():
                                                sign = sign_history[0]
                                                word = sign["motto"]
                                                timer_manager.enable()
                                                sign_history = []
                                                validate_if_a_sign_can_finish = False
                                                timer_manager.set_index(0)
                                                threading.Thread(target=play_word_in_background, args=(word,)).start()
                                            else:
                                                resultado = [item for item in sign_history if contem_sequencia(item["phonology"], params_history)]
                                                if len(resultado) == 1:
                                                    sign = resultado[0]
                                                    word = sign["motto"]
                                                    timer_manager.enable()
                                                    sign_history = []
                                                    validate_if_a_sign_can_finish = False
                                                    timer_manager.set_index(0)
                                                    threading.Thread(target=play_word_in_background, args=(word,)).start()
                                                    
                                                
                                                
                                                
                                            
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
                timer_manager.reset_timer()

            timer_manager.increase_blank_timer()

        if mode_manager.is_hand_able():
            debug_image = draw.draw_point_history(
                debug_image, point_history, mode_manager
            )
            debug_image = draw.draw_info(
                debug_image, fps, mode_manager, timer_manager.get_timer()
            )
        debug_image = draw.draw_word(debug_image, word)

        # Screen reflection #############################################################
        cv.namedWindow("Hand Gesture Recognition", cv.WINDOW_NORMAL)
        cv.imshow("Hand Gesture Recognition", debug_image)

    cap.release()
    cv.destroyAllWindows()


def play_word_in_background(word, isSpelling=False):
    Talks.play(word, isSpelling)


def contem_sequencia(phonology, criterios):
    dominant_hands = [ph["dominant_hand"] for ph in phonology]
    
    for i in range(len(dominant_hands) - len(criterios) + 1):
        if dominant_hands[i:i+len(criterios)] == criterios:
            return True
    return False

def select_mode(key, mode, number, record_on):
    if mode != 1 and mode != 2:
        number = ""
    if ord("0") <= key <= ord("9"):
        if len(number) == 1:
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

        side = "L" if point[0] < pose_landmark[13][0] else "R"
        location = "IPSILATERAL" if side == hand_side else "CONTRALATERAL"

        if pose_landmark[16][1] < point[1] < pose_landmark[8][1]:
            location = "BARRIGA " + location
        else:
            location = "PEITORAL " + location

    else:
        location = "NEUTRO"

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


cm_list = []
mov_list = []
rot_list = []


def logging_csv(
    mode_manager,
    landmark_list,
    point_history_list,
    rotation_history_list,
    received_command,
):
    global cm_list
    global mov_list
    global rot_list
    mode, train_index = mode_manager.get_current_train_mode(
        received_command
    )  # (Nothing, CM, Movement, Rotation)

    if mode == 1:
        if mode_manager.is_spelling_on():
            cm_list.append([train_index, *landmark_list])
            if received_command == "r":
                csv_path = "model/spelling_keypoint_classifier/keypoint.csv"
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    for cm in cm_list:
                        writer.writerow(cm)
                cm_list = []
        else:
            cm_list.append([train_index, *landmark_list])
            if received_command == "r":
                csv_path = "model/sign_keypoint_classifier/keypoint.csv"
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    for cm in cm_list:
                        writer.writerow(cm)
                cm_list = []

    if mode == 2:
        mov_list.append([train_index, *point_history_list])
        if received_command == "r":
            csv_path = "model/point_history_classifier/point_history.csv"
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                for mov in mov_list:
                    writer.writerow(mov)

                writer.writerow([train_index, *point_history_list])
            mov_list = []

    if mode == 3:
        rot_list.append([train_index, *rotation_history_list])
        if received_command == "r":
            csv_path = "model/rotation_history_classifier/rotation_history.csv"
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                for rot in rot_list:
                    writer.writerow(rot)

                writer.writerow([train_index, *rotation_history_list])
            rot_list = []
    return


if __name__ == "__main__":
    main()
