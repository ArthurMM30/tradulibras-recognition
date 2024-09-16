from unidecode import unidecode 
class DrawOnCamera:
    
    def __init__(self, cv_module):
        self.cv = cv_module 

    def draw_landmarks(self, image, landmark_point):
        if len(landmark_point) > 0:
            # Thumb
            self.cv.line(
                image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6
            )
            self.cv.line(
                image,
                tuple(landmark_point[2]),
                tuple(landmark_point[3]),
                (255, 255, 255),
                2,
            )
            
            self.cv.line(
                image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), 6
            )
            self.cv.line(
                image,
                tuple(landmark_point[3]),
                tuple(landmark_point[4]),
                (255, 255, 255),
                2,
            )

            # Index finger
            self.cv.line(
                image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 0, 0), 6
            )
            self.cv.line(
                image,
                tuple(landmark_point[5]),
                tuple(landmark_point[6]),
                (255, 255, 255),
                2,
            )
            self.cv.line(
                image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 0, 0), 6
            )
            self.cv.line(
                image,
                tuple(landmark_point[6]),
                tuple(landmark_point[7]),
                (255, 255, 255),
                2,
            )
            self.cv.line(
                image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 0, 0), 6
            )
            self.cv.line(
                image,
                tuple(landmark_point[7]),
                tuple(landmark_point[8]),
                (255, 255, 255),
                2,
            )

            # Middle finger
            self.cv.line(
                image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 0, 0), 6
            )
            self.cv.line(
                image,
                tuple(landmark_point[9]),
                tuple(landmark_point[10]),
                (255, 255, 255),
                2,
            )
            self.cv.line(
                image,
                tuple(landmark_point[10]),
                tuple(landmark_point[11]),
                (0, 0, 0),
                6,
            )
            self.cv.line(
                image,
                tuple(landmark_point[10]),
                tuple(landmark_point[11]),
                (255, 255, 255),
                2,
            )
            self.cv.line(
                image,
                tuple(landmark_point[11]),
                tuple(landmark_point[12]),
                (0, 0, 0),
                6,
            )
            self.cv.line(
                image,
                tuple(landmark_point[11]),
                tuple(landmark_point[12]),
                (255, 255, 255),
                2,
            )

            # Ring finger
            self.cv.line(
                image, tuple(landmark_point[13]), tuple(landmark_point[14]), (0, 0, 0), 6
            )
            self.cv.line(
                image,
                tuple(landmark_point[13]),
                tuple(landmark_point[14]),
                (255, 255, 255),
                2,
            )
            self.cv.line(
                image, tuple(landmark_point[14]), tuple(landmark_point[15]), (0, 0, 0), 6
            )
            self.cv.line(
                image,
                tuple(landmark_point[14]),
                tuple(landmark_point[15]),
                (255, 255, 255),
                2,
            )
            self.cv.line(
                image, tuple(landmark_point[15]), tuple(landmark_point[16]), (0, 0, 0), 6
            )
            self.cv.line(
                image,
                tuple(landmark_point[15]),
                tuple(landmark_point[16]),
                (255, 255, 255),
                2,
            )

            # Little finger
            self.cv.line(
                image, tuple(landmark_point[17]), tuple(landmark_point[18]), (0, 0, 0), 6
            )
            self.cv.line(
                image,
                tuple(landmark_point[17]),
                tuple(landmark_point[18]),
                (255, 255, 255),
                2,
            )
            self.cv.line(
                image, tuple(landmark_point[18]), tuple(landmark_point[19]), (0, 0, 0), 6
            )
            self.cv.line(
                image,
                tuple(landmark_point[18]),
                tuple(landmark_point[19]),
                (255, 255, 255),
                2,
            )
            self.cv.line(
                image, tuple(landmark_point[19]), tuple(landmark_point[20]), (0, 0, 0), 6
            )
            self.cv.line(
                image,
                tuple(landmark_point[19]),
                tuple(landmark_point[20]),
                (255, 255, 255),
                2,
            )

            # Palm
            self.cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 6)
            self.cv.line(
                image,
                tuple(landmark_point[0]),
                tuple(landmark_point[1]),
                (255, 255, 255),
                2,
            )
            self.cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 0, 0), 6)
            self.cv.line(
                image,
                tuple(landmark_point[1]),
                tuple(landmark_point[2]),
                (255, 255, 255),
                2,
            )
            self.cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (0, 0, 0), 6)
            self.cv.line(
                image,
                tuple(landmark_point[2]),
                tuple(landmark_point[5]),
                (255, 255, 255),
                2,
            )
            self.cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 6)
            self.cv.line(
                image,
                tuple(landmark_point[5]),
                tuple(landmark_point[9]),
                (255, 255, 255),
                2,
            )
            self.cv.line(
                image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 0, 0), 6
            )
            self.cv.line(
                image,
                tuple(landmark_point[9]),
                tuple(landmark_point[13]),
                (255, 255, 255),
                2,
            )
            self.cv.line(
                image, tuple(landmark_point[13]), tuple(landmark_point[17]), (0, 0, 0), 6
            )
            self.cv.line(
                image,
                tuple(landmark_point[13]),
                tuple(landmark_point[17]),
                (255, 255, 255),
                2,
            )
            self.cv.line(
                image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 0, 0), 6
            )
            self.cv.line(
                image,
                tuple(landmark_point[17]),
                tuple(landmark_point[0]),
                (20, 180, 240),
                2,
            )
            self.cv.line(
                image, tuple(landmark_point[17]), tuple(landmark_point[5]), (0, 0, 0), 6
            )
            self.cv.line(
                image,
                tuple(landmark_point[17]),
                tuple(landmark_point[5]),
                (20, 180, 240),
                2,
            )

        # Key Points
        for index, landmark in enumerate(landmark_point):
            if index == 0:  # Wrist 1
                self.cv.circle(image, (landmark[0], landmark[1]), 7, (45, 0, 210), -1)
                self.cv.circle(image, (landmark[0], landmark[1]), 7, (0, 0, 0), 1)
            if index == 1:  # Wrist 2
                self.cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                self.cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 2:  # Thumb: base
                self.cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                self.cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 3:  # Thumb: first joint
                self.cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                self.cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 4:  # Thumb: fingertip
                self.cv.circle(image, (landmark[0], landmark[1]), 7, (255, 255, 255), -1)
                self.cv.circle(image, (landmark[0], landmark[1]), 7, (0, 0, 0), 1)
            if index == 5:  # Index finger: base
                self.cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                self.cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 6:  # Index finger: second joint
                self.cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                self.cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 7:  # Index finger: first joint
                self.cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                self.cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 8:  # Index finger: fingertip
                self.cv.circle(image, (landmark[0], landmark[1]), 7, (255, 255, 255), -1)
                self.cv.circle(image, (landmark[0], landmark[1]), 7, (0, 0, 0), 1)
            if index == 9:  # Middle finger: base
                self.cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                self.cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 10:  # Middle finger: second joint
                self.cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                self.cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 11:  # Middle finger: first joint
                self.cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                self.cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 12:  # Middle finger: fingertip
                self.cv.circle(image, (landmark[0], landmark[1]), 7, (255, 255, 255), -1)
                self.cv.circle(image, (landmark[0], landmark[1]), 7, (0, 0, 0), 1)
            if index == 13:  # Ring finger: base
                self.cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                self.cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 14:  # Ring finger: second joint
                self.cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                self.cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 15:  # Ring finger: first joint
                self.cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                self.cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 16:  # Ring finger: fingertip
                self.cv.circle(image, (landmark[0], landmark[1]), 7, (255, 255, 255), -1)
                self.cv.circle(image, (landmark[0], landmark[1]), 7, (0, 0, 0), 1)
            if index == 17:  # Little finger: base
                self.cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                self.cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 18:  # Little finger: second joint
                self.cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                self.cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 19:  # Little finger: first joint
                self.cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                self.cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 20:  # Little finger: fingertip
                self.cv.circle(image, (landmark[0], landmark[1]), 7, (255, 255, 255), -1)
                self.cv.circle(image, (landmark[0], landmark[1]), 7, (0, 0, 0), 1)

        return image


    def draw_bounding_rect(self, use_brect, image, brect):
        if use_brect:
            # Outer rectangle
            self.cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)

        return image


    def draw_info_text(self, image, brect, hand_side, finger_gesture_text, probability_rank):
        image_width = image.shape[1]

        self.cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)

        info_text = hand_side
        if probability_rank[0][0] != None:
            info_text = info_text + ":" + probability_rank[0][0]
        self.cv.putText(
            image,
            info_text,
            (brect[0] + 5, brect[1] - 4),
            self.cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            self.cv.LINE_AA,
        )
        self.cv.putText(
            image,
            probability_rank[0][1],
            (brect[2] - 44, brect[1] - 4),
            self.cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            self.cv.LINE_AA,
        )

        if finger_gesture_text != "":
            if hand_side == "L":
                self.cv.putText(
                    image,
                    "T: " + finger_gesture_text,
                    (10, 60),
                    self.cv.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 0),
                    4,
                    self.cv.LINE_AA,
                )
                self.cv.putText(
                    image,
                    "T: " + finger_gesture_text,
                    (10, 60),
                    self.cv.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (152, 251, 152),
                    2,
                    self.cv.LINE_AA,
                )
            else:
                self.cv.putText(
                    image,
                    "T: " + finger_gesture_text,
                    (image_width - 350, 60),
                    self.cv.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 0),
                    4,
                    self.cv.LINE_AA,
                )
                self.cv.putText(
                    image,
                    "T: " + finger_gesture_text,
                    (image_width - 350, 60),
                    self.cv.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (152, 152, 251),
                    2,
                    self.cv.LINE_AA,
                )

        self.cv.rectangle(
            image, (brect[0], brect[3]), (brect[2], brect[3] + 44), (63, 63, 63), -1
        )

        self.cv.putText(
            image,
            probability_rank[1][0],
            (brect[0] + 5, brect[3] + 16),
            self.cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            self.cv.LINE_AA,
        )
        self.cv.putText(
            image,
            probability_rank[1][1],
            (brect[2] - 44, brect[3] + 16),
            self.cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            self.cv.LINE_AA,
        )

        self.cv.putText(
            image,
            probability_rank[2][0],
            (brect[0] + 5, brect[3] + 40),
            self.cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            self.cv.LINE_AA,
        )
        self.cv.putText(
            image,
            probability_rank[2][1],
            (brect[2] - 44, brect[3] + 40),
            self.cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            self.cv.LINE_AA,
        )

        return image


    def draw_point_history(self,image, point_history):
        for index, point in enumerate(point_history["L"]):
            if point[0] != 0 and point[1] != 0:
                self.cv.circle(
                    image,
                    (point[0], point[1]),
                    (point[2] ** 2 * 6) // 700 + index,
                    (152, 251, 152),
                    2,
                )

        for index, point in enumerate(point_history["R"]):
            if point[0] != 0 and point[1] != 0:
                self.cv.circle(
                    image,
                    (point[0], point[1]),
                    (point[2] ** 2 * 6) // 700 + index,
                    (152, 152, 251),
                    2,
                )

        return image


    def draw_info(self,image, fps, mode, number, timer, phrase, record_on):
        image_width, image_height = image.shape[1], image.shape[0]

        self.cv.putText(
            image,
            str(fps),
            (image_width // 2 - 30, 30),
            self.cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            4,
            self.cv.LINE_AA,
        )
        self.cv.putText(
            image,
            str(fps),
            (image_width // 2 - 30, 30),
            self.cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            self.cv.LINE_AA,
        )

        mode_string = ["Logging Key Point", "Logging Point History"]
        if number == "":
            number = "0"

        if 1 <= mode <= 2:
            active = " ON" if record_on else " OFF"
            self.cv.putText(
                image,
                "MODE:" + mode_string[mode - 1] + active,
                (image_width // 2 - 100, 70),
                self.cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                self.cv.LINE_AA,
            )
            self.cv.putText(
                image,
                "MODE:" + mode_string[mode - 1] + active,
                (image_width // 2 - 100, 70),
                self.cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                self.cv.LINE_AA,
            )
            if 0 <= int(number) <= 99:

                self.cv.putText(
                    image,
                    "K:" + number,
                    (image.shape[1] - 250, image_height - 25),
                    self.cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    4,
                    self.cv.LINE_AA,
                )
                self.cv.putText(
                    image,
                    "K:" + number,
                    (image.shape[1] - 250, image_height - 25),
                    self.cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    self.cv.LINE_AA,
                )

        if mode != 6:
            self.cv.putText(
                image,
                "TIMER:" + str(timer),
                (10, image_height - 25),
                self.cv.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 0),
                4,
                self.cv.LINE_AA,
            )
            self.cv.putText(
                image,
                "TIMER:" + str(timer),
                (10, image_height - 25),
                self.cv.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                self.cv.LINE_AA,
            )

        phrase = [unidecode(word) for word in phrase]
        self.cv.putText(
            image,
            " ".join(phrase),
            (image_width // 2 - 80, image_height - 25),
            self.cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            3,
            self.cv.LINE_AA,
        )
        self.cv.putText(
            image,
            " ".join(phrase),
            (image_width // 2 - 80, image_height - 25),
            self.cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            self.cv.LINE_AA,
        )

        return image


    def draw_pose_landmarks(self,image, landmark_point, landmark_wrist, location, hand_side):
        self.cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 6)
        self.cv.line(
            image, tuple(landmark_point[0]), tuple(landmark_point[1]), (255, 255, 255), 2
        )

        self.cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[2]), (0, 0, 0), 6)
        self.cv.line(
            image, tuple(landmark_point[0]), tuple(landmark_point[2]), (255, 255, 255), 2
        )

        self.cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[3]), (0, 0, 0), 6)
        self.cv.line(
            image, tuple(landmark_point[1]), tuple(landmark_point[3]), (255, 255, 255), 2
        )

        self.cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6)
        self.cv.line(
            image, tuple(landmark_point[2]), tuple(landmark_point[3]), (255, 255, 255), 2
        )

        self.cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[4]), (0, 0, 0), 6)
        self.cv.line(
            image, tuple(landmark_point[2]), tuple(landmark_point[4]), (255, 255, 255), 2
        )

        self.cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[5]), (0, 0, 0), 6)
        self.cv.line(
            image, tuple(landmark_point[3]), tuple(landmark_point[5]), (255, 255, 255), 2
        )

        self.cv.line(image, tuple(landmark_point[4]), tuple(landmark_point[5]), (0, 0, 0), 6)
        self.cv.line(
            image, tuple(landmark_point[4]), tuple(landmark_point[5]), (255, 255, 255), 2
        )

        self.cv.line(image, tuple(landmark_point[4]), tuple(landmark_point[6]), (0, 0, 0), 6)
        self.cv.line(
            image, tuple(landmark_point[4]), tuple(landmark_point[6]), (255, 255, 255), 2
        )

        self.cv.line(image, tuple(landmark_point[4]), tuple(landmark_point[8]), (0, 0, 0), 6)
        self.cv.line(
            image, tuple(landmark_point[4]), tuple(landmark_point[8]), (255, 255, 255), 2
        )

        self.cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[7]), (0, 0, 0), 6)
        self.cv.line(
            image, tuple(landmark_point[5]), tuple(landmark_point[7]), (255, 255, 255), 2
        )

        self.cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 6)
        self.cv.line(
            image, tuple(landmark_point[5]), tuple(landmark_point[9]), (255, 255, 255), 2
        )

        if landmark_wrist["R"] != None:
            self.cv.line(
                image, tuple(landmark_point[6]), tuple(landmark_wrist["R"]), (0, 0, 0), 6
            )
            self.cv.line(
                image,
                tuple(landmark_point[6]),
                tuple(landmark_wrist["R"]),
                (255, 255, 255),
                2,
            )

        if landmark_wrist["L"] != None:
            self.cv.line(
                image, tuple(landmark_point[7]), tuple(landmark_wrist["L"]), (0, 0, 0), 6
            )
            self.cv.line(
                image,
                tuple(landmark_point[7]),
                tuple(landmark_wrist["L"]),
                (255, 255, 255),
                2,
            )

        self.cv.line(image, tuple(landmark_point[8]), tuple(landmark_point[9]), (0, 0, 0), 6)
        self.cv.line(
            image, tuple(landmark_point[8]), tuple(landmark_point[9]), (255, 255, 255), 2
        )

        self.cv.line(
            image, tuple(landmark_point[10]), tuple(landmark_point[11]), (45, 0, 210), 2
        )

        self.cv.line(
            image, tuple(landmark_point[12]), tuple(landmark_point[13]), (45, 0, 210), 2
        )

        self.cv.line(
            image, tuple(landmark_point[13]), tuple(landmark_point[14]), (45, 0, 210), 2
        )

        for landmark in landmark_point[:10]:
            self.cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            self.cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)

        location = location.replace("LATERAL", ".")
        if hand_side == "L":
            self.cv.putText(
                image,
                "L: " + location,
                (10, 30),
                self.cv.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 0),
                4,
                self.cv.LINE_AA,
            )
            self.cv.putText(
                image,
                "L: " + location,
                (10, 30),
                self.cv.FONT_HERSHEY_SIMPLEX,
                1.0,
                (152, 251, 152),
                2,
                self.cv.LINE_AA,
            )
        else:
            self.cv.putText(
                image,
                "L: " + location,
                (image.shape[1] - 350, 30),
                self.cv.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 0),
                4,
                self.cv.LINE_AA,
            )
            self.cv.putText(
                image,
                "L: " + location,
                (image.shape[1] - 350, 30),
                self.cv.FONT_HERSHEY_SIMPLEX,
                1.0,
                (152, 152, 251),
                2,
                self.cv.LINE_AA,
            )

        return image
