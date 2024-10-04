import numpy as np
class Calcs:
    def __init__(self, cv_module) -> None:
        self.cv = cv_module
    
    def calc_bounding_rect(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point = [np.array((landmark_x, landmark_y))]

            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x, y, w, h = self.cv.boundingRect(landmark_array)

        return [x, y, x + w, y + h]


    def calc_landmark_list(self,image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        # Keypoint
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            # landmark_z = landmark.z

            landmark_point.append([landmark_x, landmark_y])

        return landmark_point


    def calc_euclidian_distance(self, x1, x2):
        return int(((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2) ** (1 / 2))


    def calc_pose_landmark_list(self, image, landmarks):
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


    def calc_new_pose_landmarks(sef, landmark_point):
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
        torso_mean_left = [(landmark_point[4][i] + landmark_point[8][i]) // 2 for i in range(2)]
        torso_mean_right = [(landmark_point[5][i] + landmark_point[9][i]) // 2 for i in range(2)]
        
        
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
            torso_mean_left,
            torso_mean_right
        ]

        return new_external_landmarks + landmark_point[4:] + new_internal_landmarks

