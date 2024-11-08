class TimerManager:
    def __init__(self):
        self.timer = 0
        self.blank_timer = 0
        self.left_last_CM = ""
        self.right_last_CM = ""
        self.last_movement = ""
        self.able = True
        self.index = 0
        self.save_result_hand = {}

    # def check_if_CM_updated(self, CM):

    def check_if_movement_updated(self, movement):
        if self.last_movement != movement:
            self.last_movement = movement
            return True
        return False

    def check_if_CM_updated(self, CM, hand_side):
        if hand_side == "L":
            if self.left_last_CM != CM:
                self.left_last_CM = CM
                return True
            return False
        else:
            if self.right_last_CM != CM:
                self.right_last_CM = CM
                return True
            return False

    def is_able(self):
        return self.able
    
    def get_timer(self):
        return self.timer
    
    def get_blank_timer(self):
        return self.timer
    
    def get_index(self):
        return self.index


    def increase_timer(self):
        self.timer += 1

    def increase_blank_timer(self):
        self.blank_timer += 1

    def reset_timer(self):
        self.timer = 0
        self.blank_timer = 0
        self.able = True

        # self.last_movement = ""

    def set_save_result_hand(self, save_result_hand):
        self.save_result_hand = save_result_hand
        
    def get_save_result_hand(self):
        return self.save_result_hand
    
    def enable(self):
        self.able = False
    
    def set_index(self, index):
        self.index = index