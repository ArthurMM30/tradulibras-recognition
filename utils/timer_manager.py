class TimerManager:
    def __init__(self):
        self.timer = 0
        self.blank_timer = 0
        self.last_CM = ""
        self.last_movement = ""
        self.able = True
        self.index = 0

    # def check_if_CM_updated(self, CM):

    def check_if_movement_updated(self, movement):
        if self.last_movement != movement:
            self.last_movement = movement
            return True
        return False

    def check_if_CM_updated(self, CM):
        if self.last_CM != CM:
            self.last_CM = CM
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

    def enable(self):
        self.able = False
    
    def set_index(self, index):
        self.index = index