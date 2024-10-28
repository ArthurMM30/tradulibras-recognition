class ModeManager:
    def __init__(self) -> None:
        self.mode = 0       # (Train Nothing, Train CM, Train History, Train Rotation)
        self.view_mode = 1  # (Show nothing, Show hands, Show hands and body)
        self.record_on = False 
        self.english_on = False
        self.train_index = 0
    
    def alter_mode_by_key(self, key):
        if ord("0") <= key <= ord("9"): # Number pictor to training
            if self.train_index > 9:
                self.train_index = key - 48
            else:
                self.train_index = self.train_index * 10 + key - 48
        if key == ord("n"):     # Clean mode
            self.mode = 0
        if key == ord("k"):     # Key configuration mode
            self.mode = 1
        if key == ord("h"):     # History point configuration mode
            self.mode = 2
        if key == ord("r"):     # Record toggle
            self.record_on = not self.record_on
        if key == ord("b"):     # Able Body
            self.view_mode += 1
            self.view_mode %= 3
        if key == ord("e"):     # Language Toggle
            self.english_on = not self.english_on


    # Train Options

    def is_train_mode(self):
        return self.mode != 0
    
    def get_train_index(self):
        return self.train_index

    def get_train_text(self):
        mode_string = ["","CM Training", "Movement Training", "Rotation Training"]

        return mode_string[self.mode]
    
    def get_current_train_mode(self):
        if not self.record_on:
            return 0, None
        

        return self.mode, self.train_index 


    # View Options

    def is_clean_mode(self):
        return self.view_mode == 0

    def is_hand_able(self):
        return self.view_mode >= 1

    def is_body_able(self):
        return self.view_mode == 2


    # Language options

    def is_english_on(self):
        return self.english_on


    # Record options

    def is_record_on(self):
        return self.record_on
        
