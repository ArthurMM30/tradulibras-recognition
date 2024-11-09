from pymongo import MongoClient

class LetterDescriptionClient():
    def __init__(self):
        self.client = MongoClient("mongodb://localhost:27017/")
        self.dbs = self.client['TraduLibras']
        self.collection = self.dbs['letterDescription']

    def getLetterByCM(self, data_request_cm):
        data_response = []
        
        cm_query = {"CM": data_request_cm}

        for sign in self.collection.find(cm_query):
            data_response.append(sign)

        return LetterDescriptionEntity(data_response)
    
    def close(self):
        self.client.close()

    
class LetterDescriptionEntity(object):
    def __init__(self, data):
        self.data = data

    def validate_if_have_rotation(self, data_index=0):
        for data in self.data:
            if data["sense"][data_index] in ("FLEXAO", "ROTACAO"):
                return True

        return False
    
    def validateSense(self, sense, index, data_index=0):
        return self.data[data_index]["sense"][index] == sense

    def filter_by_sense(self, sense):
        data_response = []

        for letter in self.data:
            if letter["sense"][0] == sense:
                data_response.append(letter)

        return LetterDescriptionEntity(data_response)

    def get(self):
        return self.data 

    def getFirstLetter(self):
        return self.data[0]["motto"] 
    
    def __len__(self):
        return len(self.data)