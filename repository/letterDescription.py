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


    def filterLetterBySense(self, data_request):
        data_response = []

        for sign in self.data:
            if sign["sense"] == data_request:
                data_response.append(sign)

        return LetterDescriptionEntity(data_response)

    def get(self):
        return self.data    

    def getFirstLetter(self):
        return self.data[0]["motto"] 
    
    def __len__(self):
        return len(self.data)