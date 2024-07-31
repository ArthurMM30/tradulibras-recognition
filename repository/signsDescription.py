from pymongo import MongoClient


class SignsDescriptionClient():
    def __init__(self):
        self.client = MongoClient("mongodb://localhost:27017/")
        self.dbs = self.client['TraduLibras']
        self.collection = self.dbs['signsDescription']


    def getSignByCM(self, data_request, index=0, is_dominant=True):
        data_response = []
        hand = "dominant_hand" if is_dominant else "auxiliar_hand"
        query = {f"phonology.{str(index)}.{hand}.CM": data_request}

        for sign in self.collection.find(query):
            data_response.append(sign)

        return SignsDescriptionEntity(data_response)

    def getSignByCMAndLocal(self, data_request_cm, data_request_local, index=0, is_dominant=True):
        data_response = []
        hand = "dominant_hand" if is_dominant else "auxiliar_hand"
        cm_query = {f"phonology.{str(index)}.{hand}.CM": data_request_cm}
        local_query = {f"phonology.{str(index)}.{hand}.final_local": data_request_local}

        for sign in self.collection.find({"$and":[cm_query, local_query]}):
            data_response.append(sign)

        return SignsDescriptionEntity(data_response)
    
    def close(self):
        self.client.close()

    
class SignsDescriptionEntity(object):
    def __init__(self, data):
        self.data = data

    def filterSignByFinalLocal(self, data_request, index, is_dominant):
        data_response = []

        hand = "dominant_hand" if is_dominant else "auxiliar_hand"

        for sign in self.data:
            if sign["phonology"][index][hand]["final_local"] == data_request:
                data_response.append(sign)

        return SignsDescriptionEntity(data_response)

    def filterSignBySense(self, data_request, index=0, is_dominant=True):
        data_response = []

        hand = "dominant_hand" if is_dominant else "auxiliar_hand"

        for sign in self.data:
            if sign["phonology"][index][hand]["sense"] == data_request:
                data_response.append(sign)

        return SignsDescriptionEntity(data_response)

    def get(self):
        return self.data    

    def getFirstMotto(self):
        return self.data[0]["motto"] 

    def getFirstMottoEn(self):
        return self.data[0]["motto_en"]
    
    def __len__(self):
        return len(self.data)
