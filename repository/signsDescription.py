from pymongo import MongoClient


class SignsDescriptionClient():
    def __init__(self):
        self.client = MongoClient("mongodb://localhost:27017/")
        self.dbs = self.client['TraduLibras']
        self.collection = self.dbs['signsDescription']

    def getAllWords(self):
        data_response = []
        for word in self.collection.find():
            data_response.append(word['motto'])
        
        return data_response

    def getSignByCM(self, data_request, index=0, is_dominant=True):
        data_response = []
        hand = "dominant_hand" if is_dominant else "auxiliar_hand"
        query = {f"phonology.{str(index)}.{hand}.CM": data_request}

        for sign in self.collection.find(query):
            data_response.append(sign)

        return SignsDescriptionEntity(data_response)

    def getSignByCMAndLocalAndTrajectory(self, data_request_cm, data_request_local, trajectory, rotation ,index=0, is_dominant=True):
        data_response = []
        neutro = data_request_local.split(" ")[0] + " NEUTRO"
        
        hand = "dominant_hand" if is_dominant else "auxiliar_hand"
        cm_query = {f"phonology.{str(index)}.{hand}.CM": data_request_cm}
        
        local_query = {
            "$or": [
                {f"phonology.{str(index)}.{hand}.final_local": data_request_local},
                {f"phonology.{str(index)}.{hand}.final_local": neutro}
            ]
        }
        
        trajectory_query = {f"phonology.{str(index)}.{hand}.sense":trajectory} 
        rotation_query = {
            "$or": [
                {f"phonology.{str(index)}.{hand}.rotation": rotation},
                {f"phonology.{str(index)}.{hand}.rotation": "REPOUSO"}
            ]
        }
        

        for sign in self.collection.find({"$and":[cm_query, local_query, trajectory_query, rotation_query]}):
            data_response.append(sign)
            
        if(len(data_response) == 0):
            local_query = {f"phonology.{str(index)}.{hand}.final_local" : "NEUTRO"}
            for sign in self.collection.find({"$and":[cm_query, local_query, trajectory_query, rotation_query]}):
                data_response.append(sign)
                
        data_response.sort(key=lambda x: len(x["phonology"]))
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
    
    def validate_if_a_sign_can_finish(self, index):
        for sign in self.data:
            if len(sign["phonology"]) == index:
                return True;
        return False;

    def get(self):
        return self.data    

    def getFirstMotto(self):
        return self.data[0]["motto"] 

    def getFirstMottoEn(self):
        return self.data[0]["motto_en"]
    
    def __len__(self):
        return len(self.data)
