db.createCollection("signsDescription");
db.signsDescription.insertMany([
  {
    _id: ObjectId("66b4bef5c27061b8ec99ef66"),
    motto: "eu",
    motto_en: "I",
    spelling: false,
    phonology: [
      {
        dominant_hand: {
          CM: "1",
          trajectory: "RETO",
          sense: "TRAS",
          final_local: "PEITORAL NEUTRO",
          rotation: "ROTAÇÃO VERTICAL",
        },
        auxiliar_hand: null,
      },
    ],
  },
  {
    _id: ObjectId("66b4bef5c27061b8ec99ef67"),
    motto: "você",
    motto_en: "you",
    spelling: false,
    phonology: [
      {
        dominant_hand: {
          CM: "1",
          trajectory: "RETO",
          sense: "FRENTE",
          final_local: "PEITORAL IPSILATERAL",
        },
        auxiliar_hand: null,
      },
    ],
  },
  {
    _id: ObjectId("66b4bef8c27061b8ec99ef68"),
    motto: "amor",
    motto_en: "love",
    spelling: false,
    phonology: [
      {
        dominant_hand: {
          CM: "S",
          trajectory: "RETO",
          sense: "ESQUERDA",
          final_local: "PEITORAL CONTRALATERAL",
        },
        auxiliar_hand: null,
      },
    ],
  },
]);
