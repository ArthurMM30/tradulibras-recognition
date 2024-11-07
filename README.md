# Tradulibras
O Tradulibras é uma solução inovadora que visa promover a inclusão e facilitar a comunicação entre surdos e ouvintes. Ele utiliza Inteligência Artificial para traduzir gestos e sinais em Libras (Língua Brasileira de Sinais) para texto e áudio, tornando a comunicação mais fluida e acessível.

Funcionalidades
Tradução de Libras para Texto e Áudio: Captura e traduz gestos em Libras para texto em português e, opcionalmente, gera áudio com a tradução.
Inclusão e Acessibilidade: Promove a comunicação entre surdos e ouvintes em diferentes contextos, proporcionando mais oportunidades de interação e integração.

## Arquitetura:
1. **Reconhecimento de Imagem** </br>
Descrição: Este módulo captura imagens e converte as mãos em coordenadas cartesianas. </br>
Tecnologia: Deep Learning. </br>
Objetivo: Identificar as posições das mãos para posterior tradução. </br>
2. **Reconhecimento de Corpo** </br>
Descrição: Detecta os movimentos do corpo e os gestos em Libras, convertendo-os em coordenadas espaciais. </br>
Tecnologia: Deep Learning. </br>
Objetivo: Identificar movimentos corporais, como gestos e expressões faciais, que são cruciais para a tradução de Libras. </br>
3. **Configuração de Mão (CM)** </br>
Descrição: Este módulo traduz as coordenadas das mãos e dos gestos em Libras para texto em português. </br>
Tecnologia: Machine Learning. </br>
Objetivo: Realizar a conversão dos gestos para um formato legível e compreensível em texto. </br>
4. **Configuração de Movimento** </br>
Descrição: Detecta movimentos complexos do corpo e das mãos, garantindo que gestos mais elaborados sejam corretamente traduzidos. </br>
Tecnologia: Machine Learning. </br>
Objetivo: Melhorar a precisão na interpretação de gestos dinâmicos. </br>
5. **Rotação** </br> 
Descrição: Este módulo ajusta a posição das mãos e do corpo, corrigindo a orientação e garantindo que a tradução seja precisa independentemente da rotação do usuário.</br>
Tecnologia: Machine Learning.</br>
Objetivo: Tratar e corrigir qualquer rotação ou mudança de posição que possa interferir na tradução.</br>

## Como realizar a instalação?

### Requisitos
Para executar o Tradulibras, você precisa de:

-> Python 3.x: A linguagem de programação principal para o desenvolvimento. </br>
-> MongoDB: Banco de dados NoSQL utilizado para armazenar dados de sinais e traduções.

**Passo 1:** </br>
 clone o repositório:
 ```bash
  git clone https://github.com/ArthurMM30/tradulibras-recognition
 ```

**Passo 2:**  </br>
configure as seguintes bibliotecas:

**OpenCV**: </br>
processamento de imagens e vídeos
```bash
pip install opencv-python
```
**MediaPipe**:</br>
detecção e rastreamento das mãos
```bash
pip install mediapipe
```
**Numpy**:</br>
Manipulação de arrays e cálculos
```bash
pip install numpy
```
**Pandas**:</br>
Análise e manipulação de dados
```bash
pip install pandas
```
**Matplotlib**:</br>
Visualização de dados
```bash
pip install matplotlib
```
**TensorFlow**:</br>
Framework de IA
```bash
pip install tensorflow
```
**pymongo**:</br>
Integração do Python com o MongoDB
```bash
pip install pymongo
```
**Unidecode**:</br>
Normalização de texto
```bash
pip install unidecode
```
**Seaborn**:</br>
Visualização de dados
```bash
pip install seaborn
```
**Playsound**:</br>
Conversão de texto para áudio
```bash
pip install playsound
```

**Passo 3:** </br>
rode o arquivo app.py
```bash
python app.py
```



