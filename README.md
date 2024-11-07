# Tradulibras
O Tradulibras é uma solução inovadora que visa promover a inclusão e facilitar a comunicação entre surdos e ouvintes. Ele utiliza Inteligência Artificial para traduzir gestos e sinais em Libras (Língua Brasileira de Sinais) para texto e áudio, tornando a comunicação mais fluida e acessível.

Funcionalidades
Tradução de Libras para Texto e Áudio: Captura e traduz gestos em Libras para texto em português e, opcionalmente, gera áudio com a tradução.
Inclusão e Acessibilidade: Promove a comunicação entre surdos e ouvintes em diferentes contextos, proporcionando mais oportunidades de interação e integração.

A branch principal do repositório é a main. Certifique-se de estar nessa branch para realizar a instalação e execução do projeto.

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
-> Docker: Usado para gerenciar o ambiente e rodar o MongoDB, o banco de dados NoSQL responsável pelo armazenamento de dados de sinais e traduções.

**Passo 1:** </br>
 Clone o repositório:
 ```bash
 git clone https://github.com/ArthurMM30/tradulibras-recognition
 ```
**Passo 2:**  </br>
Dentro da pasta do repositório clonado, instale as dependências necessárias, conforme listado no arquivo requirements.txt
```bash
pip install -r requirements.txt
```
**Passo 3:** </br>
Configuração das variáveis de ambiente: </br>
3.1 :
     Copie o arquivo .env.example e renomeie para .env.
   ```bash
   cp .env.example .env
   ```
Faça as alterações necessárias, mas as configurações padrão geralmente já funcionam para a maioria dos casos.

3.2 Subir ambiente no docker: </br>
     O MongoDB será executado utilizando Docker. Para iniciar o MongoDB com o script de inserção, execute o seguinte comando:
 ```bash
 docker compose up -d
 ```
   Este comando inicializa o MongoDB no modo detached (em segundo plano)
   
**Passo 4:** </br>
Com o ambiente configurado, ja é possível rodar o Projeto
```bash
python app.py
```

<H2>Está tudo pronto para você explorar a tradução de Libras! </H2>

### Forma de acesso às funcionalidades:
Ao executar o programa, você terá algumas possibilidades do que fazer:

**Teste do programa**:</br>
Ao entrar no programa por padrão ele estará no modo de soletragem, ou seja, identificará as letras do alfabeto.</br>
Teste algumas letras como A, B, C! Infelizmente tivemos problemas com a letra "T". Mas estamos trabalhando nisso!

Para sair da parte de soletragem, pressione a tecla "**S**" e dessa forma trocará para a parte de identificação de sinais! </br>
Como por exemplo: [🤙, 🤟, 👍]

Ainda temos as opções de visualização, podemos pressionar a tecla "**B**" algumas vezes:</br>
- Pressionando B 1 vez, será visualizado a palavra que está sendo gesticulada.</br>
- Pressionando B 2 vezes, será possível visualizar os pontos identificadores da mão.</br>
- Pressionando B 3 vezes, será possível visualizar os identificadores do corpo.</br>
- Pressionando B 4 vezes, será possível visualizar os frames de movimento, rotação, timer e FPS.

**Como gravar novas CMs?**</br>
Para gravas CMs de sinais e soletragem:</br>
Após abrir o programa, pressione a tecla "**K**".</br>
Verifique se está na área de sinais ou soletragem, para alterar é preciso apenas pressionar a tecla "**S**"</br>
Pressione o número desejado para gravar o novo sinal para determinar seu id. E por fim pressione a tecla "**R**" para iniciar a gravação, e pressione novamente para interromper a gravação.

Para treinamento de movimento:</br>
Após abrir o programa, presssione a tecla "**H**" para abrir o treinamento de movimento como frente, trás, direita, esquerda, etc.</br>
Pressione "**R**" para iniciar e interromper a gravação.
</br>
Treinamento de flexão:</br>
Após abrir o programa, presssione a tecla "**F**" para abrir o treinamento de flexão, rotação e repouso.</br>
Pressione "**R**" para iniciar e interromper a gravação.



## Colaboradores:
- [Arthur Matos Macedo](https://github.com/ArthurMM30)
- [Diogo Oliveira Lima](https://github.com/DiogoOLIVEIRAlima)
- [Helena Barbosa Costa](https://github.com/helenabc01)
- [Mirella Ayumi Miyakawa](https://github.com/MiyakawaMirella)
- [Rafaella Guimaraes Venturini](https://github.com/DriRaYV)
