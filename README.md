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
-> Docker: Usado para gerenciar o ambiente e rodar o MongoDB, o banco de dados NoSQL responsável pelo armazenamento de dados de sinais e traduções.

**Passo 1:** </br>
 clone o repositório:
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
