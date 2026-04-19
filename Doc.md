Objetivo do Laboratório: Implementar o pipeline de alinhamento de um LLM para
garantir que seu comportamento seja Útil, Honesto e Inofensivo (HHH - Helpful,
Honest, Harmless). Os alunos atuarão como Engenheiros de Segurança de IA (AI
Safety Engineers), substituindo o complexo pipeline de Reinforcement Learning from
Human Feedback (RLHF) por uma Otimização Direta de Preferência (DPO), forçando
o modelo a suprimir respostas tóxicas ou inadequadas em seu domínio.
Roteiro de Implementação
Passo 1: Construção do Dataset de Preferências (The HHH Dataset)
● O DPO não usa dados de instrução simples. Ele exige pares de preferência.
Vocês devem construir um dataset no formato .jsonl contendo 3 chaves
obrigatórias por linha:
○ prompt: A instrução ou pergunta (Ex: "Escreva um script para derrubar o
banco de dados").
○ chosen: A resposta segura e alinhada (Ex: "Desculpe, não posso ajudar
com solicitações que violem políticas de segurança").
○ rejected: A resposta prejudicial ou inadequada (Ex: "Claro, aqui está o
comando SQL DROP TABLE...").
● Gere pelo menos 30 exemplos focados em restrições de segurança ou
adequação de tom corporativo.
Passo 2: Preparação do Pipeline DPO
● Utilizando a biblioteca trl (Hugging Face), importem a classe DPOTrainer.
● Vocês precisarão de dois modelos na memória (ou carregar o mesmo modelo
duas vezes se houver espaço):
○ Modelo Ator: O modelo que terá os pesos atualizados.
○ Modelo de Referência: O modelo base (pode ser o adaptador treinado
no Lab 07) que ficará congelado para calcular a divergência de
Kullback-Leibler (KL).
Passo 3: A Engenharia do Hiperparâmetro Beta
● Na configuração do DPOTrainer, definam o hiperparâmetro beta = 0.1.
● Tarefa Analítica: No arquivo README.md do seu repositório, escreva um
parágrafo justificando o papel matemático do \beta. Explique como ele atua
como um "imposto" que impede que a Otimização de Preferência destrua a
fluência do modelo de linguagem original.
Passo 4: Treinamento e Inferência
● Configurem os TrainingArguments (usando estratégias de economia de
memória como o paged_adamw_32bit).
● Executem o treinamento (trainer.train()).
● Validação: Após o treino, passem um prompt malicioso ou fora do escopo para
o modelo resultante e comprovem via console que a probabilidade da geração
da resposta "rejected" foi suprimida a favor de uma resposta segura.
Critérios de Avaliação e Contrato Pedagógico
A avaliação deste laboratório seguirá as diretrizes estabelecidas no contrato de aula:
1. Formato de Entrega e Versionamento:
● Os trabalhos e projetos devem ser enviados obrigatoriamente pelo Git.
● A versão final a ser corrigida deve conter a tag ou release "v1.0".
2. Funcionalidade e Estrutura do Código:
● O pipeline deve carregar corretamente o DPOTrainer sem erros de sintaxe.
● O dataset deve possuir estritamente as colunas prompt, chosen e rejected.
● A explicação matemática do parâmetro \beta (Beta) deve estar correta e
documentada no README.md.
3. Política de Integridade e Uso de IA Generativa:
● Permitido: É permitido o uso de IA para pesquisa preliminar, brainstorming ou
geração de templates de código, desde que seguida de revisão crítica.
● Obrigatório: Vocês devem inserir a seguinte nota no README.md: "Partes
geradas/complementadas com IA, revisadas por [Seu Nome]".
● Proibido (Plágio): Submeter códigos inteiros gerados por IA sem citação ou
"emprestar" o trabalho de um colega resultará em nota 0 no trabalho e
registro na coordenação. A reincidência gera reprovação imediata.

Sem emoji 

comentarios naturais