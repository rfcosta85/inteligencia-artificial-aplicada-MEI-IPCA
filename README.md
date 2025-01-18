# Repositório para a cadeira de Inteligência artificial aplicada do mestrado de Engenharia informática no IPCA - Barcelos - Portugal
Objetivo: Predição de Falhas em Equipamentos Industriais

# Descrição do Projeto
Este projeto utiliza Machine Learning para prever falhas em equipamentos industriais, como turbinas, compressores e bombas. O objetivo é identificar máquinas com potencial defeito, permitindo a tomada de medidas preventivas para evitar falhas, reduzir custos de reparo e garantir a segurança das operações.

# Metodologia
O projeto possui um sanatização e adequação dos dados para o correto treinamento dos modelos, a label será a coluna faulty que informa se determinada máquina está defeituosa ou não. As colunas presentes no dataset são: 

* Temperature
* Pressure
* Vibration
* Humidity
* equipment
* location
* faulty

Conforme mencionado acima, a coluna faulty será a label e as demais colunas serão as features. Foram utilizados 6 modelos de treinamento, e foi criado uma função genérica para que se possam rodar os modelos com diversas configurações de hiperparâmetros de forma paralelizada. No final o resultado dos modelos são salvos em uma tabela no supabase, que será utilizada pela função de avaliação. Esta função cria um ranking com os 3 melhores modelos, levando em conta o contexto do trabalho, ou seja, considerando o custo de ter uma máquina parada x o custo de enviar uma equipe para reparar uma máquina que supostamente está funcionando e sem risco de parar.

# Workflow Business Proccess

![image](https://github.com/user-attachments/assets/e6dd3764-167d-4e27-8905-7aefb832f571)



# Conclusões
Este projeto demonstra a aplicação de Machine Learning para prever falhas em equipamentos industriais. A seleção do modelo e a priorização de métricas dependem do problema específico e das consequências de diferentes tipos de erros. O código pode ser adaptado e expandido para diferentes tipos de equipamentos e cenários industriais.
