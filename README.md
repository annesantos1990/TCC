# TCC
Repositório com os resultados do meu TCC em Data Science

## TCC – Data Science

Este repositório contém os códigos, experimentos e resultados desenvolvidos no meu Trabalho de Conclusão de Curso (TCC) em Data Science.

## Título do trabalho

Caracterização da conectividade funcional cerebral em indivíduos saudáveis utilizando modelos de classificação

## Descrição

O objetivo deste trabalho é caracterizar padrões de conectividade funcional cerebral a partir de sinais de EEG em estado de repouso, utilizando ferramentas de redes complexas e modelos de classificação. A conectividade funcional é estimada por meio do método de motif-synchronization, permitindo a construção de redes dinâmicas (Time-Varying Graphs) e a extração de métricas topológicas ao longo do tempo.

As features extraídas das redes são integradas a metadados demográficos e utilizadas como entrada para modelos de classificação, com foco exclusivo em indivíduos saudáveis, selecionados a partir de critérios clínicos e comportamentais.

## Dados

Os dados utilizados neste trabalho são provenientes do LEMON (Leipzig Mind-Brain-Body) Dataset, um banco de dados público contendo registros de EEG, informações demográficas e avaliações clínicas de indivíduos adultos jovens e idosos.

Os dados brutos não são disponibilizados neste repositório, conforme as diretrizes do banco de dados original.

## Metodologia

* Transformação das séries temporais de EEG em motifs simbólicos

* Estimativa de conectividade funcional via motif-synchronization

* Construção de redes dinâmicas com janelas temporais deslizantes

* Extração de métricas de redes complexas (grau, clustering, eficiência, hubs)

* Integração com metadados e construção do dataset final

* Aplicação de modelos de classificação

### Tecnologias utilizadas

Python: NumPy, Pandas, MNE, NetworkX


### Status do projeto

📌 Em desenvolvimento – resultados parciais e análises em andamento.
