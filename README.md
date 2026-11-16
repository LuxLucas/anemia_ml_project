## Estrutura de pastas

```
anemia_ml_project/
│
├── data/
│   ├── raw/
│   │   └── anemia_dataset.xlsx             # Dataset original do Kaggle (somente leitura)
│   └── processed/
│       ├── train.csv                       # Dados limpos para Árvores e Ensembles
│       ├── test.csv
│       ├── train_scaled.csv                # Dados normalizados para SVM e K-Means
│       └── test_scaled.csv
│
├── notebooks/                              # Todo o código executável, gráficos e análises
│   ├── 01_eda_e_preprocessamento.ipynb     # Lê o .xlsx -> limpa -> gera os 4 .csv em processed/
│   ├── 02_modelo_svm.ipynb                 # Treina SVM -> salva .joblib, gráfico e métrica
│   ├── 03_arvores_e_ensembles.ipynb        # Treina Árvore/Ensemble -> salva .joblib, gráficos e métricas
│   ├── 04_clustering_kmeans.ipynb          # Treina K-Means -> salva .joblib, cotovelo/silhueta e métricas
│   └── 05_comparacao_final.ipynb           # Carrega os .joblib e .csvs -> gera tabela e gráficos finais
│
├── models/                                 # Modelos já treinados salvos via joblib
│   ├── best_svm.joblib
│   ├── best_tree.joblib
│   ├── best_ensemble.joblib
│   └── kmeans_model.joblib
│
├── outputs/                                # Resultados prontos para o seu artigo/relatório
│   ├── figures/                            # Imagens (.png/.pdf) exportadas pelos notebooks via plt.savefig()
│   │   ├── matriz_confusao_svm.png
│   │   ├── matriz_confusao_ensemble.png
│   │   ├── cotovelo_kmeans.png
│   │   └── curva_roc_comparativa.png
│   └── metrics/                            # Tabelas (.csv/.xlsx) exportadas pelos notebooks via to_csv()
│       ├── metricas_svm.csv
│       ├── metricas_arvores_ensembles.csv
│       ├── metricas_kmeans.csv
│       └── comparativo_modelos_final.xlsx
│
├── src/                                    # (Opcional) Scripts .py caso queira reutilizar funções depois
│
├── requirements.txt                        # Lista de bibliotecas (pandas, scikit-learn, openpyxl, seaborn)
└── README.md                               # Documentação geral, objetivo e resumo dos resultados

```

## Guia de Instalação e Configuração (JupyterLab + GitHub)

Siga os passos abaixo para preparar o ambiente no Windows e colaborar com o projeto.

### 1. Clonar o Repositório
No terminal, navegue até a pasta desejada e clone este projeto:
```cmd
cd C:\
git clone https://github.com/LuxLucas/anemia_ml_project.git
cd anemia_ml_project
```

---

### 2. Instalar JupyterLab, Plugins e Bibliotecas de ML
Com o terminal dentro da pasta `anemia_ml_project`, instale todas as dependências em um único comando:
```cmd
pip install -r requirements.txt
```

**Extensões inclusas neste pacote:**
* `jupyterlab-git`: Controle de versão visual (Commit, Push e Pull na barra lateral).
* `jupyterlab-spreadsheet-editor`: Visualizador de arquivos `.xlsx` (clique direito no arquivo > *Open With* > *Spreadsheet Editor*).
* `jupyterlab-lsp` + `python-lsp-server`: Autocompletar inteligente e verificação de sintaxe.
* `jupyterlab_code_formatter` (`black` / `isort`): Formatação automática de código Python.
* `jupyterlab-execute-time`: Exibe o tempo de execução de cada célula.
* `nbdime`: Comparação visual limpa entre notebooks antes do commit.

---

### 3. Iniciar e Rodar o Projeto
1. Inicie o servidor do JupyterLab pelo terminal:
   ```cmd
   jupyter lab
   ```
2. Na barra lateral esquerda, abra a pasta `notebooks/` e execute os arquivos na ordem numérica (`01` a `05`).
3. **Para enviar alterações ao GitHub:** Utilize a aba do Git na barra lateral esquerda. Na primeira vez que fizer um *Push*, insira seu **Personal Access Token** do GitHub no lugar da senha.

