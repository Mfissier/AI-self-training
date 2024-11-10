# Utiliser une image officielle de Python comme base
FROM python:3.10-slim

# Installer git pour cloner Hugging Face si nécessaire
RUN apt-get update && apt-get install -y git

# Installer les bibliothèques requises
RUN pip install protobuf
RUN pip install tiktoken
RUN pip install --no-cache-dir sentencepiece transformers torch


# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier Python dans le conteneur
COPY ./app/pipeline.py /app/

# Commande par défaut pour exécuter le script
CMD ["python", "pipeline.py"]
