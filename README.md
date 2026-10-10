# PhishGuard AI : détection d'emails de phishing (ML + LLM local)

Détecteur d'emails de phishing en deux étages : un modèle ML classique (TF-IDF + régression
logistique) et un LLM local (Ollama, `qwen2.5:7b`). Le projet traite **uniquement les emails** ; les URL
ne sont pas dans le périmètre. Il est livré avec une API FastAPI et une petite interface web.

> Projet d'étude. Il aide à décider, il ne remplace pas une passerelle de sécurité mail.

## Résultat principal

Évaluation sur un jeu **difficile** de 100 emails écrits à la main (40 `dev`, 60 `test`, classes
équilibrées). Le split `dev` a servi à choisir le modèle, le prompt et la règle ; le split `test` a été
lu **une seule fois** (verrou `models/llm_test_done.lock`).

| Sur `test` (60 emails) | ML seul | LLM seul (`qwen2.5:7b`, zero-shot) |
|---|---|---|
| Recall | 0,800 [0,633 ; 0,933] | **0,967** [0,900 ; 1,000] |
| Faux positifs | 0,600 [0,433 ; 0,767] | **0,100** [0,000 ; 0,233] |
| Balanced accuracy | 0,600 | **0,933** |
| Erreurs | 24 | 4 |

Intervalles de confiance à 95 % par bootstrap. Avec 30 emails par classe, ils sont larges : le gain est
net, mais « 10 % de faux positifs » reste une estimation.

## Architecture

```
email ──► ML (TF-IDF + LR) : toujours calculé, quelques ms
      └─► LLM (Ollama, qwen2.5:7b, JSON, température 0) : décide
              │
              ├─ réponse valide  ──► verdict du LLM            (mode "llm")
              └─ Ollama en panne / réponse illisible
                                  ──► verdict du ML, signalé    (mode "ml_degrade")
```

Pourquoi « LLM seul » et pas une cascade (le LLM n'est appelé que dans la zone d'incertitude du ML) ?
Sur le jeu difficile, 11 des 24 erreurs du ML sont commises **avec assurance**, hors de la zone
d'incertitude : le LLM ne les voit jamais. Les règles en cascade plafonnent à 0,58-0,77 de balanced
accuracy. Le ML garde trois rôles : référence de comparaison, mode dégradé, explication par mots.

Autres choix, mesurés dans les notebooks : 4 prompts testés (zero-shot, grille, exemples, mini-RAG) et
2 modèles ; le 3b est inutilisable en zero-shot (faux positifs 0,85 sur `dev`) ; les prompts changent peu
les résultats du 7b. Défense contre l'injection de prompt (balises `<email>`, consigne « donnée non fiable »,
neutralisation des balises) : 4 cas sur 4 résistés, mais ce sont 4 cas écrits à la main.

## Structure du dépôt

```
notebooks/   01 nettoyage · 02 split et EDA · 03 entraînement et évaluation ·
             04 évaluation sur jeu difficile · 05 cascade ML -> LLM
src/
  features/        features numériques d'emails (non utilisées par le modèle final)
  preprocessing/   split du jeu de données
  ml/              modèle ML figé (email_ml.py)
  llm/             client Ollama + cache, prompts, parsing de la réponse
  inference/       point d'entrée : EmailDetector.analyser(texte)
  api/             API FastAPI + page web intégrée (ui.py)
models/            modèle ML, vectorizer, config et prompts du LLM, résultats, cache LLM
data/hard/         mini_jeu_etendu.csv : 100 emails difficiles écrits à la main
tests/             tests unittest (sans Ollama)
scripts/           check_parity_dev.py : vérifie que src/ redonne les chiffres du notebook 05
```

## Installation

Prérequis : Python 3.13 (testé), [Ollama](https://ollama.com) installé.

```
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS : source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen2.5:7b
```

## Utilisation

Lancer l'API (Ollama doit tourner) puis ouvrir http://127.0.0.1:8000 :

```
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

- `POST /analyze` : `{"subject": "...", "body": "..."}` ou `{"text": "..."}`
- `GET /health` : état d'Ollama, du modèle et du cache
- `GET /docs` : documentation Swagger

Exemple :

```
curl -X POST http://127.0.0.1:8000/analyze -H "Content-Type: application/json" ^
  -d "{\"subject\": \"Verify your account\", \"body\": \"Confirm your login within 12 hours: http://example.test\"}"
```

En mode `ml_degrade`, le verdict vient du ML seul et la réponse contient un champ `avertissement` : à
traiter comme une simple indication.

## Tests et vérification

```
python -m unittest discover -s tests -t . -v
python scripts/check_parity_dev.py     # doit afficher PARITÉ OK (utilise models/llm_cache.jsonl)
```

## Reproductibilité

- Le LLM n'est **pas déterministe** malgré température 0 et graine fixe (deux appels sans cache ont donné
  des sorties différentes). Les chiffres des notebooks ne se rejouent que grâce à `models/llm_cache.jsonl`.
- Ne modifiez pas `src/llm/prompts.py` sans refaire l'évaluation : un test vérifie qu'il reste identique à
  `models/llm_prompts.json`.
- Le split `test` du notebook 05 est consommé : toute nouvelle évaluation sérieuse demande de nouvelles données.

## Limites

- 100 emails écrits à la main, courts (moins de 310 caractères) et propres : les vrais emails sont plus longs
  et plus bruités ; les résultats ne sont pas garantis sur du trafic réel.
- Faux positifs fréquents sur les emails **légitimes** de réinitialisation de mot de passe et d'avis bancaires.
  Une fraude au président sans lien a aussi été manquée.
- Latence d'environ 13 s par email avec le 7b ; Ollama traite les requêtes les unes après les autres.
- Pas d'authentification sur l'API : ne l'exposez pas sur un réseau sans proxy authentifié. Le contenu des
  emails n'est ni journalisé ni stocké.
- Emails uniquement : pas de modèle d'URL, pas d'analyse des pièces jointes ni des en-têtes.

## Données

Versionnés : `data/hard/mini_jeu_etendu.csv` (100 emails écrits à la main) et `models/`. Le notebook 05, les
tests et l'API n'ont besoin que de ces deux éléments : ils fonctionnent après un simple clone.

**Non versionnés** (trop volumineux, licence à vérifier) : `data/raw/` et `data/processed/`.
Pour les reconstruire :

1. placer le corpus brut `phishing_email.csv` (environ 107 Mo) dans `data/raw/` ;
2. exécuter le notebook 01 (écrit `data/processed/emails_clean.csv`), puis le 02 (split, écrit
   `emails_test_raw.csv` et le vectorizer), puis 03 et 04.

Attention : les notebooks 02 et 03 **réécrivent** `models/emails_tfidf_vectorizer.joblib` et
`models/email_model.joblib`. Le notebook 05 et l'API dépendent de cette version exacte (colonnes retenues,
seuil, zone d'incertitude). Sauvegardez `models/` avant de les relancer, et refaites l'évaluation du
notebook 05 si le modèle change.

Un jeu d'URL (`data/raw/URL_Dataset.csv`) existe dans le projet local mais n'est pas utilisé : le dépôt
ne traite que les emails.


