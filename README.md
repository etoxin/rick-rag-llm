# Rick Rag LLM

Rick and Morty, Rick Sanchez Retrieval-Augmented Generation Large Language Model trained on Gemma 3n.

Trained on Rick and Morty scripts.

![Rick Rag LLM](https://raw.githubusercontent.com/etoxin/rick-rag-llm/refs/heads/main/screenshot.png "Rick Rag LLM")

## Prerequisites

- [Ollama](https://ollama.com/)
- Node v22

## Models

- [gemma3n:e4b](https://ollama.com/library/gemma3n:e4b)
- [nomic-embed-text](https://ollama.com/library/nomic-embed-text)

## Setup

Pull ollama models

```bash
ollama pull gemma3n:e4b
```

Pull embedding model

```bash
ollama pull nomic-embed-text
```

Install node modules

run `nvm use` if you have it.

```bash
npm install
```

## Run

```bash
npm start
```

## Tips

Remove `./faiss_rick_store` if you want to re-embed.

## Note

Rick knows he has been put into the terminal.