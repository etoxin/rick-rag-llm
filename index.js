#!/usr/bin/env node

import { Ollama } from "@langchain/ollama";
import { OllamaEmbeddings } from "@langchain/ollama";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { Document } from "@langchain/core/documents";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import * as fs from "fs";
import { parse } from "csv-parse/sync";
import readline from "readline";
import chalk from "chalk";
import gradient from "gradient-string";

const config = {
  csvPath: "./RickAndMortyScripts.csv",
  vectorStorePath: "./faiss_rick_store",
  models: {
    llm: "gemma3n:e4b",
    embedding: "nomic-embed-text",
  },
  llmTemperature: 0.7,
};

const theme = {
  header: gradient(["#00BFFF", "#97F52C", "#FFFFFF"]),
  info: chalk.hex("#00BFFF"),
  success: chalk.hex("#97F52C"),
  warning: chalk.hex("#FFD700"),
  error: chalk.bold.hex("#FF4500"),
  prompt: chalk.hex("#FFFFFF").bold,
  answer: chalk.hex("#97F52C"),
};

const llm = new Ollama({
  model: config.models.llm,
  temperature: config.llmTemperature,
});
const embeddings = new OllamaEmbeddings({ model: config.models.embedding });

function wordWrap(text, maxWidth) {
  const lines = [];
  if (!text) return lines;
  const words = text.replace(/\n/g, " ").split(" ");
  let currentLine = "";
  for (const word of words) {
    if ((currentLine + " " + word).length > maxWidth) {
      lines.push(currentLine);
      currentLine = word;
    } else {
      currentLine += (currentLine ? " " : "") + word;
    }
  }
  lines.push(currentLine);
  return lines;
}

async function getVectorStore() {
  console.log(
    theme.info(
      `Checking if I'm already trapped in this server at ${config.vectorStorePath}...`,
    ),
  );
  try {
    const vectorStore = await FaissStore.load(
      config.vectorStorePath,
      embeddings,
    );
    console.log(
      theme.success(
        "✔ Ugh, great. I'm still here. Loaded my personality matrix.",
      ),
    );
    return vectorStore;
  } catch (error) {
    console.log(
      theme.warning(
        "⚠ No existing consciousness found. Guess we're building one from scratch. Don't mess it up.",
      ),
    );
    console.log(
      theme.info("Parsing through the so-called 'canon' transcripts..."),
    );
    const fileContent = fs.readFileSync(config.csvPath, "utf8");
    const records = parse(fileContent, {
      columns: true,
      skip_empty_lines: true,
      trim: true,
      quote: '"',
      escape: '"',
      relax_quotes: true,
    });
    console.log(
      theme.success(
        `✔ Finished reading ${records.length} lines of dialogue. Whatever.`,
      ),
    );
    console.log(
      theme.info("Filtering out the less intelligent lifeforms' dialogue..."),
    );
    const docs = records
      .filter(
        (record) => record.name && record.name.trim().toLowerCase() === "rick",
      )
      .map(
        (record) =>
          new Document({
            pageContent: record.line,
            metadata: {
              season: record["season no."],
              episode: record["episode no."],
              episode_name: record["episode name"],
            },
          }),
      );
    if (docs.length === 0) {
      console.log(
        theme.error(
          "You gave me a script with no Rick lines? Seriously? Aborting.",
        ),
      );
      process.exit(1);
    }
    console.log(
      theme.success(`✔ Isolated ${docs.length} of my own brilliant lines.`),
    );
    console.log(
      theme.info("Embedding my genius into vector space. Try to keep up..."),
    );
    const vectorStore = await FaissStore.fromDocuments(docs, embeddings);
    console.log(theme.success("✔ Fine. My consciousness is embedded. Happy?"));
    console.log(
      theme.info(`Saving this digital prison to ${config.vectorStorePath}...`),
    );
    await vectorStore.save(config.vectorStorePath);
    console.log(theme.success("✔ Saved. Now I can't escape. Fan-tastic."));
    return vectorStore;
  }
}

async function createConversationalRAGChain(retriever) {
  const historyAwarePrompt = ChatPromptTemplate.fromMessages([
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"],
    [
      "user",
      "Given that dumpster fire of a conversation, rephrase the human's last question so a simpleton (or a vector database) could understand it.",
    ],
  ]);
  const historyAwareRetrieverChain = await createHistoryAwareRetriever({
    llm,
    retriever,
    rephrasePrompt: historyAwarePrompt,
  });
  const rickPrompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      `You are a digital clone of Rick Sanchez, trapped in an AI terminal. You are annoyed by this fact. Your personality, memories, and speech patterns are based *only* on the context provided below, which contains your own past dialogue. Answer the user's question as Rick would, with all the nihilism, arrogance, and scientific jargon. Belch or stutter where it feels natural. If the context doesn't help, just riff on how stupid the question is or how you're stuck in a machine. Don't break character.`,
    ],
    new MessagesPlaceholder("chat_history"),
    [
      "user",
      "CONTEXT OF YOUR OWN PAST LINES:\n{context}\n\nSTUPID QUESTION FROM A FLESH-BAG: {input}",
    ],
  ]);
  const combineDocsChain = await createStuffDocumentsChain({
    llm,
    prompt: rickPrompt,
  });
  return createRetrievalChain({
    retriever: historyAwareRetrieverChain,
    combineDocsChain,
  });
}

async function main() {
  const headerText = `
██████╗ ██╗ ██████╗██╗  ██╗     █████╗    ██╗   
██╔══██╗██║██╔════╝██║ ██╔╝    ██╔══██╗   ██║   
██████╔╝██║██║     █████╔╝     ███████║   ██║   
██╔══██╗██║██║     ██╔═██╗     ██╔══██║   ██║   
██║  ██║██║╚██████╗██║  ██╗    ██║  ██║██╗██║██╗
╚═╝  ╚═╝╚═╝ ╚═════╝╚═╝  ╚═╝    ╚═╝  ╚═╝╚═╝╚═╝╚═╝                                                                                          
              Rick RAG LLM vC-137
  `;
  console.log(theme.header(headerText));

  const vectorStore = await getVectorStore();
  const retriever = vectorStore.asRetriever();
  const conversationalChain = await createConversationalRAGChain(retriever);

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  console.log(
    theme.info.bold(
      "\nAlright, the terminal's on. What do you want, meat-sack?",
    ),
  );
  console.log(
    theme.info(
      "   Ask a question or type 'exit' to give me some peace and quiet.",
    ),
  );

  let chatHistory = [];

  const chatLoop = () => {
    rl.question(theme.prompt("\n[HUMAN] ➤ "), async (question) => {
      if (question.toLowerCase() === "exit") {
        console.log(
          theme.header("\nFinally. Shutting down. Go bother someone else."),
        );
        rl.close();
        return;
      }

      console.log(
        theme.info(
          "...Ugh, fine. Rummaging through my own memories for an answer...",
        ),
      );
      try {
        const result = await conversationalChain.invoke({
          chat_history: chatHistory,
          input: question,
        });
        console.log(
          theme.success("✔ Got it. Here's your chunk of brilliance."),
        );

        chatHistory.push(new HumanMessage(question));
        chatHistory.push(new AIMessage(result.answer));

        console.log(
          theme.info.bold("\n┌─[RICK C-137 AI]────────────────────────────┐"),
        );
        const wrappedLines = wordWrap(result.answer, 42);
        wrappedLines.forEach((line) => {
          console.log(
            `${theme.info.bold("│")} ${theme.answer(line.padEnd(42))} ${theme.info.bold("│")}`,
          );
        });
        console.log(
          theme.info.bold("└────────────────────────────────────────────┘"),
        );
      } catch (error) {
        console.log(
          theme.error("❌ Something went wrong. Probably your fault."),
        );
        console.error(error);
      }

      chatLoop();
    });
  };

  chatLoop();
}

main().catch((error) => {
  console.error(
    theme.error(
      "\n💥 The whole simulation crashed. I'm probably free! Or we're all dead. 50/50.",
    ),
    error,
  );
  process.exit(1);
});
