from __future__ import annotations

from dataclasses import dataclass
from logging import Logger
from typing import List

from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragblog.rag_evaluator import RagEvaluator, RagEvaluatorConf


@dataclass
class JSONLoaderConf:
    file_path: str
    jq_schema: str
    text_content: bool
    json_lines: bool


@dataclass
class TextSplitterConf:
    chunk_size: int
    chunk_overlap: int


@dataclass
class VectorStoreConf:
    embedding_model: str
    persist_directory: str


@dataclass
class RAGChainConf:
    prompt_model: str
    llm_model: str


@dataclass
class RagPipelineConf:
    loader: JSONLoaderConf
    splitter: TextSplitterConf
    vectorstore: VectorStoreConf
    ragchain: RAGChainConf
    evaluator: RagEvaluatorConf
    is_db_ready: bool
    is_debug: bool


class RagPipeline:
    def __init__(self, conf: RagPipelineConf, logger: Logger):
        self.conf = conf
        self.logger = logger
        self.evaluator = RagEvaluator(self.conf.evaluator, logger)

        self.logger.info(f"{self.conf}")
        # set_debug(self.conf.is_debug)

    def load_documents(self) -> List[Document]:
        """Load documents using JSONLoader."""
        loader = JSONLoader(
            file_path=self.conf.loader.file_path,
            jq_schema=self.conf.loader.jq_schema,
            text_content=self.conf.loader.text_content,
            json_lines=self.conf.loader.json_lines,
        )
        documents = loader.load()
        self.logger.info(f"Loaded {len(documents)} documents")
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.conf.splitter.chunk_size,
            chunk_overlap=self.conf.splitter.chunk_overlap,
        )
        splits = splitter.split_documents(documents)
        self.logger.info(f"Split into {len(splits)} chunks")
        return splits

    def create_vectorstore(self, splits: List[Document]) -> Chroma:
        """Create and persist vector store."""
        embeddings = OllamaEmbeddings(model=self.conf.vectorstore.embedding_model)
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=self.conf.vectorstore.persist_directory,
        )
        self.logger.info("Vector store created and persisted")
        return vectorstore

    def get_vectorstore(self) -> Chroma:
        """Load existing vector store."""
        embeddings = OllamaEmbeddings(model=self.conf.vectorstore.embedding_model)
        vectorstore = Chroma(
            persist_directory=self.conf.vectorstore.persist_directory,
            embedding_function=embeddings,
        )
        return vectorstore

    def create_prompt(self) -> PromptTemplate:
        """Create the RAG prompt."""
        template = """You are a helpful assistant.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, 
don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""
        return PromptTemplate.from_template(template)

    def format_docs(self, docs: List[Document]) -> str:
        """Format documents for context."""
        return "\n\n".join(doc.page_content for doc in docs)

    def create_rag_chain(self, retriever, prompt: PromptTemplate):
        """Create the RAG chain."""
        llm = ChatOllama(model=self.conf.ragchain.llm_model)
        rag_chain = (
            {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return rag_chain

    def query(self, question: str) -> str:
        """Query the RAG system."""
        retriever = self.get_vectorstore().as_retriever(k=10)
        prompt = self.create_prompt()
        rag_chain = self.create_rag_chain(retriever, prompt)
        response = rag_chain.invoke(question)
        return response

    def query_with_context(self, question: str, context: str) -> str:
        """Query with provided context (for evaluation)."""
        llm = ChatOllama(model=self.conf.ragchain.llm_model)
        prompt = self.create_prompt()
        # Use provided context directly
        full_prompt = prompt.format(context=context, question=question)
        response = llm.invoke(full_prompt)
        return response.content.strip()

    def evaluate_answer(self, question: str, answer: str, context: str):
        """Evaluate the quality and relevance of the answer.

        Args:
            question: The question asked.
            answer: The generated answer.
            context: The retrieved context.

        Returns:
            Evaluation results.
        """
        return self.evaluator.evaluate(question, answer, context)
