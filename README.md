# Autograph RAG

This project provides a unified interface for Retrieval Augmented Generation, wrapping the most relevant tools for LLM applications, from prototyping to production.

Its core contribution is an efficient approach to graph-based retrieval where the graph simply acts as a network linking clusters of embedding vectors instead of an accurate semantic knowledge graph, which would require domain experts to build and full traversal at query time.

## Motivation

In a general Graph RAG approach, an entity-relation extractor (typically an LLM) processes a corpus to build the nodes and links of a knowledge base. The graph is part of the retrieval process where the system answers a query by computing its embedding, matching it against entities, and navigating the graph to expand the context obtained from neighboring nodes.

The industry standard is represented by Microsoft GraphRAG, where the graph serves as a map of different embeddings to be traversed in order to find the chunks that have relationships with the query. The core of the framework is the graph context expansion which relies on embedding similarity to find the starting node and retrieve the linked embedded chunks.
This requires that the entity extraction must be enough accurate to create a knowledge base where each node, uniquely associated with a specific chunk, is connected by informative and non-redundant relationships.

Autograph RAG inverts the roles since the vector store remains the main index while the graph sits on top of it solely for context expansion. Here the nodes are just subconcepts extracted from each embedded chunk, unlike GraphRAG's where every node is a specific embedding. Several subconcepts can resolve to the same embedding, giving a single vector multiple relational entry points. The edges can connect different embeddings that are linked by explicit relations, which is especially valuable when those chunks are not semantically similar. At query time everything starts as in classic RAG, where the query enters by similarity, but the graph then widens the result set, surfacing relevant chunks that similarity alone would never reach.
Because the graph only needs to record that two nodes belonging to different embeddings are related, without needing to be a perfectly faithful map of the entire domain, it remains coarse, lightweight to build, and cost-effective to maintain.

## Installation

This project uses [uv](https://docs.astral.sh/uv/). A single command installs everything:

```bash
uv sync
```

## Configuration

When building your own entry point, `autograph_rag.config.Settings` reads typed settings from the environment or a `.env` file. Every value has a local default, so an empty environment works out of the box — see `Settings` for the current fields.