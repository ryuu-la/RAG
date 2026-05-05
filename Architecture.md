# RAG Project Architecture

This document outlines the detailed architecture of the RAG (Retrieval-Augmented Generation) system, showing how the frontend, backend, agent, and various tools connect and interact, from the UI down to the data layer and back to the UI.

## Architecture Flow (Starts and Ends with UI)

```mermaid
graph TD
    %% UI Layer (Starts Here)
    UI_Start[("React UI (Frontend)\nUser input & document uploads")]:::ui

    %% Frontend API Layer
    subgraph Frontend [Frontend (`frontend/src`)]
        API_JS[("api.js\n(API Client & SSE Streaming)")]
        App_JSX[("App.jsx\n(Session & State Management)")]
    end

    %% Backend API Layer
    subgraph Backend API [Backend FastAPI (`backend/app/main.py`)]
        Query_Endpoint[("POST /api/query/stream\n(Streaming Endpoint)")]
        Ingest_Endpoint[("POST /api/ingest/upload\n(Document Upload)")]
        Model_Endpoint[("POST /api/model/upload\n(Context Files)")]
    end

    %% Agent System Layer
    subgraph Agent System [Backend Agent (`backend/app/services/agent.py`)]
        Agent_Loop[("Agentic LLM Loop\n(Max 10 iterations)")]
        LLM[("LLM Models\n(Gemini / OpenRouter)")]
    end

    %% Tools Layer
    subgraph Tool Bindings [Agent Tools (`backend/app/services/tools.py`)]
        Tool_SearchDocs[("search_documents\n(Hybrid Search)")]
        Tool_WebSearch[("web_search\n(DuckDuckGo)")]
        Tool_ReadURL[("read_url\n(Web Page Fetcher)")]
        Tool_ImageSearch[("image_search\n(DuckDuckGo Images)")]
        Tool_LookupDoc[("lookup_document\n(Metadata)")]
        Tool_ExportPDF[("export_pdf\n(FPDF2)")]
        Tool_ExportCSV[("export_csv\n(Pandas)")]
    end

    %% Data Pipeline & Storage Layer
    subgraph Data Pipeline [Ingestion & Retrieval (`backend/app/services/ingest.py` & `retrieval.py`)]
        PDF_Processor[("PDF Extraction & Chunking")]
        Embedding[("Embedding Model\n(BAAI/bge-small)")]
        ChromaDB[("ChromaDB\n(Vector Store)")]
        BM25_Index[("BM25 Index\n(Keyword Search)")]
        Local_Store[("In-Memory Store\n(Metadata & Progress)")]
    end

    %% Flow Connections (Start -> Backend -> Tools -> Backend -> End)
    
    %% Input Flow
    UI_Start -->|User asks query / uploads document| App_JSX
    App_JSX -->|API Calls| API_JS
    API_JS -->|Streams request| Query_Endpoint
    API_JS -->|Uploads PDF| Ingest_Endpoint
    API_JS -->|Uploads Context| Model_Endpoint

    %% Ingestion Flow
    Ingest_Endpoint -->|Triggers Background Job| PDF_Processor
    PDF_Processor -->|Extracts & Chunks| Embedding
    Embedding -->|Generates Vectors| ChromaDB
    PDF_Processor -->|Indexes Words| BM25_Index
    PDF_Processor -->|Updates Status| Local_Store

    %% Query Flow
    Query_Endpoint -->|Passes query + history| Agent_Loop
    Agent_Loop <-->|Reasons & Plans| LLM
    
    %% Agent to Tools Flow
    Agent_Loop -->|Needs internal knowledge| Tool_SearchDocs
    Agent_Loop -->|Needs live info| Tool_WebSearch
    Agent_Loop -->|Needs full page| Tool_ReadURL
    Agent_Loop -->|Needs visuals| Tool_ImageSearch
    Agent_Loop -->|Needs metadata| Tool_LookupDoc
    Agent_Loop -->|Wants to create doc| Tool_ExportPDF
    Agent_Loop -->|Wants to create table| Tool_ExportCSV

    %% Tools to Data Flow
    Tool_SearchDocs -->|Semantic Search| ChromaDB
    Tool_SearchDocs -->|Keyword Search| BM25_Index
    Tool_SearchDocs -->|Retrieves merged results| Agent_Loop

    %% Final Resolution Path
    Tool_ExportPDF -->|Saves file| Backend API
    Tool_ExportCSV -->|Saves file| Backend API
    Tool_WebSearch -->|Returns snippet| Agent_Loop
    Tool_ReadURL -->|Returns text| Agent_Loop
    
    %% Streaming Back Flow
    Agent_Loop -->|Yields steps & tokens (SSE)| Query_Endpoint
    Query_Endpoint -->|Streams SSE Events| API_JS
    API_JS -->|Triggers UI Updates| App_JSX
    
    %% UI End (Ends Here)
    App_JSX -->|Renders Chat, Modals, Downloads| UI_End[("React UI (Frontend)\nDisplays Response & Citations")]:::ui

    classDef ui fill:#4a90e2,stroke:#333,stroke-width:2px,color:#fff;
```
