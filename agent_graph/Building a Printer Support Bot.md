# **Architectural Paradigms for Voice-Enabled Agentic Support Systems: A LangGraph and LangChain Implementation Study**

## **1\. Executive Summary and Theoretical Framework**

The deployment of Artificial Intelligence in technical support scenarios has historically been constrained by the linearity of interaction. Traditional Interactive Voice Response (IVR) systems and early-generation chatbots operated on rigid decision trees, incapable of handling the fluid, non-linear nature of human diagnostic conversation. However, the convergence of Large Language Models (LLMs) with graph-based orchestration frameworks represents a fundamental shift in cognitive architecture. This report provides a comprehensive architectural analysis and implementation guide for constructing a **Voice-Enabled Printer Support Agent**, leveraging **LangGraph** for stateful orchestration and **LangChain** for modular cognitive tooling.

The specific operational mandate requires an agent capable of traversing a complex decision matrix: identifying a printer model, iteratively gathering diagnostic symptoms through a strict "question-by-question" protocol, dynamically mapping this context to a specific **ChromaDB** collection for retrieval, and synthesizing a concise, step-by-step resolution suitable for audio output. This workflow necessitates a departure from the traditional Directed Acyclic Graph (DAG) approach common in simple Retrieval-Augmented Generation (RAG) applications. Instead, we must adopt a **Cyclic State Graph** architecture, which allows for recursive reasoning, self-correction, and persistent state management—features essential for replicating the behavior of a human technical expert.1

### **1.1 The Shift from Pipelines to Agents**

In standard software engineering, pipelines are linear: Input ![][image1] transforms to Output ![][image2]. In contrast, an AI Agent acts as a cognitive engine operating within a loop: Observe ![][image3] Reason ![][image3] Act. For a printer support bot, this distinction is critical. A linear pipeline assumes the user provides all necessary information (Model, Error Code, OS) in the first utterance. Real-world interactions, particularly over voice channels, are rarely so structured. Users offer partial information ("My printer is jammed"), requiring the system to enter an **Iterative Information Gathering** loop.2

LangGraph provides the primitive structures to formalize this loop. By modeling the conversation as a graph of **Nodes** (functional units) connected by **Conditional Edges** (logic gates), we can engineer a system that adapts its trajectory based on the accumulated **State**.3 This report will demonstrate how a cyclic graph architecture is the only viable solution for satisfying the user's requirement for itemized, sequential questioning.

### **1.2 The Constraints of Voice User Interfaces (VUI)**

The transition from text to voice introduces strict latency and formatting constraints. A text-based RAG system might return a 500-word excerpt from a manual, replete with Markdown tables and bold headers. In a voice context, such an output is a failure. The "Sandwich Architecture"—where the agent sits between Speech-to-Text (STT) and Text-to-Speech (TTS) layers—demands that the central graph not only retrieves information but actively acts as a transducer, converting raw technical data into conversational, linear audio scripts.4 Furthermore, the system must handle **Interrupts**—the capability for a user to verbally "barge in" during an explanation—requiring a persistent state mechanism that can pause, update, and resume execution without losing context.5

## ---

**2\. Cognitive Architecture and State Management**

At the core of any agentic system is its **State**—the shared memory structure that persists across the graph's execution steps. Unlike stateless REST APIs, a support agent must maintain a continuously evolving snapshot of the conversation, extracted entities, and internal reasoning flags.

### **2.1 Schema Design for Technical Support**

To support the requirement of mapping printer models to specific database collections, the state schema must be rigorously typed. We reject the use of unstructured dictionaries in favor of **Pydantic** models, which offer runtime validation and clearer serialization protocols for database checkpointing.7 The SupportAgentState is composed of three distinct functional layers: the **Dialogue History** (for LLM context), the **Slot Register** (for logic control), and the **Runtime Configuration** (for dynamic RAG).

**Table 1: State Schema Definition and Functional Roles**

| State Component | Data Type | Functional Role | Update Mechanism |
| :---- | :---- | :---- | :---- |
| messages | Annotated, add\_messages\] | Maintains the raw transcript of the conversation (User and AI turns). Critical for the LLM to understand previous context (e.g., "It's still not working"). | Appended via add\_messages reducer to prevent overwriting history. |
| printer\_slots | Pydantic.BaseModel | Stores the extracted entities: model\_name, error\_code, symptom, os\_environment. | Overwritten or Merged. New values replace old ones to reflect corrections (e.g., "Actually, it's a Canon, not an HP"). |
| collection\_config | Dict\[str, str\] | Maps the user-facing model\_name to the internal ChromaDB collection\_name (e.g., "HP OfficeJet" ![][image3] hp\_oj\_8710\_series). | Computed by the ModelNormalizer node. |
| missing\_info | List\[str\] | A dynamic priority queue of slots that are still required before retrieval can occur. | Re-calculated at every turn by the SlotManager node. |
| dialogue\_phase | Enum | Tracks the macro-state: GATHERING, RETRIEVING, SOLVING, HANDOFF. | Transitioned by conditional edges based on slot completeness. |

The separation of printer\_slots from messages is a crucial architectural decision. While the messages list grows indefinitely, the printer\_slots object serves as a concise "control panel" for the graph's logic gates. This decoupling allows the agent to maintain a long conversational history without confusing the strict logic required for database selection.1

### **2.2 The Graph Topology: A Cyclic State Machine**

To satisfy the requirement of "collecting information question by question," the graph cannot simply be a linear sequence. It must implement a **Supervisor-Worker** pattern or a **State Machine** pattern where the system loops until a specific "Completeness Criteria" is met.

The proposed topology consists of the following primary nodes:

1. **InputAnalyzer**: The entry point for each turn. It processes the latest user audio transcript to extract entities and update the printer\_slots.  
2. **SlotSupervisor**: The logic core. It examines the printer\_slots and determines if the "Completeness Criteria" are met. If not, it identifies the *single most high-priority missing slot*.  
3. **QuestionGenerator**: If the supervisor detects missing info, this node generates a specific question for that *one* missing item.  
4. **DynamicRetriever**: Once completeness is reached, this node configures the ChromaDB connection based on the collection\_config and executes the search.  
5. **SolutionSynthesizer**: This node transforms technical retrieval results into a voice-optimized script.

**The Iterative Cycle:**

The connection between InputAnalyzer, SlotSupervisor, and QuestionGenerator forms a **Loop**.

* *Turn 1:* User: "My printer is broken." ![][image3] Analyzer (No slots) ![][image3] Supervisor (Missing: Model) ![][image3] QuestionGen ("What is the model?") ![][image3] **END**.  
* *Turn 2:* User: "It's a Lexmark." ![][image3] Analyzer (Slot: Model=Lexmark) ![][image3] Supervisor (Missing: Symptom) ![][image3] QuestionGen ("What is the error?") ![][image3] **END**.  
* *Turn 3:* User: "Paper jam." ![][image3] Analyzer (Slot: Symptom=Jam) ![][image3] Supervisor (Complete) ![][image3] **EXIT LOOP** ![][image3] Retriever.

This topology explicitly enforces the user's requirement to avoid "asking all questions together." The QuestionGenerator is only ever aware of the *top* item in the missing\_info queue, ensuring focused, singular interactions.11

## ---

**3\. Iterative Information Gathering: The Slot-Filling Engine**

The efficacy of a support bot is defined by its ability to accurately diagnose the problem context. In Natural Language Understanding (NLU), this is known as **Slot Filling**. While traditional systems use rigid forms, an agentic approach utilizes **Zero-Shot Slot Filling** enhanced by iterative refinement.2

### **3.1 The "Next Best Question" (NBQ) Algorithm**

To support the "question by question" requirement, the SlotSupervisor node must implement a deterministic prioritization logic. We cannot rely solely on the LLM to "decide what to ask," as it might hallucinate irrelevant questions or group them inappropriately. Instead, we define a **Dependency Graph** for the slots themselves.

**Algorithm 1: Slot Prioritization Logic**

Python

def determine\_next\_slot(current\_slots: PrinterSlots) \-\> str:  
    \# 1\. Critical Dependency: Collection Selection  
    \# We cannot search without the collection name. This is absolute priority.  
    if not current\_slots.model\_name:  
        return "ask\_model\_name"  
      
    \# 2\. Critical Dependency: Search Query  
    \# We have the book (manual), now we need the query.  
    if not current\_slots.symptom\_description:  
        return "ask\_symptom"  
      
    \# 3\. Contextual Refinement (Conditional)  
    \# Only ask for Error Code if the symptom implies a digital error.  
    if "error" in current\_slots.symptom\_description and not current\_slots.error\_code:  
        return "ask\_error\_code"  
          
    \# 4\. Environment Refinement (Conditional)  
    \# Only ask for OS if the problem is connectivity/driver related.  
    if is\_connectivity\_issue(current\_slots.symptom\_description) and not current\_slots.os\_env:  
        return "ask\_os"  
          
    return "complete"

This algorithm ensures that the agent respects the logical hierarchy of technical support. Asking for the Operating System when the user has a "Paper Jam" is a common frustration with poor bots; this conditional dependency logic eliminates such redundancy.13

### **3.2 Handling Ambiguity and Corrections**

Voice inputs are prone to Automatic Speech Recognition (ASR) errors and user corrections. A user might say, "Model 50," pause, and then say, "No, wait, Model 60." A naive "append-only" state would store both.

The **State Reducer** for printer\_slots must support **Upsert** (Update/Insert) semantics. When the InputAnalyzer extracts slots, it should output a confidence score. If the user explicitly negates a previous slot ("Not the HP, the Canon"), the LLM within the InputAnalyzer must be prompted to return a Null or Overwrite instruction for the model\_name slot. This "Self-Correction" capability is a hallmark of robust agentic systems, allowing the graph to traverse back to previous states (e.g., reverting from RETRIEVING back to GATHERING if the collection name proves invalid).11

### **3.3 Prompt Engineering for Voice Slot Extraction**

The prompt for the InputAnalyzer is critical. It must distinguish between *providing information* and *conversational filler*.

* **System Prompt Strategy:** "You are a precise data extraction engine. Analyze the user's transcript. Your goal is to extract technical entities into the defined JSON schema. If a user corrects a previous statement (e.g., 'I meant Windows, not Mac'), prioritize the latest information. Ignore polite conversation ('Hello', 'Thanks')."  
* **Structured Output:** We utilize LangChain's .with\_structured\_output(PrinterSlots) method to force the LLM to return a valid JSON object matching our Pydantic schema. This eliminates parsing errors and ensures the SlotSupervisor receives clean data.8

## ---

**4\. Dynamic Retrieval Architecture: The Multi-Tenant RAG**

The user's request explicitly links the model name to the collection\_name in **ChromaDB**. This implies a **Multi-Tenant RAG** architecture. Instead of a single monolithic vector store containing all manuals for all printers, the data is partitioned.

### **4.1 The Case for Collection Partitioning**

In vector databases like ChromaDB, partitioning data into distinct collections (e.g., collection\_hp\_m404, collection\_canon\_ts9120) offers superior performance and relevance compared to a single collection with metadata filtering.

1. **Relevance Isolation:** Terminology overlaps. "Error 50" in an HP printer (Fuser) is different from "Error 50" in a Brother printer (Drum). Searching a shared index increases the risk of retrieving the wrong manual's error code. Partitioning guarantees that context is strictly limited to the user's device.17  
2. **Search Latency:** Performing an Approximate Nearest Neighbor (ANN) search on a smaller, model-specific index (e.g., 5,000 vectors) is significantly faster and more accurate than searching a global index (e.g., 5,000,000 vectors) with post-filtering.

### **4.2 Runtime Configuration of Retrievers**

Standard LangChain examples often initialize the retriever at application startup.

Python

\# Static Initialization (Bad for this use case)  
retriever \= vectorstore.as\_retriever()

For this agent, the retriever cannot be static. It does not know which collection to query until the SlotSupervisor has confirmed the model\_name. We must utilize **Runtime Configuration**.

**Implementation Strategy: The Dynamic Node** The DynamicRetriever node functions as a factory. It reads the collection\_name from the state and instantiates a *ephemeral* connection to that specific ChromaDB collection.18

Python

def dynamic\_retrieval\_node(state: SupportAgentState, config: RunnableConfig):  
    """  
    Dynamically connects to the specific Chroma collection based on state.  
    """  
    target\_collection \= state.printer\_slots.collection\_config  
    query \= state.printer\_slots.symptom\_description  
      
    \# Initialize connection to the specific collection ON THE FLY  
    vector\_db \= Chroma(  
        collection\_name=target\_collection,  
        embedding\_function=OpenAIEmbeddings(),  
        persist\_directory="./chroma\_storage"  
    )  
      
    \# Configure the retriever  
    retriever \= vector\_db.as\_retriever(search\_kwargs={"k": 5})  
      
    \# Execute retrieval  
    docs \= retriever.invoke(query)  
      
    return {"retrieved\_docs": docs}

This pattern leverages LangChain's flexibility to treat the VectorStore as a runtime dependency rather than a compile-time constant. It satisfies the user's requirement to receive the model name and *then* find the solution.19

### **4.3 Hybrid Search for Technical Precision**

Technical support queries often involve specific error codes (e.g., "0x8004005"). Semantic vector search (Dense Retrieval) is excellent for conceptual queries ("paper is stuck") but can struggle with exact alphanumeric matches, often treating "Error 50.1" and "Error 50.2" as semantically identical.

To ensure high-fidelity retrieval, the DynamicRetriever should implement a **Hybrid Search** strategy, combining:

* **Dense Vector Search (ChromaDB):** Captures meaning and natural language descriptions.  
* **Sparse Keyword Search (BM25):** Captures exact error codes and model numbers.21

By using an EnsembleRetriever within the dynamic node, we weight the results (e.g., 0.5 Dense \+ 0.5 Sparse). This ensures that if the user says an exact error code, the manual page containing that code bubbles to the top, while still handling vague descriptions like "it's making a grinding noise."

### **4.4 The Retrieval Grader (Self-Correction Loop)**

Retrieval is not infallible. If the user asks about "Jam" and the retrieval returns sections on "Network Setup," the agent should not attempt to answer. We introduce a **Retrieval Grader** node immediately after retrieval.23

* **Logic:** The LLM evaluates the retrieved documents against the user's query.  
  * *Prompt:* "Does the retrieved context contain the solution to the user's problem: '{query}'?"  
* **Conditional Edge:**  
  * **Yes:** Proceed to SolutionSynthesizer.  
  * **No:** Proceed to Queryrewriter node. This node reformulates the search query (e.g., removing specific details or adding synonyms) and loops back to DynamicRetriever.

This "Reflective RAG" pattern significantly increases success rates, preventing the bot from hallucinating an answer when the manual lookup failed.13

## ---

**5\. Solution Synthesis and Voice Optimization**

The final stage is generating the response. In a text chatbot, a 300-word excerpt is acceptable. In a voice bot, brevity and linearity are paramount. The **Solution Synthesizer** node must transform the raw manual text into a script optimized for TTS.

### **5.1 The "Sandwich" Architecture Constraints**

The voice bot operates as a continuous loop:

Audio Input ![][image3] STT ![][image3] LangGraph Agent ![][image3] TTS ![][image3] Audio Output.

The Agent represents the "Brain" of this sandwich. The critical constraint here is **Latency**. Users will tolerate a 500ms delay in text, but a 2-second silence in voice feels like a broken connection.

* **Streaming:** The SolutionSynthesizer must support **Token Streaming**. LangChain allows streaming the LLM's output token-by-token. These tokens can be piped directly to the TTS engine (if the TTS supports stream input) or buffered into sentences to reduce "Time to First Byte" (TTFB).4

### **5.2 Output Parsing and Artifact Stripping**

LLMs trained on internet data default to Markdown formatting (bolding \*\*, headers \#\#\#, lists \-). TTS engines often read these characters literally or pause awkwardly.

* *Raw Output:* "**Step 1:** Open the tray (see Fig A)."  
* *TTS Interpretation:* "Asterisk asterisk Step one asterisk asterisk Open the tray open parenthesis see Fig A close parenthesis."

We must implement a strict **Output Parser**.25

1. **Regex Cleaning:** Remove all Markdown syntax (\*, \_, \#, \`).  
2. **Visual Reference Removal:** Strip references to images ("See Figure 2.1") as the user cannot see them.  
3. **List Normalization:** Convert bullet points into transitional phrases. "Step 1... Step 2..." becomes "First... Next... Finally..."  
4. **Sentence Segmentation:** Break long compound sentences into shorter, declarative sentences for more natural breathing patterns in the synthesized speech.

### **5.3 The Concise Step-by-Step Protocol**

The user explicitly requested a "step by step instruction." The system prompt for the SolutionSynthesizer must enforce this structure.

**System Prompt:**

"You are a voice support assistant. You have retrieved the technical manual. Your goal is to explain the solution to the user.

Rules:

1. Be extremely concise. Do not read the whole page.  
2. Present the solution as a numbered list of steps.  
3. Do not use Markdown or special characters.  
4. Pause after every 3 steps to check with the user (handled by the graph logic)."

**Interactive Execution:**

Instead of reading 10 steps at once, the graph can implement a "Step Iterator."

* *Bot:* "I found the solution. First, power off the printer. Let me know when you've done that."  
* *User:* "Okay, done."  
* *Bot:* "Next, open the rear cover..." This is achieved by storing the solution\_steps list in the state and maintaining a current\_step\_index. The graph loops between ReadStep and WaitForConfirmation nodes, mirroring the iterative gathering phase.11

## ---

**6\. Advanced Graph Patterns: Persistence and Interrupts**

A robust production system requires handling long-running sessions and human interruptions.

### **6.1 Checkpointing and Persistence**

LangGraph's **Checkpointer** mechanism allows the graph to save its state (the SupportAgentState) to a database (e.g., Postgres, Redis) after every node execution.

* **Session Recovery:** If the connection drops, the user can call back. The system retrieves the thread\_id and restores the exact state (e.g., "We were discussing Error 50 on your HP Printer").  
* **Human Handoff:** If the bot fails (e.g., negative sentiment detected), the state can be loaded by a human agent's dashboard. The human sees the printer\_slots and messages history instantly, avoiding the need for the user to repeat themselves.28

### **6.2 Handling Interrupts (Human-in-the-Loop)**

In voice, a user might interrupt the bot: "Wait, I can't find that latch." LangGraph supports **Interrupts** via interrupt\_before or dynamic breakpoints.5

* **Implementation:** The voice layer (handling STT) detects user speech overlap. It sends a signal to the Agent.  
* **Graph Reaction:** The Agent halts execution of the current node (e.g., SolutionSynthesizer). The user's speech ("I can't find the latch") is injected as a new HumanMessage.  
* **State Update:** The graph resumes from the InputAnalyzer. It detects a "Problem" intent. It routes to a ClarificationNode instead of continuing the step list. This dynamic responsiveness is impossible in linear chains but native to the Graph architecture.30

## ---

**7\. Implementation Roadmap and Logic**

This section details the construction of the system, translating the theoretical architecture into concrete implementation steps.

### **7.1 Phase 1: The Core Graph & Slot Logic**

The first priority is establishing the iterative gathering loop.

**Logic Flow:**

1. **Define State:** Create the SupportAgentState Pydantic model.  
2. **Node analyze\_input:**  
   * Input: state.messages  
   * Tool: LLM with PrinterSlots structured output.  
   * Action: Update state.printer\_slots.  
3. **Node check\_completeness (Router):**  
   * Logic: Check model\_name, symptom.  
   * Return: "ask\_question" OR "retrieve".  
4. **Node generate\_question:**  
   * Input: state.printer\_slots  
   * Logic: Identify top missing priority slot. Generate question.  
   * Action: Return AIMessage.

**Key Insight:** This phase fulfills the "question by question" requirement. The router prevents the graph from progressing to retrieval until the criteria are met.3

### **7.2 Phase 2: Dynamic Retrieval Integration**

The second phase connects the cognitive graph to the data layer.

**Logic Flow:**

1. **Data Ingestion:** Script to load PDF manuals, chunk them, and upload to ChromaDB. *Crucially*, use the filename or metadata to assign collection\_name (e.g., manual\_hp\_m404).  
2. **Node retrieve\_manual:**  
   * Input: state.printer\_slots.model\_name (mapped to collection ID).  
   * Action: Instantiate Chroma(collection\_name=...). Run similarity\_search.  
   * Output: Update state.retrieved\_docs.  
3. **Validation:** Add the grade\_retrieval node to check relevance.

**Key Insight:** This phase fulfills the "receive models name (collection\_name)" requirement.17

### **7.3 Phase 3: Voice Optimization & Testing**

The final phase polishes the interaction for the voice channel.

**Logic Flow:**

1. **Node synthesize\_answer:**  
   * Prompt: "Convert these steps to a concise voice script."  
   * Output Parser: Regex strip Markdown.  
2. **Simulation:** Use **LangSmith** to run trace tests.  
   * *Test Case 1:* "My HP is broken" (Bot should ask for error).  
   * *Test Case 2:* "Error 50 on my Canon" (Bot should search Canon collection).  
   * *Test Case 3:* "My printer... wait, it's a Brother" (Bot should update slot).

**Table 2: Summary of Node Implementations and Responsibilities**

| Node Name | Input | Responsibility | Output |
| :---- | :---- | :---- | :---- |
| SlotManager | messages | Extract entities, handle corrections, update slots. | printer\_slots, missing\_info |
| RouteLogic | printer\_slots | Decide next step: Question or Retrieval? | Edge to QuestionGen or Retriever |
| QuestionGen | missing\_info | Formulate the specific question for the top missing slot. | AIMessage (Text) |
| DynRetriever | collection\_config | Connect to specific ChromaDB collection and query. | retrieved\_docs |
| Grader | retrieved\_docs | Evaluate if docs solve the problem. | Boolean (is\_relevant) |
| Synthesizer | retrieved\_docs | Generate step-by-step voice script. | AIMessage (Clean Text) |

## ---

**8\. Conclusion**

The architectural design presented herein satisfies the complex requirements of a voice-enabled, model-aware printer support agent. By leveraging **LangGraph's** cyclic state management, we move beyond simple linear interactions to a robust, iterative dialogue system that gathers information precisely "question by question." The integration of **LangChain's** runtime configuration capabilities enables true multi-tenancy in **ChromaDB**, ensuring that the retrieval process is strictly scoped to the user's specific hardware model.

This system represents a sophisticated application of Agentic RAG. It handles the ambiguity of human speech through iterative slot filling, respects the latency and formatting constraints of voice interfaces through optimized synthesis, and ensures data relevance through dynamic collection routing. This blueprint provides a scalable, resilient foundation for the next generation of automated technical support.

## **9\. References**

* **LangGraph & Cyclic State:** 1  
* **Iterative Information Gathering:** 2  
* **Slot Filling & Pydantic:** 7  
* **Dynamic RAG & ChromaDB:** 17  
* **Voice Agents & Output Parsing:** 4  
* **Human-in-the-Loop & Interrupts:** 5

#### **Works cited**

1. LangGraph \- LangChain Blog, accessed January 23, 2026, [https://blog.langchain.com/langgraph/](https://blog.langchain.com/langgraph/)  
2. Zero-shot Slot Filling in the Age of LLMs for Dialogue Systems \- ACL Anthology, accessed January 23, 2026, [https://aclanthology.org/2025.coling-industry.59.pdf](https://aclanthology.org/2025.coling-industry.59.pdf)  
3. Graph API overview \- Docs by LangChain, accessed January 23, 2026, [https://docs.langchain.com/oss/python/langgraph/graph-api](https://docs.langchain.com/oss/python/langgraph/graph-api)  
4. Build a voice agent with LangChain, accessed January 23, 2026, [https://docs.langchain.com/oss/python/langchain/voice-agent](https://docs.langchain.com/oss/python/langchain/voice-agent)  
5. Human-in-the-loop \- Docs by LangChain, accessed January 23, 2026, [https://docs.langchain.com/oss/python/langchain/human-in-the-loop](https://docs.langchain.com/oss/python/langchain/human-in-the-loop)  
6. Interrupts \- Docs by LangChain, accessed January 23, 2026, [https://docs.langchain.com/oss/python/langgraph/interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)  
7. The Architecture of Agent Memory: How LangGraph Really Works \- DEV Community, accessed January 23, 2026, [https://dev.to/sreeni5018/the-architecture-of-agent-memory-how-langgraph-really-works-59ne](https://dev.to/sreeni5018/the-architecture-of-agent-memory-how-langgraph-really-works-59ne)  
8. Understanding Pydantic for Data Validation in Langraph | by Mayur Sand \- Medium, accessed January 23, 2026, [https://medium.com/@sand.mayur/understanding-pydantic-for-data-validation-in-langraph-7d483b32e78b](https://medium.com/@sand.mayur/understanding-pydantic-for-data-validation-in-langraph-7d483b32e78b)  
9. Decisions I made when using Pydantic classes to define my LangGraph state \- Medium, accessed January 23, 2026, [https://medium.com/@martin.hodges/decisions-i-made-when-using-pydantic-classes-to-define-my-langgraph-state-264620c0efca](https://medium.com/@martin.hodges/decisions-i-made-when-using-pydantic-classes-to-define-my-langgraph-state-264620c0efca)  
10. Demystifying LangGraph: A Beginner's Guide to State, Nodes, and Reducers, accessed January 23, 2026, [https://prvnk10.medium.com/demystifying-langgraph-a-beginners-guide-to-state-nodes-and-reducers-6d0090a2604f](https://prvnk10.medium.com/demystifying-langgraph-a-beginners-guide-to-state-nodes-and-reducers-6d0090a2604f)  
11. Iterative Workflows in LangGraph | Agentic AI using LangGraph | Tutorial 8 \- Medium, accessed January 23, 2026, [https://medium.com/@frextarr.552/iterative-workflows-in-langgraph-agentic-ai-using-langgraph-tutorial-7-cf4cf07c9e92](https://medium.com/@frextarr.552/iterative-workflows-in-langgraph-agentic-ai-using-langgraph-tutorial-7-cf4cf07c9e92)  
12. An Approach to Build Zero-Shot Slot-Filling System for Industry-Grade Conversational Assistants \- arXiv, accessed January 23, 2026, [https://arxiv.org/html/2406.08848v1](https://arxiv.org/html/2406.08848v1)  
13. LangGraph 101: Let's Build A Deep Research Agent | Towards Data Science, accessed January 23, 2026, [https://towardsdatascience.com/langgraph-101-lets-build-a-deep-research-agent/](https://towardsdatascience.com/langgraph-101-lets-build-a-deep-research-agent/)  
14. Creating Task-Oriented Dialog systems with LangGraph and LangChain \- Medium, accessed January 23, 2026, [https://medium.com/data-science/creating-task-oriented-dialog-systems-with-langgraph-and-langchain-fada6c9c4983](https://medium.com/data-science/creating-task-oriented-dialog-systems-with-langgraph-and-langchain-fada6c9c4983)  
15. LangGraph Simplified \- Kaggle, accessed January 23, 2026, [https://www.kaggle.com/code/marcinrutecki/langgraph-simplified](https://www.kaggle.com/code/marcinrutecki/langgraph-simplified)  
16. Structured output \- Docs by LangChain, accessed January 23, 2026, [https://docs.langchain.com/oss/python/langchain/structured-output](https://docs.langchain.com/oss/python/langchain/structured-output)  
17. Chroma \- Docs by LangChain, accessed January 23, 2026, [https://docs.langchain.com/oss/python/integrations/vectorstores/chroma](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma)  
18. Retrievers | LangChain Reference, accessed January 23, 2026, [https://reference.langchain.com/python/langchain\_core/retrievers/](https://reference.langchain.com/python/langchain_core/retrievers/)  
19. Dynamic routing of retrievers in LangChain \- Stack Overflow, accessed January 23, 2026, [https://stackoverflow.com/questions/77573344/dynamic-routing-of-retrievers-in-langchain](https://stackoverflow.com/questions/77573344/dynamic-routing-of-retrievers-in-langchain)  
20. Agentic RAG With LangGraph \- Qdrant, accessed January 23, 2026, [https://qdrant.tech/documentation/agentic-rag-langgraph/](https://qdrant.tech/documentation/agentic-rag-langgraph/)  
21. BM25Retriever \+ ChromaDB Hybrid Search Optimization using LangChain \- Stack Overflow, accessed January 23, 2026, [https://stackoverflow.com/questions/79477745/bm25retriever-chromadb-hybrid-search-optimization-using-langchain](https://stackoverflow.com/questions/79477745/bm25retriever-chromadb-hybrid-search-optimization-using-langchain)  
22. Milvus | LangChain Reference, accessed January 23, 2026, [https://reference.langchain.com/python/integrations/langchain\_milvus/](https://reference.langchain.com/python/integrations/langchain_milvus/)  
23. Building a RAG Agent with LangGraph, LLaMA3–70b, and Scaling with Amazon Bedrock, accessed January 23, 2026, [https://medium.com/@philippkai/building-a-rag-agent-with-langgraph-llama3-70b-and-scaling-with-amazon-bedrock-2be03fb4088b](https://medium.com/@philippkai/building-a-rag-agent-with-langgraph-llama3-70b-and-scaling-with-amazon-bedrock-2be03fb4088b)  
24. langgraph/examples/rag/langgraph\_crag\_local.ipynb at main \- GitHub, accessed January 23, 2026, [https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph\_crag\_local.ipynb](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_crag_local.ipynb)  
25. Output parsers | LangChain Reference, accessed January 23, 2026, [https://reference.langchain.com/python/langchain\_core/output\_parsers/](https://reference.langchain.com/python/langchain_core/output_parsers/)  
26. Split markdown \- Docs by LangChain, accessed January 23, 2026, [https://docs.langchain.com/oss/python/integrations/splitters/markdown\_header\_metadata\_splitter](https://docs.langchain.com/oss/python/integrations/splitters/markdown_header_metadata_splitter)  
27. Workflows and agents \- Docs by LangChain, accessed January 23, 2026, [https://docs.langchain.com/oss/python/langgraph/workflows-agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)  
28. Persistence \- Docs by LangChain, accessed January 23, 2026, [https://docs.langchain.com/oss/python/langgraph/persistence](https://docs.langchain.com/oss/python/langgraph/persistence)  
29. Langgraph checkpointers with Postgres \- Let's give memory to our AI Agent \- YouTube, accessed January 23, 2026, [https://www.youtube.com/watch?v=N4VUnYRA3BY](https://www.youtube.com/watch?v=N4VUnYRA3BY)  
30. Oversee a prior art search AI agent with human-in-the-loop by using LangGraph and watsonx.ai \- IBM, accessed January 23, 2026, [https://www.ibm.com/think/tutorials/human-in-the-loop-ai-agent-langraph-watsonx-ai](https://www.ibm.com/think/tutorials/human-in-the-loop-ai-agent-langraph-watsonx-ai)  
31. Use of interrupt for human input \- LangGraph \- LangChain Forum, accessed January 23, 2026, [https://forum.langchain.com/t/use-of-interrupt-for-human-input/1729](https://forum.langchain.com/t/use-of-interrupt-for-human-input/1729)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAXCAYAAADUUxW8AAAAlklEQVR4XmNgGNbAHYivoQsSC/5DMclgKQOZmpmA+BMDmZrfADELAxmadYF4FZT9h4FEzciKQSFNtOZmIPZB4m9ggGgWQRLDCX6g8bsZIJod0cQxADbnOTNAxFvRJZDBMiA2RRcEAikGiObN6BIwIAfEv9EFkQBI8xN0wQAg/smAiMt9qNJg/neoHAifAWJjFBWjYLADACCOJ7pducxLAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAXCAYAAAAC9s/ZAAAAxElEQVR4XmNgGAXooAWIPwLxfyj+DsTvgfgDEP+Fij2Dq8YDYAagAykGiPgXdAl0AFK0CV0QCnAZDgd+DBAFBugSQCDIgPAaTnCWAbcNMNuZ0CWQAUyRMhRrAHE/VGwlkjqcAKRwHxC7ALEzlI6Dim9FUocVwPxviC4BBOwMELm76BLI4B8Dbv+DAMEYAEl+RheEghQGiPxRdAkYgCWScnQJIDBigMj9RpcAAVBAnWRAOO8GEB8E4r1QGiY+AaZhFAw7AABnZTlUC60sAgAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAXCAYAAADpwXTaAAAAVUlEQVR4XmNgGAWjgKpgL7oAJeAfugAlwAaIy9AFKQHngNgcXRAETMjEt4B4HwMa8CMTX4NiFgYKwUQg9kYXJAcoAnEnuiC54BO6ACXgMLrAKBhuAACnlhESw2iRqwAAAABJRU5ErkJggg==>