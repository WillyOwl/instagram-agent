---
trigger: always_on
---

\# AGENTS.md



Welcome to the Dynamic Agent Planning and Development Guide. This file establishes a structured, interactive workflow where the AI agent is responsible for generating its own planning questions, gathering answers, and executing development steps based on the resulting project profile. Remember to always read this file when making responses.



\---



\## 1. Environment & Setup

### 🔄 Dynamic Planning Phase
*Before executing any setup procedures, the AI agent must analyze the repository context and prompt the user with targeted questions to determine:*
1. The specific agent platform or integration framework required.
2. The preferred package manager, runtime environment, and workspace structure.
3. The messaging architecture (e.g., real-time streaming, webhooks, or polling loops).
4. The model pathway setup:
   - **Fixed-Model setup** (e.g., a single predefined local model like Ollama or a single direct API provider).
   - **Configurable-Model setup** (supporting dynamic switching between multiple model providers via LiteLLM).

> **Agent Note:** Log the generated questions and user decisions directly below this block before proceeding.

### 📋 Setup Instructions
- **Dependency Isolation:** Ensure all project dependencies match the framework determined during the planning phase. Pin version numbers strictly to prevent breaking changes in third-party API clients.
- **Workspace Integration:** For monorepos, run the platform-specific installation command from the workspace root to expose new packages to your compiler, linter, and build tools.
- **Environment Configuration:** Copy the baseline environment template to a localized `.env` file. Populate necessary credentials, API tokens, or session configurations without exposing them to source control.
  - **Fixed-Model pathway:** Ensure the specific host/model configuration variables are specified (e.g. `OLLAMA_BASE_URL`).
  - **Configurable-Model pathway:** Add an `ACTIVE_MODEL` selection variable and placeholder keys for all required model providers (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).



\---



\## 2. Agent Core \& Memory Architecture



\### 🔄 Dynamic Planning Phase

\*Before designing the agent logic, the AI agent must evaluate the data constraints and prompt the user with targeted questions to determine:\*

1\. The memory strategy to mimic user style (e.g., Few-Shot prompting patterns or dynamic Retrieval-Augmented Generation / RAG).

2\. The core language model orchestrator (e.g., cloud-based APIs, local model deployments, or a multi-provider configurable setup).

3\. The specific model providers and API key strategy (e.g., if using a Configurable-Model setup, which API keys/environment variables are needed).

4\. The source data format available for training or style injection.



> \*\*Agent Note:\*\* Log the generated questions and memory architecture decisions directly below this block before proceeding.



\### 🧠 Architectural Rules

\- \*\*Model Orchestration Pathways:\*\*

  \- \*\*Fixed-Model Pathway:\*\* Use direct, provider-specific libraries or local endpoints (e.g., Ollama or a direct provider SDK) for a dedicated configuration.

  \- \*\*Configurable-Model Pathway (LiteLLM):\*\* Transition `agent.py` to be model-agnostic by using a universal wrapper like `litellm` instead of hardcoded client libraries. Support dynamic model switching (e.g., Anthropic, OpenAI, Gemini, Xiaomi, etc.) via the `ACTIVE_MODEL` configuration variable.

\- \*\*Dynamic Startup Validation:\*\* For Configurable-Model setups, implement a startup check mapping the active model to its required API key environment variable. Validate that the key exists before initiating client calls, raising a descriptive initialization error if missing.

\- \*\*Data Parsing:\*\* Utilize a dedicated history parser module to clean input logs, converting raw exported formats into standardized message-and-response token chains.

\- \*\*Context Injection:\*\* If utilizing Few-Shot prompting, embed high-quality style examples directly into the primary system message context. If using RAG, initialize and index the vector embeddings prior to launching the main agent execution loop.

\- \*\*Rate \& Safety Management:\*\* Configure a routing wrapper to gracefully handle provider timeouts or rate limits. Implement human-like delay intervals to protect account stability on destination platforms.



\---



\## 3. Testing \& Quality Assurance



\### 🔄 Dynamic Planning Phase

\*Before verifying code changes, the AI agent must determine the testing parameters by prompting the user to establish:\*

1\. The native test runner framework and assertion libraries assigned to this specific stack.

2\. The minimum coverage expectations or validation rules required for new behavioral features.



> \*\*Agent Note:\*\* Log the test planning questions and validation constraints directly below this block before proceeding.



\### 🧪 Testing Procedures

\- \*\*Targeted Test Execution:\*\* Run the workspace-specific unit testing suite filtered to the modified agent module to confirm baseline logic compliance.

\- \*\*Behavioral Isolation:\*\* Mock all third-party external network dependencies—including live LLM API calls and social media platform login workflows—using local fixtures to guarantee deterministic test runs.
  \- **Configurable-Model pathway:** Ensure that unit tests mock the universal wrapper (e.g., `litellm.completion`) rather than individual provider SDKs, to keep test runs fully offline and deterministic.

\- \*\*Static Verification:\*\* Execute full linting and static typing validation passes across modified agent directories to prevent runtime exceptions caused by unvalidated payload structures.



\---



\## 4. Pull Requests \& Deployment



\### 🔄 Dynamic Planning Phase

\*Prior to merging changes, the AI agent must prompt the user to define:\*

1\. The code review and prompt verification policies required for landing changes.

2\. The target runtime environment for final deployment.



> **Agent Note:** Log the deployment planning questions and pipeline rules directly below this block before proceeding.
- **Q1: Code review and prompt verification policies?**
  - **Decision:** Require local linting (Ruff/Mypy) and running unit tests prior to landing changes.
- **Q2: Target runtime environment for final deployment?**
  - **Decision:** Local/Desktop environment (run manually or as a local background daemon/cron job).




\### 🚀 PR Guidelines

\- \*\*PR Naming Structure:\*\* Prefix code repository modifications with the project context tag in the following title format: `\[<project\_name>] <Descriptive Summary Title>`.

\- \*\*Pre-Commit Verification:\*\* Run the entire test and linting suite locally to ensure the codebase remains completely functional before pushing updates to remote repository branches.

\- \*\*Prompt Benchmarking:\*\* If any system prompts or memory injection mechanisms were modified during the task, append a sample evaluation log to the PR description demonstrating how the agent handles typical query edge cases.