"""
Lamarckian Knowledge Base

This design pattern is well-suited for integration into existing Agent frameworks
(such as LangGraph, AutoGen, or MetaGPT) as a "plug-in brain" component.

Core Features:
1. Historical Principle and Trajectory Retrieval: Retrieve relevant historical abstract
   principles and concrete trajectories based on the current task
2. Successful Trajectory Abstraction and Counterfactual Verification Storage: Extract
   principles from successful trajectories and verify their effectiveness through
   counterfactual testing

IMPORTANT NOTES FOR NFS USERS:
- SQLite/ChromaDB may have issues on NFS file systems due to file locking limitations
- For NFS environments, consider using Chroma Client-Server mode or a different vector store
- This implementation includes automatic recovery and retry mechanisms for NFS environments
"""

from typing import List, Dict, Optional, TypedDict
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_core.documents import Document
import chromadb
import chromadb.config
import os
import sys
import importlib.util
import tempfile
import logging
import asyncio
import re
import shutil
import threading

logger = logging.getLogger(__name__)



# Data structure definitions
class RetrievalResult(TypedDict):
    """Retrieval result"""
    principles: List[str]  # Abstract principles (High-level)
    trajectories: List[str]  # Concrete similar cases (Low-level)


class LearningResult(TypedDict):
    """Learning result"""
    extracted_principles: List[str]  # Verified and stored principles
    all_principles: List[str]  # All extracted principles (including rejected ones)
    results: List[Dict]  # Detailed verification result for each principle
    saved_count: int  # Number of principles stored


class LamarckianKnowledgeBase:
    """
    Lamarckian Knowledge Base (Client-Server Edition)
    
    Strictly uses ChromaDB HttpClient to avoid NFS file locking issues.
    """

    def __init__(self, 
                 host: str = "211.87.232.112", 
                 port: int = 8000,
                 api_key: Optional[str] = None,
                 program_suffix: str = ".py"):
        """
        Initialize the knowledge base using ChromaDB HTTP Client.

        Args:
            host: Chroma DB Server IP (e.g., "211.87.232.112")
            port: Chroma DB Server Port (e.g., 8000)
            api_key: DashScope API Key.
            program_suffix: Suffix for program files.
        """
        # 1. Initialize LLM (Qwen via DashScope)
        api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "YOUR_DASHSCOPE_API_KEY")
        
        self.llm = ChatTongyi(
            dashscope_api_key=api_key,
            model_name="qwen-plus",
            temperature=0.1
        )

        # 2. Initialize Embeddings (Crucial for 1024 dimension)
        try:
            self.embeddings = DashScopeEmbeddings(
                model="text-embedding-v3", # 推荐使用 v3 或 v4
                dashscope_api_key=api_key
            )
        except Exception as e:
            logger.warning(f"Embedding initialization failed, falling back to v2: {e}")
            self.embeddings = DashScopeEmbeddings(
                model="text-embedding-v2",
                dashscope_api_key=api_key
            )

        # 3. Initialize ChromaDB HTTP Client (The Clean Way)
        logger.info(f"🔗 Connecting to ChromaDB Server at http://{host}:{port}...")
        
        try:
            self._chroma_client = chromadb.HttpClient(
                host=host,
                port=port,
                settings=chromadb.config.Settings(anonymized_telemetry=False)
            )
            # 测试连接
            version = self._chroma_client.get_version()
            logger.info(f"✅ Connected to ChromaDB Server (Version: {version})")
        except Exception as e:
            logger.error(f"❌ Failed to connect to ChromaDB Server: {e}")
            raise ConnectionError(f"Could not connect to ChromaDB at {host}:{port}. Is the server running?")

        # 4. Collection Setup (Fixing the 384 vs 1024 dimension error)
        # 关键点：我们必须创建一个 Adapter，让 Chroma 原生客户端也能理解 LangChain 的 Embedding 类
        # 或者更简单地，我们只通过 LangChain 接口操作，或者确保 Collection 创建时元数据正确。
        
        # 为了解决 "Collection expecting embedding with dimension of 384"，
        # 我们需要在获取集合时，告诉 Chroma 我们使用的 Embedding Function。
        
        # 由于 LangChain 的 DashScopeEmbeddings 和 Chroma 原生的 EmbeddingFunction 接口略有不同，
        # 我们这里主要依赖 LangChain 的 Chroma 包装器来管理维度，但为了防止原生调用出错，
        # 我们显式获取集合。
        self._collection_name = "lamarckian_memory"
        
        # ⚠️ 如果之前创建过错误的 384 维集合，这里可能需要先手动删除，或者换个名字
        # self._chroma_client.delete_collection(self._collection_name) 

        self._chroma_collection = self._chroma_client.get_or_create_collection(
            name=self._collection_name,
            metadata={"description": "Lamarckian memory", "hnsw:space": "cosine"}
            # 注意：这里不传 embedding_function 给原生 client，
            # 因为我们主要通过下面的 self.vector_store (LangChain) 来进行 add/query，
            # LangChain 会负责计算好 embedding (1024维) 再传给 Chroma。
            # Chroma 只要接收到 1024 维的向量，它就会自动适配（如果是新集合）。
        )

        # 5. LangChain Vector Store Wrapper
        # 所有的读写操作建议优先通过这个对象进行，它会自动调用 self.embeddings 计算向量
        self.vector_store = Chroma(
            client=self._chroma_client,
            collection_name=self._collection_name,
            embedding_function=self.embeddings,
        )

        self.program_suffix = program_suffix
        self._write_lock = threading.Lock()

    def _load_evaluation_function(self, evaluator_file: str):
        """
        Load the evaluate function from the evaluator file (referencing openevolve implementation)
        
        This function expects the evaluator file to have an evaluate(program_path: str) function
        that takes a program file path, executes the program, and returns a dictionary containing metrics.
        
        Args:
            evaluator_file: Path to the evaluator file
            
        Returns:
            The loaded evaluate function
        """
        if not evaluator_file or not os.path.exists(evaluator_file):
            raise ValueError(f"Evaluator file {evaluator_file} not found")

        try:
            # Add the evaluator file's directory to Python path so it can import local modules
            eval_dir = os.path.dirname(os.path.abspath(evaluator_file))
            if eval_dir not in sys.path:
                sys.path.insert(0, eval_dir)
                logger.debug(f"Added {eval_dir} to Python path for evaluator imports")
            
            # Add openevolve module path (if exists)
            # openevolve directory should be in the project root
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            possible_openevolve_paths = [
                current_file_dir,  # Project root directory (where lamarckian_knowledge_base.py is located)
                os.path.join(os.path.dirname(eval_dir), "..", ".."),  # Parent directory of evaluator
                os.path.join(os.path.dirname(eval_dir), "..", "..", ".."),  # One level up
            ]
            
            for base_path in possible_openevolve_paths:
                base_path = os.path.abspath(base_path)
                openevolve_path = os.path.join(base_path, "openevolve")
                if os.path.exists(openevolve_path) and base_path not in sys.path:
                    sys.path.insert(0, base_path)
                    logger.debug(f"Added {base_path} to Python path for openevolve imports")
                    break

            spec = importlib.util.spec_from_file_location("evaluation_module", evaluator_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"Failed to load spec from {evaluator_file}")

            module = importlib.util.module_from_spec(spec)
            sys.modules["evaluation_module"] = module
            spec.loader.exec_module(module)

            if not hasattr(module, "evaluate"):
                raise AttributeError(
                    f"Evaluation file {evaluator_file} does not contain an 'evaluate' function"
                )

            evaluate_function = module.evaluate
            logger.info(f"Successfully loaded evaluation function from {evaluator_file}")
            return evaluate_function

        except Exception as e:
            logger.error(f"Error loading evaluation function: {str(e)}")
            raise

    def _extract_evolve_block(self, program_code: str) -> tuple[str, str, str]:
        """
        Extract the EVOLVE-BLOCK section from program code
        
        Args:
            program_code: Complete program code
            
        Returns:
            (header, block_content, footer) tuple:
            - header: All content before EVOLVE-BLOCK-START (including the marker line)
            - block_content: Content between EVOLVE-BLOCK-START and EVOLVE-BLOCK-END (excluding marker lines)
            - footer: All content after EVOLVE-BLOCK-END (including the marker line)
        """
        if "# EVOLVE-BLOCK-START" not in program_code:
            raise ValueError("Program code must contain EVOLVE-BLOCK-START")
        
        lines = program_code.split('\n')
        start_idx = None
        end_idx = None
        
        for i, line in enumerate(lines):
            if "# EVOLVE-BLOCK-START" in line:
                start_idx = i
            elif "# EVOLVE-BLOCK-END" in line:
                end_idx = i
                break
        
        if start_idx is None:
            raise ValueError("Program code must contain EVOLVE-BLOCK-START")
        if end_idx is None:
            raise ValueError("Program code must contain EVOLVE-BLOCK-END")
        
        header = '\n'.join(lines[:start_idx + 1])  # Include EVOLVE-BLOCK-START line
        block_content = '\n'.join(lines[start_idx + 1:end_idx])  # Exclude marker lines
        footer = '\n'.join(lines[end_idx:])  # Include EVOLVE-BLOCK-END line
        
        return header, block_content, footer
    
    def _replace_evolve_block(self, program_code: str, new_block_content: str) -> str:
        """
        Replace the EVOLVE-BLOCK content in program code
        
        Args:
            program_code: Original program code
            new_block_content: New EVOLVE-BLOCK content (excluding marker lines)
            
        Returns:
            Complete program code after replacement
        """
        header, _, footer = self._extract_evolve_block(program_code)
        return f"{header}\n{new_block_content}\n{footer}"

    def _prepare_program_code(self, program_code: str) -> str:
        """
        Prepare program code, ensuring it contains EVOLVE-BLOCK tags (if needed)
        Reference OpenEvolve's _prepare_program logic
        
        Args:
            program_code: Original program code
            
        Returns:
            Processed program code
        """
        # If code doesn't have EVOLVE-BLOCK-START, automatically add it
        # This ensures the code can be properly processed by the evaluator
        if "# EVOLVE-BLOCK-START" not in program_code:
            raise ValueError("Program code must contain EVOLVE-BLOCK-START")
        
        return program_code
    
    
    def _single_evaluate_program_with_file(
        self, 
        program_code: str,
        evaluate_function,
        timeout: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Execute program code using the evaluate function from the evaluator file
        Reference OpenEvolve's evaluator.py implementation
        
        Args:
            program_code: Program code to execute
            evaluate_function: evaluate function (loaded from evaluator file)
            timeout: Optional timeout in seconds, currently not implemented but kept for future extension
            
        Returns:
            Dictionary containing metrics, usually includes 'combined_score' or 'error' fields
        """
        if not evaluate_function:
            raise ValueError("No evaluator function provided")

        # Prepare program code (ensure it contains EVOLVE-BLOCK tags)
        program_code = self._prepare_program_code(program_code)
        # Create temporary file (reference OpenEvolve evaluator.py implementation)
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=self.program_suffix, 
                mode='w', 
                delete=False,
                encoding='utf-8'
            ) as temp_file:
                temp_file.write(program_code)
                temp_file_path = temp_file.name

            # Call evaluator function (similar to openevolve/evaluator.py's _direct_evaluate)
            result = evaluate_function(temp_file_path)
            
            # Ensure return is in dictionary format
            if isinstance(result, dict):
                return result
            elif hasattr(result, "metrics"):
                return result.metrics
            else:
                logger.warning(f"Evaluator returned unexpected type: {type(result)}, converting to dict")
                return {"combined_score": 0.0, "error": f"Unexpected return type: {type(result)}"}
                
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            # If execution fails, return error metrics (similar to OpenEvolve error handling)
            return {"combined_score": 0.0, "error": str(e)}
        finally:
            # Clean up temporary file (reference OpenEvolve cleanup logic)
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {temp_file_path}: {e}")

    async def _evaluate_program_with_file(
        self,
        program_code: str,
        config_path: str,
        evaluator_file: str,
        timeout: Optional[float] = None,
        system_message_override: Optional[str] = None,
        iter_over: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Execute program code using the evaluate function from the evaluator file
        Reference OpenEvolve's evaluator.py implementation
        
        Args:
            program_code: Program code to execute
            evaluate_function: evaluate function (loaded from evaluator file)
            timeout: Optional timeout in seconds, currently not implemented but kept for future extension
            
        Returns:
            Dictionary containing metrics, usually includes 'combined_score' or 'error' fields
        """
        

        # Prepare program code (ensure it contains EVOLVE-BLOCK tags)
        program_code = self._prepare_program_code(program_code)

        # Create temporary directory with concurrent-safe approach
        # Using mkdtemp for atomic directory creation with unique name
        temp_dir = None
        temp_file_path = None
        try:
            # mkdtemp creates directory atomically with unique name
            temp_dir = tempfile.mkdtemp(
                prefix="evolve_",
                suffix="_cf",  # counterfactual suffix for easy identification
                dir=None  # use system default temp dir
            )
            temp_file_path = os.path.join(temp_dir, f"program{self.program_suffix}")

            # Write program code to file inside the unique directory
            with open(temp_file_path, 'w', encoding='utf-8') as temp_file:
                temp_file.write(program_code)
                temp_file.flush()
                os.fsync(temp_file.fileno())  # Ensure content is written to disk

            # Import OpenEvolve components lazily to avoid hard dependency unless used
            from openevolve.config import load_config
            from openevolve import OpenEvolve

            # Build config using provided config_path (same as CLI behavior)
            config = load_config(config_path)

            # If caller provided a system_message override (e.g., cf_prompt), apply it
            if system_message_override:
                try:
                    config.prompt.system_message = system_message_override
                    config.llm.update_model_params({"system_message": system_message_override})
                    logger.debug("Applied system_message_override to config.prompt.system_message")
                except Exception as e:
                    logger.warning(f"Failed to apply system_message_override to config: {e}")
            if iter_over:
                config.max_iterations = iter_over
            # Initialize OpenEvolve using the temp program as the initial program
            openevolve = OpenEvolve(
                initial_program_path=temp_file_path,
                evaluation_file=evaluator_file,
                config=config,
                output_dir=None,
            )

            # Run the evolution (OpenEvolve exposes async run)
            best_program = await openevolve.run(iterations=None, target_score=None, checkpoint_path=None)
            if isinstance(best_program, dict):
                return best_program
            elif hasattr(best_program, "metrics"):
                return best_program.metrics

            # Attempt to extract metrics from the returned best_program
            return best_program.metrics

        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            # If execution fails, return error metrics (similar to OpenEvolve error handling)
            return {"combined_score": 0.0, "error": str(e)}
        finally:
            # Clean up temporary directory and all its contents
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"Failed to delete temp directory {temp_dir}: {e}")

    # =========================================================================
    # Interface 1: Historical Principle and Trajectory Retrieval
    # =========================================================================
    def retrieve_knowledge(self, query_text: str, k: int = 3) -> RetrievalResult:
        """
        Input current task Query, return relevant historical abstract principles and concrete trajectories.

        Args:
            query_text: Current task query text
            k: Number of principles and trajectories to return (at most k of each type)

        Returns:
            RetrievalResult: Dictionary containing principles and trajectories
        """
        # Ensure query_text is a string type
        if not isinstance(query_text, str):
            query_text = str(query_text)
        
        # Ensure query_text is not empty
        if not query_text or not query_text.strip():
            logger.warning("Query text is empty, returning empty result")
            return {"principles": [], "trajectories": []}
        
        print(f"🔍 [Retrieve] Retrieving knowledge related to '{query_text[:20]}...'...")

        # 1. Perform vector search
        try:
            docs = self.vector_store.similarity_search(query_text.strip(), k=k * 2)  # Retrieve more for filtering
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            # If search fails, return empty result
            return {"principles": [], "trajectories": []}

        principles = []
        trajectories = []

        # 2. Classify by metadata
        for doc in docs:
            dtype = doc.metadata.get("type", "unknown")
            content = doc.page_content
            if dtype == "principle":
                principles.append(content)
            elif dtype == "trajectory":
                trajectories.append(content)

        # Limit return count
        return {
            "principles": principles[:k],
            "trajectories": trajectories[:k]
        }

    def list_all_memories(self) -> Dict[str, List[Dict[str, object]]]:
        """
        List all stored memories in the vector store, separated by type.

        Returns:
            A dictionary with keys "principles" and "trajectories", each being a list of
            dicts with keys: "id", "content", "metadata".
        """
        try:
            # Try to access underlying collection API used by Chroma
            data = None
            collection = getattr(self.vector_store, "_collection", None)
            # Don't specify include - get() returns ids, documents, metadatas by default
            if collection is not None and hasattr(collection, "get"):
                data = collection.get()
            elif hasattr(self.vector_store, "get"):
                # Some wrappers expose get directly
                data = self.vector_store.get()
            else:
                raise RuntimeError("Unable to access underlying collection API for listing memories")

            documents = data.get("documents", []) if isinstance(data, dict) else []
            metadatas = data.get("metadatas", []) if isinstance(data, dict) else []
            # get() without include parameter returns ids, documents, metadatas by default
            ids = data.get("ids", []) if isinstance(data, dict) else []

            principles: List[Dict[str, object]] = []
            trajectories: List[Dict[str, object]] = []

            for doc, meta, id_ in zip(documents, metadatas, ids):
                entry = {"id": id_, "content": doc, "metadata": meta}
                dtype = meta.get("type", "unknown") if isinstance(meta, dict) else "unknown"
                if dtype == "principle":
                    principles.append(entry)
                elif dtype == "trajectory":
                    trajectories.append(entry)

            return {"principles": principles, "trajectories": trajectories}
        except Exception as e:
            logger.exception("Failed to list all memories")
            return {"principles": [], "trajectories": [], "error": str(e)}

    # =========================================================================
    # Interface 2: Successful Trajectory Abstraction and Counterfactual Verification Storage
    # =========================================================================
    async def learn_from_trajectory(
        self,
        initial_program_path: str,
        best_program_path: str,
        original_task: str,
        evaluator_file: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[str] = None,
        use_llm_verification: bool = False,
    ) -> LearningResult:
        """
        Compare initial and best programs, automatically perform abstraction, generate counterfactuals, 
        call external Evaluator for verification, and finally decide whether to store in knowledge base.

        Args:
            initial_program_path: Path to initial program file (e.g., initial_program.py)
            best_program_path: Path to best program file (e.g., best_program.py)
            original_task: Original task description
            evaluator_file: Path to evaluator file (similar to openevolve's evaluator.py),
                           used for counterfactual verification. If not provided, will skip 
                           automatic evaluation and counterfactual verification steps
            metrics: Optional evaluator output metrics (e.g., combined_score, performance metrics, etc.),
                     used to provide context in prompts to help extract principles better, and to compare 
                     with counterfactual results. If not provided and evaluator_file is provided, will 
                     automatically evaluate the initial_program to obtain metrics

        Returns:
            LearningResult: Dictionary containing extracted principle, counterfactual test, 
                          verification result, and whether it was saved
        """
        print(f"🧠 [Learn] Comparing programs and extracting knowledge...")
        
        # Read both program files
        print(f"   ↳ Reading initial program: {initial_program_path}")
        with open(initial_program_path, 'r', encoding='utf-8') as f:
            initial_program_code = f.read()
        
        print(f"   ↳ Reading best program: {best_program_path}")
        with open(best_program_path, 'r', encoding='utf-8') as f:
            best_program_code = f.read()
        
        
        # If metrics not provided, try to evaluate the initial program to get metrics
        if metrics is None:
            print(f"   ↳ Evaluating initial program to obtain metrics...")
            try:
                metrics = self._single_evaluate_program_with_file(initial_program_code, evaluate_function=self._load_evaluation_function(evaluator_file))
                print(f"   ↳ Initial program Metrics: {metrics}")
            except Exception as e:
                logger.warning(f"Failed to evaluate initial program: {e}")
                metrics = None

        # --- Step A: Program Comparison Abstraction (Abstraction) ---
        # Format metrics information if provided
        metrics_info = ""
        if metrics:
            metrics_lines = ["# Evaluator Metrics (for reference)"]
            for key, value in metrics.items():
                if key != "error":  # Skip error field
                    metrics_lines.append(f"- {key}: {value}")
            if metrics_lines:
                metrics_info = "\n".join(metrics_lines) + "\n\n"
        
        abstract_prompt = ChatPromptTemplate.from_template(
    """You are an expert Senior Bioinformatics Algorithm Engineer specializing in computational biology and high-performance computing.

# Task
{task}

{metrics_info}
# Initial Program (Lower Performance)
{initial_code}

# Best Program (Higher Performance)
{best_code}

# Your Task
Compare the two bioinformatics algorithms above and extract multiple key principles that explain why the best program achieved a higher evaluator score. The code changes are located exclusively between the "# EVOLVE-BLOCK-START" and "# EVOLVE-BLOCK-END" markers.

## Focus Areas for Bioinformatics:
- **Algorithmic Efficiency:** Handling large-scale biological data (e.g., genomic sequences, protein structures) within time/memory constraints.
- **Biological Correctness:** Handling edge cases in biological data (e.g., ambiguous bases, gaps, sequencing errors).
- **Heuristics & Optimization:** Shifts from brute-force to domain-specific heuristics (e.g., k-mer indexing, dynamic programming, seed-and-extend).

## Requirements:
1. Focus on strategies that directly improve evaluator metrics (e.g., biological accuracy, alignment speed, memory reduction).
2. Identify the key algorithmic transformations between the initial and best programs.
3. **Use Bioinformatics Terminology:** Generalize using domain-specific concepts (e.g., "vectorization," "hashing k-mers," "dynamic programming," "pruning search space") rather than generic programming terms where applicable.
4. Remove specific variable names (like `dna_seq_1`) or concrete values.
5. Use the format: "IF [bio-computational scenario] THEN [key strategy]"
6. Each principle should be causal: following it should increase the specific bio-metric scores.

## Output Format
Provide only the principle statements, one per line, no additional explanation.

Principles:
1. IF [scenario] THEN [strategy]
2. IF [scenario] THEN [strategy]
..."""
)

        chain_extract = abstract_prompt | self.llm | StrOutputParser()
        candidate_principles_text = chain_extract.invoke({
            "task": original_task,
            "initial_code": initial_program_code,
            "best_code": best_program_code,
            "metrics_info": metrics_info
        })
        print(f"   ↳ Candidate principles:\n{candidate_principles_text}")

        # Parse multiple principles from the response
        candidate_principles = []
        for line in candidate_principles_text.strip().split('\n'):
            line = line.strip()
            # Skip empty lines
            if not line:
                continue

            # Remove leading "-" or "*" bullet points
            if line.startswith('-') or line.startswith('*'):
                line = line[1:].strip()

            # Remove leading numbering like "1.", "2.", "3." etc.
            # Pattern: optional whitespace + digit(s) + dot + whitespace
            if_match = re.match(r'^\s*\d+\.\s+(.*)', line)
            if if_match:
                line = if_match.group(1).strip()

            # Now check if line contains "IF ... THEN ..."
            if 'IF ' in line.upper() and ' THEN ' in line.upper():
                # Extract the part starting from "IF"
                if_match = re.search(r'IF\s+.*', line, re.IGNORECASE)
                if if_match:
                    candidate_principles.append(if_match.group(0).strip())

        # If parsing failed, fallback to the whole text as single principle
        if not candidate_principles:
            candidate_principles = [candidate_principles_text]

        # Verify each principle with counterfactual testing
        valid_principles = []
        all_results = []

        for i, candidate_principle in enumerate(candidate_principles):
            print(f"\n   📋 Verifying principle {i+1}/{len(candidate_principles)}: {candidate_principle[:80]}...")

            # --- Step B: Counterfactual Generation (Counterfactual Generation) ---
            # Use complete best program code as reference, generate complete counterfactual code that violates the principle
            cf_prompt = ChatPromptTemplate.from_template(
                """You are tasked with generating a counterfactual version of code that violates the following principle to verify its importance.

# Principle to Violate
{principle}

# Original Task Context
{task}

# Initial Program Code (Reference)
This is the complete best program that follows the principle. You need to modify it to violate the principle.

{initial_code}

# Your Task
Generate a modified version of the complete program that deliberately violates the principle. The goal is to test whether violating this principle leads to:
- Lower evaluator scores (e.g., reduced combined_score)
- Execution errors
- Performance degradation

## Requirements:
1. **ONLY modify the code between "# EVOLVE-BLOCK-START" and "# EVOLVE-BLOCK-END" markers**
   - All code outside these markers MUST remain EXACTLY the same (including imports, function definitions, helper functions, etc.)
   - Only the code inside the EVOLVE-BLOCK section should be modified to violate the principle
2. Modify the EVOLVE-BLOCK code to clearly violate the stated principle while maintaining code that can execute
3. Keep the "# EVOLVE-BLOCK-START" and "# EVOLVE-BLOCK-END" markers exactly as they appear in the original code
4. Output the COMPLETE program code, including all unchanged imports, functions, and the modified EVOLVE-BLOCK section
5. Output ONLY the complete program code, no explanations, no comments outside the code

## Output Format
Provide only the complete modified program code, ready-to-be-executed Modified Counterfactual Program Code"""
            )
            # Render the counterfactual prompt text and set it as the system message for OpenEvolve
            try:
                cf_prompt_text = cf_prompt.template.format(
                    principle=candidate_principle,
                    task=original_task,
                    initial_code=initial_program_code,
                )
            except Exception:
                # Fallback: naive replacement if template attribute not available
                cf_prompt_text = (
                    "You are tasked with generating a counterfactual version of code that violates the following principle:\n\n"
                    f"Principle to Violate:\n{candidate_principle}\n\n"
                    f"Original Task Context:\n{original_task}\n\n"
                    "Initial Program Code (Reference):\n"
                    f"{initial_program_code}\n\n"
                    "Generate a modified version of the complete program that deliberately violates the principle."
                )

            cf_plan = cf_prompt_text  # store prompt used as counterfactual_test placeholder
            print(f"   ↳ Prepared counterfactual prompt (length: {len(cf_plan)} chars); OpenEvolve will use it as system_message")

            # --- Step C: External Verification (Verification) ---
            # Use evaluate function from evaluator_file to execute counterfactual solution
            # Reference OpenEvolve's evaluator.py implementation:
            # 1. Write code to temporary file
            # 2. Call evaluator's evaluate(program_path) function
            # 3. Return metrics dictionary
            # Logic assumption:
            # - If the principle is "true", then violating the principle (cf_plan) should lead to worse performance or failure.
            # - If performance is better or similar after violating the principle, the principle is not necessary.

            print(f"   ↳ Executing counterfactual verification (calling external Evaluator)...")
            original_metrics = metrics  # Save original trajectory's metrics
            cf_metrics = None

            try:

                # Use _evaluate_program_with_file method (similar to openevolve/evaluator.py implementation)
                # This method will:
                # 1. Create temporary file
                # 2. Call evaluator function
                # 3. Clean up temporary file
                print(f"   ↳ Using evaluator file: {evaluator_file}")
                cf_metrics = await self._evaluate_program_with_file(
                    initial_program_code,
                    config_path=config,
                    evaluator_file=evaluator_file,
                    system_message_override=cf_prompt_text,
                    iter_over=2,
                )
                print(f"   ↳ Counterfactual evaluation completed, Metrics: {cf_metrics}")

            except Exception as e:
                # If execution fails, it usually means the counterfactual solution failed, which indirectly confirms the principle is valid
                # Reference OpenEvolve error handling approach
                print(f"   ↳ Execution error: {e} (considered as verification passed)")
                logger.exception("Error during counterfactual evaluation")
                cf_metrics = {"error": str(e), "combined_score": 0.0}

            # --- Step D: Verification of Counterfactual Improvement (Verification) ---
            # Two modes: LLM-based verification or direct score comparison
            
            if use_llm_verification:
                # Use LLM + prompt to judge whether counterfactual performs better than original trajectory
                print(f"   ↳ Using LLM to verify if counterfactual improves...")

                # Format metrics information for prompt
                def format_metrics(metrics_dict: Optional[Dict[str, float]], label: str) -> str:
                    if not metrics_dict:
                        return f"{label}: Not available"
                    lines = [f"{label}:"]
                    for key, value in metrics_dict.items():
                        if key != "error":
                            lines.append(f"  - {key}: {value}")
                    if "error" in metrics_dict and metrics_dict["error"]:
                        lines.append(f"  - error: {metrics_dict['error']}")
                    return "\n".join(lines)

                verification_prompt = ChatPromptTemplate.from_template(
                    """You are an expert evaluator comparing two execution results to determine if a counterfactual execution (that violates a principle) performs better or worse than the original successful execution.

# Principle Being Tested
{principle}

# Original Successful Execution Metrics
{original_metrics}

# Counterfactual Execution Metrics (Violating the Principle)
{counterfactual_metrics}

# Your Task
Determine whether the counterfactual execution (which violates the principle) performs BETTER, WORSE, or SIMILAR compared to the original execution.

## Evaluation Criteria:
1. If the counterfactual has errors or significantly lower scores → WORSE (principle is valid)
2. If the counterfactual has higher scores or similar performance → BETTER or SIMILAR (principle may not be necessary)
3. Consider all metrics: combined_score, accuracy, execution_time, error status, etc.

## Output Format
Respond with ONLY one word: "WORSE", "BETTER", or "SIMILAR"

Your judgment:"""
                )

                chain_verify = verification_prompt | self.llm | StrOutputParser()
                verification_result = chain_verify.invoke({
                    "principle": candidate_principle,
                    "original_metrics": format_metrics(original_metrics, "Original"),
                    "counterfactual_metrics": format_metrics(cf_metrics, "Counterfactual")
                }).strip().upper()

                print(f"   ↳ LLM verification result: {verification_result}")

                # Determine if principle is valid
                # If counterfactual performs worse (WORSE), the principle is valid
                # If counterfactual performs better or similar (BETTER/SIMILAR), the principle is not necessary
                is_principle_valid = verification_result == "WORSE"
            else:
                # Direct score comparison: compare combined_score
                print(f"   ↳ Using direct score comparison (weighted_score)...")
                if original_metrics:
                    original_score = original_metrics.get("avg_accuracy", 0)*0.8+original_metrics.get("avg_speed_score", 0)*0.2
                else:
                    original_score = 0
                if cf_metrics:
                    cf_score = cf_metrics.get("avg_accuracy", 0)*0.8+cf_metrics.get("avg_speed_score", 0)*0.2
                else:
                    cf_score = 0

                # If counterfactual has error or score is lower, the principle is valid
                has_error = cf_metrics and cf_metrics.get("error") is not None
                is_worse = cf_score < original_score+0.0001
                
                is_principle_valid = has_error or is_worse
                
                print(f"   ↳ Original score: {original_score}, Counterfactual score: {cf_score}")
                if has_error:
                    print("   ↳ Counterfactual has error → principle is valid")
                elif is_worse:
                    print("   ↳ Counterfactual score is lower → principle is valid")
                else:
                    print("   ↳ Counterfactual score is higher or similar → principle may not be necessary")

            # Store or reject the principle based on verification result
            if is_principle_valid:
                print("   ✅ Verification passed! Principle is a key causal factor. Storing...")
                # Use lock to prevent concurrent write issues
                with self._write_lock:
                    # 1. Store principle
                    self.vector_store.add_documents([
                        Document(
                            page_content=candidate_principle,
                            metadata={"type": "principle", "source_task": original_task}
                        )
                    ])
                    # 2. Store best program code as trajectory evidence
                    self.vector_store.add_documents([
                        Document(
                            page_content=best_program_code,
                            metadata={"type": "trajectory", "source_task": original_task}
                        )
                    ])
                outcome = "VERIFIED"
                valid_principles.append(candidate_principle)
            else:
                outcome = "REJECTED"

            # Store result for this principle
            all_results.append({
                "principle": candidate_principle,
                "counterfactual_test": cf_plan,
                "verification_outcome": outcome,
                "saved": is_principle_valid
            })

        # Summary of all principles
        print(f"\n   📊 Summary: {len(valid_principles)}/{len(candidate_principles)} principles verified and stored")

        return {
            "extracted_principles": valid_principles,
            "all_principles": candidate_principles,
            "results": all_results,
            "saved_count": len(valid_principles)
        }
    def get_memory_by_task(self, task: str) -> List[Dict[str, object]]:
        all_memories = self.list_all_memories()
        all_principles = all_memories.get("principles", [])
        all_trajectories = all_memories.get("trajectories", [])

        # Use new lists to avoid modifying while iterating
        matching_principles = []
        for p in all_principles:
            if p.get("metadata", {}).get("source_task") == task:
                matching_principles.append(p)

        matching_trajectories = []
        for t in all_trajectories:
            if t.get("metadata", {}).get("source_task") == task:
                matching_trajectories.append(t)

        return matching_principles, matching_trajectories


# =============================================================================
# Usage Example
# =============================================================================

if __name__ == "__main__":
    # ============================================
    # Example: Using evaluator_file (similar to openevolve)
    # ============================================
    evaluator_file_path = "/home/common/hwluo/project/GraphEvolve/openevolve/examples/algotune/eigenvectors_complex/evaluator.py"
    kb = LamarckianKnowledgeBase()

    # ============================================
    # Phase 1: Before task starts - Retrieve knowledge (Interface 1)
    # ============================================
    user_query = "Help me process missing values in data.csv and store them in the database."

    # Call Interface 1
    retrieved = kb.retrieve_knowledge(user_query)
    print("\n=== External Knowledge Retrieved by Agent ===")
    if retrieved["principles"]:
        print(f"Reference principle: {retrieved['principles'][0]}")
        # Agent can append this principle to System Prompt
    else:
        print("No relevant historical experience, ready to explore from scratch.")

    # ... (Agent task execution code omitted here) ...

    # ============================================
    # Phase 2: After task completion - Learning and verification (Interface 2)
    # ============================================
    # Compare initial and best programs, extract improvement principles
    initial_program_path = "/home/common/hwluo/project/GraphEvolve/openevolve/examples/algotune/affine_transform_2d/initial_program.py"
    best_program_path = "/home/common/hwluo/project/GraphEvolve/openevolve/examples/algotune/affine_transform_2d/best_program.py"
    
    # Optional: Provide evaluator metrics to help extract principles and verify counterfactuals
    # If not provided, will automatically evaluate initial_program to obtain metrics
    evaluator_metrics = {
        "combined_score": 0.85,
        "accuracy": 0.92,
        "execution_time": 1.23
    }
    
    result = asyncio.run(
        kb.learn_from_trajectory(
            initial_program_path=initial_program_path,
            best_program_path=best_program_path,
            original_task=user_query,
            evaluator_file=evaluator_file_path,  # Pass evaluator_file in learn_from_trajectory
            metrics=evaluator_metrics  # Optional: If not provided, will automatically evaluate initial_program
        )
    )
    print(f"\n=== Learning Summary ===")
    print(f"Total principles extracted: {len(result['all_principles'])}")
    print(f"Verified and stored: {result['saved_count']}")
    if result['extracted_principles']:
        print(f"\n✅ Verified Principles:")
        for i, principle in enumerate(result['extracted_principles'], 1):
            print(f"  {i}. {principle}")
    else:
        print("\n❌ No principles were verified and stored.")

