# main.py
# Updated to work with CrewAI and a simple SQLite logger for history.

import os
import sqlite3
import uuid
from datetime import datetime
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Attempt to import the crew from crew_ai_setup.py
# This file should contain your CrewAI agents, tasks, and the instantiated crew.
try:
    from agentworkflow import crew # Assuming your crew object is named 'crew'
    CREW_AI_ENABLED = True
    if crew is None:
        print("WARNING: CrewAI crew object is None after import. Check crew_ai_setup.py for errors (e.g., API keys).")
        CREW_AI_ENABLED = False
except ImportError:
    print("ERROR: Could not import 'crew' from crew_ai_setup.py.")
    print("Ensure crew_ai_setup.py exists and defines the 'crew' object.")
    CREW_AI_ENABLED = False
    crew = None # Placeholder if import fails
except Exception as e:
    print(f"ERROR: An unexpected error occurred during import from crew_ai_setup.py: {e}")
    CREW_AI_ENABLED = False
    crew = None


# --- SQLite Database for Conversation History Logging ---
DATABASE_NAME = ":memory:"

def get_db_connection():
    """Gets a new database connection."""
    # Using check_same_thread=False for simplicity in FastAPI,
    # but for production, consider a more robust connection pooling strategy.
    conn = sqlite3.connect(DATABASE_NAME, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database and creates the conversation_log table."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_input TEXT NOT NULL,
                agent_response TEXT
            )
        """)
        conn.commit()
    except sqlite3.Error as e:
        print(f"SQLite error during init_db: {e}")
    finally:
        if conn:
            conn.close()

def add_to_log(thread_id: str, user_input: str, agent_response: str):
    """Adds a user input and agent response to the log."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO conversation_log (thread_id, user_input, agent_response)
            VALUES (?, ?, ?)
        """, (thread_id, user_input, agent_response))
        conn.commit()
    except sqlite3.Error as e:
        print(f"SQLite error in add_to_log for thread {thread_id}: {e}")
    finally:
        if conn:
            conn.close()

def get_log_history(thread_id: str, limit: int = 20) -> list:
    """Retrieves the last N log entries for a given thread_id."""
    history = []
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, thread_id, timestamp, user_input, agent_response
            FROM conversation_log
            WHERE thread_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (thread_id, limit))
        history = [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as e:
        print(f"SQLite error in get_log_history for thread {thread_id}: {e}")
    finally:
        if conn:
            conn.close()
    return history[::-1] # Return in chronological order


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("FastAPI startup: Initializing conversation log database...")
    init_db()
    if not CREW_AI_ENABLED or not crew:
        print("WARNING: CrewAI is not properly configured. API endpoints might not function correctly.")
        print("Please ensure OPENAI_API_KEY (or other LLM keys) are set and crew_ai_setup.py is correct.")
    else:
        print("CrewAI integration appears to be enabled.")
    yield


# --- FastAPI App Initialization ---
api = FastAPI(
    title="CrewAI Agent API",
    description="API to interact with a CrewAI-powered agent.",
    version="0.1.0-crewai",
    lifespan=lifespan,
)

# --- Request and Response Models ---
class QueryRequest(BaseModel):
    input: str
    thread_id: str = None # Optional: client can provide, or we generate one

class AgentResponse(BaseModel):
    thread_id: str
    final_output: str # CrewAI kickoff usually returns the final task's output

class HistoryEntry(BaseModel):
    id: int
    thread_id: str
    timestamp: datetime
    user_input: str
    agent_response: str = None

class HistoryResponse(BaseModel):
    thread_id: str
    history: list[HistoryEntry]


# --- API Endpoints ---
@api.post("/invoke_agent", response_model=AgentResponse)
async def invoke_agent_endpoint(request: QueryRequest, x_thread_id: str = Header(None)):
    """
    Endpoint to invoke the CrewAI agent.
    """
    if not CREW_AI_ENABLED or not crew:
        raise HTTPException(
            status_code=503, 
            detail="CrewAI service is not available or not configured correctly. Check server logs and OPENAI_API_KEY."
        )

    thread_id = request.thread_id or x_thread_id or str(uuid.uuid4())
    
    # Inputs for CrewAI kickoff must match what tasks expect.
    # Assuming the first task in your crew_ai_setup.py expects 'user_input'.
    crew_inputs = {"user_input": request.input} 
    
    try:
        print(f"Kicking off CrewAI for thread_id: {thread_id} with input: {request.input[:50]}...")
        # Note: crew.kickoff() can be blocking. For long tasks, consider background tasks.
        crew_result = crew.kickoff(inputs=crew_inputs)
        
        if crew_result is None:
            # This might happen if the crew doesn't produce a string output or fails silently
            print(f"WARNING: CrewAI kickoff for thread {thread_id} returned None.")
            crew_result_str = "Agent did not produce a textual output."
        else:
            crew_result_str = str(crew_result) # Ensure it's a string

        # Log the interaction
        add_to_log(thread_id, request.input, crew_result_str)
            
        return AgentResponse(
            thread_id=thread_id,
            final_output=crew_result_str
        )

    except Exception as e:
        print(f"Error during CrewAI kickoff for thread {thread_id}: {e}")
        import traceback
        traceback.print_exc()
        # Log the input attempt even if agent fails
        add_to_log(thread_id, request.input, f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interacting with CrewAI agent: {str(e)}")

@api.get("/session_history/{thread_id}", response_model=HistoryResponse)
async def get_history_endpoint(thread_id: str):
    """
    Endpoint to retrieve conversation history for a given thread_id from the log.
    """
    if not thread_id:
        raise HTTPException(status_code=400, detail="thread_id path parameter is required.")
    
    raw_history = get_log_history(thread_id)
    # Pydantic validation for each entry
    validated_history = [HistoryEntry(**entry) for entry in raw_history]
        
    return HistoryResponse(thread_id=thread_id, history=validated_history)

@api.get("/")
async def read_root():
    message = "Welcome to the CrewAI Agent API."
    if not CREW_AI_ENABLED or not crew:
        message += " WARNING: CrewAI is not properly configured. Please check server logs and API key (e.g., OPENAI_API_KEY) setup."
    return {"message": message}

# To run this FastAPI app locally (for testing):
# Ensure crew_ai_setup.py is in the same directory and OPENAI_API_KEY is set.
# uvicorn main:api --reload
#
# Example curl requests:
# 1. Invoke agent (thread_id will be generated if not provided):
#    curl -X POST "http://127.0.0.1:8000/invoke_agent" -H "Content-Type: application/json" -d '{"input": "Hello from FastAPI with CrewAI"}'
#
# 2. Invoke agent with a specific thread_id in body:
#    curl -X POST "http://127.0.0.1:8000/invoke_agent" -H "Content-Type: application/json" -d '{"input": "Another message for CrewAI", "thread_id": "my-crewai-thread-123"}'
#
# 3. Get history for a thread:
#    curl -X GET "http://127.0.0.1:8000/session_history/my-crewai-thread-123"
