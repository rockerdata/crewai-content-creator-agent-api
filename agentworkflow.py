# crew_ai_setup.py
# This file defines the CrewAI agents, tasks, and crew.
# It replaces the previous LangGraph agent definition.

import os
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool # If you need custom tools later


from crewai import LLM
import os

os.environ['AZURE_API_KEY']="CHA388HVZvU25sV8uP7QtgByh4JEKhtWpBT5uGJgUSagpn2UumMvJQQJ99BDAC77bzfXJ3w3AAABACOGVZPr"
os.environ['AZURE_API_BASE']="https://vidvatta-openai.openai.azure.com/"
os.environ['AZURE_API_VERSION']="2024-12-01-preview"
os.environ['AZURE_OPENAI_DEPLOYMENT_NAME']="vidvatta-35"

llm = LLM(
    model="azure/vidvatta-35",
    # api_version="2023-05-15"
)

# 1. Define Agents
# We'll create two agents, similar to the two nodes in the previous LangGraph setup.

# Agent 1: Processes the initial input
input_processor_agent = Agent(
    role='Input Analysis Specialist',
    goal='Analyze and understand the user\'s input, extracting key information or rephrasing it for clarity if needed. Prepare it for response generation.',
    backstory=(
        "You are an expert at dissecting user queries. Your primary function is to "
        "understand the core intent of the input and prepare a concise summary or "
        "the essential parts of it that the next agent will use to formulate a response. "
        "You do not generate the final answer yourself but ensure the groundwork is perfectly laid."
    ),
    verbose=True,
    allow_delegation=False, # This agent works alone on its task
    llm=llm, # Assign the LLM
    # tools=[MyCustomTool()] # Add tools if any
)

# Agent 2: Generates the final response
response_generator_agent = Agent(
    role='Content Generation Expert',
    goal='Generate a helpful and relevant final response to the user based on the processed input provided by the Input Analysis Specialist.',
    backstory=(
        "You are a skilled content creator, adept at crafting clear and concise answers. "
        "You take the analyzed input from your colleague and formulate a final response. "
        "Your aim is to be directly helpful to the user."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm, # Assign the LLM
)

# 2. Define Tasks
# Tasks will correspond to the work done by each agent.

process_input_task = Task(
    description=(
        "Take the user's original input: '{user_input}'. "
        "Analyze it, understand its core meaning, and if necessary, rephrase it or extract the most critical components. "
        "Your output should be a clear, processed version of the input, ready for the response generation stage."
    ),
    expected_output=(
        "A string containing the processed version of the input. For example, if the input is complex, "
        "this might be a summary or a key question extracted. If simple, it might be the input itself with a note that it's clear."
    ),
    agent=input_processor_agent,
)

generate_response_task = Task(
    description=(
        "Using the processed input from the 'Input Analysis Specialist', " # Context from previous task is implicitly passed
        "craft a final, user-facing response. The processed input will be provided to you. "
        "Ensure your response directly addresses the user's original query based on this processed information."
    ),
    expected_output=(
        "A string containing the final answer to be presented to the user."
    ),
    agent=response_generator_agent,
    # context=[process_input_task] # Explicitly define task dependencies if needed, though CrewAI often infers
)

# 3. Create the Crew

try:


    crew = Crew(
        agents=[input_processor_agent, response_generator_agent],
        tasks=[process_input_task, generate_response_task],
        process=Process.sequential,  # Tasks will be executed one after another
        verbose=True  # 0 for no logs, 1 for basic, 2 for detailed
        # memory=True # Enable memory for the crew (uses CrewAI's built-in memory)
        # manager_llm=llm # For hierarchical process, if you have a manager agent
    )
except ImportError:
    print("langchain_openai not installed. `pip install langchain-openai`")
    crew = None
except Exception as e: # Catches error if API key is not set for ChatOpenAI
    print(f"Error initializing LLM or Crew: {e}")
    print("Please ensure OPENAI_API_KEY is set in your environment variables if using OpenAI.")
    crew = None


# Example of how to run the Crew (for local testing)
if __name__ == "__main__":
    # if not os.getenv("OPENAI_API_KEY"):
    #     print("--------------------------------------------------------------------------------")
    #     print("WARNING: OPENAI_API_KEY is not set. CrewAI execution will likely fail.")
    #     print("Please set this environment variable with your OpenAI API key to run the example.")
    #     print("Example: export OPENAI_API_KEY='sk-...'")
    #     print("--------------------------------------------------------------------------------")

    if crew:
        print("\n--- CrewAI Agent System ---")
        user_query = "Tell me about CrewAI and how it handles complex tasks."
        
        inputs = {"user_input": user_query}
        
        print(f"\nKicking off crew with input: {user_query}")
        
        # Kick off the crew's execution
        # The result will be the output of the last task in the sequential process
        try:
            result = crew.kickoff(inputs=inputs)
            
            print("\n--- Crew Execution Finished ---")
            print("Final Result from Crew:")
            print(result)

            # The 'result' is typically the output of the last task.
            # To see intermediate steps, you rely on the verbose logging (verbose=2 for crew)
            # or by inspecting task outputs if you structure your tasks to save them explicitly.
            # For example, task.output.raw_output after execution if you run tasks individually.

        except Exception as e:
            print(f"An error occurred during crew execution: {e}")
            print("This might be due to missing API keys or LLM configuration issues.")
    else:
        print("Crew could not be initialized. Please check LLM configuration and API keys.")

