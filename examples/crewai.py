from crewai import Agent, Task, Crew, LLM
import neatlogs
from dotenv import load_dotenv
import os

load_dotenv()

neatlogs.init(
    api_key=os.getenv("NEATLOGS_API_KEY"),
    tags=["v3", "crewai", "demo"],
    instrumentations=["crewai", "openai"],
)

# Create the Azure LLM instance using CrewAI's LLM class
llm = LLM(
    model=os.getenv("model"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

# Define agents – now with the Azure LLM
researcher = Agent(
    role="Researcher",
    goal="Find the latest key facts on a given topic",
    backstory="You're an expert researcher who digs deep for accurate, up-to-date info.",
    allow_delegation=False,
    verbose=True,
    llm=llm,  # pass it here
)

writer = Agent(
    role="Writer",
    goal="Turn research into a clear, engaging short report",
    backstory="You're a skilled writer who makes complex topics easy to understand.",
    allow_delegation=False,
    verbose=True,
    llm=llm,  # and here
)

# Define tasks
research_task = Task(
    description="Research the latest developments in AI agents as of 2025.",
    expected_output="A bullet list of 5-7 key facts with sources.",
    agent=researcher,
)

write_task = Task(
    description="Write a 300-word report based on the researcher's facts.",
    expected_output="A concise, well-structured report with a title and summary.",
    agent=writer,
)

# Create the crew
crew = Crew(
    agents=[researcher, writer], tasks=[research_task, write_task], verbose=True
)

# Run it
result = crew.kickoff()

print("\nFinal Result:\n")
print(result)
