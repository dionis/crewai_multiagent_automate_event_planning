# Rename `os.environ` to `env` for nicer code
from os import environ as env
from crewai import Agent, Task, Crew, LLM, Process
from dotenv import load_dotenv,find_dotenv
import markdown
from langchain_google_genai import ChatGoogleGenerativeAI
from util import get_serper_api_key

from crewai_tools import DirectoryReadTool, \
                         FileReadTool, \
                         SerperDevTool

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


load_dotenv('config/.env')



# Rename `os.environ` to `env` for nicer code

print('GOOGLE_API_KEY:  {}'.format(env['GOOGLE_API_KEY']))

GEMINI_API_KEY = env['GOOGLE_API_KEY']

env["GEMINI_API_KEY"] = env['GOOGLE_API_KEY']
MODEL_NAME = "models/text-embedding-004"

env["SERPER_API_KEY"] = env['SERPER_KEY']


llm = LLM(
    model = "gemini/gemini-1.5-pro-latest",
    temperature = 0.7
)


from crewai_tools import ScrapeWebsiteTool, SerperDevTool

# Initialize the tools
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# Agent 1: Venue Coordinator

# Venue = Evento

venue_coordinator = Agent(
    role="Venue Coordinator",
    goal="Identify and book an appropriate venue "
    "based on event requirements",
    tools=[search_tool, scrape_tool],
    verbose = True,
    backstory=(
        "With a keen sense of space and "
        "understanding of event logistics, "
        "you excel at finding and securing "
        "the perfect venue that fits the event's theme, "
        "size, and budget constraints."
    ),
    llm = llm
)


 # Agent 2: Logistics Manager
logistics_manager = Agent(
    role='Logistics Manager',
    goal=(
        "Manage all logistics for the event "
        "including catering and equipment"
    ),
    tools=[search_tool, scrape_tool],
    verbose=True,
    backstory=(
        "Organized and detail-oriented, "
        "you ensure that every logistical aspect of the event "
        "from catering to equipment setup "
        "is flawlessly executed to create a seamless experience."
    ),
    llm = llm
)

# Agent 3: Marketing and Communications Agent
marketing_communications_agent = Agent(
    role="Marketing and Communications Agent",
    goal="Effectively market the event and "
         "communicate with participants",
    tools=[search_tool, scrape_tool],
    verbose=True,
    backstory=(
        "Creative and communicative, "
        "you craft compelling messages and "
        "engage with potential attendees "
        "to maximize event exposure and participation."
    ),
    llm = llm
)

#########################################################
# Creating Venue Pydantic Object
#
#     Create a class VenueDetails using Pydantic BaseModel.
#     Agents will populate this object with information about
#     different venues (events) by creating different instances of it.
##########################################################


from pydantic import BaseModel
# Define a Pydantic model for venue details
# (demonstrating Output as Pydantic)
class VenueDetails(BaseModel):
    name: str
    address: str
    capacity: int
    booking_status: str


#####################################################
#
#  Creating Tasks
#
#     By using output_json, you can specify the structure
#     of the output you want.
#     By using output_file, you can get your output in a file.
#     By setting human_input=True, the task will ask for human
#     feedback (whether you like the results or not) before finalising it.
#
##############################################################

venue_task = Task(
    description="Find a venue in {event_city} "
                "that meets criteria for {event_topic}.",
    expected_output="All the details of a specifically chosen"
                    "venue you found to accommodate the event.",
    human_input=True,
    output_json=VenueDetails,
    output_file="venue_details.json",
      # Outputs the venue details as a JSON file
    agent=venue_coordinator
)

##################################################
#
# By setting async_execution=True, it means the task can run in parallel with the tasks which come after it.
#
##############################################

logistics_task = Task(
    description="Coordinate catering and "
                 "equipment for an event "
                 "with {expected_participants} participants "
                 "on {tentative_date}.",
    expected_output="Confirmation of all logistics arrangements "
                    "including catering and equipment setup.",
    human_input=True,
    async_execution=True,
    agent=logistics_manager
)

marketing_task = Task(
    description="Promote the {event_topic} "
                "aiming to engage at least"
                "{expected_participants} potential attendees.",
    expected_output="Report on marketing activities "
                    "and attendee engagement formatted as markdown.",
    async_execution= False,
    output_file="marketing_report.md",  # Outputs the report as a text file
    agent=marketing_communications_agent
)

#####
#
#  Creating the Crew
#
#  Note: Since you set async_execution=True for logistics_task
#  and marketing_task tasks, now the order for them does not
#  matter in the tasks list.
#####

# Define the crew with agents and tasks
event_management_crew = Crew(
    agents=[venue_coordinator,
            logistics_manager,
            marketing_communications_agent],

    tasks=[venue_task,
           logistics_task,
           marketing_task],

    verbose=True
)

#Running the Crew

event_details = {
    'event_topic': "Tech Innovation Conference",
    'event_description': "A gathering of tech innovators "
                         "and industry leaders "
                         "to explore future technologies.",
    'event_city': "San Francisco",
    'tentative_date': "2024-09-15",
    'expected_participants': 500,
    'budget': 20000,
    'venue_type': "Conference Hall"
}

###############################
#
# Note 2:
#
#     Since you set human_input=True for some tasks, the execution will ask for
#     your input before it finishes running.
#     When it asks for feedback, use your mouse pointer to first
#     click in the text box before typing anything.
#
##################################

result = event_management_crew.kickoff( inputs = event_details)

##Display the generated venue_details.json file.
#
#

# import json
# from pprint import pprint
#
# with open('venue_details.json') as f:
#    data = json.load(f)
#
# pprint(data)
