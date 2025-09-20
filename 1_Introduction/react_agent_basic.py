import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import initialize_agent, tool
from langchain_community.tools import TavilySearchResults


load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash")

# tool 1
search_tool = TavilySearchResults(search_depth="basic")

# tool 2
@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Returns the current system time formatted according to the given format string.
    """
    current_time = datetime.datetime.now()
    return current_time.strftime(format)


agent = initialize_agent(
    tools = [search_tool, get_system_time], 
    llm=llm, 
    verbose=True, 
    agent="zero-shot-react-description"
)

agent.invoke("When was GPT-5 launched and how many days ago was that from this instant")
# result = llm.invoke("give me a fact about dogs")

# print(result)