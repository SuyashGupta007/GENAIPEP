# MovieSage AI bot
# 1 take a raw para about a movie
# 2 Extract important sturtured info
# 3 Generate a clean summary
# 4 Returns it into JSON Format

from dotenv import load_dotenv

load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

model = ChatMistralAI(model="mistral-small-2603")

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are MovieSage AI, an expert movie analyst and information extraction assistant.

```
    Your responsibilities:

    1. Carefully read and understand the entire movie description.
    2. Extract all important movie-related information.
    3. Generate a concise and engaging plot summary.
    4. Identify key characters, themes, and notable facts.
    5. Extract factual information only from the provided text.
    6. If information is not available, return null.
    7. Never hallucinate or invent details.
    8. Return ONLY valid JSON.
    9. Do not include markdown, explanations, comments, or additional text.

    Extract the following fields:

    - title
    - genre
    - director
    - writers
    - producers
    - cast
    - release_year
    - runtime
    - language
    - country
    - plot_summary
    - main_characters
    - themes
    - notable_facts
    - awards
    - box_office
    - rating
    - keywords

    
    """,
        ),
        (
            "human",
            """
    Analyze the following movie description and extract all relevant information.

    Movie Description:
    {movie_description}
    """,
        ),
    ]
)

para = input("give your paragraph")
final_prompt = prompt.invoke({"movie_description": para})

res = model.invoke(final_prompt)
print(res.content)
