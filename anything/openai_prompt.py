import os

import openai as openai
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


MODEL = "gpt-3.5-turbo"
MODEL_NOT_YET_AVAILABLE = "gpt-4"


def main():
    authenticate()
    # generate_image()

    prompt = build_prompt()
    print(prompt)

    response = response_prompt(prompt=prompt)
    print(response)


def authenticate():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = openai_api_key
    # print(openai.Model.list())


def generate_image():
    response = openai.Image.create(
        prompt=(
            "As they climbed, the world seemed to shift beneath their feet. "
            "The mountain revealed itself to be more than just a peak; it was a "
            "gateway to another realm, a place where the borders between reality "
            "and dream blurred like watercolors."
        ),
        n=1,
        size="1024x1024",
    )
    image_url = response["data"][0]["url"]
    print(image_url)


def build_prompt_summarize(text: str) -> str:
    return f"""Summarize the text delimited by ''' text.

Extract the info into this structure:
- Highlight
- index of sections
- list of short insights
- takeaway
- short takeaway, less than 240 characters
- list of most representative sentences
- tweet
Format the output as markdown. Use H2 for the sections.

Here comes the text:

'''
{text}
'''
"""


def build_prompt_connect(text_list: str) -> str:
    sep = "\n---\n"
    return f"""Find the underlying connection for the list of texts. Each is delimited by {sep} .

Extract the info into this structure:
- list of individual takeaways for each text
- takeaway of the connection among them
- short of this connection, less than 240 characters

Format the output as markdown.

Here comes the list of texts:

'''
{sep.join(text_list)}
'''
"""


def build_prompt():
    query = "what are the advantages of active learning vs weak supervision"

    results = """
    🌐 Weak Supervision: A New Programming Paradigm for Machine Learning ...
ai.stanford.edu
› blog › weak-supervision
October 3, 2019 - In active learning, the goal is to make use of SMEs more efficiently 
by having them label data points which are estimated to be most valuable to the model 
(for a good survey, see (Settles 2012)). In the standard supervised learning setting, 
this means selecting new data points to be labeled.

🌐 ML Techniques: Active Learning vs Weak Supervision
appen.com
› blog › ml-techniques-active-learning-vs-weak-supervision
March 19, 2021 - With all of these challenges in mind, teams launching artificial 
intelligence (AI) solutions turn away from fully supervised learning (which requires 
complete, hand-labeled datasets for training ML models) to active learning and weak 
supervision. The latter learning techniques are generally faster and less 
labor-intensive while still capable of training models successfully.

🌐 Weak Supervision: The New Programming Paradigm for Machine Learning ...
dawn.cs.stanford.edu
› 2017 › 07 › 16 › weak-supervision
July 16, 2017 - In active learning, the goal is to make use of subject matter 
experts more efficiently by having them label data points which are estimated 
to be most valuable to the model. Traditionally, applied to the standard supervised 
learning setting, this means selecting new data points to be labeled–for example, 
we might select mammograms that lie close to the current model decision boundary, 
and ask radiologists to label only these.

🌐 Active learning and weak supervision in NLP projects
fwdays.com
› en › event › data-science-fwdays-2020 › review › 
active-learning-and-weak-supervision-in-nlp-projects
Successful artificial intelligence solutions always require a massive amount of 
high-quality labeled data. In most cases, we don’t have a large and qualitative 
labeled set together. Weak supervision and active learning tools may help you 
optimize the labeling process and address the shortage of data labels."""

    return f"""Rank the relevance of the search results to the provided query. 
    Rank using natural numbers from 0 to n, being 0 the top result.
- query:
'''{query}'''

- results
'''
{results}
'''
"""


def response_prompt(prompt: str) -> str:
    completion = openai.ChatCompletion.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}]
    )

    return completion.choices[0].message["content"]


if __name__ == "__main__":
    main()
