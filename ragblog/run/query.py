import os

from src.const import CONST
from src.logger_custom import LOGGER
from src.rag import Rag


def main():

    rag = Rag(is_ready_vector_db=False)
    response = rag.query(
        question="""
Describe the relation between Helena and Alejandra.
Consider the author's diverse experiences and multifaceted personality,
reflecting on traits that are evident across their various blog posts.
Provide a detailed and thoughtful response.
Ensure your answer is profound and sufficiently long, 
offering deep insights and personal reflections.
"""
    )
    LOGGER.info(f"{response=}")

    output_path = os.path.join(CONST.loc.data, "output.md")
    LOGGER.info(f"{output_path}")
    os.makedirs(CONST.loc.data, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"# Response\n\n{response}")


if __name__ == "__main__":
    main()
