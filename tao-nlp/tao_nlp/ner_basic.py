TEXT = (
    "I live in New York, U.S.A. Here are my details"
    " andres.nava.alonso@protonmail.com, +34987270490. Calle peñalabra 2, 1A "
    "24008 Leon. A transition-based named entity recognition component. "
    "The entity recognizer identifies non-overlapping labelled spans of tokens. "
    "The transition-based algorithm used encodes certain assumptions that are "
    "effective for “traditional” named entity recognition tasks, but may not "
    "be a good fit, but this is not part of Rocscience Inc"
)


def sample_spacy():
    from typing import List

    import spacy

    nlp = spacy.load("en_core_web_md")

    def extract_nouns_from_text(text: str) -> List[str]:
        """Returns list of nouns from provided text."""

        text_processed = nlp(text=text)

        for entity in text_processed.ents:
            print(entity, entity.label_, entity.vector.shape, entity.vector[:3])
        nouns = [
            token.text
            for token in text_processed
            if token.pos_ == "NOUN" or token.pos_ == "PROPN"
        ]

        return nouns

    extract_nouns_from_text(text=TEXT)
    extract_nouns_from_text(text=TEXT.lower())


def sample_transformers():
    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

    model_name = "Davlan/distilbert-base-multilingual-cased-ner-hrl"
    # model_name = "dslim/bert-base-NER"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    nlp = pipeline("ner", model=model, tokenizer=tokenizer)

    ner_result_list = nlp(TEXT)

    for ner_result in ner_result_list:
        print(ner_result)


def sample_regex_email():
    import re

    email_list = re.findall("([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", TEXT)

    for email in email_list:
        print(email)


def main():
    sample_spacy()
    sample_transformers()
    sample_regex_email()


if __name__ == "__main__":
    main()
