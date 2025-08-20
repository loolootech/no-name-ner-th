import gradio as gr
from transformers import pipeline

# Load NER model. Device -1 means CPU.
# If you have a GPU, you can set device=0 or the appropriate GPU index.
ner_model = pipeline("token-classification", model="loolootech/no-name-ner-th", device=-1)

ENTITY_TO_ANONYMIZED_TOKEN_MAP = {
    "PERSON": "[PERSON]",
    "PHONE": "[PHONE]",
    "EMAIL": "[EMAIL]",
    "ADDRESS": "[LOCATION]",
    "DATE": "[DATE]",
    "NATIONAL_ID": "[NATIONAL_ID]",
    "HOSPITAL_IDS": "[HOSPITAL_IDS]",
}

def anonymize_text(original_text):
    """
    Anonymizes sensitive entities in the input text using a named entity recognition (NER) model.

    Parameters
    ----------
    original_text : str
        The text that may contain sensitive information such as names, phone numbers, emails, etc.

    Returns
    -------
    list
        A list containing:
        - original_text: the original input string
        - anonymized_text: the text with specified entities replaced by tokens
        - anonymized_entities: a list of dictionaries for each anonymized entity, containing
        the original word and its entity label

    Notes
    -----
    - Requires `ner_model` to be defined and initialized (e.g., a HuggingFace NER pipeline).
    - The mapping `ENTITY_TO_ANONYMIZED_TOKEN_MAP` defines which entity types will be replaced
    and what token will be used.
    """

    # Step 1: Perform NER on the input text
    ner_results = ner_model(original_text)

    # Step 2: Combine overlapping or adjacent entities of the same type
    combined_entities = []
    for entity in ner_results:
        # Normalize entity label (e.g., "B-PERSON" -> "PERSON")
        entity_name = entity['entity'].split('-')[-1]
        entity['entity'] = entity_name

        # Add as new entity if list is empty, different type, or non-overlapping
        if not combined_entities or combined_entities[-1]['entity'] != entity_name or \
            combined_entities[-1]['start'] + len(combined_entities[-1]['word']) < entity['start']:
            combined_entities.append(entity)
        else:
            # Merge adjacent/overlapping entities of the same type
            combined_entities[-1]['word'] += ' ' + entity['word']
            combined_entities[-1]['end'] = entity['end']

    # Step 3: Filter entities that should be anonymized
    entities_to_anonymize = [
        e for e in combined_entities if e['entity'] in ENTITY_TO_ANONYMIZED_TOKEN_MAP.keys()
    ]

    # Step 4: Sort entities in reverse order of start index to safely replace them
    entities_to_anonymize.sort(key=lambda x: x['start'], reverse=True)

    # Step 5: Replace each entity in the text with the corresponding anonymized token
    anonymized_text = original_text
    for entity in entities_to_anonymize:
        start, end = entity['start'], entity['end']
        token = ENTITY_TO_ANONYMIZED_TOKEN_MAP.get(entity['entity'])
        anonymized_text = anonymized_text[:start] + token + anonymized_text[end:]

    # Step 6: Return original text, anonymized text, and information about redacted entities
    return [
        original_text,
        anonymized_text,
        [{"word": e["word"], "label": e["entity"]} for e in entities_to_anonymize]
    ]


with gr.Blocks(title="Thai Clinical Conversation De-identification") as demo:
    gr.HTML(
        """
        <div style="text-align: center;">
            <h1 style="font-size: 3em;">Thai Clinical Conversation De-identification</h1>
            <p style="font-size: 1.2em;">Paste Thai clinical or personal text below to redact sensitive info.</p>
        </div>
        """
    )
# Use a gr.Row with gr.Column spacers to center the image
    with gr.Row():
        gr.Column(scale=1) # Left spacer
        gr.Image(
            value="assets/mascot-image-landscape.png", # Replace with your image URL/path
            width=200, # Set a smaller width for the image
            show_label=False,
            container=False # Prevent the image from being wrapped in a default Gradio container
        )
        gr.Column(scale=1) # Right spacer

    # Add the main interface components
    gr.Interface(
        fn=anonymize_text,
        inputs=gr.Textbox(lines=10, label="Input Text"),
        outputs=[
            gr.Textbox(label="Original Text"),
            gr.Textbox(label="Anonymized Text"),
            gr.JSON(label="Entities")
        ],
        live=False,  # Set live=False since we are using Blocks now
    )

if __name__ == "__main__":
    demo.launch(share=False)
