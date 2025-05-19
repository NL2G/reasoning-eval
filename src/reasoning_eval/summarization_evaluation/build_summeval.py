import sienna

from .utils import DATA_BASE_DIR


def parse_story_file(content):
    """
    Remove article highlights and unnecessary white characters.
    """
    content_raw = content.split("@highlight")[0]
    content = " ".join(filter(None, [x.strip() for x in content_raw.split("\n")]))
    return content


if __name__ == "__main__":
    annotation_data = sienna.load(DATA_BASE_DIR / "model_annotations.aligned.jsonl")

    for sample in annotation_data:
        fpath = DATA_BASE_DIR / sample["filepath"]
        text = "\n".join(sienna.load(fpath))
        text = parse_story_file(text)
        sample["src"] = text

    sienna.save(
        annotation_data, DATA_BASE_DIR / "model_annotations.aligned.paired.jsonl"
    )
