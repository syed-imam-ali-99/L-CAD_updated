import json


def load_image_caption_pairs(path):
    """Load image-caption pairs from either list-pair or mapping JSON format."""
    with open(path, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict):
        return list(data.items())

    if isinstance(data, list):
        pairs = []
        for item in data:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                pairs.append((item[0], item[1]))
            elif isinstance(item, dict):
                img_name = item.get('image') or item.get('img') or item.get('file_name') or item.get('filename')
                caption = item.get('caption') or item.get('text') or item.get('prompt')
                if img_name is None or caption is None:
                    raise ValueError(f"Invalid pair object in {path}: {item}")
                pairs.append((img_name, caption))
            else:
                raise ValueError(f"Invalid pair entry in {path}: {item}")
        return pairs

    raise ValueError(f"Unsupported pairs JSON format in {path}: {type(data).__name__}")
