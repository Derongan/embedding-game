from dataclasses import dataclass
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_text as text  # Registers the ops.
import numpy as np
from numpy import dot
from numpy.linalg import norm


@dataclass
class ComponentOptionGroup():
    """Group of mutually exclusive component options."""

    options: tuple['ComponentOption']


@dataclass
class ComponentGenerationNode():
    """Node in component generation graph."""

    groups: tuple[ComponentOptionGroup]


@dataclass
class ComponentOption:
    """Component description, child nodes, and output components."""

    description: str
    components: tuple[str]
    child: ComponentGenerationNode = None
    embedding: np.ndarray = None


class EmbeddingPopulator:
    def __init__(self) -> None:
        hub_url = "https://tfhub.dev/google/sentence-t5/st5-base/1"
        self._encoder = hub.KerasLayer(hub_url)

    def populate(self, to_encode: dict[str, ComponentOption]) -> None:
        embeddings = self.generate_embeddings(tuple(to_encode))

        for idx, option in enumerate(to_encode):
            to_encode[option].embedding = embeddings[idx]

    def generate_embeddings(self, values: tuple[str]) -> np.ndarray:
        return self._encoder(tf.constant(values))[0]


def gen_option_dict(root: ComponentGenerationNode) -> dict[str, ComponentOption]:

    if not root:
        return {}

    if not root.groups:
        return {}

    option_dict = {}

    for group in root.groups:
        for option in group.options:
            option_dict[option.description] = option

            option_dict.update(gen_option_dict(option.child))

    return option_dict


def get_components(descriptions: tuple[str], populator: EmbeddingPopulator, root: ComponentGenerationNode) -> dict[str, tuple[str]]:
    embeddings = populator.generate_embeddings(descriptions)

    embedding_dict = {}

    for idx in range(len(descriptions)):
        embedding_dict[descriptions[idx]] = embeddings[idx]

    return _get_all_components(embedding_dict, root)


def _get_all_components(embeddings: dict[str, tuple[np.ndarray]], root: ComponentGenerationNode) -> dict[str, tuple[str]]:
    return {key: _get_components(value, root) for (key, value) in embeddings.items()}


def _get_components(embedding: np.ndarray, root: ComponentGenerationNode) -> tuple[str]:
    if not root:
        return ()

    if not root.groups:
        return ()

    components = ()
    for group in root.groups:
        if not group.options:
            continue

        type_index = np.argmax(
            [cos_sim(embedding, option.embedding) for option in group.options])

        option = group.options[type_index]

        components += option.components + \
            _get_components(embedding, option.child)

    return components


def cos_sim(a, b): return dot(a, b)/(norm(a)*norm(b))
