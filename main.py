from component_gen import ComponentGenerationNode, ComponentOption, ComponentOptionGroup, EmbeddingPopulator, gen_option_dict, get_components


if __name__ == "__main__":
    colors = ["red", "yellow", "green", "blue",
              "orange", "black", "white", "purple", "grey", "silver", "gold", "brown"]

    color_options = ComponentOptionGroup(tuple(ComponentOption(
        f'colored {color}', (color,)) for color in colors))

    weapon_type_options = ComponentOptionGroup((
        ComponentOption(
            "a melee weapon such as a sword or mace", ("melee",)),
        ComponentOption(
            "a ranged weapon such as a bow or thrown spear", ("ranged",))))

    entity_type_options = ComponentOptionGroup((
        ComponentOption("a tool or weapon used for fighting",
                        ("weapon",),  ComponentGenerationNode((weapon_type_options,))),
        ComponentOption("a magical spell", ("spell",))))

    root = ComponentGenerationNode((entity_type_options, color_options))

    option_dict = gen_option_dict(root)

    populator = EmbeddingPopulator()

    populator.populate(option_dict)

    weapons = ["a freezing red bolt of cold", "firebolt",
               "a firey magma sword", "a sharpened stick", "a dark powerful longbow", "a pitch-black deadly sharp dagger"]

    print(get_components(weapons, populator, root))
