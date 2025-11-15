from enum import Enum

from pydantic import Field, create_model


# json_schema_str = """
# {
#     "type": "object",
#     "properties": {
#         "name": {"type": "string"},
#         "age": {"type": "integer"},
#         "is_active": {"type": "boolean", "default": true}
#     },
#     "required": ["name", "age"]
# }
# """


def create_field_type(model_name, field_name, field_schema, required):
    field_type = None
    match field_schema:
        case {"enum": options}:
            field_type = Enum(field_name, {str(v): v for v in options})
        case {"type": "string"}:
            field_type = str
        case {"type": "integer"}:
            field_type = int
        case {"type": "number"}:
            field_type = float
        case {"type": "boolean"}:
            field_type = bool
        case {"type": "object", "properties": _}:
            field_type = create_pydantic_model_from_json_schema(
                field_schema.get("name", f"{model_name}__{field_name}"), field_schema
            )
        case {"type": "object"}:
            field_type = dict
        case {"type": "array"}:
            t, _ = create_field_type(model_name, field_name, field_schema["items"], True)
            field_type = list[t]
        case x:
            print(field_schema)
            raise ValueError(f"unknown schema type: {x}")

    # additional kwargs
    field_kwargs = {}
    if "maximum" in field_schema:
        field_kwargs["le"] = field_schema["maximum"]
    if "maxItems" in field_schema:
        field_kwargs["max_length"] = field_schema["maxItems"]
    if "minItems" in field_schema:
        field_kwargs["min_length"] = field_schema["minItems"]
    if field_name.startswith("_"):
        field_kwargs["alias"] = field_name

    default_value = field_schema.get("default")
    if required:
        if default_value is not None:
            return field_type, Field(default_value, **field_kwargs)
        else:
            return field_type, Field(**field_kwargs)
    return field_type, Field(default_value, **field_kwargs)


def create_pydantic_model_from_json_schema(model_name: str, schema: dict):
    model_fields = {}

    properties = schema.get("properties", {})
    required_fields = schema.get("required", [])

    for field_name, field_schema in properties.items():
        field_type, field_args = create_field_type(model_name, field_name, field_schema, required=field_name in required_fields)
        if field_name.startswith("_"):
            field_name = field_name.lstrip("_")
        model_fields[field_name] = field_type, field_args

    DynamicModel = create_model(model_name, **model_fields)
    return DynamicModel


# # Example usage
# MyDynamicModel = create_pydantic_model_from_json_schema(json.loads(json_schema_str))
#
# # Instantiate and validate
# data = {"name": "Alice", "age": 30}
# instance = MyDynamicModel(**data)
# print(instance.model_dump_json(indent=2))
