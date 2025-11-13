from enum import Enum

from pydantic import create_model

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


def create_field_type(model_name, field_name, field_schema):
    field_type = None
    match field_schema:
        case {"enum": options}:
            field_type = Enum(field_name, {v: v for v in options})
        case {"type": "string"}:
            field_type = str
        case {"type": "integer"}:
            field_type = int
        case {"type": "number"}:
            field_type = float
        case {"type": "boolean"}:
            field_type = bool
        case {"type": "object"}:
            field_type = create_pydantic_model_from_json_schema(
                field_schema.get("name", f"{model_name}__{field_name}"), field_schema
            )
        case {"type": "array"}:
            field_type = list[create_field_type(model_name, field_name, field_schema["items"])]
        case x:
            print(field_schema)
            raise ValueError(f"unknown schema type: {x}")
    return field_type


def create_pydantic_model_from_json_schema(model_name: str, schema: dict):
    model_fields = {}

    properties = schema.get("properties", {})
    required_fields = schema.get("required", [])

    for field_name, field_schema in properties.items():
        field_type = create_field_type(model_name, field_name, field_schema)
        default_value = field_schema.get("default")

        if field_name in required_fields:
            if default_value is not None:
                model_fields[field_name] = (field_type, default_value)
            else:
                model_fields[field_name] = field_type
        else:
            model_fields[field_name] = (field_type, default_value)

    DynamicModel = create_model(model_name, **model_fields)
    return DynamicModel


# # Example usage
# MyDynamicModel = create_pydantic_model_from_json_schema(json.loads(json_schema_str))
#
# # Instantiate and validate
# data = {"name": "Alice", "age": 30}
# instance = MyDynamicModel(**data)
# print(instance.model_dump_json(indent=2))
