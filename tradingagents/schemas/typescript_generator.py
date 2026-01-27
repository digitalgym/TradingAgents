"""
Generate TypeScript types from Pydantic schemas.

Usage:
    python -m tradingagents.schemas.typescript_generator > web/frontend/src/types/schemas.ts

Or programmatically:
    from tradingagents.schemas.typescript_generator import generate_typescript
    ts_code = generate_typescript()
"""

from typing import Dict, Any, List, Set
from enum import Enum


def json_type_to_ts(
    schema: Dict[str, Any], defs: Dict[str, Any], indent: int = 0
) -> str:
    """Convert JSON schema type to TypeScript type."""
    # Handle $ref
    if "$ref" in schema:
        ref = schema["$ref"].split("/")[-1]
        return ref

    # Handle allOf (Pydantic uses this for inheritance)
    if "allOf" in schema:
        # Usually just one item pointing to a ref
        types = [json_type_to_ts(s, defs, indent) for s in schema["allOf"]]
        return " & ".join(types)

    # Handle anyOf (union types, including Optional)
    if "anyOf" in schema:
        types = []
        for s in schema["anyOf"]:
            if s.get("type") == "null":
                types.append("null")
            else:
                types.append(json_type_to_ts(s, defs, indent))
        return " | ".join(types)

    # Handle const (literal types)
    if "const" in schema:
        val = schema["const"]
        if isinstance(val, str):
            return f'"{val}"'
        return str(val)

    # Handle enum
    if "enum" in schema:
        return " | ".join(f'"{v}"' for v in schema["enum"])

    schema_type = schema.get("type")

    # Handle array of types (e.g., ["string", "null"])
    if isinstance(schema_type, list):
        types = []
        for t in schema_type:
            if t == "null":
                types.append("null")
            else:
                types.append(json_type_to_ts({"type": t}, defs, indent))
        return " | ".join(types)

    if schema_type == "string":
        return "string"
    elif schema_type == "number" or schema_type == "integer":
        return "number"
    elif schema_type == "boolean":
        return "boolean"
    elif schema_type == "null":
        return "null"
    elif schema_type == "array":
        items = schema.get("items", {})
        item_type = json_type_to_ts(items, defs, indent)
        return f"{item_type}[]"
    elif schema_type == "object":
        if "properties" in schema:
            # Inline object definition
            lines = ["{"]
            props = schema.get("properties", {})
            required = set(schema.get("required", []))
            for prop_name, prop_schema in props.items():
                ts_type = json_type_to_ts(prop_schema, defs, indent + 2)
                optional = "" if prop_name in required else "?"
                lines.append(f"{'  ' * (indent + 1)}{prop_name}{optional}: {ts_type};")
            lines.append(f"{'  ' * indent}}}")
            return "\n".join(lines)
        elif "additionalProperties" in schema:
            val_type = json_type_to_ts(schema["additionalProperties"], defs, indent)
            return f"Record<string, {val_type}>"
        return "Record<string, unknown>"

    return "unknown"


def schema_to_typescript(
    name: str, schema: Dict[str, Any], defs: Dict[str, Any]
) -> str:
    """Convert a JSON schema to TypeScript interface or type."""
    lines = []

    # Add description as JSDoc comment
    if "description" in schema:
        lines.append(f"/** {schema['description']} */")

    # Handle enums
    if "enum" in schema:
        values = " | ".join(f'"{v}"' for v in schema["enum"])
        lines.append(f"export type {name} = {values};")
        return "\n".join(lines)

    # Handle objects (interfaces)
    if schema.get("type") == "object" or "properties" in schema:
        lines.append(f"export interface {name} {{")

        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        for prop_name, prop_schema in properties.items():
            ts_type = json_type_to_ts(prop_schema, defs, 1)
            optional = "" if prop_name in required else "?"

            # Add description as inline comment
            if "description" in prop_schema:
                lines.append(f"  /** {prop_schema['description']} */")

            lines.append(f"  {prop_name}{optional}: {ts_type};")

        lines.append("}")
        return "\n".join(lines)

    # Fallback: type alias
    ts_type = json_type_to_ts(schema, defs, 0)
    lines.append(f"export type {name} = {ts_type};")
    return "\n".join(lines)


def generate_typescript() -> str:
    """Generate TypeScript definitions from all schemas."""
    from tradingagents.schemas import get_all_schemas

    schemas = get_all_schemas()

    output_lines = [
        "// Auto-generated TypeScript types from TradingAgents Pydantic schemas",
        "// Do not edit manually - regenerate with:",
        "//   python -m tradingagents.schemas.typescript_generator > web/frontend/src/types/schemas.ts",
        "",
    ]

    # Process enums first
    enum_names: Set[str] = set()
    for schema_class in schemas:
        if isinstance(schema_class, type) and issubclass(schema_class, Enum):
            name = schema_class.__name__
            values = " | ".join(f'"{m.value}"' for m in schema_class)
            output_lines.append(f"export type {name} = {values};")
            output_lines.append("")
            enum_names.add(name)

    # Process Pydantic models
    for schema_class in schemas:
        if isinstance(schema_class, type) and issubclass(schema_class, Enum):
            continue  # Already processed

        if not hasattr(schema_class, "model_json_schema"):
            continue

        name = schema_class.__name__
        json_schema = schema_class.model_json_schema()
        defs = json_schema.get("$defs", {})

        # Generate the main interface
        ts_code = schema_to_typescript(name, json_schema, defs)
        output_lines.append(ts_code)
        output_lines.append("")

    return "\n".join(output_lines)


def generate_to_file(output_path: str) -> None:
    """Generate TypeScript definitions and write to file."""
    ts_code = generate_typescript()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(ts_code)
    print(f"Generated TypeScript types to: {output_path}")


if __name__ == "__main__":
    print(generate_typescript())
