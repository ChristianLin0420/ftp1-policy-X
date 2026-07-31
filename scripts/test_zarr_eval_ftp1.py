import ast
import pathlib


def test_evaluation_entrypoint_has_no_debugger_traps() -> None:
    source_path = pathlib.Path(__file__).with_name("zarr_eval_ftp1_pytorch.py")
    tree = ast.parse(source_path.read_text())

    debugger_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "breakpoint":
            debugger_calls.append(node.lineno)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "set_trace":
            debugger_calls.append(node.lineno)

    assert debugger_calls == []
