import ast
import inspect
import textwrap
from pathlib import Path


def inline(func):
    # grab function codes
    source = inspect.getsource(func)
    # remove indentation
    source = textwrap.dedent(source)
    tree = ast.parse(source)
    
    # extract function body visitor
    class InlineTransformer(ast.NodeTransformer):
        def __init__(self, func_name):
            self.func_name = func_name
            self.function_body = None
        
        def visit_FunctionDef(self, node):
            if node.name == self.func_name:
                self.function_body = node.body
            return node
        
        def visit_Call(self, node):
            # make inline
            if isinstance(node.func, ast.Name) and node.func.id == self.func_name:
                return ast.Module(
                    body=self.function_body,
                    type_ignores=[]
                )
            return node
    
    transformer = InlineTransformer(func.__name__)
    
    def wrapper(*args, **kwargs):
        frame = inspect.currentframe()
        calling_frame = frame.f_back
        local_vars = calling_frame.f_locals
        
        # bind local variables
        params = inspect.signature(func).parameters
        for param, arg in zip(params.values(), args):
            local_vars[param.name] = arg
        for k, v in kwargs.items():
            local_vars[k] = v
        
        # run function body directly
        result = None
        for stmt in transformer.function_body:
            if isinstance(stmt, ast.Return):
                result = eval(compile(ast.Expression(stmt.value), '<string>', 'eval'), globals(), local_vars)
                break
            else:
                exec(compile(ast.Module([stmt], type_ignores=[]), '<string>', 'exec'), globals(), local_vars)
        
        return result
    
    return wrapper


def inline_module(file_path: str):
    """Decorator to inline an entire module"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Find project root directory
            project_root = Path(file_path).parent
            
            class ModuleInliner(ast.NodeTransformer):
                def __init__(self):
                    self.inlined_modules = set()
                    self.root = project_root
                
                def inline_module_content(self, module_name: str) -> list:
                    if module_name in self.inlined_modules:
                        return []
                    
                    # Create full path
                    module_path = self.root / f"{module_name}.py"
                    
                    if not module_path.exists():
                        print(f"Warning: Cannot find module {module_path}")
                        return []
                    
                    # Read module source code
                    module_source = module_path.read_text()
                    module_tree = ast.parse(module_source)
                    
                    # Recursively process imports in this module
                    transformed = self.visit(module_tree)
                    self.inlined_modules.add(module_name)
                    
                    return transformed.body
                
                def visit_Import(self, node):
                    replacements = []
                    for name in node.names:
                        # Only inline local modules
                        if (self.root / f"{name.name}.py").exists():
                            module_content = self.inline_module_content(name.name)
                            replacements.extend(module_content)
                    
                    return replacements if replacements else node
                
                def visit_ImportFrom(self, node):
                    if node.module and (self.root / f"{node.module}.py").exists():
                        module_content = self.inline_module_content(node.module)
                        
                        # Handle 'from module import specific_names'
                        if node.names[0].name != '*':
                            wanted_names = {alias.name: alias.asname or alias.name 
                                         for alias in node.names}
                            filtered_content = []
                            
                            for item in module_content:
                                if (isinstance(item, (ast.FunctionDef, ast.ClassDef)) and 
                                    item.name in wanted_names):
                                    if wanted_names[item.name] != item.name:
                                        item.name = wanted_names[item.name]
                                    filtered_content.append(item)
                            return filtered_content
                        
                        return module_content
                    return node

            # Get original function source
            source = inspect.getsource(func)
            tree = ast.parse(source)
            
            # Perform inline transformation
            inliner = ModuleInliner()
            transformed = inliner.visit(tree)
            
            # Print inlined source for debugging
            print("=== Inlined Source ===")
            print(ast.unparse(transformed))
            print("=====================")
            
            # Compile and execute transformed code
            module = ast.Module(body=transformed.body, type_ignores=[])
            code = compile(ast.fix_missing_locations(module), '<string>', 'exec')
            
            # Execute in new namespace
            namespace = {}
            exec(code, globals(), namespace)
            
            # Execute original function
            return namespace[func.__name__](*args, **kwargs)
            
        return wrapper
    return decorator
