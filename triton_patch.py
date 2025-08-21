"""
Aggressive triton compatibility patch - import this FIRST before any transformers imports
"""
import sys

class TritonStub:
    """Mock module that returns False for any triton check"""
    def __getattr__(self, name):
        if 'triton' in name.lower():
            return lambda *args, **kwargs: False
        raise AttributeError(f"module '{self.__class__.__name__}' has no attribute '{name}'")

# Pre-emptively add stub modules to sys.modules
sys.modules['triton'] = TritonStub()

# Monkey patch transformers before it's imported
def patch_transformers():
    import importlib
    import transformers
    import transformers.utils
    
    # Create stub function
    def triton_stub(*args, **kwargs):
        return False
    
    # Patch all possible locations
    modules_to_patch = [
        transformers,
        transformers.utils,
    ]
    
    # Try to patch import_utils if it exists
    try:
        import transformers.utils.import_utils
        modules_to_patch.append(transformers.utils.import_utils)
    except:
        pass
    
    # All possible triton function names (including typos)
    triton_names = [
        'is_triton_available',
        'is_triton_kernels_available', 
        'is_triton_kernels_availalble',  # typo version
        '_is_triton_available',
        '_is_triton_kernels_available',
        '_is_triton_kernels_availalble',
    ]
    
    for module in modules_to_patch:
        for name in triton_names:
            setattr(module, name, triton_stub)
            
    # Also patch in the module's __dict__ directly
    for module in modules_to_patch:
        for name in triton_names:
            module.__dict__[name] = triton_stub
    
    print("✅ Triton compatibility patches applied")

# Apply patches
patch_transformers()