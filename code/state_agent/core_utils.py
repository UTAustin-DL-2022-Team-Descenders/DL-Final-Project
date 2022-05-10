import os.path
import torch

# StateAgent agnostic Save & Load model functions. Used in state_agent.py Match
def save_model(model, model_name="state_agent", save_path=os.path.abspath(os.path.dirname(__file__)), use_jit=False, verbose=False):
    from os import path
    if use_jit:
        model.eval()
        model_scripted = torch.jit.script(model)
        model_scripted.save(path.join(save_path, f"{model_name}.pt"))
    else: # Otherwise use Pickle
        torch.save(model.state_dict(), path.join(save_path, f"{model_name}.th"))

    if verbose:
        print(f"Saved {model_name} to {save_path}")


def load_model(model_name="state_agent", load_path=os.path.abspath(os.path.dirname(__file__)), use_jit=False, model=None, conversion = None, verbose=False):

    try:
        if use_jit:
            model = torch.jit.load(os.path.join(load_path, f"{model_name}.pt"))
            model.eval()
        else: # Otherwise use Pickle. Need to use model_class for this
            loaded = torch.load(os.path.join(load_path, f"{model_name}.th"))
            if conversion:
                loaded = conversion(loaded)
            model.load_state_dict(loaded)
            model.eval()

        if verbose:
            print("Loaded pre-existing network from", load_path)
        return model
    except FileNotFoundError as e:
        sys.exit(f"Problem loading model: {e.strerror}")
    except ValueError as e:
        raise e

class DictObj:
    def __init__(self, in_dict:dict):
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
               setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val])
            else:
               setattr(self, key, DictObj(val) if isinstance(val, dict) else val)