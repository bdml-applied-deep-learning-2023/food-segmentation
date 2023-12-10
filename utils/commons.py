import torch
import os


def save(context, path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

    state_dict = dict()

    for key in context.keys():
        if hasattr(context[key], "state_dict"):
            state_dict[key] = context[key].state_dict()
        else:
            state_dict[key] = context[key]

    torch.save(state_dict, path)


# def load(context, path, device=None):
#     state_dict = torch.load(path, map_location="cpu" if device is None else device)

#     for key in context.keys():
#         if hasattr(context[key], "load_state_dict"):
#             context[key].load_state_dict(state_dict[key])
#         else:
#             context[key] = state_dict[key]

#     print(f"Context has been loaded from {path}.")


def load(context, path, device=None):
    state_dict = torch.load(path, map_location="cpu" if device is None else device)

    for key in context.keys():
        if hasattr(context[key], "state_dict") and key in state_dict.keys():
            context_state_dict = context[key].state_dict()
            loaded_state_dict = state_dict[key]

            for name in loaded_state_dict.keys():
                if name in context_state_dict.keys():
                    context_state_dict[name] = loaded_state_dict[name]

            context[key].load_state_dict(context_state_dict)
        else:
            context[key] = state_dict[key]

    print(f"Context has been loaded from {path}.")
    
