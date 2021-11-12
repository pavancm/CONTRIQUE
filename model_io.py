import os
import torch

def save_model(args, model, optimizer):
    out = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.current_epoch))

    # To save a DataParallel model generically, save the model.module.state_dict().
    # This way, you have the flexibility to load the model any way you want to any device you want.
    if args.nodes > 1:
        torch.save(model.module.state_dict(), out)
    else:
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), out)
        else:
            torch.save(model.state_dict(), out)