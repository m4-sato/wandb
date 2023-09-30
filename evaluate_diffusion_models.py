from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
import numpy as np
from utilities import *

import wandb

wandb.login(anonymous="allow")

MODEL_ARTIFACT = "dlai-course/model-registry/SpriteGen:latest"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

config = SimpleNamespace(
    num_samples = 30, 
    timesteps = 500,
    beta1 = 1e-4,
    beta2 = 0.02,
    height = 16,
    ddim_n = 25
)

def load_model(model_artifact_name):
    api = wandb.Api()
    artifact = api.artifact(model_artifact_name, type="model")
    model_path = Path(artifact.download())
    
    producer_run = artifact.logged_by()
    
    model_weights = torch.load(model_path/"context_model.pth", map_location="cpu")
    
    model = ContextUnet(in_channels=3,
                        n_feat=producer_run.config["n_feat"],
                        n_cfeat=producer_run.config["n_cfeat"],
                        height=producer_run.config["height"])
    
    model.load_state_dict(model_weights)
    
    model.eval()
    
    return model.to(DEVICE)

nn_model = load_model(MODEL_ARTIFACT)

_, sample_ddpm_context = setup_ddpm(config.beta1,
                                    config.beta2,
                                    config.timesteps,
                                    DEVICE)

noises = torch.randn(config.num_samples, 3, config.height, config.height).to(DEVICE)

ctx_vector = F.one_hot(torch.tensor([0, 0, 0, 0, 0, 0,
                                    1, 1, 1, 1, 1, 1,
                                    2, 2, 2, 2, 2, 2,
                                    3, 3, 3, 3, 3, 3,
                                    4, 4, 4, 4, 4, 4]),
                        5).to(DEVICE).float()

sample_ddim_context= setup_ddim(config.beta1,
                                config.beta2,
                                config.timesteps,
                                DEVICE)

ddpm_samples, _ = sample_ddpm_context(nn_model, noises, ctx_vector)

ddim_samples, _ = sample_ddim_context(nn_model, noises, ctx_vector, n=config.ddim_n)

table = wandb.Table(columns=["input_noise", "ddpm", "ddim", "class"])

for noise, ddpm_s, ddim_s, c in zip(noises, ddpm_samples, ddim_samples, to_classes(ctx_vector)):
    table.add_data(wandb.Image(noise), wandb.Image(ddpm_s), wandb.Image(ddim_s), c)

with wandb.init(project="dlai_sprite_diffusion", job_type= "samplers_battle", config=config):
    wandb.log({"samples_table": table})
