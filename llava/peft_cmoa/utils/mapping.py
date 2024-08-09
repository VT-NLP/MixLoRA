from transformers import PreTrainedModel
from peft.utils.config import PeftConfig

from .peft_model import PeftModel, PeftModelForCausalLM

from ..cmoa_lora import CMoALoraConfig

MODEL_TYPE_TO_PEFT_MODEL_MAPPING = {
    "CAUSAL_LM": PeftModelForCausalLM,
}

PEFT_TYPE_TO_CONFIG_MAPPING = {
    "LORA": CMoALoraConfig,
}


def get_peft_model(model: PreTrainedModel, peft_config: PeftConfig, adapter_name: str = "default") -> PeftModel:
    """
    Returns a Peft model object from a model and a config.

    Args:
        model ([`transformers.PreTrainedModel`]): Model to be wrapped.
        peft_config ([`PeftConfig`]): Configuration object containing the parameters of the Peft model.
    """
    model_config = getattr(model, "config", {"model_type": "custom"})
    if hasattr(model_config, "to_dict"):
        model_config = model_config.to_dict()

    peft_config.base_model_name_or_path = model.__dict__.get(
        "name_or_path", None)

    return MODEL_TYPE_TO_PEFT_MODEL_MAPPING[peft_config.task_type](model, peft_config, adapter_name=adapter_name)
